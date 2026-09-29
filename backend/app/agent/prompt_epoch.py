"""Prompt-cache epochs: when the cached prefix is cold, and what a turn sees.

Anthropic's prompt cache keeps a prefix for a fixed lifetime after it was
last touched. Once a user has been quiet for longer than that, the next turn
rewrites the whole prefix (tools, system, history) at the cache-write rate,
whatever it contains. An *epoch* is the run of turns that share one cached
prefix: it opens on a cold start and lasts until the next one.

Two things are keyed to epochs, each behind its own setting:

- ``prompt_stable_prefix_enabled``: the workspace documents in the system
  block (SOUL.md, USER.md, MEMORY.md) are a snapshot taken when the epoch
  opens, so the system block stays byte-identical for every turn in it. An
  edit made mid-epoch reaches the model once, as a delta on the current turn
  (``system_prompt.render_workspace_updates``), and stays on that turn's
  message in every later history of the epoch, where it is cached. It folds
  into the snapshot at the next cold start. See :class:`WorkspaceTurn`.
- ``cold_start_compaction_enabled``: the history before the epoch's first
  message is rebuilt to a budget, once, when the epoch opens. Old results lose
  any SKILL.md guidance delivered on them, old results of calls that only
  read become stubs (a write's result stays, as it may be the only record of
  an ID), and only if the rest is still over budget are the
  oldest turns dropped (and compacted into memory, as a trim would). The
  rebuild is a pure function of the stored rows, their timestamps and the
  turn's tool schema, so every later turn in the epoch renders the same bytes
  and the history only grows by appending. See :func:`build_history_view`.

The mid-session trim (``core.assemble_prompt``) sheds history the same way
before it drops whole turns: old read results become stubs, and the row the
stubbing stops at is stored as ``sessions.history_stub_seq``, which every
later render applies. See :func:`shed_history_view`.

The definition of a cold start lives in one place, :func:`is_cold_gap`, and
is computed from message timestamps rather than from process state. That is
what lets the model-comparison replay rebuild the cold-start history a live
turn would have seen from the transcript alone. The trim's stub seq is the
exception: it is session state written by an earlier turn, and the replay
does not reconstruct it (see ``model_comparison.sampling``).
"""

from __future__ import annotations

import datetime
import json
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from backend.app.agent.context import (
    _advance_trim_watermark_only,
    _set_history_stub_seq,
    _stored_messages_to_agent_messages,
    trigger_compaction_for_dropped,
)
from backend.app.agent.dto import StoredMessage
from backend.app.agent.messages import (
    AgentMessage,
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from backend.app.agent.skills.loader import strip_skill_guidance
from backend.app.agent.system_prompt import (
    WorkspaceSnapshot,
    append_workspace_updates,
    render_workspace_updates,
)
from backend.app.agent.tools.base import Tool, is_mutating_call
from backend.app.config import settings
from backend.app.enums import MessageDirection
from backend.app.services.llm_service import breakpoint_ttls

logger = logging.getLogger(__name__)

# Anthropic's two cache lifetimes. The threshold follows the history
# breakpoint's unless ``prompt_cache_idle_seconds`` pins it.
_CACHE_TTLS = {"1h": datetime.timedelta(hours=1), "5m": datetime.timedelta(minutes=5)}

# Calibrated on production prompts rather than the trimmer's four: tool
# results are JSON, IDs and URLs, and on the current Claude tokenizer a
# captured 179k-character history billed as about 96k tokens. Erring low on
# characters per token errs towards a smaller rebuilt history.
_CHARS_PER_TOKEN = 2

# A result this short costs about as much as its stub, and short results are
# the confirmations ("Saved.", "Event created: id 123") most worth keeping.
_ELIDE_MIN_CHARS = 300

# When the prose alone is over budget, drop turns until the rebuilt history is
# at this fraction of the budget, not just under it. Without the headroom the
# next cold start is over budget again by one turn, and every cold start
# would fire a compaction call and rewrite MEMORY.md.
_DROP_TO_FRACTION = 0.6

# Bound on the process-local snapshot store, as for the other per-user caches.
_SNAPSHOT_STORE_MAX = 1024


def cache_idle_threshold() -> datetime.timedelta:
    """How long the history cache survives without a call touching it.

    The lifetime of the history-tail breakpoint (``llm_cache_history_ttl``,
    capped at the tools and system lifetime as the request caps it). Once it
    lapses the next turn rewrites the history whatever it contains, so that
    turn is where an epoch opens.
    """
    if settings.prompt_cache_idle_seconds is not None:
        return datetime.timedelta(seconds=settings.prompt_cache_idle_seconds)
    return _CACHE_TTLS[breakpoint_ttls().history]


def is_cold_gap(previous: datetime.datetime | None, current: datetime.datetime) -> bool:
    """True when a turn at *current* finds the cache from *previous* expired.

    *previous* is when this conversation's last agent turn touched the cache
    (see :func:`find_epoch`); None means there was no such turn. This is the
    only definition of a cold start; everything else asks it.
    """
    if previous is None:
        return True
    return current - previous > cache_idle_threshold()


def _parse_ts(raw: str) -> datetime.datetime | None:
    try:
        parsed = datetime.datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=datetime.UTC)


def _touches_cache(rows: list[StoredMessage], index: int) -> bool:
    """Whether *rows[index]* marks an agent turn that read or wrote the cache.

    An inbound row starts a turn, and the reply that follows it ends one, so
    its timestamp is when the turn's last LLM call returned. An outbound row
    that does not follow an inbound is a heartbeat: its call carries no
    conversation history, so it leaves the history cache as it found it.
    """
    row = rows[index]
    if row.direction == MessageDirection.INBOUND:
        return True
    return index > 0 and rows[index - 1].direction == MessageDirection.INBOUND


@dataclass(frozen=True)
class PromptEpoch:
    """The cache epoch a turn belongs to.

    ``key`` is the seq of the row that opened it and is the same for every
    turn in the epoch. ``cold_start`` is True for the turn that opened it,
    which is the turn that pays for rewriting the prefix. ``turn_seq`` is
    the seq of the row the turn answers, where a workspace update delivered
    on this turn is kept (see :class:`WorkspaceTurn`). It is about the turn,
    not the epoch, so it takes no part in equality.
    """

    key: int
    cold_start: bool
    turn_seq: int | None = field(default=None, compare=False)


def find_epoch(rows: list[StoredMessage], current: StoredMessage) -> tuple[PromptEpoch, int]:
    """Locate the epoch *current* belongs to.

    *rows* is the history window in seq order, without *current*. Returns
    the epoch and the index into ``rows + [current]`` of the row that opened
    it: the latest inbound row that arrived after a cold gap. Rows before that
    index are the history the epoch inherited; rows from it onward were
    appended during the epoch. With no cold gap in the window the whole window
    is one epoch and nothing before it is visible.

    A row with an unparseable timestamp never opens an epoch. Treating it as
    warm keeps the prefix stable, which is the failure worth having.
    """
    seq = [*rows, current]
    boundary = 0
    for index in range(len(seq) - 1, -1, -1):
        if seq[index].direction != MessageDirection.INBOUND:
            continue
        at = _parse_ts(seq[index].timestamp)
        if at is None:
            continue
        prev_index = next((b for b in range(index - 1, -1, -1) if _touches_cache(seq, b)), None)
        previous: datetime.datetime | None = None
        if prev_index is not None:
            previous = _parse_ts(seq[prev_index].timestamp)
            if previous is None:
                continue
        if is_cold_gap(previous, at):
            boundary = index
            break
    # The epoch is still opening while no reply has been written inside it:
    # a batch of inbound rows is answered by one turn, after the last of them.
    cold_start = all(
        seq[index].direction == MessageDirection.INBOUND for index in range(boundary, len(seq))
    )
    return (
        PromptEpoch(key=seq[boundary].seq, cold_start=cold_start, turn_seq=current.seq),
        boundary,
    )


@dataclass
class HistoryView:
    """The history one turn sends, and what rebuilding it cost."""

    messages: list[AgentMessage]
    epoch: PromptEpoch
    # Rows removed for the budget. The caller compacts them into memory.
    dropped_rows: list[StoredMessage] = field(default_factory=list)
    elided_results: int = 0


def _estimate_tokens(messages: list[AgentMessage]) -> int:
    chars = 0
    for m in messages:
        if isinstance(m, SystemMessage | UserMessage | ToolResultMessage):
            chars += len(m.content or "")
        elif isinstance(m, AssistantMessage):
            chars += len(m.content or "")
            for tc in m.tool_calls:
                chars += len(tc.name) + len(json.dumps(tc.arguments, default=str))
    return chars // _CHARS_PER_TOKEN


def _group_turns(rows: list[StoredMessage]) -> list[list[StoredMessage]]:
    """Split *rows* into turns, each starting at an inbound row.

    Rows before the first inbound (a heartbeat reply at the top of the
    window) form a turn of their own.
    """
    groups: list[list[StoredMessage]] = []
    for row in rows:
        if row.direction == MessageDirection.INBOUND or not groups:
            groups.append([row])
        else:
            groups[-1].append(row)
    return groups


def elided_result_stub(tool_name: str, chars: int) -> str:
    """What an old tool result is replaced with in a rebuilt history."""
    # Not "re-run the tool": the call above the stub may have been a write
    # (created an event, sent an email), and repeating it would repeat the
    # action. Only a lookup is safe to redo.
    return f"[tool result elided: {tool_name}, {chars} chars; re-run only if it just reads]"


def _is_write(tools_by_name: Mapping[str, Tool], name: str, args: dict[str, Any]) -> bool:
    """Whether a stored call's result must stay verbatim in a rebuilt history.

    A write's result is often the only place the record it made is named
    ("created invoice 643"), and a stub would invite the model to guess the
    ID or repeat the write. A tool missing from this turn's schema, or a
    result with no call above it, cannot be classified and counts as a write.
    """
    tool = tools_by_name.get(name)
    return tool is None or is_mutating_call(tool, args)


def _render(
    pre_groups: list[list[StoredMessage]],
    post: list[StoredMessage],
    tz_name: str,
    verbatim_from: int,
    preceding: Sequence[StoredMessage],
    tools_by_name: Mapping[str, Tool],
    stub_before: int | None = None,
) -> tuple[list[AgentMessage], int, int]:
    """Render the rows, eliding read results in turns before *verbatim_from*.

    Rows with a seq below *stub_before* are treated the same way wherever
    they sit, inherited or appended during the epoch: that is where a
    mid-session trim left its stubs (see :func:`shed_history_view`).

    Only results of calls that just read are elided. A write's result stays
    verbatim at any age and size, error or not (see :func:`_is_write`), and
    counts towards the tokens returned, so the budget steps see it.

    Every result in those turns, read or write, first loses any SKILL.md block
    delivered on it (``skills.loader.strip_skill_guidance``). Guidance on a
    write would otherwise stay forever. Once stripped, no marker for the
    category is left in the history, so the agent delivers the guidance again
    on the category's next use, into a row appended during the epoch. Turns
    inside the verbatim window keep theirs: it is recent and may be in use.

    Rendered in one pass so timestamp markers read the same as they would
    without the rebuild. *preceding* is the rows just before the first one
    rendered, including any turns the rebuild dropped, so the first kept row
    renders as it will on the next turn, when the dropped rows sit below the
    trim watermark. Returns the messages, the estimated tokens of the
    inherited part, and how many results were elided.
    """
    group_of: dict[int, int] = {}
    for g, rows in enumerate(pre_groups):
        for row in rows:
            group_of[row.seq] = g
    flat = [row for rows in pre_groups for row in rows]
    rendered = _stored_messages_to_agent_messages(
        [*flat, *post], tz_name=tz_name, preceding=preceding
    )

    out: list[AgentMessage] = []
    pre_part: list[AgentMessage] = []
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    group: int | None = None
    # The row a result belongs to: results carry no seq of their own and
    # follow the assistant message of the row that stored them.
    row_seq: int | None = None
    elided = 0
    for m in rendered:
        if isinstance(m, UserMessage | AssistantMessage) and m.seq is not None:
            group = group_of.get(m.seq)
            row_seq = m.seq
        if isinstance(m, AssistantMessage):
            for tc in m.tool_calls:
                calls[tc.id] = (tc.name, tc.arguments)
        old = (group is not None and group < verbatim_from) or (
            stub_before is not None and row_seq is not None and row_seq < stub_before
        )
        if isinstance(m, ToolResultMessage) and old:
            # Guidance is not the result, and re-delivers on the category's
            # next use, so it goes from reads and writes alike.
            content = strip_skill_guidance(m.content)
            if content != m.content:
                m = ToolResultMessage(
                    tool_call_id=m.tool_call_id, content=content, is_error=m.is_error
                )
            name, args = calls.get(m.tool_call_id, (None, {}))
            if (
                len(m.content) > _ELIDE_MIN_CHARS
                and name is not None
                and not _is_write(tools_by_name, name, args)
            ):
                m = ToolResultMessage(
                    tool_call_id=m.tool_call_id,
                    content=elided_result_stub(name, len(m.content)),
                    is_error=m.is_error,
                )
                elided += 1
        out.append(m)
        if group is not None:
            pre_part.append(m)
    return out, _estimate_tokens(pre_part), elided


def build_history_view(
    rows: list[StoredMessage],
    current: StoredMessage | None,
    tz_name: str,
    *,
    compact: bool,
    tools_by_name: Mapping[str, Tool],
    preceding: Sequence[StoredMessage] = (),
    stub_before: int | None = None,
) -> HistoryView:
    """Render the history for the turn answering *current*.

    *preceding* is the rows just before *rows* (see
    ``context._stored_messages_to_agent_messages``). They render nothing and
    only seed the timestamp markers, so the first row renders the same bytes
    whether it is mid-history or the first row after the trim watermark.

    *tools_by_name* is the turn's tool schema, read only to tell a call that
    reads from one that writes (``tools.base.is_mutating_call``).

    With *compact* False this is the plain rendering, and only the epoch is
    worked out. With it True the history the epoch inherited is rebuilt:

    1. In turns older than the last ``cold_start_verbatim_turns``, every
       tool result loses any SKILL.md guidance delivered on it, and results
       of read calls become stubs. Each stub keeps its
       ``tool_use_id``, so every ``tool_use`` still has its result. Write
       results are never stubbed: they may be the only record of an ID.
    2. While the rebuilt history is over ``cold_start_history_budget_tokens``,
       the verbatim window shrinks, one turn at a time, to nothing. Read
       results can be fetched again; prose and write results cannot.
    3. While it is still over budget, or holds more than
       ``context_trim_target_turns`` turns, the oldest turns are dropped
       whole until it is at ``_DROP_TO_FRACTION`` of the budget. The caller
       compacts them into memory, write results included. Kept write results
       count towards the budget, so they are what drives the drop when they
       alone exceed it.

    Rows appended during the epoch are never touched, and the rebuild reads
    nothing but the rows, so it renders the same bytes on every turn of the
    epoch. After a drop the next turn loads fewer rows, finds them within
    budget and drops nothing more.

    *stub_before* is ``sessions.history_stub_seq``, where a mid-session trim
    stubbed (:func:`shed_history_view`). With *compact* True, results in rows
    below it lose their guidance and read results become stubs, as in step 1,
    whether or not an epoch opened since. It is persisted, so every turn
    after the trim applies it the same way.
    """
    if current is None:
        return HistoryView(
            messages=_stored_messages_to_agent_messages(rows, tz_name=tz_name, preceding=preceding),
            epoch=PromptEpoch(key=0, cold_start=True),
        )
    epoch, boundary = find_epoch(rows, current)
    if not compact or (boundary == 0 and stub_before is None):
        return HistoryView(
            messages=_stored_messages_to_agent_messages(rows, tz_name=tz_name, preceding=preceding),
            epoch=epoch,
        )
    if boundary == 0:
        messages, _, elided = _render(
            [],
            rows,
            tz_name,
            verbatim_from=0,
            preceding=preceding,
            tools_by_name=tools_by_name,
            stub_before=stub_before,
        )
        return HistoryView(messages=messages, epoch=epoch, elided_results=elided)

    groups = _group_turns(rows[:boundary])
    post = rows[boundary:]
    budget = settings.cold_start_history_budget_tokens
    max_turns = settings.context_trim_target_turns

    def before(first: int) -> list[StoredMessage]:
        """The rows ahead of ``groups[first:]``: *preceding*, then the dropped turns."""
        return [*preceding, *(row for group in groups[:first] for row in group)]

    def fit_verbatim(first: int) -> tuple[list[AgentMessage], int, int]:
        """Steps 1 and 2: the widest verbatim window that fits the budget."""
        kept = groups[first:]
        verbatim = min(settings.cold_start_verbatim_turns, len(kept))
        while True:
            messages, tokens, elided = _render(
                kept,
                post,
                tz_name,
                verbatim_from=len(kept) - verbatim,
                preceding=before(first),
                tools_by_name=tools_by_name,
                stub_before=stub_before,
            )
            if tokens <= budget or verbatim == 0:
                return messages, tokens, elided
            verbatim -= 1

    start = 0
    messages, tokens, elided = fit_verbatim(0)
    if tokens > budget or len(groups) > max_turns:
        # Step 3. Every read result is already a stub here (write results
        # never are), so drop from the front with no verbatim window,
        # re-rendering after each drop so the check reads the bytes that
        # will be sent. Always keep the newest turn.
        drop_to = int(budget * _DROP_TO_FRACTION)
        while start < len(groups) - 1 and (tokens > drop_to or len(groups) - start > max_turns):
            start += 1
            _, tokens, _ = _render(
                groups[start:],
                post,
                tz_name,
                verbatim_from=len(groups),
                preceding=before(start),
                tools_by_name=tools_by_name,
                stub_before=stub_before,
            )
        # Then widen the verbatim window again as far as the budget allows.
        # This is the same computation the next turn runs on the rows that
        # remain, so the result is a fixed point: it drops nothing more and
        # renders the same bytes.
        messages, tokens, elided = fit_verbatim(start)

    dropped = [row for group in groups[:start] for row in group]
    return HistoryView(messages=messages, epoch=epoch, dropped_rows=dropped, elided_results=elided)


@dataclass
class ShedView:
    """What a mid-session trim sends in place of the history it was given."""

    view: HistoryView
    # The new ``sessions.history_stub_seq``: read results below it are stubs.
    stub_before: int


def shed_history_view(
    rows: list[StoredMessage],
    current: StoredMessage,
    tz_name: str,
    *,
    tools_by_name: Mapping[str, Tool],
    fits: Callable[[list[AgentMessage]], bool],
    preceding: Sequence[StoredMessage] = (),
    stub_before: int | None = None,
) -> ShedView:
    """Shrink the history for a mid-session trim, lookups before prose.

    The trim in ``core.assemble_prompt`` calls this when the prompt crosses
    ``context_trim_trigger_tokens``, instead of dropping whole turns first.
    *fits* is the trimmer's test of whether a rendered history brings the
    prompt within its target. The steps mirror the cold-start rebuild's
    (:func:`build_history_view`), with *fits* as the budget:

    1. Read results in turns older than the last ``cold_start_verbatim_turns``
       become stubs and lose their SKILL.md guidance. Write results stay
       verbatim.
    2. While that does not fit, the verbatim window shrinks, one turn at a
       time, to nothing. Read results can be fetched again; prose and write
       results cannot.
    3. Only if every read result is a stub and it still does not fit are the
       oldest turns dropped, as few as fit (the newest is always kept), and
       the verbatim window then widens again as far as *fits* allows. The
       caller compacts the dropped turns into memory and advances the trim
       watermark past them.

    The first row of the verbatim window (or *current*, with no window) is
    the returned ``stub_before``, which the caller persists. It never moves
    back past the *stub_before* given.

    Every candidate is rendered by :func:`build_history_view` with its
    ``stub_before`` and the rows that would remain, which is the call every
    later turn makes once the stub seq and the watermark are stored. So the
    history this returns is the one the next turns render, byte for byte,
    and they grow it only by appending.
    """
    groups = _group_turns(rows)

    def stub_line(start: int, verbatim: int) -> int:
        """The stub seq that leaves the last *verbatim* of ``groups[start:]`` whole."""
        line = groups[len(groups) - verbatim][0].seq if verbatim else current.seq
        return line if stub_before is None else max(line, stub_before)

    def render(start: int, verbatim: int) -> HistoryView:
        dropped = [row for group in groups[:start] for row in group]
        view = build_history_view(
            [row for group in groups[start:] for row in group],
            current,
            tz_name,
            compact=True,
            tools_by_name=tools_by_name,
            preceding=[*preceding, *dropped],
            stub_before=stub_line(start, verbatim),
        )
        # A cold-start rebuild inside the render may drop more from the front.
        view.dropped_rows = [*dropped, *view.dropped_rows]
        return view

    def widest(start: int) -> tuple[HistoryView, int] | None:
        """Steps 1 and 2 over ``groups[start:]``: the widest window that fits."""
        for verbatim in range(min(settings.cold_start_verbatim_turns, len(groups) - start), -1, -1):
            view = render(start, verbatim)
            if fits(view.messages):
                return view, stub_line(start, verbatim)
        return None

    found = widest(0)
    last = len(groups) - 1
    if found is None and last >= 1:
        # Step 3. Fewer turns never render longer, so search for the fewest
        # drops that fit with every read stubbed, rather than re-rendering
        # the whole window once per turn.
        low, high = 1, last
        while low < high:
            middle = (low + high) // 2
            if fits(render(middle, 0).messages):
                high = middle
            else:
                low = middle + 1
        found = widest(low)
        if found is None:
            # Even the newest turn alone does not fit: it is what is kept.
            found = render(low, 0), stub_line(low, 0)
    if found is None:
        # A single turn that does not fit even with its reads stubbed.
        found = render(0, 0), stub_line(0, 0)
    view, new_stub = found
    return ShedView(view=view, stub_before=new_stub)


@dataclass
class ShedHistory:
    """A trim's replacement history, and the turns it dropped (rendered)."""

    messages: list[AgentMessage]
    dropped: list[AgentMessage]


# The hook ``core.assemble_prompt`` calls to shed history when the prompt is
# over the trim trigger. It takes the trimmer's fit test and returns the new
# history, or None when there is nothing it can shed.
HistoryShedder = Callable[[Callable[[list[AgentMessage]], bool]], Awaitable[ShedHistory | None]]


class EpochHistoryRenderer:
    """The live loop's history renderer. Records the epoch it found.

    Passed to ``load_conversation_history`` as its ``render`` hook, so row
    selection (the trim watermark, the window limit, overflow compaction)
    stays where it is and only the rendering changes. A compacting renderer
    also serves the turn's trim: see :meth:`shed`.

    With *persist* False it renders and sheds exactly as the live loop does
    but writes nothing: no compaction, no watermark, no stub seq. That is the
    model-comparison replay's renderer, which must not change the session it
    reads.
    """

    def __init__(
        self,
        user_id: str,
        *,
        compact: bool,
        tools_by_name: Mapping[str, Tool] | None = None,
        stub_before: int | None = None,
        persist: bool = True,
    ) -> None:
        # The rebuild needs the turn's tools to tell reads from writes. Without
        # them every call would count as a write and nothing would be elided,
        # so a compacting renderer must be given them.
        if compact and tools_by_name is None:
            raise ValueError("a compacting EpochHistoryRenderer needs the turn's tools_by_name")
        self._user_id = user_id
        self._compact = compact
        self._tools_by_name: Mapping[str, Tool] = tools_by_name or {}
        # ``sessions.history_stub_seq`` as the turn loaded it.
        self._stub_before = stub_before
        self._persist = persist
        self.epoch: PromptEpoch | None = None
        # What the last render kept: rows, current row, timezone, and the
        # rows ahead of them (dropped ones included). :meth:`shed` starts here.
        self._rendered: (
            tuple[list[StoredMessage], StoredMessage, str, list[StoredMessage]] | None
        ) = None

    async def __call__(
        self,
        rows: list[StoredMessage],
        current: StoredMessage | None,
        tz_name: str,
        preceding: Sequence[StoredMessage] = (),
    ) -> list[AgentMessage]:
        view = build_history_view(
            rows,
            current,
            tz_name,
            compact=self._compact,
            tools_by_name=self._tools_by_name,
            preceding=preceding,
            stub_before=self._stub_before,
        )
        self.epoch = view.epoch
        if self._compact and view.epoch.cold_start:
            logger.info(
                "Cold start for user %s (epoch %d): %d tool result(s) elided, %d row(s) dropped",
                self._user_id,
                view.epoch.key,
                view.elided_results,
                len(view.dropped_rows),
            )
        await self._compact_dropped(view.dropped_rows, tz_name, preceding)
        if current is not None:
            kept = rows[len(view.dropped_rows) :]
            self._rendered = (kept, current, tz_name, [*preceding, *view.dropped_rows])
        return view.messages

    async def shed(self, fits: Callable[[list[AgentMessage]], bool]) -> ShedHistory | None:
        """Shed the rendered history for a trim (:func:`shed_history_view`).

        Stores the new stub seq and compacts the dropped turns before
        returning (unless the renderer does not persist), so the next turn
        loads what this one sends. None when the renderer does not compact or
        has nothing to shed; the caller falls back to the plain trim.
        """
        if not self._compact or self._rendered is None or not self._rendered[0]:
            return None
        rows, current, tz_name, preceding = self._rendered
        shed = shed_history_view(
            rows,
            current,
            tz_name,
            tools_by_name=self._tools_by_name,
            fits=fits,
            preceding=preceding,
            stub_before=self._stub_before,
        )
        if self._persist:
            await _set_history_stub_seq(self._user_id, shed.stub_before)
        self._stub_before = shed.stub_before
        dropped_rows = shed.view.dropped_rows
        dropped = await self._compact_dropped(dropped_rows, tz_name, preceding)
        self._rendered = (rows[len(dropped_rows) :], current, tz_name, [*preceding, *dropped_rows])
        logger.info(
            "Trim for user %s shed history: read results stubbed below seq %d"
            " (%d elided), %d row(s) dropped",
            self._user_id,
            shed.stub_before,
            shed.view.elided_results,
            len(dropped_rows),
        )
        return ShedHistory(messages=shed.view.messages, dropped=dropped)

    async def _compact_dropped(
        self, dropped_rows: list[StoredMessage], tz_name: str, preceding: Sequence[StoredMessage]
    ) -> list[AgentMessage]:
        """Compact *dropped_rows* into memory and move the watermark past them.

        Returns them rendered in full, results unstubbed, which is what the
        compaction reads.

        With compaction off the watermark stays put, as it does for the plain
        trim (``context.trigger_compaction_for_dropped`` is a no-op then).
        After a cold-start drop the next turn loads the same rows, drops the
        same ones, and renders the same bytes. After a trim's drop it is not
        a fixed point: the next turn loads the dropped rows again, rendered
        with the stored stub seq, so their read results are stubs. If that
        history is under the trigger it is sent whole, dropped turns
        included, and the prefix the trim turn cached is not read; if it is
        over, the trim sheds again. The plain trim with compaction off does
        the same: it reloads the dropped rows and trims afresh every turn.
        """
        dropped = _stored_messages_to_agent_messages(
            dropped_rows, tz_name=tz_name, preceding=preceding
        )
        if not dropped_rows or not self._persist or not settings.compaction_enabled:
            return dropped
        await trigger_compaction_for_dropped(self._user_id, dropped)
        # The compaction event's range ends at the last row that renders
        # a message. Rows after it that render nothing (an approval
        # prompt) must go too, or the next turn would load them and
        # render its first turn differently from this one.
        await _advance_trim_watermark_only(self._user_id, dropped_rows[-1].seq)
        return dropped


@dataclass(frozen=True)
class _Delivery:
    """A workspace update delivered on the turn answering row *seq*."""

    seq: int
    text: str
    # The documents as the model has them once it has read *text*.
    shown: WorkspaceSnapshot


@dataclass
class _EpochWorkspace:
    key: int
    snapshot: WorkspaceSnapshot
    deliveries: list[_Delivery] = field(default_factory=list)


# The workspace per user, keyed by epoch: the snapshot and the updates
# delivered since. Process-local on purpose: after a restart the next turn
# takes a fresh snapshot, which is byte-identical to the lost one unless a
# document changed mid-epoch, so the worst case is one early cache rewrite.
_SNAPSHOTS: OrderedDict[str, _EpochWorkspace] = OrderedDict()


def _remember(user_id: str, epoch: PromptEpoch, live: WorkspaceSnapshot) -> _EpochWorkspace:
    """The workspace entry for *epoch*, taking *live* as its snapshot when new.

    Only a cold start takes a new snapshot, and drops the updates delivered in
    the epoch before. A warm turn whose key moved (a trim or window overflow
    advanced the watermark past the row that opened the epoch, so
    ``find_epoch`` keys on the first row left) is still inside the same cached
    prefix: it keeps the stored entry.
    """
    stored = _SNAPSHOTS.get(user_id)
    if stored is not None and (stored.key == epoch.key or not epoch.cold_start):
        stored.key = epoch.key
    else:
        stored = _EpochWorkspace(key=epoch.key, snapshot=live)
        _SNAPSHOTS[user_id] = stored
    _SNAPSHOTS.move_to_end(user_id)
    while len(_SNAPSHOTS) > _SNAPSHOT_STORE_MAX:
        _SNAPSHOTS.popitem(last=False)
    return stored


@dataclass(frozen=True)
class WorkspaceView:
    """What one turn shows of the workspace documents, around its history."""

    # The history, with each earlier update on the turn it was delivered on.
    history: list[AgentMessage]
    # For the current turn: what changed since the model last saw it, or "".
    updates: str


class WorkspaceTurn:
    """One turn's view of the workspace documents in a cache epoch.

    ``snapshot`` goes in the system block: the copy taken when the epoch
    opened. :meth:`apply` lays the updates delivered since onto the history
    the turn sends, and returns the one the current turn carries.

    An update is delivered once. The turn it first rides records it against
    the row it answers (``epoch.turn_seq``), and every later turn of the epoch
    appends it to that row's message in the history. The history cache then
    holds it, so a large change (a compaction rewriting MEMORY.md) is sent
    uncached once rather than on every turn until the next cold start. A
    later change is diffed against what the model last saw, not the snapshot.

    :meth:`apply` takes the history after the trim has decided which rows
    survive (``core.ClawboltAgent._trim_history``), never before: a shed
    re-renders the history from the stored rows, and the trim and compaction
    must see those rows as stored.
    """

    def __init__(self, user_id: str, epoch: PromptEpoch, live: WorkspaceSnapshot) -> None:
        self._entry = _remember(user_id, epoch, live)
        self._live = live
        self._turn_seq = epoch.turn_seq

    @property
    def snapshot(self) -> WorkspaceSnapshot:
        return self._entry.snapshot

    def _seen(self, history: Sequence[AgentMessage]) -> int:
        """How many of the recorded deliveries the model sees in *history*.

        An update counts as seen only while its row is in *history*, along
        with the rows of every update before it (each is a diff on the one
        before). The current turn's own delivery, recorded by an earlier
        :meth:`apply` of the same turn, never is: its row is the current turn.
        """
        present = {m.seq for m in history if isinstance(m, UserMessage) and m.seq is not None}
        seen = 0
        deliveries = self._entry.deliveries
        while seen < len(deliveries) and deliveries[seen].seq in present:
            seen += 1
        return seen

    def _updates(self, seen: int) -> str:
        deliveries = self._entry.deliveries
        shown = deliveries[seen - 1].shown if seen else self._entry.snapshot
        return render_workspace_updates(shown, self._live)

    def preview(self, history: Sequence[AgentMessage]) -> str:
        """The update the current turn would carry after *history*, recording nothing.

        What the trim measures the current turn with, before it has decided
        the history :meth:`apply` then takes.
        """
        return self._updates(self._seen(history))

    def apply(self, history: Sequence[AgentMessage]) -> WorkspaceView:
        """Lay earlier updates onto *history*, and record the current turn's.

        From the first delivery whose row is missing from *history* (dropped
        or shed by a trim, or a row that renders nothing, as an approval reply
        does) the deliveries are forgotten, and the current turn carries the
        change against the last one still visible. A turn with no
        ``turn_seq`` records nothing, so every turn sends the change afresh.
        Calling it again in the same turn (the overflow retry's trim) replaces
        what the first call recorded.
        """
        seen = self._seen(history)
        del self._entry.deliveries[seen:]
        updates = self._updates(seen)
        by_seq = {d.seq: d.text for d in self._entry.deliveries}
        view_history = [
            replace(m, content=append_workspace_updates(m.content, by_seq[m.seq]))
            if isinstance(m, UserMessage) and m.seq in by_seq
            else m
            for m in history
        ]
        if updates and self._turn_seq is not None:
            self._entry.deliveries.append(
                _Delivery(seq=self._turn_seq, text=updates, shown=self._live)
            )
        return WorkspaceView(history=view_history, updates=updates)


def reset_workspace_snapshots() -> None:
    """Clear the snapshot store (for tests)."""
    _SNAPSHOTS.clear()
