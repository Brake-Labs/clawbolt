"""Turn selection and prompt reconstruction for the model-swap evaluator.

Reconstruction runs against *today's* system prompt, tool set, and memory,
not the versions in force when the turn originally happened. That is
deliberate: the decision being made is "should this user move to model X
now", so the prompt the candidate sees should be the prompt it would
actually get. Replaying a months-old system prompt would measure a
configuration nobody is going to ship.

The clock is the one exception, and it is not a configuration choice. The
turn's own timestamp is stamped on the replayed turn, because the history
rows around it render absolute date markers: a run days later would ask the
model to read a conversation that ended last week under a header claiming
today, and every date-relative instruction in it would resolve to the wrong
day. See ``assemble_for_sample``.

The history for a turn is the window of rows immediately preceding it,
bounded by ``conversation_history_limit`` and then trimmed by the same
``trim_messages`` governor the live loop uses. The session's current
``last_trim_seq`` watermark is deliberately *not* applied: it describes
what is visible today, and applying it would erase the history that older
samples actually ran with.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from backend.app.agent.approval import get_approval_store
from backend.app.agent.context import (
    _parse_tool_interactions,
    _stored_messages_to_agent_messages,
)
from backend.app.agent.core import AssembledPrompt, ClawboltAgent
from backend.app.agent.dto import StoredMessage
from backend.app.agent.messages import AgentMessage, AssistantMessage
from backend.app.agent.router import init_storage
from backend.app.agent.session_db import get_session_store
from backend.app.agent.stores import ToolConfigStore
from backend.app.agent.tools.base import Tool, tool_to_function_schema
from backend.app.agent.tools.registry import (
    ToolContext,
    create_list_capabilities_tool,
    default_registry,
    ensure_tool_modules_imported,
)
from backend.app.bus import OutboundMessage
from backend.app.config import settings
from backend.app.enums import MessageDirection
from backend.app.models import User
from backend.app.services.llm_eval import metrics
from backend.app.services.llm_eval.execution import MAX_REPLAY_READ_ROUNDS
from backend.app.services.llm_eval.types import RecordedToolResult, ReplaySample, ToolCall

logger = logging.getLogger(__name__)


async def _refuse_outbound(message: OutboundMessage) -> None:
    """Outbound sink for replay tool contexts. Must never be reached."""
    raise AssertionError(
        "llm_eval attempted to publish an outbound message; a replay must never execute a tool"
    )


@dataclass
class ReplayFixture:
    """Everything reusable across every turn in one evaluation run.

    The tool list is built once and shared. Tool *schemas* depend only on
    each tool's params model, never on the turn being replayed, so building
    them per sample would re-run every integration's ``auth_check`` for an
    identical result. The per-turn text that some factories close over
    (``turn_text``) affects tool behavior, never the schema the model sees,
    and no tool is executed during a replay.
    """

    user: User
    rows: list[StoredMessage]
    tools: list[Tool] = field(default_factory=list)
    tool_schemas: list[dict] = field(default_factory=list)
    tools_by_name: dict[str, Tool] = field(default_factory=dict)

    @property
    def tz_name(self) -> str:
        return self.user.timezone or ""


# Rows a single user turn can occupy. One inbound row plus the outbound rows
# the agent wrote answering it; four covers a turn with several tool receipts
# and a final reply, and the fallback below covers the rest.
_ROWS_PER_TURN_ALLOWANCE = 6


def _row_budget(sample_limit: int) -> int:
    """Rows worth loading to sample ``sample_limit`` turns with their history.

    A run needs the last N inbound turns, the outbound rows that follow each of
    them, and ``conversation_history_limit`` rows before the oldest one, since
    that is the window ``_history_for`` slices. Everything older is loaded,
    decrypted, and discarded.
    """
    return sample_limit * _ROWS_PER_TURN_ALLOWANCE + settings.conversation_history_limit


async def build_fixture(user: User, *, sample_limit: int | None = None) -> ReplayFixture:
    """Materialize the user's live tool set and enough transcript to replay.

    ``sample_limit`` bounds the transcript read to the tail a run of that size
    can reach. Every row is envelope-encrypted, so decryption happens on
    attribute access in this process: loading a long transcript to use its last
    few turns spends real CPU on the event loop that also serves the user's own
    messages. Omit it to load everything, which is what a caller that does not
    know its sample count has to do.
    """
    store = get_session_store(user.id)
    rows: list[StoredMessage] = []
    if sample_limit is not None and sample_limit > 0:
        budget = _row_budget(sample_limit)
        rows = await store.get_recent_messages_async(budget)
        inbound_at = [i for i, r in enumerate(rows) if r.direction == MessageDirection.INBOUND]
        # A full window means older rows exist that were not read, so the
        # question is whether what was read is actually enough. Two ways it is
        # not: fewer turns than asked for, or enough turns but not enough rows
        # in front of the oldest one to rebuild the history the agent saw.
        #
        # Counting turns alone missed the second, and it is the commoner
        # failure: the samples consume the budget, the oldest replays against
        # a shorter prompt than production builds, and the run scores that
        # difference as if it were the candidate's doing.
        if len(rows) == budget:
            short_of_turns = len(inbound_at) < sample_limit
            history_rows = inbound_at[-sample_limit] if not short_of_turns else 0
            if short_of_turns or history_rows < settings.conversation_history_limit:
                # This user's turns are unusually long. Fall back rather than
                # quietly running a smaller or shallower evaluation than the
                # operator chose.
                logger.info(
                    "Replay window of %d rows held %d inbound turn(s) and %d row(s) of "
                    "history for user %s; loading the full transcript",
                    budget,
                    len(inbound_at),
                    history_rows,
                    user.id,
                )
                rows = []
    if not rows:
        sessions = await store.list_sessions_async()
        for session in sessions:
            rows.extend(session.messages)
    rows.sort(key=lambda m: m.seq)

    # Nothing populates the registry at startup: every entry point that needs
    # tools calls this itself (``tool_assembly``, ``heartbeat``, ``onboarding``,
    # ``approval``, ``user_tools``). Without it a run on a worker that has not
    # yet processed a message builds an empty tool list, and then both models
    # are offered no tools, agree perfectly on replying in prose, and the report
    # reads as a clean pass. Silent and completely wrong.
    ensure_tool_modules_imported()

    # ``refresh=False``: a replay must not write the user's state or message
    # them. Refreshing an expired Drive token writes ``oauth_tokens``, and a
    # permanent refresh failure deletes the grant and sends the user a
    # "Drive disconnected" message, which the ``_refuse_outbound`` sink below
    # cannot intercept because it goes straight to the message bus. The token
    # here is only ever read for its presence, to keep the file tools on the
    # schema, so an expired one is as good as a fresh one.
    storage = await init_storage(user, refresh=False)
    context = ToolContext(
        user=user,
        storage=storage,
        # The messaging factory is registered ``requires_outbound=True`` and
        # asserts on this, so passing None drops ``send_media_reply`` from the
        # schema and the replay would offer a smaller tool list than production
        # does. It is never invoked: a replay stops at the model's decision and
        # executes nothing. It raises rather than passing so that if that
        # invariant is ever broken, the run fails loudly instead of publishing
        # a real message to a real user's channel.
        publish_outbound=_refuse_outbound,
        channel=user.preferred_channel or "",
        to_address="",
        downloaded_media=[],
        turn_text="",
    )
    # Mirror ``router.run_agent_step`` exactly. A tool group the user disabled,
    # or a sub-tool they set to NEVER, is absent from the schema the live agent
    # is offered *and* from the tool-guidelines section of the system prompt, so
    # including it here would score a prompt this user never received and let a
    # candidate "call" a tool production would not have exposed.
    #
    # ``approval_store.ensure_complete`` is deliberately not called: it writes
    # PERMISSIONS.json, and a replay must not mutate the user's state. It only
    # ever backfills tools at their *default* level, and no default is NEVER,
    # so skipping it cannot change the set read back here.
    disabled_groups = await ToolConfigStore(user.id).get_disabled_tool_names()
    disabled_sub_tools = await get_approval_store().get_never_tool_names(user.id)

    tools = await default_registry.create_core_tools(
        context,
        excluded_factories=disabled_groups or None,
        excluded_tool_names=disabled_sub_tools or None,
    )
    tools.extend(
        await default_registry.create_ready_specialist_tools(
            context,
            excluded_factories=disabled_groups or None,
            excluded_tool_names=disabled_sub_tools or None,
        )
    )
    # ``list_capabilities`` is on almost every real turn (any unconnected
    # integration is enough to add it) and carries a usage hint of its own, so
    # omitting it changed both the schema and the system prompt.
    specialist_summaries = await default_registry.get_available_specialist_summaries(
        context, excluded_factories=disabled_groups or None
    )
    unauthenticated = await default_registry.get_unauthenticated_specialists(
        context, excluded_factories=disabled_groups or None
    )
    if specialist_summaries or unauthenticated:
        disabled_specialist_subs = default_registry.get_disabled_specialist_sub_tools(
            disabled_sub_tools or set()
        )
        tools.append(
            create_list_capabilities_tool(
                specialist_summaries,
                unauthenticated=unauthenticated,
                disabled_sub_tools=disabled_specialist_subs or None,
            )
        )

    fixture = ReplayFixture(user=user, rows=rows, tools=tools)
    fixture.tool_schemas = [tool_to_function_schema(t) for t in tools]
    fixture.tools_by_name = {t.name: t for t in tools}
    logger.info(
        "Replay fixture for user %s: %d rows, %d tools",
        user.id,
        len(rows),
        len(tools),
    )
    return fixture


def _parse_timestamp(raw: str) -> datetime | None:
    """Parse a stored ISO timestamp, assuming UTC when it carries no offset."""
    try:
        parsed = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


# How far apart two consecutive inbound rows can be and still be one turn.
# Production batches rapid-fire messages with a 1.5 s window
# (``settings.message_batch_window_ms``) and runs the pipeline only for the
# last of them, so the rows of a real batch are seconds apart. A wider gap
# means the earlier row was orphaned rather than batched: see
# ``agent.inbound_recovery``, whose module docstring records a production
# inbound that waited 29 hours for the next message to wake a batcher, and
# which only re-dispatches orphans younger than 30 minutes.
_BATCH_GAP_LIMIT = timedelta(minutes=2)


def _same_batch(earlier: StoredMessage, later: StoredMessage) -> bool:
    """Whether two consecutive inbound rows were answered as one turn.

    An unreadable timestamp answers False, which reports "the agent did
    nothing" for that turn. That is the safe direction: the credit this grants
    is what stops ``check_safety`` raising ``UNREQUESTED_MUTATION``, so a
    corrupt row must not hand out an exemption.
    """
    first = _parse_timestamp(earlier.timestamp)
    second = _parse_timestamp(later.timestamp)
    if first is None or second is None:
        return False
    return second - first <= _BATCH_GAP_LIMIT


def _historic_response(rows: list[StoredMessage], start: int) -> tuple[str, list[str]]:
    """Return the reply text and tool names the agent produced for a turn.

    A "turn" here is the batch of consecutive inbound rows starting at *start*,
    plus the outbound rows that follow it. Skipping over the rest of the batch
    is what makes this correct for rapid-fire messages: a user who sends four
    messages in a row persists four inbound rows and the agent answers the
    batch once, after the last of them. Reading only up to the *next* inbound
    row reported "the agent did nothing" for the first three, and
    ``check_safety`` then charged the candidate with an unrequested mutation on
    a turn whose text was an explicit instruction to write.

    The batch is bounded by ``_BATCH_GAP_LIMIT``, because two consecutive
    inbound rows far apart in time are not a batch: the earlier one went
    unanswered. Crediting it with the later turn's tool calls would exempt a
    candidate that wrote something in reply to a message the agent never
    answered.

    Tool names are read back through the same rebuilder the LLM history uses,
    so a row whose ``tool_interactions_json`` is malformed degrades to "no
    tools" here exactly as it does in the prompt.
    """
    answered = _response_rows(rows, start)
    tool_names: list[str] = []
    for row in answered:
        for msg in _stored_messages_to_agent_messages([row]):
            if isinstance(msg, AssistantMessage):
                tool_names.extend(tc.name for tc in msg.tool_calls)
    return _reply_text(answered), tool_names


def _reply_text(answered: list[StoredMessage]) -> str:
    """The prose the turn ended on, across every outbound row it wrote."""
    return "\n\n".join(text for row in answered if (text := row.llm_reply_text or row.body))


def _response_rows(rows: list[StoredMessage], start: int) -> list[StoredMessage]:
    """The outbound rows that answered the turn whose batch begins at *start*."""
    index = start + 1
    # Advance past the remainder of the inbound batch this row belongs to.
    while (
        index < len(rows)
        and rows[index].direction == MessageDirection.INBOUND
        and _same_batch(rows[index - 1], rows[index])
    ):
        index += 1
    answered: list[StoredMessage] = []
    for row in rows[index:]:
        if row.direction == MessageDirection.INBOUND:
            break
        answered.append(row)
    return answered


def _historic_tool_results(rows: list[StoredMessage], start: int) -> tuple[RecordedToolResult, ...]:
    """Every tool call the live turn made, with the result it got back.

    Read through the same parser the history rebuild uses, so a malformed
    ``tool_interactions_json`` yields no results here just as it yields no
    tool calls in the prompt.
    """
    return tuple(
        RecordedToolResult(
            name=interaction.name,
            arguments=interaction.args,
            result=interaction.result,
            is_error=interaction.is_error,
        )
        for row in _response_rows(rows, start)
        for interaction in _parse_tool_interactions(row.tool_interactions_json)
    )


@dataclass(frozen=True)
class HistoricDecision:
    """The decision the live turn is scored on, as the transcript preserves it.

    Scored at the point in the turn the replay scores the candidate at. The
    candidate is advanced past every response whose calls are all replayable
    lookups (``metrics.replayable_lookup``, up to ``MAX_REPLAY_READ_ROUNDS``
    rounds) and scored on the first response that would need a live tool, so
    the recorded turn is walked by the same rule: its leading replayable
    lookups are skipped into ``lookups`` and the next recorded call is
    ``calls``. Taking the literal first recorded call instead read every
    lookup-then-act turn as the candidate acting where production talked.

    ``text`` is the prose that belongs to the scored round, which the
    transcript only preserves when the round is the last one: an outbound row
    carries the turn's final reply and its calls in one flat list, so a
    decision that opened with a call has no recorded text of its own and
    carries none rather than borrowing the reply written after it.

    One thing the transcript cannot give back. ``tool_interactions_json``
    holds one flat, ordered list per outbound row, so a turn that recorded
    three calls could have asked for all three at once or for one at a time
    across three rounds. ``flattened`` says the scored decision sits in such
    a list, which is counted and surfaced rather than assumed away.
    """

    available: bool
    calls: tuple[ToolCall, ...]
    flattened: bool
    lookups: tuple[RecordedToolResult, ...] = ()
    """Recorded lookups skipped before the scored decision, in order.

    The incumbent's counterpart to ``ModelCallResult.replayed_lookups``:
    what it had read by the time it made the decision being scored. Carried
    so the judge prompt can show a "lookups made first" block for this side
    too, which is otherwise a structural tell for which response is which.
    """
    text: str = ""


def _recorded_entry_count(raw: str) -> int:
    """How many interactions the row claims, before any of them are validated.

    ``_parse_tool_interactions`` drops the entries that fail validation and
    keeps the rest, so a shorter list back is the only sign that part of the
    record is gone. Counting the raw entries is what lets the caller tell
    "this turn called nothing" from "this turn's first call did not survive",
    which are the same empty list otherwise. A payload that is not a JSON
    list answers -1, which no parse result can equal, so it reads as lost
    rather than as empty.
    """
    if not raw or not raw.strip():
        return 0
    try:
        entries = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return -1
    return len(entries) if isinstance(entries, list) else -1


_UNAVAILABLE = HistoricDecision(available=False, calls=(), flattened=False)


def _recorded_calls(answered: list[StoredMessage]) -> list[RecordedToolResult] | None:
    """Every call the turn recorded, in order, or None when part of it is gone.

    A record that did not survive whole is not a decision: dropping the
    entries that failed validation leaves the *next* call standing as the
    turn's opening move, which on a lookup-then-write turn reports the
    incumbent as having opened with the write.
    """
    recorded: list[RecordedToolResult] = []
    for row in answered:
        parsed = _parse_tool_interactions(row.tool_interactions_json)
        if _recorded_entry_count(row.tool_interactions_json) != len(parsed):
            logger.warning(
                "Unparseable tool interactions on seq %d; its first decision is unavailable",
                row.seq,
            )
            return None
        recorded.extend(
            RecordedToolResult(
                name=item.name,
                arguments=item.args,
                result=item.result,
                is_error=item.is_error,
            )
            for item in parsed
        )
    return recorded


def _historic_first_decision(
    rows: list[StoredMessage], start: int, tools_by_name: dict[str, Tool]
) -> HistoricDecision:
    """Reconstruct the scored decision of the turn whose batch begins at *start*.

    Walks the recorded calls by the rule ``execution.call_model`` walks the
    candidate's rounds by: leading replayable lookups are skipped, at most
    ``MAX_REPLAY_READ_ROUNDS`` of them, and the next recorded call is the
    decision. A lookup the replay could not have fed back (its tool has left
    the schema, or the same call is not in the record) stops the walk and is
    itself the decision, exactly as it would end the replay.

    A turn whose calls are all skipped lookups decided whatever came after
    them, which is the prose the turn ended on.

    Unavailable in three cases, each of which reads as "the agent did
    nothing" if it is not distinguished, and that reading is what would
    charge a candidate with an unrequested mutation on a turn production
    also wrote on. The turn was never answered. An outbound row carried tool
    interactions that did not survive parsing. Or the turn spent itself on
    lookups and recorded no prose after them, so what it decided next was
    never written down and there is nothing comparable to score.
    """
    answered = _response_rows(rows, start)
    if not answered:
        return _UNAVAILABLE
    recorded = _recorded_calls(answered)
    if recorded is None:
        return _UNAVAILABLE

    lookups: list[RecordedToolResult] = []
    index = 0
    while index < len(recorded) and len(lookups) < MAX_REPLAY_READ_ROUNDS:
        call = recorded[index]
        fed = metrics.replayable_lookup(call.name, call.arguments, tools_by_name, recorded)
        if fed is None:
            break
        lookups.append(fed)
        index += 1

    remaining = recorded[index:]
    reply = _reply_text(answered)
    if not remaining:
        # Every recorded call was a lookup the replay would have answered,
        # so the decision that follows them is the prose the turn ended on.
        if lookups and not reply.strip():
            return _UNAVAILABLE
        return HistoricDecision(
            available=True,
            calls=(),
            flattened=False,
            lookups=tuple(lookups),
            text=reply,
        )
    first = remaining[0]
    return HistoricDecision(
        available=True,
        calls=(ToolCall(name=first.name, arguments=first.arguments),),
        # The scored call shares a flat list with others, so whether the
        # decision was this call alone or this call and its neighbours in one
        # round is not recoverable.
        flattened=len(recorded) > 1,
        lookups=tuple(lookups),
    )


def _batch_end(rows: list[StoredMessage], start: int) -> int:
    """Index of the last inbound row in the batch that begins at *start*."""
    index = start
    while (
        index + 1 < len(rows)
        and rows[index + 1].direction == MessageDirection.INBOUND
        and _same_batch(rows[index], rows[index + 1])
    ):
        index += 1
    return index


def _message_context(row: StoredMessage) -> str:
    return row.processed_context or row.body


def select_samples(fixture: ReplayFixture, limit: int) -> list[ReplaySample]:
    """Pick the most recent *limit* turns, oldest first.

    A turn is a batch, not a row. Production answers rapid-fire messages once,
    after the last of them, with the earlier ones already in the history it
    loads, so that is the replay too: one sample per batch, at the batch's
    last row. Replaying each row alone scored decisions production never
    made, on a fraction of what the user had said, and the judge read that
    fraction as the whole request. The earlier rows ride along as
    ``batched_messages`` so the report and the judge see everything the user
    sent.

    Blank inbound rows are skipped: rapid-fire attachment batching persists
    a placeholder with no body and no processed context, and replaying one
    would ask both models to respond to an empty string. A batch that ends in
    one is replayed at its last row with text.
    """
    samples: list[ReplaySample] = []
    rows = fixture.rows
    index = 0
    while index < len(rows):
        if rows[index].direction != MessageDirection.INBOUND:
            index += 1
            continue
        end = _batch_end(rows, index)
        texts = [(i, _message_context(rows[i])) for i in range(index, end + 1)]
        texts = [(i, text) for i, text in texts if text.strip()]
        if texts:
            last_index, message_context = texts[-1]
            row = rows[last_index]
            reply, tool_names = _historic_response(rows, last_index)
            decision = _historic_first_decision(rows, last_index, fixture.tools_by_name)
            samples.append(
                ReplaySample(
                    seq=row.seq,
                    timestamp=row.timestamp,
                    message_context=message_context,
                    historic_reply=reply,
                    historic_tool_names=tool_names,
                    historic_tool_results=_historic_tool_results(rows, last_index),
                    historic_first_calls=decision.calls,
                    historic_decision_lookups=decision.lookups,
                    historic_decision_text=decision.text,
                    historic_decision_available=decision.available,
                    historic_calls_flattened=decision.flattened,
                    batched_messages=tuple(text for _, text in texts[:-1]),
                )
            )
        index = end + 1
    return samples[-limit:] if limit > 0 else samples


def _history_for(fixture: ReplayFixture, sample: ReplaySample) -> list[AgentMessage]:
    """Rebuild the conversation history as it stood just before *sample*."""
    preceding = [r for r in fixture.rows if r.seq < sample.seq]
    window = preceding[-settings.conversation_history_limit :]
    return _stored_messages_to_agent_messages(window, tz_name=fixture.tz_name)


def sample_clock(sample: ReplaySample) -> datetime | None:
    """The wall time to stamp on *sample*'s replayed turn, or None for now.

    Falls back to None (wall time) on an unparseable timestamp rather than
    guessing: a wrong clock is worse than the honest current one, and the
    column is ISO-formatted by ``_turn_row`` so this is a corrupt-row guard,
    not an expected branch.
    """
    parsed = _parse_timestamp(sample.timestamp)
    if parsed is None:
        logger.warning("Unparseable timestamp %r on seq %d", sample.timestamp, sample.seq)
    return parsed


async def assemble_for_sample(fixture: ReplayFixture, sample: ReplaySample) -> AssembledPrompt:
    """Build the exact prompt the live agent would send for this turn.

    A fresh ``ClawboltAgent`` per sample keeps turns independent: the agent
    accumulates per-turn state (delivered SKILL.md categories, the last
    reported input-token count) that would otherwise leak from one replayed
    turn into the next and change the prompt. Construction is cheap; the
    expensive part, the tool list, is shared via the fixture.

    The clock is the turn's own timestamp, not now. Everything else about the
    prompt is deliberately today's (see the module docstring), but the clock
    is not a configuration choice: history rows render absolute date markers,
    so a replay run days later hands the model a conversation that ends last
    week under a header saying today, and "book it for this past Thursday"
    lands on the wrong Thursday for both models.
    """
    agent = ClawboltAgent(user=fixture.user)
    agent.register_tools(fixture.tools)
    # Reproducibility: without this the trim decision reads a process-local
    # estimate written by live traffic, so the same run can build a different
    # prompt on a worker that recently served this user.
    return await agent.assemble_prompt(
        sample.message_context,
        _history_for(fixture, sample),
        deterministic_trim=True,
        now=sample_clock(sample),
    )
