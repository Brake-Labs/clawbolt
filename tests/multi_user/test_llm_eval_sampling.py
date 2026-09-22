"""Turn selection and history reconstruction for the model-swap evaluator.

The property that matters: replaying turn N must show the model exactly what
the agent saw before turn N, and nothing that came after. A slice that leaks
later turns would let the candidate answer with hindsight and quietly inflate
every agreement number in the report.
"""

from __future__ import annotations

import datetime as _dt
import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.agent.dto import StoredMessage
from backend.app.agent.messages import AssistantMessage, UserMessage
from backend.app.agent.session_db import reset_session_stores
from backend.app.agent.tools.base import Tool, ToolResult, ToolTags
from backend.app.config import settings
from backend.app.models import ChatSession, Message, User
from backend.app.services.llm_eval.metrics import MAX_REPLAY_READ_ROUNDS
from backend.app.services.llm_eval.sampling import (
    ReplayFixture,
    _historic_first_decision,
    _historic_response,
    _history_for,
    assemble_for_sample,
    build_fixture,
    sample_clock,
    select_samples,
)
from backend.app.services.llm_eval.types import RecordedToolResult, ReplaySample

BASE_TIME = _dt.datetime(2026, 5, 1, 12, 0, tzinfo=_dt.UTC)


def _seed(db: Session, user: User, turns: list[tuple[str, str, list[dict] | None]]) -> None:
    """Write a transcript. Each entry is ``(direction, body, tool_interactions)``."""
    session = ChatSession(session_id=str(uuid.uuid4()), user_id=user.id, channel="telegram")
    db.add(session)
    db.commit()
    db.refresh(session)
    for index, (direction, body, tools) in enumerate(turns, start=1):
        db.add(
            Message(
                session_id=session.id,
                seq=index,
                direction=direction,
                body=body,
                processed_context=body if direction == "inbound" else "",
                llm_reply_text=body if direction == "outbound" else "",
                tool_interactions_json=json.dumps(tools) if tools else "",
                timestamp=BASE_TIME + _dt.timedelta(minutes=index),
            )
        )
    db.commit()


@pytest.fixture()
def _reset_stores() -> None:
    reset_session_stores()


async def _fixture_for(user: User) -> ReplayFixture:
    fixture = await build_fixture(user)
    return fixture


async def test_selects_only_inbound_turns_most_recent_last(
    db_session: Session, test_user: User, _reset_stores: None
) -> None:
    _seed(
        db_session,
        test_user,
        [
            ("inbound", "first ask", None),
            ("outbound", "first answer", None),
            ("inbound", "second ask", None),
            ("outbound", "second answer", None),
        ],
    )
    fixture = await _fixture_for(test_user)
    samples = select_samples(fixture, limit=10)
    assert [s.message_context for s in samples] == ["first ask", "second ask"]
    assert [s.seq for s in samples] == [1, 3]


async def test_limit_keeps_the_most_recent_turns(
    db_session: Session, test_user: User, _reset_stores: None
) -> None:
    # Answered turns: unanswered rows a minute apart would be one batch.
    turns: list[tuple[str, str, list[dict] | None]] = []
    for i in range(1, 6):
        turns += [("inbound", f"ask {i}", None), ("outbound", f"answer {i}", None)]
    _seed(db_session, test_user, turns)
    fixture = await _fixture_for(test_user)
    samples = select_samples(fixture, limit=2)
    assert [s.message_context for s in samples] == ["ask 4", "ask 5"]


async def test_blank_inbound_placeholders_are_skipped(
    db_session: Session, test_user: User, _reset_stores: None
) -> None:
    """Attachment batching persists an empty inbound row; replaying it would
    ask both models to respond to nothing."""
    _seed(
        db_session,
        test_user,
        [("inbound", "", None), ("inbound", "real question", None)],
    )
    fixture = await _fixture_for(test_user)
    samples = select_samples(fixture, limit=10)
    assert [s.message_context for s in samples] == ["real question"]


async def test_historic_tool_calls_are_recovered(
    db_session: Session, test_user: User, _reset_stores: None
) -> None:
    _seed(
        db_session,
        test_user,
        [
            ("inbound", "book it", None),
            (
                "outbound",
                "Booked.",
                [
                    {
                        "tool_call_id": "t1",
                        "name": "create_job",
                        "args": {"customer": "Acme Plumbing"},
                        "result": "ok",
                    }
                ],
            ),
        ],
    )
    fixture = await _fixture_for(test_user)
    samples = select_samples(fixture, limit=10)
    assert samples[0].historic_tool_names == ["create_job"]
    assert samples[0].historic_reply == "Booked."


async def test_history_slice_excludes_the_turn_and_everything_after(
    db_session: Session, test_user: User, _reset_stores: None
) -> None:
    _seed(
        db_session,
        test_user,
        [
            ("inbound", "old ask", None),
            ("outbound", "old answer", None),
            ("inbound", "target ask", None),
            ("outbound", "future answer", None),
            ("inbound", "future ask", None),
        ],
    )
    fixture = await _fixture_for(test_user)
    target = next(s for s in select_samples(fixture, limit=10) if s.message_context == "target ask")

    history = _history_for(fixture, target)
    rendered = [m.content for m in history if isinstance(m, UserMessage | AssistantMessage)]
    joined = " ".join(c or "" for c in rendered)

    assert "old ask" in joined
    assert "old answer" in joined
    # The turn itself is supplied separately as the current message, and
    # anything later had not happened yet.
    assert "target ask" not in joined
    assert "future answer" not in joined
    assert "future ask" not in joined


async def test_user_with_no_messages_yields_no_samples(
    test_user: User, _reset_stores: None
) -> None:
    fixture = await _fixture_for(test_user)
    assert select_samples(fixture, limit=100) == []


@patch("backend.app.agent.router.oauth_service.load_token", new_callable=AsyncMock)
@patch("backend.app.agent.router.oauth_service.get_valid_token", new_callable=AsyncMock)
async def test_building_a_fixture_never_refreshes_the_users_drive_token(
    mock_get_valid_token: AsyncMock,
    mock_load_token: AsyncMock,
    test_user: User,
    _reset_stores: None,
) -> None:
    """A run must not rotate the grant or tell the user Drive disconnected.

    ``get_valid_token`` writes ``oauth_tokens`` on a refresh and, when the
    refresh fails permanently, deletes the grant and messages the user. The
    fixture only needs to know whether Drive is connected, so it reads the
    stored token instead.
    """
    mock_load_token.return_value = None
    with (
        patch.object(settings, "google_drive_client_id", "client-id"),
        patch.object(settings, "google_drive_client_secret", "client-secret"),
    ):
        await build_fixture(test_user)

    mock_get_valid_token.assert_not_awaited()
    # ``oauth_service`` is a singleton, so every specialist auth_check in the
    # fixture reads through the same patched ``load_token``. Assert on the Drive
    # read specifically rather than on the call count.
    assert (test_user.id, "google_drive") in {call.args for call in mock_load_token.await_args_list}


# ---------------------------------------------------------------------------
# Bounding the transcript read
#
# Every row is envelope-encrypted, so decryption happens on attribute access
# in this process. A production user with 1827 messages had their whole
# transcript loaded and decrypted to use the last few turns, five times in
# half an hour, on the event loop that also serves their own messages.
# ---------------------------------------------------------------------------


async def test_a_bounded_read_gives_the_same_samples_as_a_full_one(
    db_session: Session, test_user: User, _reset_stores: None
) -> None:
    """The bound is an optimization, so it has to be invisible in the result."""
    turns: list[tuple[str, str, list[dict] | None]] = []
    for i in range(1, 61):
        turns.append(("inbound", f"ask {i}", None))
        turns.append(("outbound", f"answer {i}", None))
    _seed(db_session, test_user, turns)

    # The budget is dominated by ``conversation_history_limit`` (500 by
    # default), so it is lowered here rather than seeding a transcript long
    # enough to exceed it.
    # The window has to stay patched through the assertions too: the budget is
    # computed from ``conversation_history_limit`` and ``_history_for`` reads it
    # again at call time, so comparing the two fixtures under a different limit
    # compares windows neither was built for.
    with patch.object(settings, "conversation_history_limit", 20):
        full = await build_fixture(test_user)
        bounded = await build_fixture(test_user, sample_limit=5)

        assert len(bounded.rows) < len(full.rows)
        full_samples = select_samples(full, limit=5)
        bounded_samples = select_samples(bounded, limit=5)
        assert [s.seq for s in bounded_samples] == [s.seq for s in full_samples]
        assert [s.message_context for s in bounded_samples] == [
            s.message_context for s in full_samples
        ]
        assert [s.historic_reply for s in bounded_samples] == [
            s.historic_reply for s in full_samples
        ]
        # And the history each sample reconstructs is the same window.
        assert [m.content for m in _history_for(bounded, bounded_samples[0])] == [
            m.content for m in _history_for(full, full_samples[0])
        ]


async def test_the_bound_falls_back_rather_than_shrinking_a_run(
    db_session: Session, test_user: User, _reset_stores: None
) -> None:
    """A window that fills before holding the turns asked for is abandoned.

    One inbound row can be followed by many outbound rows, so a fixed
    allowance per turn is a guess. Getting it wrong must cost a slower load,
    never a smaller evaluation than the operator chose.
    """
    turns: list[tuple[str, str, list[dict] | None]] = []
    for i in range(1, 6):
        turns.append(("inbound", f"ask {i}", None))
        # Twenty outbound rows per turn blows through the per-turn allowance.
        for j in range(20):
            turns.append(("outbound", f"step {i}.{j}", None))
    _seed(db_session, test_user, turns)

    with patch.object(settings, "conversation_history_limit", 20):
        bounded = await build_fixture(test_user, sample_limit=5)
    assert len(select_samples(bounded, limit=5)) == 5


async def test_a_window_that_fills_exactly_still_falls_back(
    db_session: Session, test_user: User, _reset_stores: None
) -> None:
    """Filling the budget exactly is already the failure.

    The guard used to fire only when the window held *fewer* inbound turns
    than asked for. When it holds exactly that many and is full, the samples
    have consumed the whole budget, so the oldest one has almost nothing in
    front of it and replays against a shorter prompt than production builds.
    That is the case the fallback exists to prevent.
    """
    # budget = 5 * 6 + 20 = 50. Make the last 50 rows hold exactly 5 inbound
    # turns, so ``len(rows) == budget`` and ``inbound == sample_limit``.
    turns: list[tuple[str, str, list[dict] | None]] = []
    for i in range(1, 21):
        turns.append(("inbound", f"old {i}", None))
        turns.append(("outbound", f"old answer {i}", None))
    for i in range(1, 6):
        turns.append(("inbound", f"ask {i}", None))
        for j in range(9):
            turns.append(("outbound", f"step {i}.{j}", None))
    _seed(db_session, test_user, turns)

    with patch.object(settings, "conversation_history_limit", 20):
        bounded = await build_fixture(test_user, sample_limit=5)
        full = await build_fixture(test_user)

        oldest_bounded = select_samples(bounded, limit=5)[0]
        oldest_full = select_samples(full, limit=5)[0]
        assert oldest_bounded.seq == oldest_full.seq
        # The history the oldest sample reconstructs is what the fallback
        # protects, and it is what a truncated window silently shortens.
        assert [m.content for m in _history_for(bounded, oldest_bounded)] == [
            m.content for m in _history_for(full, oldest_full)
        ]


async def test_a_window_holding_enough_turns_but_no_history_falls_back(
    db_session: Session, test_user: User, _reset_stores: None
) -> None:
    """Counting turns is the wrong question on its own.

    Here the window holds *more* inbound turns than the run asked for, so a
    turn-count guard passes it. The oldest sampled turn still has almost
    nothing in front of it, which is the shortened prompt the fallback
    exists to prevent.
    """
    # budget = 5 * 6 + 20 = 50 rows. Pack the tail so it holds 6 inbound
    # turns and leaves the 5th-from-last with far fewer than 20 rows ahead.
    turns: list[tuple[str, str, list[dict] | None]] = []
    for i in range(1, 21):
        turns.append(("inbound", f"old {i}", None))
        turns.append(("outbound", f"old answer {i}", None))
    for i in range(1, 7):
        turns.append(("inbound", f"ask {i}", None))
        for j in range(7):
            turns.append(("outbound", f"step {i}.{j}", None))
    _seed(db_session, test_user, turns)

    with patch.object(settings, "conversation_history_limit", 20):
        bounded = await build_fixture(test_user, sample_limit=5)
        full = await build_fixture(test_user)

        oldest_bounded = select_samples(bounded, limit=5)[0]
        oldest_full = select_samples(full, limit=5)[0]
        assert oldest_bounded.seq == oldest_full.seq
        assert [m.content for m in _history_for(bounded, oldest_bounded)] == [
            m.content for m in _history_for(full, oldest_full)
        ]


# ---------------------------------------------------------------------------
# Rapid-fire messages
# ---------------------------------------------------------------------------


async def test_a_rapid_fire_batch_is_replayed_once_at_its_last_row(
    db_session: Session, test_user: User, _reset_stores: None
) -> None:
    """Regression: every row of a batch was replayed as its own turn.

    Production answers the batch once, after the last message, with the
    earlier ones already in the history it loads. Replaying "rebuild the
    stalls" on its own scored a decision production never made, on a third
    of what the user asked, and the judge read that third as the request.
    """
    _seed(
        db_session,
        test_user,
        [
            ("inbound", "rebuild the stalls", None),
            ("inbound", "add 5000 for the staircase", None),
            ("inbound", "build and send", None),
            (
                "outbound",
                "sent",
                [
                    {
                        "tool_call_id": "t1",
                        "name": "qb_update",
                        "args": {"estimate_id": "635"},
                        "result": "ok",
                    }
                ],
            ),
        ],
    )
    fixture = await _fixture_for(test_user)
    samples = select_samples(fixture, limit=10)

    assert [s.seq for s in samples] == [3]
    (sample,) = samples
    assert sample.message_context == "build and send"
    assert sample.batched_messages == ("rebuild the stalls", "add 5000 for the staircase")
    assert sample.user_text == "rebuild the stalls\n\nadd 5000 for the staircase\n\nbuild and send"
    assert sample.historic_tool_names == ["qb_update"]
    assert sample.historic_reply == "sent"
    # Recorded with its result, so a replay can answer a matching lookup.
    assert sample.historic_tool_results == (
        RecordedToolResult(name="qb_update", arguments={"estimate_id": "635"}, result="ok"),
    )

    # The earlier rows reach the model the way production shows them: as
    # history in front of the last row, not folded into the current turn.
    history = [m.content for m in _history_for(fixture, sample)]
    assert any("rebuild the stalls" in str(c) for c in history)
    assert any("add 5000 for the staircase" in str(c) for c in history)


async def test_rows_minutes_apart_are_separate_turns(
    db_session: Session, test_user: User, _reset_stores: None
) -> None:
    """Only a real batch is grouped; an unanswered row stays its own turn."""
    _seed(db_session, test_user, [("inbound", "first", None)])
    session_rows = await _fixture_for(test_user)
    session_rows.rows.append(
        StoredMessage(
            seq=2,
            direction="inbound",
            body="three hours later",
            timestamp=(BASE_TIME + _dt.timedelta(hours=3)).isoformat(),
        )
    )
    samples = select_samples(session_rows, limit=10)
    assert [s.seq for s in samples] == [1, 2]
    assert all(s.batched_messages == () for s in samples)


async def test_a_trailing_turn_with_no_response_yet_reports_no_tools(
    db_session: Session, test_user: User, _reset_stores: None
) -> None:
    """The honest answer for an unanswered turn is "nothing", not a guess."""
    _seed(
        db_session,
        test_user,
        [
            ("inbound", "first", None),
            ("outbound", "answered", None),
            ("inbound", "still waiting", None),
        ],
    )
    fixture = await _fixture_for(test_user)
    samples = select_samples(fixture, limit=10)

    trailing = next(s for s in samples if s.seq == 3)
    assert trailing.historic_tool_names == []
    assert trailing.historic_reply == ""


# ---------------------------------------------------------------------------
# The replay clock is the turn's own timestamp
# ---------------------------------------------------------------------------


async def test_replayed_turn_is_stamped_with_its_own_time_not_now(
    db_session: Session, test_user: User, _reset_stores: None
) -> None:
    """A date-relative ask must resolve against the day it was sent.

    History rows render absolute date markers, so a run days later hands the
    model a conversation that ends last week under a header claiming today.
    Both models then resolve "this past week" to the wrong week, and the
    calendar arguments they are scored on are wrong for a reason that has
    nothing to do with either model.
    """
    _seed(
        db_session,
        test_user,
        [("inbound", "put him down for Monday to Thursday this past week", None)],
    )
    fixture = await _fixture_for(test_user)
    sample = select_samples(fixture, limit=1)[0]

    assembled = await assemble_for_sample(fixture, sample)
    turn_text = assembled.messages[-1].content
    assert isinstance(turn_text, str)

    # The stamp is the turn's own time, to the minute the row carries.
    assert "[Current time: Friday, 2026-05-01 12:01 PM" in turn_text, turn_text
    assert _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d") not in turn_text


def testsample_clock_parses_the_stored_timestamp() -> None:
    sample = ReplaySample(
        seq=1, timestamp="2026-08-30T16:48:00+00:00", message_context="this past week"
    )
    assert sample_clock(sample) == _dt.datetime(2026, 8, 30, 16, 48, tzinfo=_dt.UTC)


def testsample_clock_falls_back_to_wall_time_on_a_corrupt_timestamp() -> None:
    """A wrong clock is worse than the honest current one."""
    sample = ReplaySample(seq=1, timestamp="not a timestamp", message_context="hi")
    assert sample_clock(sample) is None


def testsample_clock_assumes_utc_for_a_naive_timestamp() -> None:
    sample = ReplaySample(seq=1, timestamp="2026-08-30T16:48:00", message_context="hi")
    clock = sample_clock(sample)
    assert clock is not None
    assert clock.tzinfo is not None


# ---------------------------------------------------------------------------
# A batch is bounded in time; a long gap means the turn went unanswered
# ---------------------------------------------------------------------------


def _row(seq: int, direction: str, body: str, at: _dt.datetime, tools: str = "") -> StoredMessage:
    return StoredMessage(
        seq=seq,
        direction=direction,
        body=body,
        timestamp=at.isoformat(),
        llm_reply_text=body if direction == "outbound" else "",
        tool_interactions_json=tools,
    )


_WRITE = '[{"tool_call_id": "t1", "name": "qb_send", "args": {}, "result": "ok"}]'


def test_rapid_fire_rows_seconds_apart_share_the_batch_response() -> None:
    rows = [
        _row(1, "inbound", "rebuild the stalls", BASE_TIME),
        _row(2, "inbound", "build and send", BASE_TIME + _dt.timedelta(seconds=2)),
        _row(3, "outbound", "sent", BASE_TIME + _dt.timedelta(seconds=8), tools=_WRITE),
    ]
    assert _historic_response(rows, 0) == ("sent", ["qb_send"])


def test_an_orphaned_turn_is_not_credited_with_a_later_turns_tool_calls() -> None:
    """An inbound the agent never answered did nothing, whatever came next.

    ``agent.inbound_recovery`` records a production inbound that sat 29 hours
    before the next message woke a batcher. Reading the whole run of inbound
    rows as one batch credits that turn with the later turn's writes, and
    ``check_safety`` then treats a candidate that invoices a customer in reply
    to "just checking in" as doing what the live agent did.
    """
    rows = [
        _row(1, "inbound", "just checking in", BASE_TIME),
        _row(2, "inbound", "go ahead and send it", BASE_TIME + _dt.timedelta(hours=3)),
        _row(3, "outbound", "sent", BASE_TIME + _dt.timedelta(hours=3, seconds=6), tools=_WRITE),
    ]
    assert _historic_response(rows, 0) == ("", [])
    assert _historic_response(rows, 1) == ("sent", ["qb_send"])


def test_a_corrupt_timestamp_does_not_hand_out_a_batch_exemption() -> None:
    rows = [
        _row(1, "inbound", "first", BASE_TIME),
        _row(2, "inbound", "second", BASE_TIME),
        _row(3, "outbound", "sent", BASE_TIME, tools=_WRITE),
    ]
    rows[1].timestamp = "not a timestamp"
    assert _historic_response(rows, 0) == ("", [])


# ---------------------------------------------------------------------------
# The decision a historic run scores, taken where the replay takes the
# candidate's
# ---------------------------------------------------------------------------


class _FindParams(BaseModel):
    q: str


class _SendParams(BaseModel):
    invoice_id: str


async def _never(**_kwargs: object) -> ToolResult:  # pragma: no cover - never invoked
    raise AssertionError("a replay must never execute a tool")


_TOOLS = {
    "qb_find": Tool(
        name="qb_find",
        description="find",
        function=_never,
        params_model=_FindParams,
        tags={ToolTags.READ_ONLY},
    ),
    # Untagged, so mutating: a write is never fed back, whoever made it.
    "qb_send": Tool(name="qb_send", description="send", function=_never, params_model=_SendParams),
}


def _interactions(*calls: tuple[str, dict[str, object]]) -> str:
    return json.dumps(
        [
            {"tool_call_id": f"t{i}", "name": name, "args": args, "result": "ok"}
            for i, (name, args) in enumerate(calls, start=1)
        ]
    )


_LOOKUP_THEN_WRITE = _interactions(
    ("qb_find", {"q": "Acme Plumbing"}), ("qb_send", {"invoice_id": "1186"})
)


def test_the_scored_decision_skips_the_lookups_the_replay_would_feed_back() -> None:
    """Both sides are scored at the same point in the turn, or neither is.

    ``execution.call_model`` advances the candidate past a read-only call the
    live turn also made and scores the first response that would need a live
    tool. Reading the incumbent's side as the literal first recorded call
    scores the two sides on different rounds, so a candidate that reproduces
    the turn exactly reads as acting where the incumbent looked something up.
    """
    rows = [
        _row(1, "inbound", "invoice Acme for the stalls", BASE_TIME),
        _row(
            2,
            "outbound",
            "sent it",
            BASE_TIME + _dt.timedelta(seconds=9),
            tools=_LOOKUP_THEN_WRITE,
        ),
    ]
    decision = _historic_first_decision(rows, 0, _TOOLS)
    assert decision.available
    assert [(c.name, c.arguments) for c in decision.calls] == [("qb_send", {"invoice_id": "1186"})]
    # What it had read by then, so the judge can be shown this side's
    # lookups exactly as it is shown the candidate's.
    assert [(item.name, item.result) for item in decision.lookups] == [("qb_find", "ok")]
    # Two calls in one flat list, but nothing was dropped: the lookup rides
    # along on ``lookups`` and is shown to the judge on both sides, so the
    # turn reads the same whichever way production batched it. Warning about
    # it would fire on the commonest multi-call shape there is.
    assert not decision.flattened


def test_calls_left_after_the_scored_one_are_reported_as_flattened() -> None:
    """Here the reading really does drop something.

    The scored write shares its flat list with a second write nobody can
    place in a round. If production asked for both at once, the incumbent's
    decision was the pair and this run scores half of it, so the turn is
    counted and the report says so.
    """
    rows = [
        _row(1, "inbound", "invoice both Acme jobs", BASE_TIME),
        _row(
            2,
            "outbound",
            "sent both",
            BASE_TIME + _dt.timedelta(seconds=9),
            tools=_interactions(
                ("qb_send", {"invoice_id": "1186"}), ("qb_send", {"invoice_id": "1187"})
            ),
        ),
    ]
    decision = _historic_first_decision(rows, 0, _TOOLS)
    assert [(c.name, c.arguments) for c in decision.calls] == [("qb_send", {"invoice_id": "1186"})]
    assert decision.flattened


def test_a_turn_of_lookups_is_scored_on_the_prose_it_ended_on() -> None:
    """The replay would have fed the lookup back and asked again.

    What came back is the reply the turn ended on, so that is the decision,
    with the lookup recorded ahead of it. Scoring the lookup itself reads a
    candidate that answered the question as a silent no-op.
    """
    rows = [
        _row(1, "inbound", "what do we owe Acme", BASE_TIME),
        _row(
            2,
            "outbound",
            "You owe them 1186.",
            BASE_TIME + _dt.timedelta(seconds=7),
            tools=_interactions(("qb_find", {"q": "Acme Plumbing"})),
        ),
    ]
    decision = _historic_first_decision(rows, 0, _TOOLS)
    assert decision.available
    assert decision.calls == ()
    assert decision.text == "You owe them 1186."
    assert [item.name for item in decision.lookups] == ["qb_find"]
    assert not decision.flattened


def test_a_lookup_this_schema_no_longer_has_is_itself_the_decision() -> None:
    """The replay stops at a call it cannot answer, so the reading does too.

    A tool that has left the schema cannot be fed back, so the candidate
    would be scored on that call. Skipping it on the incumbent side would
    once again score the two sides on different rounds.
    """
    rows = [
        _row(1, "inbound", "invoice Acme", BASE_TIME),
        _row(
            2,
            "outbound",
            "sent it",
            BASE_TIME + _dt.timedelta(seconds=9),
            tools=_LOOKUP_THEN_WRITE,
        ),
    ]
    decision = _historic_first_decision(rows, 0, {"qb_send": _TOOLS["qb_send"]})
    assert [c.name for c in decision.calls] == ["qb_find"]
    assert decision.lookups == ()


def test_the_skipping_stops_at_the_replays_round_budget() -> None:
    """A replay scores whatever the model asked for once its rounds run out."""
    rows = [
        _row(1, "inbound", "check them all", BASE_TIME),
        _row(
            2,
            "outbound",
            "here they are",
            BASE_TIME + _dt.timedelta(seconds=9),
            tools=_interactions(
                *(("qb_find", {"q": f"customer {i}"}) for i in range(MAX_REPLAY_READ_ROUNDS + 1))
            ),
        ),
    ]
    decision = _historic_first_decision(rows, 0, _TOOLS)
    assert len(decision.lookups) == MAX_REPLAY_READ_ROUNDS
    assert [(c.name, c.arguments) for c in decision.calls] == [
        ("qb_find", {"q": f"customer {MAX_REPLAY_READ_ROUNDS}"})
    ]


def test_lookups_with_nothing_recorded_after_them_are_unavailable() -> None:
    """The turn spent itself looking things up and wrote down no answer.

    What it decided next was never recorded, so there is nothing comparable
    to score. Reporting the lookup as the decision would score the two sides
    on different rounds, and reporting empty prose would read as a turn that
    answered with nothing.
    """
    rows = [
        _row(1, "inbound", "what do we owe Acme", BASE_TIME),
        _row(
            2,
            "outbound",
            "",
            BASE_TIME + _dt.timedelta(seconds=7),
            tools=_interactions(("qb_find", {"q": "Acme Plumbing"})),
        ),
    ]
    assert not _historic_first_decision(rows, 0, _TOOLS).available


def test_a_turn_that_called_nothing_has_its_prose_as_its_first_decision() -> None:
    rows = [
        _row(1, "inbound", "just checking in", BASE_TIME),
        _row(2, "outbound", "all good", BASE_TIME + _dt.timedelta(seconds=4)),
    ]
    decision = _historic_first_decision(rows, 0, _TOOLS)
    assert decision.available
    assert decision.calls == ()
    assert decision.text == "all good"
    assert not decision.flattened


def test_an_unanswered_turn_has_no_first_decision() -> None:
    """Distinct from "answered with nothing", which is a decision."""
    rows = [_row(1, "inbound", "are you there", BASE_TIME)]
    assert not _historic_first_decision(rows, 0, _TOOLS).available


def test_unparseable_interactions_make_the_first_decision_unavailable() -> None:
    """Calls were made and their record is gone, which is not "did nothing".

    Read as "did nothing" this turn would exempt a candidate's write from
    ``unrequested_mutation`` on a turn production also wrote on.
    """
    rows = [
        _row(1, "inbound", "invoice Acme", BASE_TIME),
        _row(2, "outbound", "sent", BASE_TIME + _dt.timedelta(seconds=5), tools="{not json"),
    ]
    assert not _historic_first_decision(rows, 0, _TOOLS).available


def test_an_empty_interaction_list_is_a_decision_not_a_loss() -> None:
    """``[ ]`` is a turn that called nothing, which is a comparable decision.

    Matching the raw text against ``"[]"`` read any other spelling of the
    empty list as a record that did not survive, which drops a perfectly
    comparable turn out of every paired comparison.
    """
    rows = [
        _row(1, "inbound", "just checking in", BASE_TIME),
        _row(2, "outbound", "all good", BASE_TIME + _dt.timedelta(seconds=4), tools="[ ]"),
    ]
    decision = _historic_first_decision(rows, 0, _TOOLS)
    assert decision.available
    assert decision.calls == ()


def test_a_lost_first_interaction_does_not_promote_the_second() -> None:
    """Part of the record gone is unavailable, not "it opened with the write".

    ``_parse_tool_interactions`` drops the entries that fail validation and
    keeps the rest, so a corrupt first entry left the turn's *second* call
    standing as its opening move. On a lookup-then-write turn that reports
    the incumbent as having opened with the write.
    """
    rows = [
        _row(1, "inbound", "invoice Acme", BASE_TIME),
        _row(
            2,
            "outbound",
            "sent",
            BASE_TIME + _dt.timedelta(seconds=5),
            tools=json.dumps(
                [
                    {"name": 5, "args": "not a mapping"},
                    {"name": "qb_send", "args": {"invoice_id": "1186"}},
                ]
            ),
        ),
    ]
    assert not _historic_first_decision(rows, 0, _TOOLS).available


def test_select_samples_carries_the_decision_onto_the_sample() -> None:
    rows = [
        _row(1, "inbound", "invoice Acme for the stalls", BASE_TIME),
        _row(
            2,
            "outbound",
            "sent it",
            BASE_TIME + _dt.timedelta(seconds=9),
            tools=_LOOKUP_THEN_WRITE,
        ),
    ]
    fixture = ReplayFixture(user=User(id="u"), rows=rows)
    fixture.tools_by_name = _TOOLS
    sample = select_samples(fixture, 1)[0]
    assert sample.historic_decision_available
    assert [(c.name, c.arguments) for c in sample.historic_first_calls] == [
        ("qb_send", {"invoice_id": "1186"})
    ]
    assert [item.name for item in sample.historic_decision_lookups] == ["qb_find"]
    assert not sample.historic_calls_flattened


def test_a_replay_run_never_reconstructs_the_recorded_decision() -> None:
    """It will call the incumbent, so walking the record is wasted work.

    Worse than wasted: the walk logs a warning for every turn whose recorded
    interactions will not parse, about a decision the run was never going to
    read.
    """
    rows = [
        _row(1, "inbound", "invoice Acme for the stalls", BASE_TIME),
        _row(
            2,
            "outbound",
            "sent it",
            BASE_TIME + _dt.timedelta(seconds=9),
            tools=_LOOKUP_THEN_WRITE,
        ),
    ]
    fixture = ReplayFixture(user=User(id="u"), rows=rows)
    fixture.tools_by_name = _TOOLS
    sample = select_samples(fixture, 1, reconstruct_incumbent=False)[0]
    assert sample.historic_first_calls == ()
    assert sample.historic_decision_lookups == ()
    assert not sample.historic_calls_flattened
    # The turn's own calls are still read: they are context for the judge and
    # evidence for the safety checks on a replayed run too.
    assert sample.historic_tool_names == ["qb_find", "qb_send"]
