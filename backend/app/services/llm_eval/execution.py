"""Model dispatch for the evaluator.

Mirrors ``ClawboltAgent._call_llm_with_retry`` in everything that shapes the
request (cache breakpoints, system-prompt caching, tool caching, thinking
config), and the live loop's retry of a reply truncated with no tool call,
and deliberately drops everything that shapes the *conversation*:
no observers fire, no typing indicator is sent, no context-overflow retry
re-trims the prompt, and no usage is logged against the user's quota. An
evaluation must not appear in the user's spend or in the operator's
telemetry as if it were real traffic.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Sequence
from typing import Any, cast

from any_llm import amessages
from any_llm.exceptions import AnyLLMError
from any_llm.types.messages import MessageResponse

from backend.app.agent.core import _MAX_TOKENS_CEILING, AssembledPrompt
from backend.app.agent.llm_parsing import get_response_text, parse_tool_calls
from backend.app.agent.messages import (
    AgentMessage,
    AssistantMessage,
    ToolCallRequest,
    ToolResultMessage,
    messages_to_messages_api,
)
from backend.app.agent.tools.base import Tool
from backend.app.config import settings
from backend.app.services.llm_eval import metrics
from backend.app.services.llm_eval.types import ModelCallResult, RecordedToolResult, ToolCall
from backend.app.services.llm_service import (
    LLMTarget,
    apply_history_cache_breakpoint,
    apply_in_turn_cache_breakpoint,
    apply_tool_caching,
    fit_max_tokens_to_reasoning,
    prepare_system_with_caching,
)

logger = logging.getLogger(__name__)


def _serialize_blocks(response: MessageResponse) -> list[dict[str, Any]]:
    """Dump response content blocks to plain JSON-safe dicts."""
    blocks: list[dict[str, Any]] = []
    for block in response.content:
        try:
            blocks.append(block.model_dump(mode="json"))
        except Exception:
            logger.exception("Failed to serialize an eval response block; skipping")
    return blocks


def _add_usage(total: ModelCallResult, attempt: ModelCallResult) -> None:
    """Fold a spent attempt's usage into the result that replaced it.

    Production pays for a truncated attempt before it retries, so the retry's
    cost alone would understate what the turn costs on this model.
    """
    total.input_tokens += attempt.input_tokens
    total.output_tokens += attempt.output_tokens
    total.cache_creation_input_tokens += attempt.cache_creation_input_tokens
    total.cache_read_input_tokens += attempt.cache_read_input_tokens
    total.latency_ms += attempt.latency_ms


def _needs_truncation_retry(result: ModelCallResult, max_tokens: int) -> bool:
    """Whether production would discard *result* and ask again with a larger budget.

    Mirrors the guard in ``ClawboltAgent._run_loop``: a response cut off at
    ``max_tokens`` with no parseable tool call is not a reply (typically the
    reasoning spent the budget), so the loop doubles the budget up to
    ``_MAX_TOKENS_CEILING`` and retries rather than delivering it.
    """
    return (
        not result.error
        and not result.tool_calls
        and result.stop_reason == "max_tokens"
        and max_tokens < _MAX_TOKENS_CEILING
    )


# Extra rounds a replay may spend on lookups before its decision is scored.
# Production runs up to ``MAX_TOOL_ROUNDS``, but a lookup-then-act turn needs
# one or two, and every round is another paid call on both sides. A model
# still looking things up after this many is scored on the lookup it asked for.
MAX_REPLAY_READ_ROUNDS = 3


async def call_model(
    assembled: AssembledPrompt,
    tool_schemas: list[dict[str, Any]] | None,
    *,
    target: LLMTarget,
    reasoning_effort: str,
    max_tokens: int | None = None,
    tools_by_name: dict[str, Tool] | None = None,
    recorded: Sequence[RecordedToolResult] = (),
) -> ModelCallResult:
    """Replay one turn through one model and return the decision to score.

    Provider errors are captured onto the result rather than raised: one
    model failing on one turn is a data point about that model, not a
    reason to abandon a run that may be 90 turns deep.

    *reasoning_effort* is per side and recorded on the run. The two models in
    a comparison need not share one: effort is not portable across families,
    so forcing the candidate to the incumbent's setting measures the setting
    rather than the candidate, and can be rejected outright by a model whose
    endpoint spells reasoning differently.

    **Lookups.** Production looks things up before it writes, so scoring only
    a model's first call penalized a model that did the same against one that
    guessed an ID and wrote. When every call in a response is read-only and
    matches a call the live turn made (*recorded*, from
    ``ReplaySample.historic_tool_results``), the recorded result is fed back
    and the model is asked again, for up to ``MAX_REPLAY_READ_ROUNDS`` extra
    rounds. Nothing is executed: a write, a read the live turn never made, or
    a read with different arguments ends the replay, and that response is the
    decision scored. Without *tools_by_name* the replay is single-round.

    Two budget rules keep the replay from charging a model for limits
    production does not impose. A thinking budget at or above ``max_tokens``
    raises ``max_tokens`` to fit it (``fit_max_tokens_to_reasoning``), and a
    reply truncated with no tool call is retried at a doubled budget the way
    the live loop retries it.
    """
    reasoning = target.reasoning_kwargs(reasoning_effort)
    base_budget = fit_max_tokens_to_reasoning(
        max_tokens or settings.llm_max_tokens_agent, reasoning
    )
    messages: list[AgentMessage] = list(assembled.messages)
    lookups: list[RecordedToolResult] = []
    spent: ModelCallResult | None = None
    for round_number in range(MAX_REPLAY_READ_ROUNDS + 1):
        result = await _decide(
            messages, tool_schemas, target=target, reasoning=reasoning, max_tokens=base_budget
        )
        if spent is not None:
            _add_usage(result, spent)
            result.truncation_retries += spent.truncation_retries
        result.replayed_lookups = list(lookups)
        if tools_by_name is None or round_number == MAX_REPLAY_READ_ROUNDS:
            return result
        fed = replayable_lookups(result, tools_by_name, recorded)
        if fed is None:
            return result
        messages.extend(_lookup_round_messages(result, fed, round_number))
        lookups.extend(fed)
        spent = result
    raise AssertionError("unreachable: the last round always returns")


def replayable_lookups(
    result: ModelCallResult,
    tools_by_name: dict[str, Tool],
    recorded: Sequence[RecordedToolResult],
) -> list[RecordedToolResult] | None:
    """The recorded results to feed back for *result*, or None to stop here.

    Continues only when every call is a read the live turn also made with the
    same arguments (compared after the params model fills defaults). One call
    that would need a live tool, a write included, ends the replay: a partial
    round cannot be answered without executing something.
    """
    if result.error or not result.tool_calls or result.stop_reason == "max_tokens":
        return None
    fed: list[RecordedToolResult] = []
    for call in result.tool_calls:
        tool = tools_by_name.get(call.name)
        if tool is None or metrics.is_mutating_call(tool, call.arguments):
            return None
        wanted = metrics.normalized_args(tool, call.arguments)
        match = next(
            (
                r
                for r in recorded
                if r.name == call.name and metrics.normalized_args(tool, r.arguments) == wanted
            ),
            None,
        )
        if match is None:
            return None
        fed.append(
            RecordedToolResult(
                name=call.name,
                arguments=call.arguments,
                result=match.result,
                is_error=match.is_error,
            )
        )
    return fed


def _lookup_round_messages(
    result: ModelCallResult, fed: list[RecordedToolResult], round_number: int
) -> list[AgentMessage]:
    """The assistant turn and tool results the live loop appends after a round.

    Built the way ``ClawboltAgent._run_loop`` builds them (text plus
    ``tool_use`` blocks, then one result per call), so the next round reads
    the same shape production's next round reads.
    """
    requests = [
        ToolCallRequest(
            id=call.id or f"eval_replay_{round_number}_{index}",
            name=call.name,
            arguments=call.arguments,
        )
        for index, call in enumerate(result.tool_calls)
    ]
    messages: list[AgentMessage] = [
        AssistantMessage(content=result.text or None, tool_calls=requests)
    ]
    messages.extend(
        ToolResultMessage(tool_call_id=request.id, content=item.result, is_error=item.is_error)
        for request, item in zip(requests, fed, strict=True)
    )
    return messages


async def _decide(
    messages: list[AgentMessage],
    tool_schemas: list[dict[str, Any]] | None,
    *,
    target: LLMTarget,
    reasoning: dict[str, Any],
    max_tokens: int,
) -> ModelCallResult:
    """One round, retried at a larger budget when production would retry it."""
    budget = max_tokens
    result = await _dispatch(
        messages, tool_schemas, target=target, reasoning=reasoning, max_tokens=budget
    )
    while _needs_truncation_retry(result, budget):
        budget = min(budget * 2, _MAX_TOKENS_CEILING)
        logger.info(
            "Eval call for %s truncated with no tool call; retrying at %d",
            target.describe(),
            budget,
        )
        retry = await _dispatch(
            messages, tool_schemas, target=target, reasoning=reasoning, max_tokens=budget
        )
        _add_usage(retry, result)
        retry.truncation_retries = result.truncation_retries + 1
        result = retry
    return result


async def _dispatch(
    messages: list[AgentMessage],
    tool_schemas: list[dict[str, Any]] | None,
    *,
    target: LLMTarget,
    reasoning: dict[str, Any],
    max_tokens: int,
) -> ModelCallResult:
    """One provider call, with the same request shaping as the live loop."""
    system_str, msg_dicts = messages_to_messages_api(messages)

    # The cache helpers stamp ``cache_control`` markers onto the dicts they
    # are given. Both models in a comparison are handed the same assembled
    # prompt, and the two providers may not agree on whether markers belong,
    # so each call works on its own copy. Without this the second model
    # would inherit the first's markers.
    msg_dicts = copy.deepcopy(msg_dicts)
    schemas = copy.deepcopy(tool_schemas) if tool_schemas else None

    msg_dicts = apply_history_cache_breakpoint(msg_dicts, target)
    msg_dicts = apply_in_turn_cache_breakpoint(msg_dicts, target)
    system: str | list[dict[str, Any]] | None = system_str
    if system is not None:
        system = prepare_system_with_caching(system, target)
    if schemas:
        schemas = apply_tool_caching(schemas, target)

    started = time.monotonic()
    try:
        response = cast(
            MessageResponse,
            await amessages(
                **target.connection_kwargs(),
                system=system,
                messages=msg_dicts,
                tools=schemas,
                max_tokens=max_tokens,
                **reasoning,
            ),
        )
    except AnyLLMError as exc:
        logger.warning("Eval call failed for %s: %s", target.describe(), exc)
        return ModelCallResult(
            provider=target.provider,
            model=target.model,
            latency_ms=(time.monotonic() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected eval call failure for %s", target.describe())
        return ModelCallResult(
            provider=target.provider,
            model=target.model,
            latency_ms=(time.monotonic() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )

    latency_ms = (time.monotonic() - started) * 1000
    usage = response.usage
    # ``arguments`` is None when the provider returned a tool input that was
    # not a dict. Recorded as empty so the args validator sees it and reports
    # ``invalid_args`` rather than the call silently vanishing from the diff.
    tool_calls = [
        ToolCall(name=c.name, arguments=c.arguments or {}, id=c.id)
        for c in parse_tool_calls(response)
    ]
    return ModelCallResult(
        provider=target.provider,
        model=target.model,
        text=get_response_text(response),
        tool_calls=tool_calls,
        content_blocks=_serialize_blocks(response),
        stop_reason=response.stop_reason,
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        cache_creation_input_tokens=(usage.cache_creation_input_tokens or 0) if usage else 0,
        cache_read_input_tokens=(usage.cache_read_input_tokens or 0) if usage else 0,
        latency_ms=latency_ms,
    )
