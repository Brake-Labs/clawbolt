"""Admin error alerting: turn application-level ERROR logs into operator email.

Reactive half of the monitoring stack. ``health_monitor`` is the proactive half;
this module answers "something just broke for a user, tell me now", while the
health monitor answers "something is broken and nobody has hit it yet".

Design:

- **A logging handler, not call sites.** Every error either layer already logs
  becomes an alert, including the ones nobody thought to instrument. Adding a
  new failure path costs nothing: ``logger.exception(...)`` is already the
  convention in both repos.
- **Grouped by fingerprint.** A broken integration logs the same error
  thousands of times a minute. Records collapse on
  ``(logger, exception type, log template)``. The log *template*
  (``record.msg`` before %-substitution) is the grouping key rather than the
  formatted message, so "user abc failed" and "user def failed" are one group
  instead of two.
- **Emailed on transitions, not on recurrence.** A group is an incident: its
  first occurrence emails, repeats while it is open are counted silently, and
  once it has gone ``alert_dedupe_minutes`` without recurring a single
  "resolved" email closes it, carrying the total count and duration. A
  background sweep failing the same way every two minutes for a day is two
  emails, not 48. Open incidents live in memory, so a restart forgets them: a
  problem still happening after a deploy opens again, and one that stopped
  across the deploy never gets its resolved email.
- **Batched.** A flush loop coalesces every eligible group into a single email
  per tick, so one deploy going bad is one email listing five problems.
- **Never blocks the logging call.** ``emit()`` only mutates an in-memory dict
  under a lock. SMTP happens on the flush task. Logging is called from sync
  code, worker threads, and inside exception handlers; doing I/O there would
  be a deadlock and latency risk.
- **Tracebacks are formatted immediately.** Holding a live ``exc_info`` tuple
  pins every frame's locals (request objects, DB sessions, LLM payloads) alive
  until the next flush. Format to text on capture, drop the tuple.
- **Dormant without config.** No SMTP host or no resolvable recipient means
  the handler is never installed, so dev/local and CI never send.

Privacy: alert bodies carry tracebacks and log messages, which can contain user
data. They go only to the single operator address resolved from
``alert_email`` / ``admin_email``, and the alert body is never itself logged.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from backend.app.config import settings
from backend.app.observability import get_request_id
from backend.app.services import email_service

logger = logging.getLogger(__name__)

# Logger trees whose ERROR records become alerts. ``uvicorn.error`` carries
# unhandled ASGI exceptions ("Exception in ASGI application"), which never
# reach a ``backend`` logger, so a 500 from a route would otherwise be
# invisible here.
CAPTURED_LOGGERS = ("backend", "uvicorn.error")

# Never capture from the alert delivery path itself. Without this, an SES
# outage logs ERROR from email_service, which enqueues an alert about the
# failure to send alerts, which fails to send, forever.
EXCLUDED_LOGGERS = (
    "backend.app.services.admin_alerts",
    "backend.app.services.email_service",
)

# Truncation bounds. A runaway recursion produces megabyte tracebacks, and an
# email with 200 distinct problems in it is unreadable anyway.
_MAX_TRACEBACK_CHARS = 4000
_MAX_MESSAGE_CHARS = 1000
_MAX_PENDING_GROUPS = 200
_MAX_GROUPS_PER_EMAIL = 25
# Distinct argument/result samples kept per tool-failure group, and the cap on
# each. One example of a broken call is diagnostic; twenty is a data dump.
_MAX_SAMPLES_PER_GROUP = 3
_MAX_SAMPLE_CHARS = 500


def _recipient() -> str:
    """Resolve the operator address, preferring the alert-specific override."""
    return settings.alert_email or settings.admin_email


def is_enabled() -> bool:
    """True when alerts are configured well enough to actually deliver."""
    return bool(
        settings.alerts_enabled and settings.smtp_host and settings.smtp_from_email and _recipient()
    )


def _should_capture(logger_name: str) -> bool:
    """True when *logger_name* is in a captured tree and not on the send path.

    Excluded prefixes are checked first so a more specific exclusion beats the
    broad ``backend`` inclusion.
    """
    if logger_name.startswith(EXCLUDED_LOGGERS):
        return False
    return logger_name.startswith(CAPTURED_LOGGERS)


@dataclass(frozen=True)
class AlertSummary:
    """One grouped error, ready to render into the operator email."""

    title: str
    logger_name: str
    level: str
    count: int
    message: str
    traceback_text: str | None
    request_id: str
    first_seen: datetime
    last_seen: datetime


@dataclass(frozen=True)
class ToolFailureSummary:
    """One grouped tool failure, ready to render into the operator email.

    ``user_count`` counts distinct users, which is the number that separates
    "one tenant's token expired" from "the integration is down for everyone".
    ``samples`` carries argument and result detail and is populated only from
    users who opted into data sharing; failures from everyone else still raise
    ``count`` and ``user_count`` so an incident is never invisible.
    """

    tool_name: str
    error_kind: str
    count: int
    user_count: int
    consented_user_count: int
    samples: list[str]
    first_seen: datetime
    last_seen: datetime


@dataclass(frozen=True)
class ResolvedSummary:
    """One incident that has stopped recurring, ready for the resolved email.

    ``count`` is every occurrence since the incident opened, including the
    ones in its opening email. ``user_count`` is populated for tool failures
    only; application errors are not attributed to users.
    """

    title: str
    count: int
    user_count: int
    first_seen: datetime
    last_seen: datetime


@dataclass
class _OpenIncident:
    """A group whose opening email has gone out, tracked until it goes quiet."""

    title: str
    first_seen: datetime
    last_seen: datetime
    count: int
    user_ids: set[str] = field(default_factory=set)

    def touch(self, user_id: str | None = None) -> None:
        self.count += 1
        self.last_seen = datetime.now(UTC)
        if user_id is not None:
            self.user_ids.add(user_id)

    def to_resolved(self) -> ResolvedSummary:
        return ResolvedSummary(
            title=self.title,
            count=self.count,
            user_count=len(self.user_ids),
            first_seen=self.first_seen,
            last_seen=self.last_seen,
        )


@dataclass
class _PendingToolFailure:
    """Mutable accumulator for one (tool, error kind) between flushes."""

    tool_name: str
    error_kind: str
    first_seen: datetime
    last_seen: datetime
    count: int = 1
    user_ids: set[str] = field(default_factory=set)
    consented_user_ids: set[str] = field(default_factory=set)
    samples: list[str] = field(default_factory=list)

    def merge(self, user_id: str, sample: str | None) -> None:
        self.count += 1
        self.last_seen = datetime.now(UTC)
        self.user_ids.add(user_id)
        if sample is not None:
            self.consented_user_ids.add(user_id)
            # Keep a bounded handful. Twenty copies of one stack of arguments
            # tells you nothing the first one did not.
            if len(self.samples) < _MAX_SAMPLES_PER_GROUP and sample not in self.samples:
                self.samples.append(sample)

    def to_summary(self) -> ToolFailureSummary:
        return ToolFailureSummary(
            tool_name=self.tool_name,
            error_kind=self.error_kind,
            count=self.count,
            user_count=len(self.user_ids),
            consented_user_count=len(self.consented_user_ids),
            samples=list(self.samples),
            first_seen=self.first_seen,
            last_seen=self.last_seen,
        )


@dataclass
class _PendingGroup:
    """Mutable accumulator for one fingerprint between flushes."""

    logger_name: str
    level: str
    exc_type: str | None
    message: str
    traceback_text: str | None
    request_id: str
    first_seen: datetime
    last_seen: datetime
    count: int = 1

    def merge(self, message: str, traceback_text: str | None, request_id: str) -> None:
        """Fold a repeat occurrence in, keeping the most recent detail.

        The newest traceback wins over the first: when an error recurs, the
        latest occurrence is the one still reproducible.
        """
        self.count += 1
        self.last_seen = datetime.now(UTC)
        self.message = message
        if traceback_text is not None:
            self.traceback_text = traceback_text
        if request_id != "-":
            self.request_id = request_id

    def to_summary(self) -> AlertSummary:
        exc = f"{self.exc_type}: " if self.exc_type else ""
        return AlertSummary(
            title=f"{exc}{self.message}"[:160],
            logger_name=self.logger_name,
            level=self.level,
            count=self.count,
            message=self.message,
            traceback_text=self.traceback_text,
            request_id=self.request_id,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
        )


@dataclass
class _AlertStore:
    """Thread-safe accumulator and incident state.

    ``_pending`` holds groups whose opening email has not gone out yet.
    ``_open`` holds groups that have been emailed and are still recurring; a
    record for one of those only bumps its count. Both are written from
    arbitrary threads via ``emit()`` and drained by the flush task, so every
    access takes ``_lock``. The lock is held only for dict mutation, never
    across the SMTP call.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _pending: dict[str, _PendingGroup] = field(default_factory=dict)
    _pending_tools: dict[str, _PendingToolFailure] = field(default_factory=dict)
    _open: dict[str, _OpenIncident] = field(default_factory=dict)
    _email_times: deque[float] = field(default_factory=deque)
    _overflow_dropped: int = 0

    def record(
        self,
        fingerprint: str,
        logger_name: str,
        level: str,
        exc_type: str | None,
        message: str,
        traceback_text: str | None,
        request_id: str,
    ) -> None:
        """Accumulate one error occurrence. Cheap, non-blocking, thread-safe."""
        now = datetime.now(UTC)
        with self._lock:
            incident = self._open.get(fingerprint)
            if incident is not None:
                # Already reported and still happening: count it, stay quiet.
                incident.touch()
                return
            existing = self._pending.get(fingerprint)
            if existing is not None:
                existing.merge(message, traceback_text, request_id)
                return
            if len(self._pending) >= _MAX_PENDING_GROUPS:
                # Distinct-fingerprint flood. Drop rather than grow unbounded;
                # the count surfaces in the next email so the truncation is
                # never silent.
                self._overflow_dropped += 1
                return
            self._pending[fingerprint] = _PendingGroup(
                logger_name=logger_name,
                level=level,
                exc_type=exc_type,
                message=message,
                traceback_text=traceback_text,
                request_id=request_id,
                first_seen=now,
                last_seen=now,
            )

    def record_tool_failure(
        self,
        tool_name: str,
        error_kind: str,
        user_id: str,
        sample: str | None,
    ) -> None:
        """Accumulate one tool failure. Cheap, non-blocking, thread-safe.

        *sample* carries argument and result detail and must already be
        consent-gated and redacted by the caller; pass ``None`` for a user who
        has not opted into data sharing. The occurrence still counts either
        way, so an outage confined to non-consenting users is visible as a
        number even when nothing about it can be shown.
        """
        now = datetime.now(UTC)
        fingerprint = f"tool|{tool_name}|{error_kind}"
        with self._lock:
            incident = self._open.get(fingerprint)
            if incident is not None:
                incident.touch(user_id)
                return
            existing = self._pending_tools.get(fingerprint)
            if existing is not None:
                existing.merge(user_id, sample)
                return
            if len(self._pending_tools) >= _MAX_PENDING_GROUPS:
                self._overflow_dropped += 1
                return
            group = _PendingToolFailure(
                tool_name=tool_name,
                error_kind=error_kind,
                first_seen=now,
                last_seen=now,
            )
            group.user_ids.add(user_id)
            if sample is not None:
                group.consented_user_ids.add(user_id)
                group.samples.append(sample)
            self._pending_tools[fingerprint] = group

    def _at_hourly_cap(self) -> bool:
        """True when the hourly email ceiling has been reached.

        Counts emails sent, not alerts raised. Caller holds ``_lock``.
        """
        cap = max(1, settings.alert_max_emails_per_hour)
        now = time.monotonic()
        while self._email_times and now - self._email_times[0] > 3600:
            self._email_times.popleft()
        return len(self._email_times) >= cap

    def _take_new(
        self,
    ) -> tuple[list[tuple[str, _PendingGroup]], list[tuple[str, _PendingToolFailure]], int]:
        """Remove and return the groups whose opening email is due.

        Returns ``(errors, tools, dropped)``. The accumulators are returned
        rather than summaries: a failed send has to put them back, and a tool
        summary has already collapsed its distinct-user sets into counts that
        cannot be merged with new arrivals without double counting. Both kinds
        share the per-email cap, so a tool-failure storm cannot crowd
        application errors out of the message.
        """
        with self._lock:
            errors: list[tuple[str, _PendingGroup]] = []
            for fingerprint, group in list(self._pending.items())[:_MAX_GROUPS_PER_EMAIL]:
                errors.append((fingerprint, group))
                del self._pending[fingerprint]
            tools: list[tuple[str, _PendingToolFailure]] = []
            for fingerprint, tool_group in list(self._pending_tools.items())[
                :_MAX_GROUPS_PER_EMAIL
            ]:
                tools.append((fingerprint, tool_group))
                del self._pending_tools[fingerprint]
            dropped = self._overflow_dropped
            self._overflow_dropped = 0
            return errors, tools, dropped

    def _take_resolved(self) -> list[tuple[str, ResolvedSummary]]:
        """Return open incidents that have gone quiet for the whole window.

        They stay in ``_open`` until the resolved email is confirmed sent, so a
        failed send retries on the next tick instead of losing the notice.
        """
        quiet = timedelta(minutes=max(0, settings.alert_dedupe_minutes))
        now = datetime.now(UTC)
        with self._lock:
            resolved: list[tuple[str, ResolvedSummary]] = []
            for fingerprint, incident in self._open.items():
                if now - incident.last_seen < quiet:
                    continue
                resolved.append((fingerprint, incident.to_resolved()))
                if len(resolved) >= _MAX_GROUPS_PER_EMAIL:
                    break
            return resolved

    def _mark_sent(
        self,
        errors: list[tuple[str, _PendingGroup]],
        tools: list[tuple[str, _PendingToolFailure]],
        resolved: list[tuple[str, ResolvedSummary]],
    ) -> None:
        """Open the newly reported incidents and close the resolved ones."""
        with self._lock:
            self._email_times.append(time.monotonic())
            for fingerprint, _ in resolved:
                self._open.pop(fingerprint, None)
            for fingerprint, group in errors:
                incident = _OpenIncident(
                    title=group.to_summary().title,
                    first_seen=group.first_seen,
                    last_seen=group.last_seen,
                    count=group.count,
                )
                # Occurrences that arrived during the send landed in pending,
                # because the incident was not open yet. Fold them in.
                late = self._pending.pop(fingerprint, None)
                if late is not None:
                    incident.count += late.count
                    incident.last_seen = late.last_seen
                self._open[fingerprint] = incident
            for fingerprint, tool_group in tools:
                incident = _OpenIncident(
                    title=f"{tool_group.tool_name} {tool_group.error_kind}",
                    first_seen=tool_group.first_seen,
                    last_seen=tool_group.last_seen,
                    count=tool_group.count,
                    user_ids=set(tool_group.user_ids),
                )
                late_tool = self._pending_tools.pop(fingerprint, None)
                if late_tool is not None:
                    incident.count += late_tool.count
                    incident.last_seen = late_tool.last_seen
                    incident.user_ids |= late_tool.user_ids
                self._open[fingerprint] = incident

    async def flush(self) -> bool:
        """Send one batched email: new incidents opened, old ones resolved.

        Returns True when an email was sent. Best-effort: on send failure every
        group goes back where it came from, so the next tick retries the same
        notice rather than dropping it.
        """
        with self._lock:
            if self._at_hourly_cap():
                return False
        errors, tools, dropped = self._take_new()
        resolved = self._take_resolved()
        if not errors and not tools and not resolved:
            # Nothing to send; an overflow count alone waits for a real email.
            self._restore(errors, tools, dropped)
            return False
        sent = await email_service.send_admin_alert(
            _recipient(),
            [group.to_summary() for _, group in errors],
            dropped,
            tool_failures=[group.to_summary() for _, group in tools],
            resolved=[summary for _, summary in resolved],
        )
        if sent:
            self._mark_sent(errors, tools, resolved)
        else:
            self._restore(errors, tools, dropped)
        return sent

    def _restore(
        self,
        errors: list[tuple[str, _PendingGroup]],
        tools: list[tuple[str, _PendingToolFailure]],
        dropped: int,
    ) -> None:
        """Return groups to pending after a failed send, merging with new arrivals.

        Without this a failed SMTP call silently discards the opening email,
        and since the group never opened, nothing would ever report it.
        """
        with self._lock:
            self._overflow_dropped += dropped
            for fingerprint, group in errors:
                current = self._pending.get(fingerprint)
                if current is None:
                    self._pending[fingerprint] = group
                    continue
                # Arrivals during the send attempt are newer; keep their detail.
                current.count += group.count
                current.first_seen = min(current.first_seen, group.first_seen)
            for fingerprint, tool_group in tools:
                current_tool = self._pending_tools.get(fingerprint)
                if current_tool is None:
                    self._pending_tools[fingerprint] = tool_group
                    continue
                current_tool.count += tool_group.count
                current_tool.first_seen = min(current_tool.first_seen, tool_group.first_seen)
                current_tool.user_ids |= tool_group.user_ids
                current_tool.consented_user_ids |= tool_group.consented_user_ids
                for sample in tool_group.samples:
                    if (
                        len(current_tool.samples) < _MAX_SAMPLES_PER_GROUP
                        and sample not in current_tool.samples
                    ):
                        current_tool.samples.append(sample)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending) + len(self._pending_tools)

    def open_count(self) -> int:
        with self._lock:
            return len(self._open)

    def reset(self) -> None:
        """Clear all state. For tests."""
        with self._lock:
            self._pending.clear()
            self._pending_tools.clear()
            self._open.clear()
            self._email_times.clear()
            self._overflow_dropped = 0


_store = _AlertStore()


def record_tool_failure(
    tool_name: str,
    error_kind: str,
    user_id: str,
    sample: str | None,
) -> None:
    """Accumulate one agent tool failure into the next operator email.

    Module-level entry point mirroring the logging handler's ``emit``: callers
    outside this module should not reach into the store. *sample* must already
    be consent-gated and redacted, or ``None``.
    """
    _store.record_tool_failure(
        tool_name=tool_name,
        error_kind=error_kind,
        user_id=user_id,
        sample=sample[:_MAX_SAMPLE_CHARS] if sample is not None else None,
    )


class AdminAlertHandler(logging.Handler):
    """Logging handler that accumulates ERROR records for the operator email."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)

    def emit(self, record: logging.LogRecord) -> None:
        """Fingerprint and accumulate. Must never raise, never block."""
        try:
            if not _should_capture(record.name):
                return

            exc_type: str | None = None
            traceback_text: str | None = None
            if record.exc_info and record.exc_info[1] is not None:
                exc_class, exc_value, exc_tb = record.exc_info
                exc_type = type(exc_value).__name__ if exc_class is not None else None
                # Format now: holding exc_tb would pin every frame's locals
                # alive until the next flush.
                traceback_text = "".join(traceback.format_exception(exc_class, exc_value, exc_tb))[
                    :_MAX_TRACEBACK_CHARS
                ]

            # Group on the unformatted template so per-user / per-id variation
            # in the arguments collapses into one alert.
            template = str(record.msg)
            fingerprint = f"{record.name}|{exc_type or ''}|{template}"

            _store.record(
                fingerprint=fingerprint,
                logger_name=record.name,
                level=record.levelname,
                exc_type=exc_type,
                message=record.getMessage()[:_MAX_MESSAGE_CHARS],
                traceback_text=traceback_text,
                request_id=get_request_id(),
            )
        except Exception:
            # Handler contract: never propagate out of a logging call.
            self.handleError(record)


_handler: AdminAlertHandler | None = None
_flush_task: asyncio.Task[None] | None = None


def install_alert_handler() -> bool:
    """Attach the handler to the captured logger trees. Idempotent.

    Returns True when installed (or already installed), False when alerting is
    not configured.

    Attached per-tree rather than on the root logger because ``uvicorn.error``
    propagates to ``uvicorn``, which uvicorn's own log config marks
    ``propagate: False`` -- records from it never reach root handlers.

    Safe to call more than once, and it must be: uvicorn applies its
    ``dictConfig`` after this module's import-time install, and ``dictConfig``
    strips existing handlers from every logger it names (including
    ``uvicorn.error``). ``start_alert_flusher()`` re-installs from the
    lifespan, which runs after that.
    """
    global _handler
    if not is_enabled():
        return False
    if _handler is None:
        _handler = AdminAlertHandler()
    for name in CAPTURED_LOGGERS:
        target = logging.getLogger(name)
        if not any(isinstance(h, AdminAlertHandler) for h in target.handlers):
            target.addHandler(_handler)
    return True


async def _flush_loop() -> None:
    """Drain accumulated alerts on a timer until cancelled."""
    interval = max(5, settings.alert_flush_interval_seconds)
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await _store.flush()
            except Exception:
                # Excluded logger, so this cannot feed back into the store.
                logger.exception("Admin alert flush failed")
    except asyncio.CancelledError:
        # Final best-effort drain so a clean shutdown does not swallow the
        # errors that may have caused it.
        try:
            await _store.flush()
        except Exception:
            logger.exception("Admin alert final flush failed")
        raise


def start_alert_flusher() -> bool:
    """Re-install the handler and start the flush task. Call from the lifespan."""
    global _flush_task
    if not install_alert_handler():
        logger.info("Admin error alerts disabled (no SMTP host or no recipient configured)")
        return False
    if _flush_task is None or _flush_task.done():
        _flush_task = asyncio.create_task(_flush_loop(), name="admin-alert-flush")
    logger.info(
        "Admin error alerts active: checked every %ds, one email when a group "
        "starts and one when it has been quiet for %dm",
        settings.alert_flush_interval_seconds,
        settings.alert_dedupe_minutes,
    )
    return True


def stop_alert_flusher() -> None:
    """Cancel the flush task. Call from lifespan shutdown."""
    global _flush_task
    if _flush_task is not None and not _flush_task.done():
        _flush_task.cancel()
    _flush_task = None


def pending_group_count() -> int:
    """Number of error groups waiting for the next flush."""
    return _store.pending_count()


async def send_test_alert() -> bool:
    """Send a synthetic alert so an operator can verify the pipeline end to end.

    Bypasses the store entirely: dedupe and the hourly cap would otherwise make
    a repeated test look like a broken pipeline.
    """
    if not is_enabled():
        return False
    now = datetime.now(UTC)
    summary = AlertSummary(
        title="Test alert: alerting is configured correctly",
        logger_name="backend.app.services.admin_alerts",
        level="ERROR",
        count=1,
        message=(
            "This is a synthetic alert triggered from the admin monitoring endpoint. "
            "No application error occurred."
        ),
        traceback_text=None,
        request_id=get_request_id(),
        first_seen=now,
        last_seen=now,
    )
    return await email_service.send_admin_alert(_recipient(), [summary], 0)


def reset_for_tests() -> None:
    """Detach the handler and clear state so tests do not leak into each other."""
    global _handler, _flush_task
    for name in CAPTURED_LOGGERS:
        target = logging.getLogger(name)
        for handler in [h for h in target.handlers if isinstance(h, AdminAlertHandler)]:
            target.removeHandler(handler)
    _handler = None
    _flush_task = None
    _store.reset()
