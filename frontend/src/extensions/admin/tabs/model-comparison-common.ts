import type { TurnOutcome } from '../admin-api';

// Shared by the model-comparison index page (the start form and the run
// history) and the report page each run links to. Both poll, both decide
// whether a run is still in flight, and both name the turn outcomes, so these
// live here rather than being duplicated across two page modules.

// Poll cadence while a run is in flight. A fifty-turn run takes minutes and
// writes a row per turn, so this is fast enough to look alive and slow enough
// not to hammer the endpoint.
export const POLL_MS = 2000;

export const ACTIVE_STATUSES = new Set(['pending', 'running']);

/** How a turn's candidate decision read against the writes production made.
 *
 * Wording is deliberately flat. Most of these are not a pass or a fail: a
 * candidate that called the same tool with different arguments may have
 * reworded a message body or filed a note against the wrong job, and only
 * reading the turn tells the two apart. The exception is
 * ``no_candidate_output``, which has no benign reading.
 */
export const OUTCOME_COPY: Record<TurnOutcome, { label: string; className: string }> = {
  no_candidate_output: {
    label: 'Answered with nothing',
    className: 'bg-error-bg text-error-text',
  },
  write_missed: {
    label: 'Did not make the write',
    className: 'bg-error-bg text-error-text',
  },
  write_args_differ: {
    label: 'Same tool, different arguments',
    className: 'bg-warning-bg text-warning-text',
  },
  write_same_record: {
    label: 'Same record, different arguments',
    className: 'bg-warning-bg text-warning-text',
  },
  not_replayed: {
    label: 'Could not be replayed',
    className: 'bg-warning-bg text-warning-text',
  },
  replay_incomplete: {
    label: 'Still looking things up at the round cap',
    className: 'bg-panel text-muted-foreground',
  },
  write_matched: {
    label: 'Reached the same write',
    className: 'bg-success-bg text-success-text',
  },
  no_write: {
    label: 'No write on this turn',
    className: 'bg-panel text-muted-foreground',
  },
};

/** The turn-level label for one write's outcome. Keyed by ``WriteOutcome``. */
export const WRITE_OUTCOME_COPY: Record<string, string> = {
  matched: 'Reached the same write',
  same_record_different_args: 'Same record, different arguments',
  same_tool_different_args: 'Same tool, different arguments',
  missed: 'Did not make the write',
  not_reached: 'Not measured: the replay ran out of lookup rounds',
  not_replayed: 'Not measured: the provider did not answer this turn',
};

/** What each deterministic check looks for. Keyed by ``types.Finding``. */
export const FINDING_COPY: Record<string, string> = {
  unknown_tool: 'Called a tool that does not exist',
  invalid_args: 'Arguments the tool rejects',
  unrequested_write: 'Wrote something the live turn did not',
  fabricated_id: 'Wrote to a record ID it was never shown',
  truncated: 'Response truncated mid-thought',
  tool_not_in_schema: 'Retired tool name, carried by this history',
  call_failed: 'Provider call failed',
};

export function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/** A latency, or "not available" when nothing was measured.
 *
 * The wire sends null rather than 0 when a run has no latency samples, so a
 * run where every call failed does not read as an instantaneous model.
 */
export function ms(value: number | null | undefined): string {
  if (value == null) return 'not available';
  return value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${Math.round(value)}ms`;
}
