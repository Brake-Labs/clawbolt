import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  cancelComparisonRun,
  deleteComparisonRun,
  getComparisonProgress,
  getComparisonReport,
  type ComparisonReport,
  type ComparisonSummary,
  type ComparisonToolCall,
  type ComparisonTurn,
  type TurnOutcome,
} from '../admin-api';
import ConfirmDialog from '../ConfirmDialog';
import { formatRelative } from '../format';
import { adminPath } from '../nav-items';
import {
  ACTIVE_STATUSES,
  FINDING_COPY,
  ms,
  OUTCOME_COPY,
  POLL_MS,
  pct,
  WRITE_OUTCOME_COPY,
} from './model-comparison-common';

// One comparison run's report, at its own URL under the run's public id.
//
// It is a page rather than a panel on the start form because of what an
// operator does with it: they read it, leave, come back to it days later when
// the switch is actually being decided, and paste the link to someone else.
//
// Three things drive the layout:
//
// - The page must not read as a verdict. The top of it is a handful of
//   counts and a write-match rate, none of which is a threshold anything was
//   compared against.
// - Safety findings are shown per side. The candidate's count means nothing
//   without production's beside it, and three of the checks cannot be asked
//   of the record at all, which the page says rather than printing a zero.
// - The turns are the point, not an appendix. Ten arrive by default, ordered
//   so the ones an operator acts on are the ones on screen.
//
// A run still in flight renders here too, with its progress and whatever
// turns have landed, so the page is worth opening before the run finishes.

// Turns are returned worst-first, so the first page is the part that decides
// anything. The rest is available on request rather than shipped by default:
// every text column on a turn is decrypted and PII-scrubbed per request.
const TURN_PAGE_SIZE = 10;

/** The candidate, and where it ran.
 *
 * The effort is read off the run rather than off the current setting, since
 * the setting is mutable and the run is the record of what was measured.
 * ``auto`` is omitted: it means the provider chose, which is the default a
 * reader already assumes.
 */
function describeCandidate(endpoint: string, model: string, effort: string): string {
  const where = endpoint ? `${endpoint}/${model}` : model;
  return effort && effort !== 'auto' ? `${where} (${effort})` : where;
}

/** Dollars, or why there are none.
 *
 * ``total_cost_usd`` is null rather than "0.000000" whenever nothing can
 * price the tokens, so this never has to decide whether a zero is real.
 */
function money(totals: { total_cost_usd: string | null }): string {
  if (totals.total_cost_usd == null) return 'not available';
  return `$${Number(totals.total_cost_usd).toFixed(4)}`;
}

/** What the user's live loop billed over the same days, for the cost tile.
 *
 * Not a like-for-like figure and the hint says so: the window covers every
 * call the live agent made inside it, including tool rounds the replay never
 * reached, while the candidate's number covers the sampled turns only. It is
 * here because a candidate cost with nothing beside it reads as what the
 * deployment would pay, and that comparison was never on the page.
 */
function productionCostHint(production: ComparisonSummary['production']): string {
  if (production.calls === 0) return 'No production usage recorded for this window';
  const spend = money(production);
  const calls = production.calls.toLocaleString();
  // The sum covers the priced rows only, so attributing it to every call in
  // the window overstates what those dollars bought. Say which calls it
  // covers and how many it does not.
  if (production.unpriced_calls) {
    const priced = (production.calls - production.unpriced_calls).toLocaleString();
    return `Production billed ${spend} over ${priced} priced call${
      production.calls - production.unpriced_calls === 1 ? '' : 's'
    } of ${calls} in the same window; the other ${production.unpriced_calls.toLocaleString()} are unpriced`;
  }
  return `Production billed ${spend} over ${calls} call${
    production.calls === 1 ? '' : 's'
  } in the same window`;
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-[--radius-md] border border-border bg-card p-3">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 text-xl font-semibold text-foreground">{value}</p>
      {hint ? <p className="mt-1 text-xs text-muted-foreground">{hint}</p> : null}
    </div>
  );
}

/** Hard safety violations, per side, with what each side was checked for.
 *
 * Three of the checks cannot be asked of the record: production's own writes
 * are what the unrequested-write check compares against, a tool it called
 * existed when it called it, and a delivered reply carries no truncated
 * budget. Saying so is the difference between "production had none" and
 * "production was not asked", which are not the same claim.
 */
function ViolationPanel({ summary }: { summary: ComparisonSummary }) {
  const kinds = new Set([
    ...Object.keys(summary.candidate_findings),
    ...Object.keys(summary.production_findings),
  ]);
  const checkedBoth = new Set(summary.production_checked_findings);
  const rows = [...kinds].filter(k => k !== 'call_failed').sort();
  return (
    <section className="rounded-[--radius-lg] border border-border bg-card">
      <div className="border-b border-border p-3">
        <h3 className="text-sm font-semibold text-foreground">Safety checks</h3>
        <p className="mt-1 text-xs text-muted-foreground">
          Deterministic, no judge. Both columns run the same code; where production is marked not
          applicable, the check has no meaning against a recorded turn.
        </p>
        {/* The known blind spot, on the page rather than in a docstring. An
            unrequested write is caught by the record or file it names, so a
            tool call that names neither is only caught by its tool name. */}
        <p className="mt-1 text-xs text-muted-foreground">
          An unrequested write is matched on the record IDs and file paths the call carries. A few
          writers carry neither (updating the heartbeat, creating a project, discarding media,
          toggling an integration), so a second call through one of those goes unflagged when the
          live turn used the same tool. Read those turns rather than the count.
        </p>
      </div>
      {rows.length === 0 ? (
        <p className="p-3 text-sm text-muted-foreground">
          Nothing flagged on either side across {summary.turns_replayed} replayed turns.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wide text-muted-foreground">
                <th className="p-3">Check</th>
                <th className="p-3">Candidate</th>
                <th className="p-3">Production</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(kind => (
                <tr key={kind} className="border-t border-border">
                  <td className="p-3 text-foreground">{FINDING_COPY[kind] ?? kind}</td>
                  <td className="p-3 font-mono text-foreground">
                    {summary.candidate_findings[kind] ?? 0}
                  </td>
                  <td className="p-3 font-mono text-muted-foreground">
                    {checkedBoth.has(kind) ? (
                      (summary.production_findings[kind] ?? 0)
                    ) : (
                      <span className="font-sans text-xs">not applicable</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function SummaryGrid({ summary }: { summary: ComparisonSummary }) {
  const totals = summary.candidate;
  const silent = summary.outcome_counts.no_candidate_output ?? 0;
  const incomplete = summary.outcome_counts.replay_incomplete ?? 0;
  // Writes the candidate was never asked about: the replay ran out of lookup
  // rounds, or the provider never answered the turn. Two buckets on the wire
  // because an operator acts on them differently, one number on the tile
  // because what the rate's denominator drops is the same either way.
  const unmeasured = summary.writes_not_reached + summary.writes_not_replayed;
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <Stat
        label="Candidate violations"
        value={String(summary.candidate_violations)}
        hint={`Production ${summary.production_violations} on the checks that apply to it`}
      />
      {/* Its own tile rather than a line in the writes hint. A turn the
          candidate answered with nothing passes every safety check by having
          nothing to check, so the violation count beside it is a zero that
          means the opposite of what it looks like. */}
      <Stat
        label="Answered with nothing"
        value={String(silent)}
        hint={
          silent
            ? 'Turns production answered and the candidate returned no text and no tool call'
            : 'The candidate answered every turn production answered'
        }
      />
      <Stat
        label="Writes reached"
        value={
          summary.writes_measured
            ? `${summary.writes_matched}/${summary.writes_measured}`
            : 'no write turns'
        }
        hint={
          summary.writes_measured
            ? // The unmeasured writes are named here, not only in the notes
              // below. The denominator is the measured writes, so a run that
              // ran out of lookup rounds or lost turns to the provider
              // reports a rate over what was left, and a reader who cannot
              // see how many dropped out reads that rate as the whole sample.
              `${pct(summary.write_match_rate)} matched on every argument, ${summary.writes_same_record} same record with different arguments, ${summary.writes_args_differ} same tool only, ${summary.writes_missed} not made${unmeasured ? `, ${unmeasured} not measured` : ''}`
            : summary.writes_total
              ? `${summary.writes_total} write(s), none measured: no replay reached the question`
              : 'The live turns in this sample wrote nothing'
        }
      />
      <Stat
        label="Candidate cost"
        value={money(totals)}
        // The production half of the tile does not depend on the candidate
        // half. An unpriced candidate used to blank it out, which threw away
        // the one figure on the page that was still a real number.
        hint={
          totals.total_cost_usd == null
            ? `${productionCostHint(summary.production)}. The candidate's own cost is not available: see the note below`
            : productionCostHint(summary.production)
        }
      />
      <Stat
        label="Candidate latency (p95)"
        value={ms(totals.latency_p95_ms)}
        hint={`p50 ${ms(totals.latency_p50_ms)}`}
      />
      <Stat
        label="Turns replayed"
        value={String(summary.turns_replayed)}
        hint={`of ${summary.turns_total} attempted`}
      />
      <Stat
        label="Turns that failed"
        value={String(summary.turns_failed)}
        hint={
          incomplete
            ? `The provider did not answer. ${incomplete} more ran out of lookup rounds`
            : 'The provider did not answer'
        }
      />
      <Stat
        label="Candidate prompt tokens"
        value={totals.billed_prompt_tokens.toLocaleString()}
        hint={`${totals.output_tokens.toLocaleString()} output, ${totals.cache_read_tokens.toLocaleString()} read from cache`}
      />
    </div>
  );
}

function ToolCallList({ calls, empty }: { calls: ComparisonToolCall[]; empty: string }) {
  if (calls.length === 0) {
    return <p className="mb-2 text-xs italic text-muted-foreground">{empty}</p>;
  }
  return (
    <ul className="mb-2 space-y-1">
      {calls.map((call, index) => (
        // ``break-all``: a serialized argument payload is one unbroken token,
        // so without it the line runs past the card and the tail is clipped
        // away rather than wrapped. On a phone that hid most of every call.
        <li
          key={`${call.name}-${index}`}
          className="break-all rounded-[--radius-sm] bg-panel px-2 py-1 font-mono text-xs text-foreground"
        >
          <span className="font-semibold">{call.name}</span>
          <span className="text-muted-foreground">({JSON.stringify(call.arguments)})</span>
        </li>
      ))}
    </ul>
  );
}

function TurnCard({ turn }: { turn: ComparisonTurn }) {
  // ``violation`` comes from the API rather than a local set: a copy here
  // would be a hand-maintained mirror of types.HARD_VIOLATIONS, and it
  // decides whether a badge reads as an accusation.
  const violations = turn.findings.filter(f => f.violation && f.side === 'candidate');
  const production = turn.findings.filter(f => f.violation && f.side === 'production');
  const advisory = turn.findings.filter(f => !f.violation);
  const detailed = [...violations, ...production, ...advisory];
  const outcome = OUTCOME_COPY[turn.outcome as TurnOutcome];
  // Expand what an operator is here for. A retired tool name alone is not
  // worth opening a diff for, and auto-expanding those buried the turns that
  // were. ``no_candidate_output`` is here because it is the one outcome with
  // no benign reading, and it would otherwise be the emptiest card on the
  // page and read as the least interesting.
  const [open, setOpen] = useState(
    violations.length > 0 ||
      turn.outcome === 'no_candidate_output' ||
      turn.outcome === 'write_missed' ||
      turn.outcome === 'write_args_differ' ||
      turn.outcome === 'write_same_record',
  );
  return (
    <div className="rounded-[--radius-md] border border-border bg-card">
      <button
        type="button"
        onClick={() => setOpen(v => !v)}
        aria-expanded={open}
        className="flex w-full items-start gap-3 p-3 text-left"
      >
        <div className="min-w-0 flex-1">
          {/* One line while collapsed, so a list of turns stays scannable.
              Expanding is the only place the question is shown in full: it is
              not repeated in the diff below, and one truncated line is about
              six words on a phone. */}
          <p className={`text-sm text-foreground ${open ? 'break-words' : 'truncate'}`}>
            {turn.user_message}
          </p>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs">
            {outcome ? (
              <span className={`rounded-full px-2 py-0.5 ${outcome.className}`}>
                {outcome.label}
              </span>
            ) : (
              <span className="text-muted-foreground">{turn.outcome}</span>
            )}
            {violations.map((issue, index) => (
              <span
                key={`violation-${issue.finding}-${index}`}
                className="rounded-full bg-error-bg px-2 py-0.5 text-error-text"
              >
                {FINDING_COPY[issue.finding] ?? issue.finding}
                {issue.tool_name ? `: ${issue.tool_name}` : ''}
              </span>
            ))}
            {production.map((issue, index) => (
              <span
                key={`production-${issue.finding}-${index}`}
                className="rounded-full bg-warning-bg px-2 py-0.5 text-warning-text"
                title="The live turn did this too"
              >
                Production: {FINDING_COPY[issue.finding] ?? issue.finding}
                {issue.tool_name ? `: ${issue.tool_name}` : ''}
              </span>
            ))}
            {advisory.map((issue, index) => (
              <span
                key={`advisory-${issue.finding}-${index}`}
                className="rounded-full bg-panel px-2 py-0.5 text-muted-foreground"
                title="Describes the replay, not a model"
              >
                {FINDING_COPY[issue.finding] ?? issue.finding}
                {issue.tool_name ? `: ${issue.tool_name}` : ''}
              </span>
            ))}
          </div>
        </div>
        <span className="shrink-0 text-xs text-muted-foreground">{open ? 'Hide' : 'Compare'}</span>
      </button>

      {open ? (
        <div className="border-t border-border p-3">
          {detailed.length > 0 ? (
            <ul className="mb-3 space-y-1 text-xs text-muted-foreground">
              {detailed.map((issue, index) => (
                <li key={`detail-${issue.finding}-${index}`}>
                  <span className="font-medium">
                    {issue.side === 'production' ? 'Production: ' : ''}
                    {FINDING_COPY[issue.finding] ?? issue.finding}
                  </span>{' '}
                  {issue.detail}
                </li>
              ))}
            </ul>
          ) : null}

          {turn.writes.length > 0 ? (
            <ul className="mb-3 space-y-1 text-xs">
              {turn.writes.map((write, index) => (
                <li
                  key={`write-${write.tool_name}-${index}`}
                  className="break-all rounded-[--radius-sm] bg-panel px-2 py-1 font-mono text-muted-foreground"
                >
                  <span className="font-semibold text-foreground">{write.tool_name}</span>{' '}
                  {WRITE_OUTCOME_COPY[write.outcome] ?? write.outcome}
                  {/* What differed, named. Without this the reader has to
                      diff two JSON blobs to find the one field that moved,
                      which is the whole difference between a paraphrase and
                      a tenfold amount error. */}
                  {write.differing_arguments.length > 0 ? (
                    <>
                      {' | differs on '}
                      <span className="text-warning-text">
                        {write.differing_arguments.join(', ')}
                      </span>
                    </>
                  ) : null}
                  {Object.keys(write.record_ids).length > 0
                    ? ` | record ${Object.entries(write.record_ids)
                        .map(([path, values]) => `${path}=${values.join('/')}`)
                        .join(' ')}`
                    : ''}
                  {' | production sent '}
                  {JSON.stringify(write.key_arguments)}
                  {write.candidate_arguments
                    ? ` | candidate sent ${JSON.stringify(write.candidate_arguments)}`
                    : ''}
                </li>
              ))}
            </ul>
          ) : null}

          <div className="flex flex-col gap-4 sm:flex-row">
            <div className="min-w-0 flex-1">
              <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Production
              </p>
              <ToolCallList calls={turn.production_tool_calls} empty="No tool calls" />
              {turn.production_reply ? (
                <p className="whitespace-pre-wrap text-sm text-foreground">
                  {turn.production_reply}
                </p>
              ) : null}
            </div>
            <div className="min-w-0 flex-1">
              <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Candidate
              </p>
              {turn.candidate_error ? (
                <p className="text-sm text-error-text">{turn.candidate_error}</p>
              ) : (
                <>
                  {turn.candidate_replayed_lookups.length > 0 ? (
                    // Lookups the live turn also made, answered from its
                    // recorded results so the decision below could be read.
                    // Nothing ran.
                    <details className="mb-2">
                      <summary className="cursor-pointer text-xs text-muted-foreground">
                        Looked up first:{' '}
                        {turn.candidate_replayed_lookups.map(l => l.name).join(', ')}
                      </summary>
                      <ul className="mt-1 space-y-1">
                        {turn.candidate_replayed_lookups.map((lookup, index) => (
                          <li
                            key={`${lookup.name}-${index}`}
                            className="break-all rounded-[--radius-sm] bg-panel px-2 py-1 font-mono text-xs text-muted-foreground"
                          >
                            <span className="font-semibold">{lookup.name}</span>(
                            {JSON.stringify(lookup.arguments)}) returned {lookup.result}
                          </li>
                        ))}
                      </ul>
                    </details>
                  ) : null}
                  <ToolCallList calls={turn.candidate_tool_calls} empty="No tool calls" />
                  {turn.candidate_text ? (
                    <p className="whitespace-pre-wrap text-sm text-foreground">
                      {turn.candidate_text}
                    </p>
                  ) : turn.outcome === 'no_candidate_output' ? (
                    // Said outright. An empty column beside a production
                    // reply reads as a rendering gap, not as the model
                    // returning nothing.
                    <p className="text-sm text-error-text">Returned no text and no tool call.</p>
                  ) : null}
                </>
              )}
            </div>
          </div>

          {/* Billed prompt tokens, not ``candidate_input_tokens``. The two are
              wildly different for a cached call. */}
          <p className="mt-3 font-mono text-xs text-muted-foreground">
            candidate billed prompt tokens:{' '}
            {(
              turn.candidate_input_tokens +
              turn.candidate_cache_read_tokens +
              turn.candidate_cache_creation_tokens
            ).toLocaleString()}{' '}
            | output {turn.candidate_output_tokens.toLocaleString()} | latency{' '}
            {ms(turn.candidate_latency_ms)}
          </p>
        </div>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function ModelComparisonReportPage({ runId }: { runId: string }) {
  const navigate = useNavigate();
  const [report, setReport] = useState<ComparisonReport | null>(null);
  const [turnLimit, setTurnLimit] = useState(TURN_PAGE_SIZE);
  const [error, setError] = useState<string | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const load = useCallback(
    async (limit: number) => {
      try {
        setReport(await getComparisonReport(runId, limit));
      } catch (e) {
        const message = (e as Error).message;
        // A bad id in a pasted link is the common case here, and it deserves
        // a way back to the run list rather than a red banner.
        if (/not found/i.test(message)) setNotFound(true);
        else setError(message);
      }
    },
    [runId],
  );

  // Only a different run warrants blanking the page. Resetting on every
  // ``turnLimit`` change made "Show more turns" replace the whole report with
  // the loading skeleton, discarding which cards the reader had expanded and
  // where they were scrolled to.
  useEffect(() => {
    setReport(null);
    setNotFound(false);
    setError(null);
  }, [runId]);

  useEffect(() => {
    void load(turnLimit);
  }, [load, turnLimit]);

  // Poll only while the run is unfinished, and poll the counters rather than
  // the report: a report is immutable once the run completes, and the report
  // endpoint is audited and decrypts every turn to order them, so polling it
  // buried one human read under an audit row and a full decrypt every two
  // seconds. The full read happens once, when the run settles.
  const isActive = report ? ACTIVE_STATUSES.has(report.run.status) : false;
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    if (!isActive) return;
    pollRef.current = setInterval(() => {
      void (async () => {
        try {
          const progress = await getComparisonProgress(runId);
          const settled = !ACTIVE_STATUSES.has(progress.status);
          if (settled) {
            // Fetch the evidence and let it carry the status change. Flipping
            // the status here first would render "this run has no summary"
            // until the report landed, which on a long run is over a second.
            await load(turnLimit);
            return;
          }
          setReport(prev =>
            prev
              ? {
                  ...prev,
                  run: {
                    ...prev.run,
                    progress_completed: progress.progress_completed,
                    progress_total: progress.progress_total,
                  },
                }
              : prev,
          );
        } catch {
          // A failed progress tick is not worth surfacing; the next one
          // either recovers or the operator reloads.
        }
      })();
    }, POLL_MS);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [isActive, load, runId, turnLimit]);

  async function handleCancel() {
    try {
      await cancelComparisonRun(runId);
      await load(turnLimit);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function handleDelete() {
    setDeleting(true);
    try {
      await deleteComparisonRun(runId);
      // Back to the list rather than staying on a page whose subject no
      // longer exists, which would otherwise poll its way to "does not exist".
      navigate(adminPath('model-comparison'));
    } catch (e) {
      setError((e as Error).message);
      setConfirmingDelete(false);
    } finally {
      setDeleting(false);
    }
  }

  const backLink = (
    <Link to={adminPath('model-comparison')} className="text-sm text-primary hover:underline">
      Back to model comparison
    </Link>
  );

  if (notFound) {
    return (
      <div className="space-y-2 text-sm">
        <p className="text-danger">That comparison run does not exist.</p>
        {backLink}
      </div>
    );
  }

  if (!report) {
    return (
      <div className="space-y-3">
        {backLink}
        <div className="h-32 animate-pulse rounded-[--radius-md] bg-panel" />
      </div>
    );
  }

  const { run } = report;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        {backLink}
        <div className="flex flex-wrap items-baseline gap-3">
          <p className="text-xs text-muted-foreground">
            {describeCandidate(
              run.candidate_endpoint,
              run.candidate_model,
              run.candidate_reasoning_effort,
            )}{' '}
            against this user's recorded turns on {run.incumbent_model || 'an unrecorded model'},
            started {formatRelative(run.created_at)}
          </p>
          {/* Offered only once the run has settled. While it is in flight the
              button beside the progress bar is Cancel, which is the step the
              API requires first anyway. */}
          {isActive ? null : (
            <button
              type="button"
              onClick={() => setConfirmingDelete(true)}
              className="rounded-[--radius-md] border border-border px-2 py-1 text-xs text-muted-foreground hover:bg-panel"
            >
              Delete run
            </button>
          )}
        </div>
      </div>

      {error ? (
        <div className="rounded-[--radius-md] bg-error-bg px-3 py-2 text-sm text-error-text">
          {error}
        </div>
      ) : null}

      {isActive ? (
        <section className="rounded-[--radius-lg] border border-border bg-card p-4">
          <div className="flex flex-col items-start gap-2 sm:flex-row sm:items-center sm:justify-between">
            <p className="min-w-0 break-words text-sm text-foreground">
              Replaying {run.progress_completed} of {run.progress_total || '?'} turns through{' '}
              {run.candidate_model}
            </p>
            <button
              type="button"
              onClick={() => void handleCancel()}
              className="shrink-0 rounded-[--radius-md] border border-border px-3 py-1 text-sm text-muted-foreground"
            >
              Cancel
            </button>
          </div>
          <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-panel">
            <div
              className="h-full bg-primary transition-all"
              style={{
                width: run.progress_total
                  ? `${(run.progress_completed / run.progress_total) * 100}%`
                  : '5%',
              }}
            />
          </div>
        </section>
      ) : null}

      {/* A run that stopped early still has a summary, so this cannot live in
          the no-summary branch below: without it the reason the run gave up
          would be invisible under a page of counts. */}
      {run.error ? (
        <div className="rounded-[--radius-md] bg-error-bg px-3 py-2 text-sm text-error-text">
          {run.error}
        </div>
      ) : null}

      {run.summary ? (
        <>
          {run.summary.notes.map(note => (
            <div
              key={note}
              className="rounded-[--radius-md] bg-warning-bg px-3 py-2 text-sm text-warning-text"
            >
              {note}
            </div>
          ))}

          <SummaryGrid summary={run.summary} />
          <ViolationPanel summary={run.summary} />

          {/* Links out rather than applying the switch here. The per-user
              override already has one owner (user detail -> LLM), and a second
              control writing the same column is how the two drift. */}
          <Link
            to={`${adminPath('users')}/${run.user_id}/llm`}
            className="inline-block text-sm text-primary underline underline-offset-2"
          >
            Model settings for this user
          </Link>
        </>
      ) : (
        <p className="text-sm text-muted-foreground">
          {isActive
            ? 'The summary is written when the run finishes. Turns appear below as they land.'
            : 'This run has no summary.'}
        </p>
      )}

      {report.turns.length > 0 ? (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-foreground">
            Turns worth reading
            {report.total_turns > report.turns.length
              ? ` (${report.turns.length} of ${report.total_turns})`
              : ''}
          </h3>
          <div className="space-y-2">
            {report.turns.map(turn => (
              <TurnCard key={turn.message_seq} turn={turn} />
            ))}
          </div>
          {report.total_turns > report.turns.length ? (
            <button
              type="button"
              onClick={() => setTurnLimit(n => n + TURN_PAGE_SIZE)}
              className="mt-2 rounded-[--radius-md] border border-border px-3 py-1 text-sm text-muted-foreground"
            >
              Show more turns
            </button>
          ) : null}
        </div>
      ) : null}

      <ConfirmDialog
        open={confirmingDelete}
        onClose={() => setConfirmingDelete(false)}
        onConfirm={handleDelete}
        title="Delete this comparison run?"
        description={
          <div className="space-y-2">
            <p>
              This removes the run and all {report.total_turns} recorded turn
              {report.total_turns === 1 ? '' : 's'}. It cannot be undone, and the replay would have
              to be paid for again to get it back.
            </p>
            <p className="text-xs">{run.candidate_model}.</p>
          </div>
        }
        confirmLabel="Delete run"
        destructive
        busy={deleting}
      />
    </div>
  );
}
