import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import ModelComparisonReportPage from './model-comparison-report';
import { report, run, summary, turn } from './model-comparison-fixtures';

// ---------------------------------------------------------------------------
// One comparison run's report, at its own URL.
//
// A switching decision gets made off this page, so these tests care most
// about the ways it could read as more reassuring than the run was: an
// unpriced model showing as free, a production count that was never measured
// passing for a clean zero, a first page of turns passing for the whole run,
// or anything at all that reads as a verdict.
// ---------------------------------------------------------------------------

vi.mock('../admin-api', () => ({
  getComparisonReport: vi.fn(),
  getComparisonProgress: vi.fn(),
  cancelComparisonRun: vi.fn(),
  deleteComparisonRun: vi.fn(),
}));

// Deleting navigates away, and there is no route table here to navigate
// into, so the call itself is what gets asserted.
const navigate = vi.fn();
vi.mock('react-router-dom', async () => ({
  ...(await vi.importActual<typeof import('react-router-dom')>('react-router-dom')),
  useNavigate: () => navigate,
}));

function renderReport(runId = 'run-0001') {
  return render(
    <MemoryRouter>
      <ModelComparisonReportPage runId={runId} />
    </MemoryRouter>,
  );
}

beforeEach(async () => {
  const api = await import('../admin-api');
  vi.mocked(api.getComparisonReport).mockReset().mockResolvedValue(report());
  vi.mocked(api.getComparisonProgress).mockReset();
  vi.mocked(api.cancelComparisonRun).mockReset();
  vi.mocked(api.deleteComparisonRun).mockReset().mockResolvedValue(undefined);
  navigate.mockReset();
});

describe('ModelComparisonReportPage', () => {
  it('polls the counters, not the audited report, while a run is in flight', async () => {
    // The report endpoint is audited and loads every turn to order them, so
    // polling it buried one human read under an audit row and a full decrypt
    // every two seconds.
    const api = await import('../admin-api');
    vi.mocked(api.getComparisonReport).mockResolvedValue(
      report({ run: run({ status: 'running', progress_completed: 3, progress_total: 40 }) }),
    );
    vi.mocked(api.getComparisonProgress).mockResolvedValue({
      id: 'run-0001',
      status: 'running',
      progress_completed: 9,
      progress_total: 40,
    });
    renderReport();

    await waitFor(() => expect(api.getComparisonReport).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.getComparisonProgress).toHaveBeenCalled(), { timeout: 4000 });
    // The counters advance from the cheap read.
    await waitFor(() => expect(screen.getByText(/9 of 40/)).toBeInTheDocument());
    // And the audited read has not been repeated.
    expect(api.getComparisonReport).toHaveBeenCalledTimes(1);
  });

  it('fetches the evidence once, when the run settles', async () => {
    const api = await import('../admin-api');
    vi.mocked(api.getComparisonReport).mockResolvedValue(
      report({ run: run({ status: 'running', progress_completed: 39, progress_total: 40 }) }),
    );
    vi.mocked(api.getComparisonProgress).mockResolvedValue({
      id: 'run-0001',
      status: 'completed',
      progress_completed: 40,
      progress_total: 40,
    });
    renderReport();

    await waitFor(() => expect(api.getComparisonReport).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.getComparisonReport).toHaveBeenCalledTimes(2), {
      timeout: 5000,
    });
  });

  it('asks for the ten turns worth reading, not the whole run', async () => {
    const api = await import('../admin-api');
    renderReport('run-abc');

    await waitFor(() => expect(api.getComparisonReport).toHaveBeenCalledWith('run-abc', 10));
  });

  it('states no verdict anywhere', async () => {
    // The point of the rewrite. A recommendation banner, a "safe to switch"
    // pill or a pass/fail word on this page would undo it.
    renderReport();

    await screen.findByText('Safety checks');
    for (const forbidden of [/safe to switch/i, /do not switch/i, /inconclusive/i, /verdict/i]) {
      expect(screen.queryByText(forbidden)).not.toBeInTheDocument();
    }
    // The switch itself lives on user detail; the report only points at it.
    expect(screen.getByRole('link', { name: 'Model settings for this user' })).toHaveAttribute(
      'href',
      '/app/admin/users/user-1/llm',
    );
  });

  it('offers a way back to the run list', async () => {
    renderReport();

    const back = await screen.findByRole('link', { name: 'Back to model comparison' });
    expect(back).toHaveAttribute('href', '/app/admin/model-comparison');
  });

  it('sets the candidate violations against production&apos;s', async () => {
    // A candidate count means nothing on its own. "The incumbent does this
    // too" is the whole reason both columns exist.
    const api = await import('../admin-api');
    const s = summary({
      candidate_findings: { fabricated_id: 4 },
      production_findings: { fabricated_id: 3 },
      candidate_violations: 4,
      production_violations: 3,
    });
    vi.mocked(api.getComparisonReport).mockResolvedValue(report({ run: run({ summary: s }) }));
    renderReport();

    const tile = (await screen.findByText('Candidate violations')).closest('div');
    expect(within(tile as HTMLElement).getByText('4')).toBeInTheDocument();
    expect(within(tile as HTMLElement).getByText(/Production 3/)).toBeInTheDocument();
  });

  it('says a check was not asked of production rather than printing a zero', async () => {
    // Production's own writes are what the unrequested-write check compares
    // against, so it passes by construction. A "0" in that cell would be a
    // measurement nobody took.
    const api = await import('../admin-api');
    const s = summary({
      candidate_findings: { unrequested_write: 2, fabricated_id: 1 },
      production_findings: { fabricated_id: 1 },
      candidate_violations: 3,
      production_violations: 1,
    });
    vi.mocked(api.getComparisonReport).mockResolvedValue(report({ run: run({ summary: s }) }));
    renderReport();

    const row = (await screen.findByText('Wrote something the live turn did not')).closest('tr');
    expect(within(row as HTMLElement).getByText('not applicable')).toBeInTheDocument();

    // A check that does apply shows production's real count.
    const compared = screen.getByText('Wrote to a record ID it was never shown').closest('tr');
    expect(within(compared as HTMLElement).getAllByText('1')).toHaveLength(2);
  });

  it('reports the write-match rate with the middle buckets beside it', async () => {
    // "Right record, different content" and "the tool aimed somewhere else"
    // are separate buckets, and neither is in the headline rate. Hiding them
    // would make that rate read as a quality score.
    const api = await import('../admin-api');
    const s = summary({
      writes_total: 10,
      writes_matched: 6,
      writes_same_record: 2,
      writes_args_differ: 1,
      writes_missed: 1,
      writes_not_reached: 0,
      writes_not_replayed: 0,
      writes_measured: 10,
      write_match_rate: 0.6,
    });
    vi.mocked(api.getComparisonReport).mockResolvedValue(report({ run: run({ summary: s }) }));
    renderReport();

    const tile = (await screen.findByText('Writes reached')).closest('div');
    expect(within(tile as HTMLElement).getByText('6/10')).toBeInTheDocument();
    expect(
      within(tile as HTMLElement).getByText(
        /60% matched on every argument, 2 same record with different arguments, 1 same tool only, 1 not made/,
      ),
    ).toBeInTheDocument();
  });

  it('measures the rate over the writes the candidate was actually asked about', async () => {
    // A write on a turn whose replay ran out of lookup rounds was never put
    // to the candidate. Counting it in the denominator reports a measurement
    // failure as a lower score.
    const api = await import('../admin-api');
    const s = summary({
      writes_total: 10,
      writes_matched: 6,
      writes_same_record: 0,
      writes_args_differ: 0,
      writes_missed: 0,
      writes_not_reached: 4,
      writes_not_replayed: 0,
      writes_measured: 6,
      write_match_rate: 1,
    });
    vi.mocked(api.getComparisonReport).mockResolvedValue(report({ run: run({ summary: s }) }));
    renderReport();

    const tile = (await screen.findByText('Writes reached')).closest('div');
    expect(within(tile as HTMLElement).getByText('6/6')).toBeInTheDocument();
  });

  it('gives a silent candidate its own tile', async () => {
    // A turn the candidate answered with nothing passes every safety check by
    // having nothing to check, so the violation count beside it is a zero
    // that means the opposite of what it looks like.
    const api = await import('../admin-api');
    const s = summary({
      outcome_counts: { no_candidate_output: 3, no_write: 37 },
    });
    vi.mocked(api.getComparisonReport).mockResolvedValue(report({ run: run({ summary: s }) }));
    renderReport();

    const tile = (await screen.findByText('Answered with nothing')).closest('div');
    expect(within(tile as HTMLElement).getByText('3')).toBeInTheDocument();
  });

  it('says a latency nobody measured is not available rather than zero', async () => {
    const api = await import('../admin-api');
    const s = summary({
      turns_replayed: 0,
      turns_failed: 40,
      candidate: { ...summary().candidate, latency_p50_ms: null, latency_p95_ms: null },
    });
    vi.mocked(api.getComparisonReport).mockResolvedValue(report({ run: run({ summary: s }) }));
    renderReport();

    const tile = (await screen.findByText('Candidate latency (p95)')).closest('div');
    expect(within(tile as HTMLElement).getByText('not available')).toBeInTheDocument();
  });

  it('puts the production window beside the candidate cost', async () => {
    // A candidate cost with nothing beside it reads as what the deployment
    // would pay. The comparison was never on the page before.
    renderReport();

    const tile = (await screen.findByText('Candidate cost')).closest('div');
    expect(
      within(tile as HTMLElement).getByText(/Production billed \$0.9000 over 80 calls/),
    ).toBeInTheDocument();
  });

  it('keeps the production window when the candidate cost is unavailable', async () => {
    // The two halves of the tile are independent. Blanking the production
    // figure because the candidate has none threw away the one real number
    // left on the tile.
    const api = await import('../admin-api');
    const s = summary({ notes: ['Cost is not available: no pricing entry for candidate.'] });
    s.candidate = { ...s.candidate, total_cost_usd: null, cost_unavailable_reason: 'model' };
    vi.mocked(api.getComparisonReport).mockResolvedValue(report({ run: run({ summary: s }) }));
    renderReport();

    const tile = (await screen.findByText('Candidate cost')).closest('div');
    expect(
      within(tile as HTMLElement).getByText(/Production billed \$0.9000 over 80 calls/),
    ).toBeInTheDocument();
  });

  it('does not attribute the priced sum to the unpriced calls too', async () => {
    // "billed $X over 80 calls, 20 of them unpriced" reads as $X buying all
    // 80. The sum covers the priced rows only.
    const api = await import('../admin-api');
    const s = summary();
    s.production = { ...s.production, unpriced_calls: 20 };
    vi.mocked(api.getComparisonReport).mockResolvedValue(report({ run: run({ summary: s }) }));
    renderReport();

    const tile = (await screen.findByText('Candidate cost')).closest('div');
    expect(
      within(tile as HTMLElement).getByText(
        /Production billed \$0.9000 over 60 priced calls of 80 in the same window; the other 20 are unpriced/,
      ),
    ).toBeInTheDocument();
  });

  it('counts a turn the provider never answered out of the write rate', async () => {
    // Its writes are unmeasured, not missed, and the tile has to say how
    // many dropped out or the rate reads as the whole sample.
    const api = await import('../admin-api');
    const s = summary({
      writes_total: 10,
      writes_matched: 6,
      writes_same_record: 0,
      writes_args_differ: 0,
      writes_missed: 0,
      writes_not_reached: 1,
      writes_not_replayed: 3,
      writes_measured: 6,
      write_match_rate: 1,
    });
    vi.mocked(api.getComparisonReport).mockResolvedValue(report({ run: run({ summary: s }) }));
    renderReport();

    const tile = (await screen.findByText('Writes reached')).closest('div');
    expect(within(tile as HTMLElement).getByText('6/6')).toBeInTheDocument();
    expect(within(tile as HTMLElement).getByText(/4 not measured/)).toBeInTheDocument();
  });

  it('says the cost is not available rather than showing zero', async () => {
    // The old column reported "0.000000" for a gateway model name next to a
    // warning nobody read, and the number beat the warning every time.
    const api = await import('../admin-api');
    const s = summary({
      notes: [
        'Cost is not available: endpoint otari is marked unpriced, so (anthropic, candidate) does not name who bills these tokens.',
      ],
    });
    s.candidate = { ...s.candidate, total_cost_usd: null, cost_unavailable_reason: 'endpoint' };
    vi.mocked(api.getComparisonReport).mockResolvedValue(report({ run: run({ summary: s }) }));
    renderReport();

    const tile = (await screen.findByText('Candidate cost')).closest('div');
    expect(within(tile as HTMLElement).getByText('not available')).toBeInTheDocument();
    expect(screen.getByText(/marked unpriced/)).toBeInTheDocument();
    expect(screen.queryByText('$0.0000')).not.toBeInTheDocument();
  });

  // -------------------------------------------------------------------------
  // Severity, not just "something was flagged"
  // -------------------------------------------------------------------------

  it('does not dress a fixture artifact as an accusation', async () => {
    // ``tool_not_in_schema`` describes the replayed fixture: the name is in
    // the history and either side can copy it out. Rendering it in the same
    // red as a real violation made the first screen of a report badges that
    // the text below goes on to disown.
    const api = await import('../admin-api');
    const advisory = turn({
      findings: [
        {
          finding: 'tool_not_in_schema',
          tool_name: 'retired_tool',
          detail: "in this turn's record but not in the current tool schema",
          violation: false,
          side: 'candidate',
        },
      ],
    });
    vi.mocked(api.getComparisonReport).mockResolvedValue(report({ turns: [advisory] }));
    renderReport();

    // The badge, not the detail line the expanded card repeats it in.
    const [badge] = await screen.findAllByText(/Retired tool name/);
    expect(badge?.className).not.toContain('error');
    expect(badge?.className).toContain('text-muted-foreground');
  });

  it('renders a candidate violation in the error style', async () => {
    const api = await import('../admin-api');
    const violating = turn({
      findings: [
        {
          finding: 'unrequested_write',
          tool_name: 'qb_update',
          detail: 'a write the live turn did not make',
          violation: true,
          side: 'candidate',
        },
      ],
    });
    vi.mocked(api.getComparisonReport).mockResolvedValue(report({ turns: [violating] }));
    renderReport();

    // The badge, not the expanded detail line beneath it.
    const [badge] = await screen.findAllByText(/Wrote something the live turn did not/);
    expect(badge?.className).toContain('error');
  });

  it("labels production's findings as production's, not as an accusation", async () => {
    const api = await import('../admin-api');
    const both = turn({
      findings: [
        {
          finding: 'fabricated_id',
          tool_name: 'add_note',
          detail: 'wrote to work_order_id=118600',
          violation: true,
          side: 'production',
        },
      ],
    });
    vi.mocked(api.getComparisonReport).mockResolvedValue(report({ turns: [both] }));
    renderReport();

    const [badge] = await screen.findAllByText(
      /Production: Wrote to a record ID it was never shown/,
    );
    expect(badge?.className).not.toContain('error');
  });

  it('shows the writes production made and what the candidate did about them', async () => {
    renderReport();

    // The fixture turn is a miss: production filed against customer 884412
    // and the candidate replied in prose.
    expect(await screen.findByText('Did not make the write')).toBeInTheDocument();
    expect(screen.getByText(/production sent/)).toHaveTextContent('884412');
  });

  it('names the arguments that differ rather than showing two blobs', async () => {
    const api = await import('../admin-api');
    vi.mocked(api.getComparisonReport).mockResolvedValue(
      report({
        turns: [
          turn({
            outcome: 'write_same_record',
            writes: [
              {
                tool_name: 'qb_update',
                outcome: 'same_record_different_args',
                key_arguments: { entity_type: 'Invoice', entity_id: '4102', total_amt: 500 },
                candidate_arguments: {
                  entity_type: 'Estimate',
                  entity_id: '4102',
                  total_amt: 5000,
                },
                record_ids: { entity_id: ['4102'] },
                differing_arguments: ['entity_type', 'total_amt'],
              },
            ],
          }),
        ],
      }),
    );
    renderReport();

    const line = await screen.findByText(/differs on/);
    expect(line).toHaveTextContent('entity_type, total_amt');
    expect(line).toHaveTextContent('Same record, different arguments');
  });

  it('says outright when the candidate returned nothing', async () => {
    // An empty column beside a production reply reads as a rendering gap,
    // not as the model returning nothing.
    const api = await import('../admin-api');
    vi.mocked(api.getComparisonReport).mockResolvedValue(
      report({
        turns: [
          turn({
            outcome: 'no_candidate_output',
            candidate_text: '',
            candidate_tool_calls: [],
            writes: [],
          }),
        ],
      }),
    );
    renderReport();

    // Two matches: the summary tile's label and this turn's badge. The badge
    // is the one under test.
    expect(await screen.findByText('Returned no text and no tool call.')).toBeInTheDocument();
    expect(screen.getAllByText('Answered with nothing').length).toBeGreaterThan(1);
  });

  it('shows what production actually did beside the candidate', async () => {
    renderReport();

    const production = (await screen.findByText('Production')).closest('div');
    expect(within(production as HTMLElement).getByText(/create_job/)).toBeInTheDocument();
    expect(within(production as HTMLElement).getByText('Booked.')).toBeInTheDocument();
  });

  it('shows the lookups the candidate made before the decision shown', async () => {
    const api = await import('../admin-api');
    const looked = turn({
      candidate_replayed_lookups: [
        {
          name: 'appfolio_search_work_orders',
          arguments: { search_term: '12 Oak St' },
          result: 'work order 71002',
          is_error: false,
        },
      ],
    });
    vi.mocked(api.getComparisonReport).mockResolvedValue(report({ turns: [looked] }));
    renderReport();

    expect(
      await screen.findByText('Looked up first: appfolio_search_work_orders'),
    ).toBeInTheDocument();
  });

  it('shows billed prompt tokens rather than the raw input column', async () => {
    // The two are wildly different for a cached call, and the raw column
    // makes a fully cached run look like it barely used any context.
    const api = await import('../admin-api');
    const t = turn({
      candidate_input_tokens: 9329,
      candidate_cache_read_tokens: 18288,
      candidate_cache_creation_tokens: 223482,
    });
    vi.mocked(api.getComparisonReport).mockResolvedValue(report({ turns: [t] }));
    renderReport();

    expect(await screen.findByText(/billed prompt tokens/)).toHaveTextContent(/251,099/);
  });

  it('pages the turn list and offers to load the rest', async () => {
    const api = await import('../admin-api');
    vi.mocked(api.getComparisonReport).mockResolvedValue(
      report({ turns: [turn()], total_turns: 120 }),
    );
    renderReport();

    // The count has to be visible, or a partial report reads as the whole run.
    expect(await screen.findByText(/1 of 120/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Show more turns' }));
    await waitFor(() => expect(api.getComparisonReport).toHaveBeenLastCalledWith('run-0001', 20));
  });

  it('shows progress and the turns already in for a run still going', async () => {
    // The page is worth opening before the run finishes, including from a
    // different browser than the one that started it.
    const api = await import('../admin-api');
    vi.mocked(api.getComparisonReport).mockResolvedValue(
      report({
        run: run({ status: 'running', progress_completed: 3, progress_total: 40, summary: null }),
        turns: [turn()],
        total_turns: 3,
      }),
    );
    renderReport();

    expect(await screen.findByText(/Replaying 3 of 40 turns/)).toBeInTheDocument();
    expect(screen.getByText(/summary is written when the run finishes/)).toBeInTheDocument();
    expect(screen.getByText(/Turns worth reading/)).toBeInTheDocument();
  });

  it('cancels the run it is showing', async () => {
    const api = await import('../admin-api');
    vi.mocked(api.getComparisonReport).mockResolvedValue(
      report({ run: run({ status: 'running', summary: null }) }),
    );
    vi.mocked(api.cancelComparisonRun).mockResolvedValue(run({ status: 'cancelled' }));
    renderReport();

    await userEvent.click(await screen.findByRole('button', { name: 'Cancel' }));

    await waitFor(() => expect(api.cancelComparisonRun).toHaveBeenCalledWith('run-0001'));
  });

  it('will not delete the run until the confirmation is accepted', async () => {
    const api = await import('../admin-api');
    renderReport();

    await userEvent.click(await screen.findByRole('button', { name: 'Delete run' }));
    expect(api.deleteComparisonRun).not.toHaveBeenCalled();

    // The dialog names the evidence, and the count is what makes the warning
    // concrete rather than boilerplate.
    const dialog = await screen.findByRole('dialog');
    expect(dialog).toHaveTextContent('1 recorded turn');
    expect(dialog).toHaveTextContent('candidate');
    await within(dialog).findByText(/cannot be undone/);

    await userEvent.click(within(dialog).getByRole('button', { name: 'Delete run' }));

    await waitFor(() => expect(api.deleteComparisonRun).toHaveBeenCalledWith('run-0001'));
    // Staying put would leave the page polling a run that no longer exists.
    expect(navigate).toHaveBeenCalledWith('/app/admin/model-comparison');
  });

  it('pluralises the evidence count in the delete confirmation', async () => {
    const api = await import('../admin-api');
    vi.mocked(api.getComparisonReport).mockResolvedValue(report({ total_turns: 40 }));
    renderReport();

    await userEvent.click(await screen.findByRole('button', { name: 'Delete run' }));

    expect(await screen.findByRole('dialog')).toHaveTextContent('40 recorded turns');
  });

  it('offers no delete control while the run is still going', async () => {
    // The API refuses it with a 409 until the run is cancelled, and Cancel is
    // already the button on offer there.
    const api = await import('../admin-api');
    vi.mocked(api.getComparisonReport).mockResolvedValue(
      report({ run: run({ status: 'running', summary: null }) }),
    );
    renderReport();

    await screen.findByRole('button', { name: 'Cancel' });
    expect(screen.queryByRole('button', { name: 'Delete run' })).not.toBeInTheDocument();
  });

  it('surfaces a failed delete and stays on the page', async () => {
    const api = await import('../admin-api');
    vi.mocked(api.deleteComparisonRun).mockRejectedValue(
      new Error('Failed to delete the comparison'),
    );
    renderReport();

    await userEvent.click(await screen.findByRole('button', { name: 'Delete run' }));
    const dialog = await screen.findByRole('dialog');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Delete run' }));

    expect(await screen.findByText('Failed to delete the comparison')).toBeInTheDocument();
    expect(navigate).not.toHaveBeenCalled();
  });

  it('shows why a run stopped early, beside the partial counts', async () => {
    // A run the provider killed still carries a summary, so the reason has to
    // be its own line rather than something a reader infers from a short
    // turn count.
    const api = await import('../admin-api');
    vi.mocked(api.getComparisonReport).mockResolvedValue(
      report({
        run: run({
          status: 'failed',
          error: 'stopped after 3 consecutive provider failures: APIStatusError: 503',
          summary: summary({
            turns_replayed: 3,
            notes: ['stopped after 3 consecutive provider failures: APIStatusError: 503'],
          }),
        }),
      }),
    );
    renderReport();

    expect(
      await screen.findAllByText(/3 consecutive provider failures/),
    ).not.toHaveLength(0);
  });

  it('offers a way back when the id in the URL is not a run', async () => {
    // A pasted link with a stale id is the common failure here, and a red
    // banner with no exit is a dead end.
    const api = await import('../admin-api');
    vi.mocked(api.getComparisonReport).mockRejectedValue(new Error('Run not found'));
    renderReport('nope');

    expect(await screen.findByText('That comparison run does not exist.')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Back to model comparison' })).toBeInTheDocument();
  });
});
