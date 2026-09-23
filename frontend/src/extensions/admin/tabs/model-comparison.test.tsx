import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import ModelComparisonTab from './model-comparison';
import { report, run, runList, USERS } from './model-comparison-fixtures';

// ---------------------------------------------------------------------------
// Admin Model Comparison tab: starting runs and listing them.
//
// One run's evidence is a separate page (model-comparison-report.test.tsx).
// What is left here is the form, and the tests care most about the ways it
// could mislead: offering a user who cannot legally be compared, offering a
// sample size the API will reject, or letting a second run start while one is
// in flight.
// ---------------------------------------------------------------------------

vi.mock('../admin-api', () => ({
  getAdminUsers: vi.fn(),
  listComparisonRuns: vi.fn(),
  startComparisonRun: vi.fn(),
  getComparisonReport: vi.fn(),
  getComparisonProgress: vi.fn(),
  cancelComparisonRun: vi.fn(),
  deleteComparisonRun: vi.fn(),
  listLLMEndpoints: vi.fn().mockResolvedValue([]),
}));

vi.mock('../llm-picker', () => ({
  LLMProviderSelect: ({ value, onChange }: { value: string; onChange: (v: string) => void }) => (
    <input aria-label="provider" value={value} onChange={e => onChange(e.target.value)} />
  ),
  LLMModelField: ({ value, onChange }: { value: string; onChange: (v: string) => void }) => (
    <input aria-label="model" value={value} onChange={e => onChange(e.target.value)} />
  ),
  LLMEndpointSelect: ({ value, onChange }: { value: string; onChange: (v: string) => void }) => (
    <input aria-label="endpoint" value={value} onChange={e => onChange(e.target.value)} />
  ),
  ReasoningEffortSelect: ({
    value,
    onChange,
    inheritLabel,
  }: {
    value: string;
    onChange: (v: string) => void;
    inheritLabel?: string;
  }) => (
    <input
      aria-label={inheritLabel ?? 'effort'}
      value={value}
      onChange={e => onChange(e.target.value)}
    />
  ),
}));

// The run history renders twice: a card list up to ``xl`` and a table above
// it, swapped by a CSS media query that jsdom does not evaluate. So a query
// naming a run matches once per layout. Tests that are not about the layouts
// themselves go through these; the layouts are checked against each other in
// "shows every run in both layouts".
const listed = (text: string) => screen.queryAllByText(text).length;
// Both layouts render the same ``DeleteRunButton``, so a test about the
// delete flow rather than about the layouts gets nothing from clicking each.
const deleteControl = (model: string) => {
  const [control] = screen.getAllByRole('button', { name: `Delete run against ${model}` });
  if (!control) throw new Error(`no delete control for ${model}`);
  return control;
};

function renderTab() {
  return render(
    <MemoryRouter>
      <ModelComparisonTab />
    </MemoryRouter>,
  );
}

beforeEach(async () => {
  const api = await import('../admin-api');
  vi.mocked(api.getAdminUsers).mockReset().mockResolvedValue(USERS as never);
  vi.mocked(api.listComparisonRuns).mockReset().mockResolvedValue(runList([]));
  vi.mocked(api.startComparisonRun).mockReset();
  vi.mocked(api.getComparisonReport).mockReset().mockResolvedValue(report());
  vi.mocked(api.cancelComparisonRun).mockReset();
  vi.mocked(api.deleteComparisonRun).mockReset().mockResolvedValue(undefined);
});

describe('ModelComparisonTab', () => {
  it('only offers users who consented to data sharing', async () => {
    const api = await import('../admin-api');
    renderTab();

    await waitFor(() => expect(api.getAdminUsers).toHaveBeenCalled());
    // A run reads real conversations, so a non-consenting user must not be
    // reachable from the picker at all.
    expect(vi.mocked(api.getAdminUsers)).toHaveBeenCalledWith(
      expect.objectContaining({ consent: 'shared' }),
    );
  });

  it('starts a run with the chosen candidate and the slider size', async () => {
    const api = await import('../admin-api');
    vi.mocked(api.startComparisonRun).mockResolvedValue(run({ status: 'pending' }));
    renderTab();

    await screen.findByRole('option', { name: 'consenting@example.com' });
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'User' }), 'user-1');
    await userEvent.type(screen.getByLabelText('provider'), 'anthropic');
    await userEvent.type(screen.getByLabelText('model'), 'candidate');

    const slider = screen.getByRole('slider', { name: 'Turns to replay' });
    fireEvent.change(slider, { target: { value: '65' } });
    expect(await screen.findByText('Most recent 65')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Run comparison' }));

    await waitFor(() =>
      expect(api.startComparisonRun).toHaveBeenCalledWith('user-1', {
        candidateEndpoint: '',
        candidateProvider: 'anthropic',
        candidateModel: 'candidate',
        // Empty means "the deployment's setting", which the API resolves and
        // freezes onto the run rather than reading per call.
        candidateReasoningEffort: '',
        sampleCount: 65,
        // Empty follows the live loop's history setting, resolved the same way.
        historyMode: '',
      }),
    );
  });

  it('replays with the history mode the operator picks', async () => {
    // Replaying the same turns once each way is how the cold-start rebuild
    // is read, so the form must send the choice through.
    const api = await import('../admin-api');
    vi.mocked(api.startComparisonRun).mockResolvedValue(run({ status: 'pending' }));
    renderTab();

    await screen.findByRole('option', { name: 'consenting@example.com' });
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'User' }), 'user-1');
    await userEvent.type(screen.getByLabelText('provider'), 'anthropic');
    await userEvent.type(screen.getByLabelText('model'), 'candidate');
    await userEvent.selectOptions(
      screen.getByRole('combobox', { name: 'History' }),
      'cold_start_compaction',
    );

    await userEvent.click(screen.getByRole('button', { name: 'Run comparison' }));

    await waitFor(() =>
      expect(api.startComparisonRun).toHaveBeenCalledWith(
        'user-1',
        expect.objectContaining({ historyMode: 'cold_start_compaction' }),
      ),
    );
  });

  it('offers one reasoning effort, the candidate&apos;s', async () => {
    // Nothing is sent to the incumbent: the baseline is the recorded turn.
    // A second effort control would be asking for a value nothing reads.
    const api = await import('../admin-api');
    vi.mocked(api.startComparisonRun).mockResolvedValue(run({ status: 'pending' }));
    renderTab();

    await screen.findByRole('option', { name: 'consenting@example.com' });
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'User' }), 'user-1');
    await userEvent.type(screen.getByLabelText('provider'), 'anthropic');
    await userEvent.type(screen.getByLabelText('model'), 'candidate');

    const efforts = screen.getAllByLabelText('Deployment default');
    expect(efforts).toHaveLength(1);
    await userEvent.type(efforts[0] as HTMLElement, 'high');

    await userEvent.click(screen.getByRole('button', { name: 'Run comparison' }));

    await waitFor(() =>
      expect(api.startComparisonRun).toHaveBeenCalledWith(
        'user-1',
        expect.objectContaining({ candidateReasoningEffort: 'high' }),
      ),
    );
  });

  it('starts a run against an endpoint with no provider', async () => {
    // An endpoint carries its own dialect, so it is a complete destination.
    // Requiring a provider beside it would be asking for a value that the
    // endpoint then supersedes.
    const api = await import('../admin-api');
    vi.mocked(api.startComparisonRun).mockResolvedValue(run({ status: 'pending' }));
    renderTab();

    await screen.findByRole('option', { name: 'consenting@example.com' });
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'User' }), 'user-1');
    await userEvent.type(screen.getByLabelText('endpoint'), 'otari');
    await userEvent.type(screen.getByLabelText('model'), 'candidate');

    await userEvent.click(screen.getByRole('button', { name: 'Run comparison' }));

    await waitFor(() =>
      expect(api.startComparisonRun).toHaveBeenCalledWith(
        'user-1',
        expect.objectContaining({ candidateEndpoint: 'otari', candidateProvider: '' }),
      ),
    );
  });

  it('bounds the slider by the cap the API reports', async () => {
    // MODEL_COMPARISON_MAX_SAMPLES is configurable, so a slider holding its
    // own ceiling would offer a size start_run rejects with a bare 422.
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(runList([], { max_samples: 40 }));
    renderTab();

    await screen.findByRole('option', { name: 'consenting@example.com' });
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'User' }), 'user-1');

    const slider = await screen.findByRole('slider', { name: 'Turns to replay' });
    await waitFor(() => expect(slider).toHaveAttribute('max', '40'));
    // The default of 50 sits above that cap, so it has to come down with it.
    expect(await screen.findByText('Most recent 40')).toBeInTheDocument();
  });

  it('promises no verdict anywhere on the form', async () => {
    // The evaluator this replaced warned that a short run would report
    // "inconclusive", because it produced a recommendation. Nothing here
    // does, and a leftover hint that one is coming is the regression worth
    // guarding.
    renderTab();

    await screen.findByRole('option', { name: 'consenting@example.com' });
    expect(screen.queryByText(/inconclusive/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/safe to switch/i)).not.toBeInTheDocument();
    expect(screen.getByText(/No tool is ever executed/)).toBeInTheDocument();
  });

  it('tracks an in-flight run through the counters, not the audited list', async () => {
    // Re-reading the whole list every two seconds wrote an audit row per
    // tick, so a long run left open buried one human read under hundreds.
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(
      runList([run({ status: 'running', progress_completed: 2, progress_total: 40 })]),
    );
    vi.mocked(api.getComparisonProgress).mockResolvedValue({
      id: 'run-0001',
      status: 'running',
      progress_completed: 11,
      progress_total: 40,
    });
    renderTab();

    await screen.findByRole('option', { name: 'consenting@example.com' });
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'User' }), 'user-1');
    const listCallsAfterSelect = vi.mocked(api.listComparisonRuns).mock.calls.length;

    await waitFor(() => expect(api.getComparisonProgress).toHaveBeenCalledWith('run-0001'), {
      timeout: 4000,
    });
    await waitFor(() => expect(listed('Replaying 11 of 40 turns through candidate')).toBe(1));
    // The audited listing was not re-read to get that.
    expect(vi.mocked(api.listComparisonRuns).mock.calls.length).toBe(listCallsAfterSelect);
  });

  it("does not claim another tenant's run belongs to this form", async () => {
    // The unfiltered table is the default view. Matching any active row there
    // put a stranger's progress bar under the words "already running for this
    // user", which is the opposite of what the guard is for.
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(
      runList([run({ id: 'run-0002', user_id: 'user-2', status: 'running' })]),
    );
    renderTab();

    expect(await screen.findByText('Recent comparisons')).toBeInTheDocument();
    expect(
      screen.queryByText('A comparison is already running for this user.'),
    ).not.toBeInTheDocument();
  });

  it('stops growing the page at the ceiling the API reports', async () => {
    // Growing past it 422s, and the poll closes over the size, so every later
    // tick fails too and the table stops updating rather than just growing.
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(
      runList([run()], { total: 500, max_page_size: 50 }),
    );
    renderTab();

    const more = await screen.findByRole('button', { name: 'Show more runs' });
    await userEvent.click(more);
    await waitFor(() => expect(api.listComparisonRuns).toHaveBeenCalledWith({ limit: 50 }));

    // At the ceiling the control goes away and says why, rather than
    // offering a click that fails.
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: 'Show more runs' })).not.toBeInTheDocument(),
    );
    expect(screen.getByText(/Showing the 50 most recent of 500/)).toBeInTheDocument();
    expect(
      vi.mocked(api.listComparisonRuns).mock.calls.every(([opts]) => (opts?.limit ?? 0) <= 50),
    ).toBe(true);
  });

  it('lists runs across every user before one is picked', async () => {
    // The table is how a run is found again weeks later, when nobody
    // remembers whose it was.
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(
      runList([
        run(),
        run({
          id: 'run-0002',
          user_id: 'user-2',
          user_email: 'other@example.com',
          candidate_model: 'other-candidate',
        }),
      ]),
    );
    renderTab();

    expect(await screen.findByText('Recent comparisons')).toBeInTheDocument();
    await screen.findAllByText('other@example.com');
    expect(listed('other-candidate')).toBeGreaterThan(0);
    // Unfiltered: the API is asked for every user's runs.
    expect(api.listComparisonRuns).toHaveBeenCalledWith(
      expect.objectContaining({ userId: undefined }),
    );
  });

  it('narrows the table to the selected user', async () => {
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(runList([run()]));
    renderTab();

    await screen.findByRole('option', { name: 'consenting@example.com' });
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'User' }), 'user-1');

    await waitFor(() =>
      expect(api.listComparisonRuns).toHaveBeenCalledWith(
        expect.objectContaining({ userId: 'user-1' }),
      ),
    );
    expect(await screen.findByText('Runs for this user')).toBeInTheDocument();
  });

  it('does not offer a report for a run whose user withdrew consent', async () => {
    // The run row survives, because it is metadata rather than conversation
    // content, but its evidence is no longer readable and the link would 403.
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(runList([run({ user_consented: false })]));
    renderTab();

    await screen.findAllByText('consent withdrawn');
    expect(screen.queryByRole('link', { name: /ago|2026/ })).not.toBeInTheDocument();
  });

  it('pages the run table', async () => {
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(runList([run()], { total: 60 }));
    renderTab();

    expect(await screen.findByText('1 of 60')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Show more runs' }));
    await waitFor(() =>
      expect(api.listComparisonRuns).toHaveBeenLastCalledWith(
        expect.objectContaining({ limit: 50 }),
      ),
    );
  });

  it('lets a run start while another user has one in flight', async () => {
    // start_run allows one active run per user, so someone else's run in the
    // unfiltered table must not disable this form.
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(
      runList([
        run({
          id: 'run-0009',
          user_id: 'user-2',
          user_email: 'other@example.com',
          status: 'running',
          summary: null,
        }),
      ]),
    );
    renderTab();

    await screen.findByRole('option', { name: 'consenting@example.com' });
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'User' }), 'user-1');
    await userEvent.type(screen.getByLabelText('provider'), 'anthropic');
    await userEvent.type(screen.getByLabelText('model'), 'candidate');

    expect(screen.getByRole('button', { name: 'Run comparison' })).not.toBeDisabled();
  });

  it('links each past run to its own report URL', async () => {
    // The report is a page an operator returns to and shares, so the row has
    // to carry a real, copyable link rather than only a click handler.
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(runList([run()]));
    renderTab();

    await screen.findByRole('option', { name: 'consenting@example.com' });
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'User' }), 'user-1');

    // One per layout, and both have to point at the run: a card whose
    // timestamp went nowhere would strand every phone visitor on the list.
    const links = await screen.findAllByRole('link', { name: /ago|2026/ });
    expect(links).toHaveLength(2);
    for (const link of links) {
      expect(link).toHaveAttribute('href', '/app/admin/model-comparison/run-0001');
    }
  });

  it('does not render a report inline', async () => {
    // Regression: the report used to expand under the form, which meant no
    // URL for it and no way back to one.
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(runList([run()]));
    renderTab();

    await screen.findByRole('option', { name: 'consenting@example.com' });
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'User' }), 'user-1');
    await screen.findAllByText('candidate');

    expect(api.getComparisonReport).not.toHaveBeenCalled();
    expect(screen.queryByText('Turns worth reading')).not.toBeInTheDocument();
  });

  it('blocks a second run while one is in flight and links to the one running', async () => {
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(
      runList([
        run({ status: 'running', progress_completed: 12, progress_total: 40, summary: null }),
      ]),
    );
    renderTab();

    await screen.findByRole('option', { name: 'consenting@example.com' });
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'User' }), 'user-1');

    expect(await screen.findByText(/Replaying 12 of 40 turns/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run comparison' })).toBeDisabled();
    expect(screen.getByRole('link', { name: 'Open report' })).toHaveAttribute(
      'href',
      '/app/admin/model-comparison/run-0001',
    );
  });

  it('shows every run in both layouts', async () => {
    // Two pieces of markup for one row can drift, and the direction it drifts
    // is invisible to whoever changed it: a column added to the table and
    // forgotten on the card costs nothing on a desktop and hides the field on
    // every phone. So whatever the table says about a run, the card says too.
    //
    // The field list below catches a field dropped from either layout. It
    // cannot catch one added to only the table, since it never asked about
    // that field, hence the column count: a ninth column fails here until
    // whoever added it decides where it goes on the card.
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(runList([run()]));
    renderTab();

    const table = await screen.findByRole('table');
    const cards = screen.getByRole('list', { name: 'Comparison runs' });

    // Started, User, Candidate, User is on, Turns, Status, counts, actions.
    // The list is unfiltered here, so the User column is present.
    expect(within(table).getAllByRole('columnheader')).toHaveLength(8);

    for (const layout of [table, cards]) {
      expect(within(layout).getByText('candidate')).toBeInTheDocument();
      expect(within(layout).getByText(/incumbent/)).toBeInTheDocument();
      expect(within(layout).getByText('consenting@example.com')).toBeInTheDocument();
      // The headline counts: violations per side, and writes reached.
      expect(within(layout).getByText(/violations/)).toBeInTheDocument();
      expect(within(layout).getByText(/10\/10 writes/)).toBeInTheDocument();
      expect(within(layout).getByText(/completed/)).toBeInTheDocument();
      expect(within(layout).getByRole('link', { name: /ago|2026/ })).toBeInTheDocument();
      expect(
        within(layout).getByRole('button', { name: 'Delete run against candidate' }),
      ).toBeInTheDocument();
    }
  });

  it('keeps the table scroller a containing block', async () => {
    // Not styling: dropping this class stretches the document past the
    // viewport and the browser zooms the page out to fit. See the comment on
    // the wrapper for the mechanism. jsdom has no layout engine, so the class
    // is the only part of it that can be asserted here.
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(runList([run()]));
    renderTab();

    const scroller = (await screen.findByRole('table')).parentElement;
    expect(scroller).toHaveClass('overflow-x-auto');
    expect(scroller).toHaveClass('relative');
  });

  it('will not delete a run until the confirmation is accepted', async () => {
    // The gate is the whole feature: the row itself navigates to the report,
    // so a delete control that fired on the first click would be one stray
    // click away from destroying evidence that costs a replay to rebuild.
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(runList([run()]));
    renderTab();

    await screen.findAllByText('candidate');
    await userEvent.click(deleteControl('candidate'));
    expect(api.deleteComparisonRun).not.toHaveBeenCalled();

    const dialog = await screen.findByRole('dialog');
    expect(dialog).toHaveTextContent('cannot be undone');
    await userEvent.click(screen.getByRole('button', { name: 'Delete run' }));

    await waitFor(() => expect(api.deleteComparisonRun).toHaveBeenCalledWith('run-0001'));
  });

  it('drops the deleted row without refetching the list', async () => {
    // Refetching would snap a list paged several clicks deep back to page one.
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(
      runList([run(), run({ id: 'run-0002', candidate_model: 'other-candidate' })]),
    );
    renderTab();

    await screen.findAllByText('candidate');
    await userEvent.click(deleteControl('candidate'));
    await userEvent.click(screen.getByRole('button', { name: 'Delete run' }));

    // Gone from both layouts, not just the one the click went through.
    await waitFor(() => expect(listed('candidate')).toBe(0));
    expect(listed('other-candidate')).toBeGreaterThan(0);
    expect(vi.mocked(api.listComparisonRuns).mock.calls.length).toBe(1);
  });

  it('cancelling the dialog leaves the run alone', async () => {
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(runList([run()]));
    renderTab();

    await screen.findAllByText('candidate');
    await userEvent.click(deleteControl('candidate'));
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));

    expect(api.deleteComparisonRun).not.toHaveBeenCalled();
    expect(listed('candidate')).toBeGreaterThan(0);
  });

  it('offers no delete control while a run is still going', async () => {
    // The API refuses it with a 409, so the remedy is Cancel, not Delete.
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(
      runList([run({ status: 'running', summary: null })]),
    );
    renderTab();

    await screen.findAllByText('candidate');
    // In neither layout, so a phone cannot reach what the table withholds.
    expect(screen.queryAllByRole('button', { name: 'Delete run against candidate' })).toHaveLength(
      0,
    );
  });

  it('keeps the delete control on a run whose user withdrew consent', async () => {
    // Its report already 403s, so this is the row most worth clearing out and
    // the only control that can still act on it.
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(runList([run({ user_consented: false })]));
    renderTab();

    expect(
      await screen.findAllByRole('button', { name: 'Delete run against candidate' }),
    ).toHaveLength(2);
  });

  it('surfaces a failed delete and keeps the row', async () => {
    const api = await import('../admin-api');
    vi.mocked(api.listComparisonRuns).mockResolvedValue(runList([run()]));
    vi.mocked(api.deleteComparisonRun).mockRejectedValue(
      new Error('Run is still running. Cancel it before deleting.'),
    );
    renderTab();

    await screen.findAllByText('candidate');
    await userEvent.click(deleteControl('candidate'));
    await userEvent.click(screen.getByRole('button', { name: 'Delete run' }));

    expect(await screen.findByText(/Cancel it before deleting/)).toBeInTheDocument();
    expect(listed('candidate')).toBeGreaterThan(0);
  });
});
