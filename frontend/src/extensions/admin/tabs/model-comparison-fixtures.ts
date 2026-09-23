import type {
  ComparisonReport,
  ComparisonRun,
  ComparisonRunList,
  ComparisonSummary,
  ComparisonTurn,
} from '../admin-api';

// Shared by model-comparison.test.tsx (the start form and history) and
// model-comparison-report.test.tsx (one run's report), which exercise two
// pages against the same shapes.

export const USERS = {
  total: 1,
  skip: 0,
  limit: 200,
  items: [
    {
      id: 'user-1',
      user_id: 'google_abc',
      email: 'consenting@example.com',
      plan: 'pro',
      status: 'active',
      role: 'user',
      is_active: true,
      onboarding_complete: true,
      data_sharing_consent: true,
    },
  ],
};

export function summary(overrides: Partial<ComparisonSummary> = {}): ComparisonSummary {
  return {
    turns_total: 40,
    turns_replayed: 40,
    turns_failed: 0,
    outcome_counts: { write_matched: 10, no_write: 30 },
    candidate_findings: {},
    production_findings: {},
    candidate_violations: 0,
    production_violations: 0,
    production_checked_findings: ['fabricated_id', 'invalid_args', 'tool_not_in_schema'],
    writes_total: 10,
    writes_matched: 10,
    writes_same_record: 0,
    writes_args_differ: 0,
    writes_missed: 0,
    writes_not_reached: 0,
    writes_not_replayed: 0,
    writes_measured: 10,
    write_match_rate: 1,
    candidate: {
      provider: 'anthropic',
      model: 'candidate',
      input_tokens: 1000,
      output_tokens: 120,
      cache_read_tokens: 0,
      cache_creation_tokens: 0,
      billed_prompt_tokens: 1000,
      total_cost_usd: '0.400000',
      cost_unavailable_reason: '',
      latency_p50_ms: 600,
      latency_p95_ms: 900,
    },
    production: {
      calls: 80,
      input_tokens: 4000,
      output_tokens: 900,
      cache_read_tokens: 200,
      cache_creation_tokens: 100,
      billed_prompt_tokens: 4300,
      total_cost_usd: '0.900000',
      unpriced_calls: 0,
      window_start: '2026-04-28T09:00:00Z',
      window_end: '2026-05-01T12:00:00Z',
    },
    notes: [],
    ...overrides,
  };
}

export function run(overrides: Partial<ComparisonRun> = {}): ComparisonRun {
  return {
    id: 'run-0001',
    user_id: 'user-1',
    user_email: 'consenting@example.com',
    user_consented: true,
    incumbent_endpoint: '',
    incumbent_provider: 'anthropic',
    incumbent_model: 'incumbent',
    candidate_endpoint: '',
    candidate_provider: 'anthropic',
    candidate_model: 'candidate',
    candidate_reasoning_effort: 'high',
    history_mode: 'full',
    requested_samples: 40,
    status: 'completed',
    progress_completed: 40,
    progress_total: 40,
    error: '',
    created_at: '2026-05-01T12:00:00Z',
    started_at: null,
    completed_at: null,
    summary: summary(),
    ...overrides,
  };
}

export function turn(overrides: Partial<ComparisonTurn> = {}): ComparisonTurn {
  return {
    message_seq: 1,
    message_timestamp: '2026-05-01T12:00:00Z',
    user_message: 'can you book that job',
    production_reply: 'Booked.',
    production_tool_calls: [
      {
        name: 'create_job',
        arguments: { customer_id: '884412' },
        result: 'ok',
        is_error: false,
      },
    ],
    candidate_text: 'Sure, I can help with that.',
    candidate_tool_calls: [],
    candidate_replayed_lookups: [],
    candidate_stop_reason: 'end_turn',
    candidate_input_tokens: 0,
    candidate_output_tokens: 0,
    candidate_cache_read_tokens: 0,
    candidate_cache_creation_tokens: 0,
    candidate_latency_ms: 0,
    candidate_error: '',
    outcome: 'write_missed',
    writes: [
      {
        tool_name: 'create_job',
        outcome: 'missed',
        key_arguments: { customer_id: '884412' },
        candidate_arguments: null,
        record_ids: { customer_id: ['884412'] },
        differing_arguments: [],
      },
    ],
    findings: [],
    ...overrides,
  };
}

export function report(overrides: Partial<ComparisonReport> = {}): ComparisonReport {
  return { run: run(), turns: [turn()], total_turns: 1, ...overrides };
}

// ``listComparisonRuns`` returns the run list plus the bounds the sample
// slider has to respect, so a mock has to carry them or the slider falls back
// to its own defaults and the test stops exercising the wired value.
export function runList(
  runs: ComparisonRun[] = [],
  overrides: Partial<ComparisonRunList> = {},
): ComparisonRunList {
  return {
    runs,
    total: runs.length,
    max_samples: 200,
    max_page_size: 100,
    ...overrides,
  };
}
