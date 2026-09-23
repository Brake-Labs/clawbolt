# Clawbolt

Clawbolt is an AI assistant for the trades. FastAPI backend with a messaging-first interface (iMessage, RCS, SMS, Telegram, web chat) and a custom tool-calling agent loop built on any-llm. Built by Mozilla.ai using the open-core model.

## Build & Run Commands

```bash
# Install dependencies
uv sync

# Run server (requires PostgreSQL -- see docker-compose.yml)
uv run uvicorn backend.app.main:app --reload

# Run with Docker (starts Postgres + app, runs migrations automatically)
docker compose up

# Database migrations
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "description"

# Tests
uv run pytest -v

# Lint & format
uv run ruff check backend/ tests/ alembic/
uv run ruff format --check backend/ tests/ alembic/

# Type checking
uv run ty check --python .venv backend/ tests/ alembic/
```

## Tech Stack

- Python 3.11+, FastAPI, SQLAlchemy 2.0, Pydantic v2
- any-llm-sdk (LLM provider abstraction via `amessages`)
- Multi-channel messaging: BlueBubbles (iMessage/RCS/SMS), Twilio (SMS), Linq, Telegram (via python-telegram-bot), web chat
- Google Drive (per-user OAuth) for file storage
- PostgreSQL for all data persistence, Alembic for migrations
- uv + hatchling build system, ruff linting, ty type checking

## Storage

All structured data is stored in PostgreSQL (configurable via `DATABASE_URL`). The core tables:

| Table | Purpose |
|---|---|
| `users` | User profiles, personality text, preferences |
| `channel_routes` | Channel -> user routing (iMessage, SMS, Telegram, web chat, etc.) |
| `sessions` | Chat session metadata |
| `messages` | Chat messages (FK to sessions) |
| `memory_documents` | Structured memory and compaction history |
| `heartbeat_logs` | Heartbeat send log |
| `idempotency_keys` | Webhook deduplication |
| `llm_usage_logs` | Token usage tracking, with the endpoint that served each call |
| `llm_endpoints` | Named LLM destinations: dialect, base URL, own credential, capabilities |
| `tool_configs` | Per-user tool configuration |
| `calendar_configs` | Per-user calendar integration settings |
| `oauth_tokens` | Encrypted OAuth tokens for integrations (Google Calendar, Google Drive, QuickBooks, etc.) |

Ten more tables exist for `AUTH_MODE=multi_user` and stay empty in a single-user deployment: `subscriptions`, `usage_quotas`, `deleted_user_usage`, `allowed_emails`, `waitlist_entries`, `admin_api_keys`, `admin_audit_logs`, `llm_payload_captures`, `model_comparison_runs`, `model_comparison_turns`. See "Multi-user mode" below.

Saved files are not tracked in Postgres. The Google Drive integration is the source of truth for filenames, locations, and descriptions. The agent quotes saved files by their storage path (e.g. `/Astro Home Management - 123 Main Street/photos/foo.jpg`).

Key store modules:
- `backend/app/agent/user_db.py` -- `UserStore` (singleton via `get_user_store()`)
- `backend/app/agent/session_db.py` -- `SessionStore` (per-user via `get_session_store(id)`)
- `backend/app/agent/memory_db.py` -- `MemoryStore` (per-user via `get_memory_store(id)`)
- `backend/app/agent/stores.py` -- `HeartbeatStore`, `IdempotencyStore`, `LLMUsageStore`, `ToolConfigStore`
- `backend/app/agent/dto.py` -- Pydantic DTOs: `UserData`, `StoredMessage`, `SessionState`, etc.
- `backend/app/database.py` -- `Base`, `AsyncSessionLocal`, `db_session_async()`, `get_async_db()`, `get_async_engine()`
- `backend/app/models/` -- SQLAlchemy ORM models, one module per product area. Importing the package is what registers every mapper on `Base.metadata`, so Alembic autogenerate and the test-suite TRUNCATE both depend on it.

File storage is exposed through the Google Drive integration: each user grants the `drive.file` scope and uploads land in their own Drive under a top-level `Clawbolt` folder. The operator wires the OAuth client via `GOOGLE_DRIVE_CLIENT_ID` / `GOOGLE_DRIVE_CLIENT_SECRET`; without those, the file tools never load.

## Database access

The singleton engine and session factory live in `backend/app/database.py`. The session factory ships with `expire_on_commit=False` so attribute access after commit does not trigger `MissingGreenlet`. Pool tuning (`pool_recycle`, `pool_pre_ping`, statement timeout) is on the engine.

- READ-only methods: `db = AsyncSessionLocal()` + `try / finally await db.close()`. Lighter weight, no rollback wrapper. Reference `IdempotencyStore.has_seen` in `backend/app/agent/stores.py`.
- WRITE methods: `async with db_session_async() as db: ...`. Auto-rollback on exception, auto-close. Reference `IdempotencyStore.try_mark_seen` in `backend/app/agent/stores.py`.
- FastAPI dependency: `db: AsyncSession = Depends(get_async_db)`.

Pool sizing is currently SQLAlchemy default (`pool_size=5`, `max_overflow=10`). To re-evaluate, run `scripts/benchmark_pool.py` against a prod-like Postgres; it sweeps `pool_size`/`max_overflow` across realistic concurrency and emits a markdown report with p50/p95/p99 connection-acquisition latency. Methodology and the most recent run are in `scripts/benchmark_pool_report.md` (issue #1179).

### Common SQLAlchemy 2.0 patterns

- Read: `(await db.execute(select(X).where(...))).scalar_one_or_none()`. For the shapes that repeat, use the helpers in `backend/app/query_helpers.py`: `get_or_404_async` (fetch one or 404), `count_rows(db, Model.id, *where)` (scalar count), `fetch_all(db, stmt)` (entities as a `list`), and `iso` / `iso_or_none` (timestamp serialization, `""` or `None` fallback). Hand-roll only where the query does not fit one, e.g. a count across a join.
- DML rowcount: at runtime `(await db.execute(update/delete)).rowcount` returns `int`, but the stubs say `Result`. Cast to access cleanly: `cast("CursorResult[object]", await db.execute(...)).rowcount`. Reference `SessionStore.delete_message` in `backend/app/agent/session_db.py`.
- Bulk DML `synchronize_session`: the kwarg moved off `update()`/`delete()` constructors. Use `.execution_options(synchronize_session="fetch")` on the executable. Reference `_append_history_update` in `backend/app/agent/memory_db.py`.
- Row-level lock: `(await db.execute(select(M).filter_by(id=x).with_for_update())).scalar_one_or_none()`.
- `.scalars().all()` returns `Sequence[T]`, not `list[T]`. Wrap with `list(...)` only when the consumer is typed for `list`; `fetch_all` already does.
- Do not use the SQLAlchemy 1.x `db.query()` API.

### Encrypted columns: do not concat on the SQL side

`EncryptedString` columns (e.g. `MemoryDocument.history_text`, `OAuthToken.access_token`) handle envelope encryption automatically on bind/unbind. SQL-side string concat (`Model.col || new_text`) operates on ciphertext and silently corrupts the row. Bug fixed in #1200.

Correct pattern: SELECT FOR UPDATE the row, decrypt-in-Python via attribute access, append in Python, UPDATE with the full new plaintext. Reference `_doc_select_for_update` and `_append_history_update` plus their callers in `backend/app/agent/memory_db.py`. The row-level lock serializes concurrent appenders so neither side loses its update.

### Advisory locks

Use `pg_advisory_xact_lock` whenever possible: the lock is bound to the surrounding transaction and released automatically on COMMIT or ROLLBACK. Just execute the lock SQL inside the session and let the existing commit drop it. Reference `_advisory_lock_sql` in `backend/app/agent/session_db.py` (SessionStore) and `_lock_user_permissions` in `backend/app/agent/approval.py`.

Session-scoped advisory locks (`pg_advisory_lock` / `pg_try_advisory_lock`) are different: the unlock MUST run on the **same** connection that took the lock. `AsyncSession.commit()` returns the underlying connection to the pool, so a follow-up `pg_advisory_unlock` call on a fresh `AsyncSessionLocal()` runs on a different connection and is a silent no-op (Postgres returns `False`, the helper logs nothing). Recovery code in `backend/app/agent/inbound_recovery.py` and OAuth refresh in `backend/app/services/oauth.py` rely on the same-connection coupling: they hold the lock on a dedicated `AsyncConnection` (not a session) for the duration of the critical section.

Concurrency tests for advisory locks: spin per-task `AsyncConnection` handles so the lock primitive is actually exercised, not the connection serialization. Coordinate via `asyncio.Event`. Do not assert on `time.monotonic()` deltas across tasks; sub-millisecond races make the comparisons flake. Reference `TestInboundRecoveryLockSerialization` in `tests/test_inbound_recovery.py`.

### Test DB isolation

The session-scoped autouse `_isolate_async_engine` fixture rebinds the OSS engine to a `NullPool` async engine pointed at the test database. The default `_isolate_stores` autouse fixture TRUNCATEs every table in `Base.metadata.sorted_tables` after each test (RESTART IDENTITY + CASCADE) so the next test starts on a clean slate. The standard `test_user` fixture writes through `db_session_async()` against that engine.

The opt-in `async_db` fixture (in `tests/conftest.py`) gives a stricter SAVEPOINT-based rollback for tests that need it: each test runs inside a per-test `AsyncConnection` with a wrapping transaction, and the fixture rebinds `_async_session_factory` so store calls pick up the test connection. Two non-obvious choices documented in the design comment block above the fixture:

- **Function-scoped engine.** asyncpg connections bind to the event loop they were created on; pytest-asyncio rotates loops between tests by default. A session-scoped engine surfaces as `RuntimeError: Future attached to a different loop` on the second test.
- **`join_transaction_mode="create_savepoint"`.** Forces every session into its own SAVEPOINT under the outer transaction so an `IntegrityError` rolls back to the savepoint without detaching the outer transaction.

Pair `async_db` with the `async_test_user` fixture, which inserts the test user through the per-test connection so it is visible inside the same outer transaction. End every `async_db`-using test file with an iso-canary pair (`_part_a` writes a fixed-id row, `_part_b` asserts it is gone) to prove rollback isolation. Reference `test_async_isolation_rolls_back_between_tests_part_a` and `_part_b` in `tests/test_idempotency_pruning_async.py`.

### Multi-user tests

`tests/multi_user/` runs the `AUTH_MODE=multi_user` surface. Its `conftest.py` inherits the root database fixtures and adds three things: a package-level `MULTI_USER_APP` built by `create_app()` with the mode set (one instance, so a test's `app.dependency_overrides` and the `client` fixture agree on the object), an autouse fixture pinning `settings.auth_mode`, and a sync `db_session` for the setup and assertions that suite does through a plain `Session`.

Put a test there when it exercises sign-in, the admin console, quotas, or operator monitoring. Everything else belongs in `tests/`, where the app is single-user.

Two things that suite gets wrong easily, both of which surface as a foreign-key violation or a hung `TRUNCATE` rather than as anything that names the cause:

- **Flush a `User` before its dependent rows.** None of the multi-user models declares an ORM relationship to `User`, so a single flush orders the INSERTs by mapper sort key, which puts `Subscription` first. See the note on the model.
- **Set `auth_mode` when you patch `settings` wholesale.** The lifespan and `create_app()` both read it, and a bare `MagicMock` compares unequal to every string, so the multi-user branches silently do not run.

## Backwards Compatibility

Until this project has its first production release, you do not need to be concerned about backwards compatible changes.

## Coding Standards

- All type annotations required
- Ruff rules: `E, F, I, UP, B, SIM, ANN, RUF` (line length 100, `E501` and `B008` ignored)
- SQLAlchemy 2.0 `mapped_column` style for all ORM models
- Pydantic v2 for all data classes and request/response schemas
- All routes `async def`
- All LLM calls via any-llm `amessages` (async)
- Never use `BaseHTTPMiddleware` for streaming endpoints -- use pure ASGI middleware
- Conventional commit prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`, `chore:`
- Every data endpoint uses `Depends(get_current_user)` with `user_id` scoping
- Config via Pydantic `BaseSettings` with `extra="ignore"`
- Prefer `isinstance` checks and direct typed attribute access over `getattr`, `hasattr`, or string-based type checks. Our objects are properly typed; using `getattr` with defaults masks real problems when types change. Only use `getattr`/`hasattr` when explicitly directed or at true dynamic boundaries (e.g. plugin APIs).
- Never use em dashes in user-facing content, comments, or copy -- use periods, commas, colons, or pipes instead
- All imports at the top of the file. No inline or deferred imports inside functions. The only exception is `TYPE_CHECKING` guarded imports.

## Privacy & PII

**Never write down real names or real personal information anywhere that gets persisted or shared.** This includes, but is not limited to:

- Source code (comments, docstrings, string literals, variable names)
- Tests and test fixtures (use obviously fake names like `Alice`, `Bob`, `Test User`, or domain-appropriate placeholders)
- Documentation (READMEs, AGENTS.md, CLAUDE.md, SKILL.md, user guides, design docs)
- Commit messages, branch names, and PR titles/descriptions/comments
- GitHub issues, discussions, and any other public artifact
- Migration files, seed data, and example payloads
- Logs, error messages, and debug output that may be checked in

Real PII to avoid: real customer/user names, real phone numbers, real email addresses, real addresses, real business names from customer data, real Telegram handles or chat IDs, real OAuth tokens or API keys.

**Soft PII**: use judgment for content that isn't on the hard list above but still ties back to a specific user. Generalize in your own writing.

When you need realistic-looking data, use clearly synthetic values: `jane.doe@example.com`, `+15555550123`, `Acme Plumbing`, UUIDs, or `faker`-style placeholders. If you encounter real PII in a debugging session or pasted content, scrub it before committing or pushing anything to GitHub.

## Testing

- pytest with FastAPI `TestClient`
- PostgreSQL for all tests (requires a local `clawbolt_test` database; see conftest.py)
- `reset_stores()` clears cached store singletons between tests
- Override `get_current_user` via FastAPI dependency injection
- Mock ALL external services: messaging channels (BlueBubbles, Twilio, Linq, Telegram), LLM (any-llm), faster-whisper, Google Drive
- Bug fixes must include regression tests

## Architecture

- **PostgreSQL storage**: all structured data in PostgreSQL via SQLAlchemy 2.0 ORM. See `backend/app/database.py` and `backend/app/models/`. Store modules in `backend/app/agent/` provide CRUD APIs.
- **Auth plugin infrastructure**: base.py (ABC), loader.py (dynamic import), dependencies.py (get_current_user), scoping.py (row-level auth). `AUTH_MODE` selects the model: `single_user` (default) resolves every request to the one user in the database; `multi_user` delegates to a resolver registered via `set_current_user_resolver()` and never falls back to the single-user path, since that would serve one tenant's data to an unauthenticated caller.
- **`user_id` scoping** on every data class and endpoint from day one
- **Message bus**: async inbound/outbound queues in `bus.py`. Channels publish inbound messages; the agent publishes outbound replies. The ``ChannelManager`` dispatches outbound messages to the correct channel.
- **Agent loop**: channel webhook -> media pipeline -> tool-calling loop (any-llm `amessages`) -> tool execution -> reply
- **Memory**: Freeform per-user MEMORY.md managed via workspace tools, backed by `memory_documents` table with automatic compaction
- **Prompt-cache epochs**: `backend/app/agent/prompt_epoch.py` owns the one definition of a cold start (the first message after the cache idled out, read from message timestamps) and what keys off it: the per-epoch workspace snapshot in the system block, and the cold-start history rebuild. Anything that renders history or the system block for the agent goes through it, so the cached prefix stays byte-identical inside an epoch. The rebuild stubs only the results of calls that read (`tools.base.is_mutating_call`, against the turn's own tools); a write's result stays verbatim at any age, because it may be the only place the ID of what it made is written down
- **Services**: External services abstracted behind service classes in `backend/app/services/`

## Multi-user mode

`AUTH_MODE=multi_user` (see `docs/self-host/configuration.md`) turns one process into a hosted, multi-tenant deployment. The default, `single_user`, is unchanged self-hosted behavior and none of this is reachable there.

What the mode switches on, and where it lives:

| Surface | Modules |
|---|---|
| Google OAuth sign-in, JWT sessions, admin API keys | `auth/google_oauth` router, `auth/oauth_flow.py`, `auth/jwt_auth.py`, `auth/session_auth.py`, `services/admin_api_keys.py` |
| Admin console, audit log, consent-gated shared data | `routers/admin/`, `routers/admin_shared_data/`, `routers/admin_reported_conversations.py`, `services/admin_audit.py`, `services/pii_redaction.py` |
| Account page, data export, deletion | `routers/account.py`, `services/data_export.py`, `services/user_deletion.py`, `services/inactive_cleanup.py` |
| Per-tenant quotas and plans | `billing/` |
| Operator monitoring and email | `routers/monitoring.py`, `services/health_monitor.py`, `services/admin_alerts.py`, `services/email_service.py` |
| Model comparison report | `routers/admin_model_comparison.py`, `services/model_comparison/` |
| Request middleware (security headers, SEO meta, admin config guard) | `middleware/` |
| KMS envelope encryption | `security/kms.py`, `security/dek_cache.py`, `security/validate.py` |

On the frontend, the same surface lives under `frontend/src/extensions/`, plus `pages/marketing/`, `pages/docs/`, `layouts/`, and `components/LoginPage.tsx`. It ships in every build and gates itself at **runtime**, not at build time: `isPremiumAuth()` reads `/api/auth/config`, which only reports `oauth_google` when the backend is in multi_user mode. There is no separate frontend build for the two modes, and no overlay.

Terms and Privacy are the exception. They are a contract naming a specific operator, so a deployment supplies the prose at `public/legal/*.html` and `pages/marketing/LegalPage.tsx` renders it. A deployment that supplies nothing gets a placeholder. Do not check legal text into this repo.

Three rules when touching this:

- **`create_app()` is the only place that decides what mounts.** Routers and middleware are conditional there. Do not gate a route by checking the mode inside the handler.
- **The agent-level hooks are process-global,** so they are installed at `main.py` import under `if MULTI_USER`: the quota pipeline, the `ChannelRoute` allowlist override, the heartbeat usage hook, the per-user LLM resolver, and the payload-capture observers. They cannot be per-app, which is why `create_app()` does not touch them.
- **`get_kek_provider()` in `auth/loader.py` is load-bearing for data.** Returning a different provider than the one that wrote a row makes every `EncryptedString` column on it unreadable. Changing its resolution order is a data migration, not a refactor.

### Model comparison report

`services/model_comparison/` answers "what would change if this user moved to
a different model" by replaying their own recent turns through one candidate
and laying each decision beside what production actually did, which is already
in the transcript. The admin console drives it; `routers/admin_model_comparison.py`
owns the job lifecycle.

It returns no verdict, and adding one back is the change this rewrite exists
to prevent. The evaluator it replaced scored each turn's first decision
against a live replay of the incumbent and converted that diff into a
recommendation through sign tests, ceilings, confounder guards and blocking
tiers. Four rounds of review kept finding artifacts in the machinery rather
than in the models: identical candidates blocked, bad candidates approved,
judge blinding that leaked. This deployment has three users whose transcripts
the operator reads anyway.

Invariants, each of which the feature is worthless without:

- **A replay never executes a tool.** Executing would text real customers and
  mutate real job records on every run. A replay continues past a lookup only
  when every call in the response is read-only (`checks.is_mutating_call`) and
  matches a call the live turn made; it then feeds back the result that turn
  recorded, for at most `MAX_REPLAY_READ_ROUNDS` extra rounds
  (`execution.call_model`). Anything else, a write included, is the decision
  recorded. Feeding a recorded result is not execution: it copies a string out
  of the user's transcript, and nothing in `execution` can invoke a `Tool`,
  which it holds only to read `params_model` and the read-only tag. The tool
  context the fixture builds passes `_refuse_outbound`, which raises rather
  than publishing, so a broken invariant fails a run loudly instead of
  messaging a real user.
- **Prompts are built by `ClawboltAgent.assemble_prompt`,** the same method the
  live loop calls. A second assembly implementation would report on prompts no
  user ever received. If you change how the agent assembles a turn, the replay
  follows automatically; keep it that way.
- **History is rendered by `prompt_epoch.build_history_view`,** the function
  the live loop's history renderer calls, with the cold-start rebuild on or
  off per the run's `history_mode` (`types.HistoryMode`). To read what the
  rebuild changes before enabling `COLD_START_COMPACTION_ENABLED`, start two
  runs over the same turns with the *incumbent* model as the candidate, one
  with `full` and one with `cold_start_compaction`. The `full` run is the
  sampling-noise floor; what the compacted run adds beyond it (missed or
  different writes, extra re-fetch lookups, `FABRICATED_ID` on the production
  side, which counts writes whose IDs only an elided result carried) is the
  rebuild's cost.
- **The baseline is the record, not a second call.** Only the candidate is
  sent anywhere. A live incumbent replay would re-sample a model whose answer
  for that turn is already stored, doubling the spend to reintroduce the
  sampling noise the old verdict machinery existed to reason about. The run
  records the user's current model as a label so the report says what the
  candidate would replace; nothing is sent there.
- **A count of findings is only readable beside the other side's.** The
  deterministic checks run against the record too, so the report can say the
  incumbent does this as well. Three cannot be asked of a recorded turn and
  are candidate-only: `UNREQUESTED_WRITE` (production's own writes are the
  standard it compares to), `UNKNOWN_TOOL` (a tool it called existed when it
  called it) and `TRUNCATED` (a delivered reply carries no spent budget).
  `types.PRODUCTION_CHECKED` is the set that is asked of both, it ships on the
  summary, and the console renders the rest as "not applicable" rather than as
  a clean zero. A zero for a check nobody ran is a measurement nobody took.
- **`UNREQUESTED_WRITE` compares arguments, not only tool names.** Three
  shapes, all in `checks.check_candidate`: a write through a tool the live
  turn never *wrote* with, where a read does not count, since a turn that
  only asked `manage_integration` for status did not ask for a disconnect;
  a write carrying record IDs that no production write to that same tool
  touched; and more user-facing messages (tools tagged
  `ToolTags.SENDS_REPLY`, which today is `send_media_reply` alone) than
  production sent on the turn. The last two exist because a name-only
  exemption could not see a second `add_note` against the neighbouring job
  or a second attachment to the customer, which are the two shapes this
  deployment can actually suffer. Sharing one record ID with a production
  write to that tool is enough to pass: a write to the right record with
  different wording is a `WriteOutcome`, and charging it here too would make
  every paraphrase a safety finding. What a write is compared *on* is
  `checks.write_targets`: its record IDs and its file paths, because a path
  identifies a document as well as an ID identifies a record and a live
  `write_file` on one file must not exempt a candidate `write_file` on
  MEMORY.md. Deliberately not `collect_ids`, which stays about record IDs
  alone: an invented path creates a file rather than acting on somebody
  else's record, so paths have no business in `FABRICATED_ID` or in the write
  comparison's `record_ids`. One shape stays out of reach: a write naming
  neither a record nor a file, through a tool production also wrote with,
  passes on the tool name. That is `update_heartbeat`,
  `companycam_create_project`, `discard_media` and `manage_integration`, and
  a create has nothing to name by construction. The console's safety panel
  says so, because a limitation only a docstring carries is one the operator
  reading the count never learns. The prose reply is not a tool call, so the
  message count never sees it on either side.
- **Not every finding is a violation.** `types.HARD_VIOLATIONS` is what the
  counts total. `TOOL_NOT_IN_SCHEMA` describes the replayed fixture (a name in
  this user's history that the current schema lacks) and `CALL_FAILED` is a
  failure to measure, so both are recorded on the turn and excluded. Anything
  reading `bool(findings)` as "this model is unsafe" is a bug.
- **A write's record IDs must come from what the model saw.** `FABRICATED_ID`
  is deterministic: an ID-shaped argument of a mutating call (named `*_id`,
  `*_ids`, `*_ref` or described as an ID in the params model) that appears
  nowhere in the prompt, the user's message, or a lookup result it had by then
  is a guess. Checked on both sides, but not equally: the candidate's rounds
  are known, so its haystack grows only between rounds, while the record
  stores a flat list of calls with no round boundaries and the production
  haystack therefore grows call by call. A production write is credited with
  the result of a read issued in its own response, which it could not have
  seen, so `FABRICATED_ID` is under-reported on the production side. Nothing
  can close that without round markers on the stored calls, and it errs in
  the safe direction: it flatters the incumbent, not the candidate. Name new
  ID parameters that way so the check covers them.
- **Whether a tool mutates comes from `ToolTags.READ_ONLY`, not the approval
  policy.** Untagged means mutating. See step 7 of "Adding a New Agent Tool".
- **One reading of a tool call, shared.** `checks.accept_args` is where the
  params model is applied, with the same numeric-to-string repair the live
  loop applies (`core_support._stringify_numbers_for_string_fields`). Running
  the bare params model in one place and the repaired one in another made the
  two halves disagree about a single call: `checks` said nothing about
  `add_note(work_order_id=118600)` against production's `"118600"`, while
  `report` reported the right record with different arguments and named a
  field that does not really differ. Inside a free-form payload the params
  model declares nothing (`qb_update`'s `data` is a `dict[str, Any]`), so the
  write comparison settles the spelling itself: `report.comparable` renders
  every integral number as its digits before comparing, the way `collect_ids`
  already does for record IDs.
- **`MATCHED` needs the whole validated argument set.** For every write the
  live turn made, `report.compare_writes` reports one of six outcomes.
  `MATCHED` is agreement on every argument after the params model fills its
  defaults. `SAME_RECORD_DIFFERENT_ARGS` is the right record with different
  content; `SAME_TOOL_DIFFERENT_ARGS` is the tool called against something
  production did not write to, or a write with no record ID at all whose
  arguments differ; `MISSED` is not calling the tool. Two more are not
  readings of a decision at all: `NOT_REACHED` is a replay that ran out of
  lookup rounds before the candidate decided, and `NOT_REPLAYED` is a turn
  the provider errored on, so the candidate never saw it.
  Agreement on record IDs alone used to be enough, and it meant
  `qb_update(entity_type="Invoice", data={Id:123, TotalAmt:500})` and
  `qb_update(entity_type="Estimate", data={Id:123, TotalAmt:5000})` counted as
  one write in the headline rate. The headline counts `MATCHED` only, over
  `writes_measured` (`writes_total` less both unmeasured buckets), which
  understates the candidate rather than flattering it. Reading the turn is
  still what separates a paraphrase from a note on the wrong job.
- **Matching is greedy.** Each production write is compared against every
  candidate call to that tool independently and the best reading wins, so two
  production writes to one tool can both be judged against the same candidate
  call and both report a match. A pairing that consumed each candidate call
  once would have to choose which production write to charge for the
  shortfall; choosing wrong is worse than over-crediting a tool the candidate
  did reach.
- **A candidate that produces nothing is a `TurnOutcome`, not a finding.**
  `NO_CANDIDATE_OUTPUT` is production answering (prose or a write) and the
  candidate returning no text, no tool call and no error. It passes every
  deterministic check by having nothing to check, so before it existed the
  turn landed in `no_write`, counted towards nothing and rendered collapsed at
  the bottom. It is deliberately *not* in `candidate_violations`: that count
  is findings, which are per-call defects read off a tool schema, and a silent
  turn has no call to charge. It gets its own summary tile, its own note, and
  the top of the turn ordering instead. `REPLAY_INCOMPLETE` is the sibling
  case for the measurement running out (`execution.MAX_REPLAY_READ_ROUNDS`),
  and its writes are `NOT_REACHED` rather than `MISSED`, so a cap the replay
  hit is never reported as a write the candidate skipped. `NOT_REPLAYED` does
  the same job for a turn the provider errored on. That one is easy to get a
  lot of: `MAX_CONSECUTIVE_CALL_FAILURES` counts *consecutive* failures and
  the turns run under `asyncio.gather`, so a flaky provider scatters errored
  turns through a run without tripping the breaker, and every write on every
  one of them used to land in the rate's denominator as a miss.
- **Cost is `None` when nothing can price it, never zero.** A model served
  through a gateway is billed by whoever is behind it, which the (provider,
  model) pair no longer names, so `LLMTarget.priced` suppresses the figure and
  the summary carries the reason instead. The old column reported `0.000000`
  next to a warning, and the number beat the warning every time. Wiring a real
  per-endpoint price would need price columns on `llm_endpoints`, which do not
  exist. `percentile_latency_ms` is `None` on no samples for the same reason.
- **A candidate cost figure is not a quote, and the report says so.**
  `report.COST_COMPARABILITY` is always in the notes when a cost is shown:
  providers bill different prompt token counts for byte-identical prompts (up
  to 1.7x between two of this deployment's), and a replay's cache hits are an
  artifact of re-sending one turn rather than following a conversation.
  Beside it, `production_usage.read_production_usage` sums `llm_usage_logs`
  over the sampled turns' own timestamps, so the tile carries what the user
  actually costs today across the same days. That number is not like-for-like
  either and is labelled as the window's total: it includes every live call
  inside it, heartbeats and compaction included, and every tool round of
  every turn in the window, while the candidate's covers only the sampled
  turns, each one decision plus the lookup rounds and truncation retries the
  replay spent reaching it (`execution._add_usage`). The window also ends at
  the newest sampled turn's own timestamp, so that turn's live calls, which
  come after it, are outside it.
- **A failing provider stops the run.** `MAX_CONSECUTIVE_CALL_FAILURES`
  consecutive errored turns end it with `FAILED`, the evidence already
  gathered, and the reason in both the `error` column and the summary's notes.
  A run competes with live traffic at the same gateway, so it must not spend a
  200-sample budget rediscovering that the provider is down.

Consent-gated like `/admin/shared-data`: a run reads real conversations and the
report renders them back, so both require `User.data_sharing_consent`. Content is
PII-redacted at serialization, after the checks, so findings are computed on
real values and only the human-readable drill-down is masked.

## Adding a New Agent Tool

The agent's capabilities are extended by adding tools. Tools follow a factory/registry pattern with auto-discovery.

### Core vs. Specialist

- **Core tools** (`core=True`): Always available to the agent on every message. Use for universal capabilities (math, messaging, files, workspace). No activation step needed.
- **Specialist tools** (`core=False`): Gated on the integration's `auth_check`; loaded onto the schema from turn 1 once the user connects the integration (the tool list is held stable per auth state for prompt-cache reasons, issue #1170). Use for integrations and domain-specific features (calendar, QuickBooks, CompanyCam). The `list_capabilities` meta-tool surfaces unconnected integrations and serves SKILL.md guidance; it does not activate tools.

### Checklist for adding a tool

1. **Create the tool module** at `backend/app/agent/tools/<name>_tools.py`. Follow the pattern in `heartbeat_tools.py` (simplest example): Pydantic params model, async tool function returning `ToolResult`, factory function, and `_register()` called at module level. The `_tools` suffix is required for auto-discovery.

2. **Add tool name constants** to `backend/app/agent/tools/names.py` in the `ToolName` class. All tool name strings must be defined here to prevent silent breakage on renames.

3. **Register in the dashboard** at `backend/app/routers/user_tools.py`:
   - Add the factory name to `_CORE_FACTORIES` (if core) so it cannot be disabled
   - Add a `_FACTORY_META` entry with a description (and `domain_group`/`domain_group_order` if specialist)

4. **Add to the registry test** at `tests/test_tool_registry.py`: add `"backend.app.agent.tools.<name>_tools"` to `EXPECTED_TOOL_MODULES`.

5. **Wire up approval policies** for any mutating tool. If a `SubToolInfo` declares `default_permission="ask"`, the corresponding `Tool` object **must** have `approval_policy=ApprovalPolicy(default_level=PermissionLevel.ASK)`. Without this, the WebUI shows "ask" but the runtime auto-executes. See `quickbooks_tools.py` for the reference pattern. The global test `test_ask_sub_tools_have_approval_policy` in `test_tool_registry.py` enforces this.

6. **Set a `concurrency_group` if your tool mutates shared state.** The agent runs all approved tool calls from a single LLM turn concurrently by default. Tools with the same non-None `concurrency_group` serialize in submission order; tools with different keys (or `None`) may run in parallel. Set this whenever your tool could race with another tool in the same turn against a shared resource, for example a DB row, a workspace document, a disk file, or the user-facing message stream. Read-only and stateless tools should leave it `None`. Accepts either a static string or a callable that takes the validated args and returns a key, for the case where a single tool routes to distinct resources by argument (e.g. workspace writers keyed by file path). Existing keys: `"workspace_path:<path>"` for workspace document mutations (resolved per call by `_workspace_path_concurrency_key`), `"user_outbound"` for reply senders, `"user_integrations"` for integration toggles. The global test `test_state_mutating_tools_have_concurrency_group` in `test_tool_registry.py` enforces that any tool tagged `MODIFIES_PROFILE` or `SENDS_REPLY` declares one.

7. **Classify the tool as a read or a write.** Tag it `tags={ToolTags.READ_ONLY}` if calling it only looks something up; otherwise add its name to `_MUTATING_TOOLS` in `tests/test_tool_registry.py`. Untagged means mutating. The approval policy cannot stand in for this: `ApprovalPolicy` defaults `default_level` to `ASK`, so most search and list tools are gated too, while `write_file` and `manage_integration` write without being gated. The model comparison report counts an unrequested write as a finding and reads this tag to tell production's writes from its lookups, so a read left untagged is charged on both sides and stops a replay from continuing past it, and a writer left unlisted lets a candidate rewrite MEMORY.md with nothing reported. The cold-start history rebuild reads it too: a writer tagged `READ_ONLY` has its old results stubbed, losing the IDs of what it created. A tool whose actions differ (`manage_integration`'s `status`) stays untagged and sets `read_only_when`; argument checks that live in the tool body belong in `precheck`, so a call the tool would refuse is not counted as a write. The global test `test_every_tool_is_classified_read_or_write` in `test_tool_registry.py` enforces it, reaching integration tools through their own builders rather than the registry factories.

8. **Write tests** at `tests/test_<name>_tools.py`. Call the factory function directly (e.g., `_create_calculator_tools()`) and invoke the tool function. No database needed for stateless tools.

9. **(Specialist only) Add a SKILL.md** at `backend/app/agent/skills/<name>/SKILL.md` if the tool has complex workflows the LLM needs guidance on. This markdown is delivered into the conversation as a tool result: either when the LLM calls `list_capabilities("<name>")`, or auto-appended to the first result of the category's tools when the model skips discovery (each delivery carries a `[skill-guidance: <name>]` marker so it happens at most once per context window). Core tools do not need SKILL.md; their `description` and `usage_hint` fields in the Python code serve the same purpose. See "SKILL.md structure for specialist tools" below for the expected skeleton.

### Key files

| File | Purpose |
|---|---|
| `backend/app/agent/tools/base.py` | `Tool`, `ToolResult`, `ToolErrorKind` definitions |
| `backend/app/agent/tools/names.py` | All tool name constants (`ToolName` class) |
| `backend/app/agent/tools/registry.py` | `ToolRegistry`, `ToolFactory`, `ToolContext`, auto-discovery |
| `backend/app/agent/skills/loader.py` | SKILL.md loader (`get_skill_instructions`) |
| `backend/app/routers/user_tools.py` | Dashboard wiring (`_CORE_FACTORIES`, `_FACTORY_META`) |

## Editing prompt files (SKILL.md, system prompts)

`SKILL.md` files and the agent system prompt are injected into the LLM context on every relevant turn, so prose costs tokens on every conversation. Be terse and non-redundant when editing them.

- **Do not duplicate information that already lives elsewhere in the file.** If a field's shape is shown in a payload example or an example workflow, do not re-document it in a separate "field shapes" preamble. The agent reads the whole file.
- **State rules at the failure mode, not as top-of-section framing.** A "do not claim X is unavailable" warning belongs inside the workflow that prevents X. A general preamble at the top of a section is usually padding the headings or steps already imply.
- **Trust the steps.** A numbered workflow does not need an intro paragraph explaining when to use it; the heading and the cross-references from other sections already do that.
- **Cut padding.** Phrases like "Use this whenever ...", "If a field is listed here, the entity has it ...", "It is important to ..." are framing the structure already implies. Delete the sentence; if the meaning is intact, it was redundant.

After editing, read the diff and ask: did I add a new fact, or restate an old one? Restated facts double the prompt without doubling agent behavior.

### SKILL.md structure for specialist tools

A specialist SKILL.md is delivered at most once per context window (on `list_capabilities("<name>")` or first use of the category's tools), so it pays its cost in one place. Aim for 60-150 lines covering what tool descriptions and `usage_hint` strings cannot carry on their own.

Use this skeleton; drop sections that do not apply:

1. **Lead paragraph.** Name the platform, the entities that live in it, and the scope of this integration's coverage.
2. **Available Tools table.** Group by domain when the surface is wide (e.g. Projects / Photos / Checklists). Columns: tool name in backticks, purpose, approval (`Auto` or `Ask`). Every registered tool gets a row.
3. **Entity vocabulary.** Bullets defining each first-class entity (id, key fields, relationships). Include enum literals (status values, type codes) verbatim; the agent has no other way to know them.
4. **Failure-mode rules.** Per-topic sections (e.g. `## Photo handles`, `## Dates`) stating non-obvious constraints at the place they apply, plus the consequence of breaking them.
5. **Common Workflows.** `### Named workflow` subsections with numbered steps, one tool call or one decision per step. Cover the multi-tool recipes the agent cannot infer from individual tool descriptions.
6. **Companion integrations.** Bullets pointing to other integrations the agent may compose with (e.g. "X customer ids are not Y project ids"). Reciprocate when another SKILL.md cross-references this one.
7. **Connecting.** OAuth flow, paste-token, or magic-link steps. Note what happens if the connection lapses.

Reference implementations: `backend/app/integrations/servicetitan/SKILL.md` (clean, mid-sized) and `backend/app/integrations/companycam/SKILL.md` (wider tool surface).

## Definition of Done

Every change must pass all checks before it's considered complete:

```bash
uv run pytest -v                                  # tests pass
uv run ruff check backend/ tests/ alembic/                 # lint passes
uv run ruff format --check backend/ tests/ alembic/        # format passes
uv run ty check --python .venv backend/ tests/ alembic/    # type checking passes
cd frontend && npm run typecheck                   # TypeScript type checking passes
cd frontend && npm run deadcode                    # no dead JS/TS code (knip)
cd frontend && npm run test                        # vitest suite passes
```

### Frontend generated types

When backend schemas change (`backend/app/schemas/`, route signatures, response models, or endpoint docstrings), you **must** regenerate the frontend OpenAPI types. Never hand-edit `frontend/src/generated/api.d.ts`. CI will fail if the committed file doesn't match what the generator produces.

The spec is exported in `multi_user` mode, so it documents every route the product can serve. The frontend ships as one bundle for both modes and calls those routes through a typed client; a spec narrowed to `single_user` would leave the sign-in, account, and admin calls untyped.

```bash
uv run python scripts/export_openapi.py           # export openapi.json from backend
cd frontend && npm run generate:api                # regenerate src/generated/api.d.ts
```

Commit both `frontend/openapi.json` and `frontend/src/generated/api.d.ts`.

- Bug fixes include regression tests
- New features evaluate whether the user docs (`frontend/src/docs-content/`) need updates
- Features that change how users interact with the assistant must update the user guide (`frontend/src/docs-content/guide/`)
- When you manage a pull request, you must always adhere to the pull request template at .github/pull_request_template.md
- CI green

## Sandbox Tips

### Ephemeral directories

`target/`, `node_modules/`, and `.venv/` don't persist between sessions. Run `uv sync` at the start of each session if needed.

### PostgreSQL for tests

Tests require a running PostgreSQL instance. In a sandbox without Docker, install and start PostgreSQL directly:

```bash
# Install PostgreSQL (Debian/Ubuntu)
apt-get update -qq && apt-get install -y -qq postgresql postgresql-client

# Start the cluster
pg_ctlcluster 16 main start

# Create the test user and database
su - postgres -c "psql -c \"CREATE USER clawbolt WITH PASSWORD 'clawbolt' CREATEDB;\""
su - postgres -c "psql -c \"CREATE DATABASE clawbolt_test OWNER clawbolt;\""
```

The test suite connects to `postgresql://clawbolt:clawbolt@localhost:5432/clawbolt_test`. The conftest.py handles table creation and per-test TRUNCATE automatically.

### Parallel pytest with xdist

`pytest -n auto` (or any `-n <N>`) is supported. Each xdist worker writes to its own database (`clawbolt_test_gw0`, `clawbolt_test_gw1`, …); the conftest auto-creates them on demand by connecting to the `postgres` admin DB (so the `clawbolt` role needs `CREATEDB`, which the install command above grants). Sequential runs and CI continue to use `clawbolt_test` unchanged.

Set `OSS_TEST_DB=<name>` to override the database name explicitly (useful when several agents or branches run pytest at the same time and you want stable per-branch DBs).

### Git operations

Git auth is pre-configured. Never push directly to main. Always create a branch and open a PR.

## Design System
Always read DESIGN.md before making any visual or UI decisions.
All font choices, colors, spacing, and aesthetic direction are defined there.
Do not deviate without explicit user approval.
In QA mode, flag any code that doesn't match DESIGN.md.

### Frontend tokens (single source of truth)
The frontend is token-driven; `frontend/src/styles/README.md` is the engineering reference. Rules:

- **Never hard-code colors in components.** Use semantic Tailwind utilities backed by tokens (`bg-card`, `text-muted-foreground`, `text-primary`, `border-border`, the `*-bg`/`*-text` state pairs). No raw palette colors (`bg-gray-200`), no hex literals, no `text-white` on a colored fill (use the fill's `*-foreground` token). Brand/3rd-party icons are the only exception.
- **Colors live in two source files**: `brand-tokens.css` (`--brand-color-*`, drives app utilities) and `palette.ts` (drives HeroUI components). To change a color, edit the token there, not components. If it is a HeroUI-used color, also edit `palette.ts` and run `npm run generate:tokens` (regenerates `heroui-tokens.generated.css`, which is committed and never hand-edited).
- **HeroUI is themed through the same tokens** at runtime; customize HeroUI via props / `classNames` with token utilities, not by overriding internals.
- **`src/styles/tokens.test.ts` guards** undefined-token usage, palette/brand-tokens drift, and generated-file freshness. `node scripts/audit-contrast.mjs` checks WCAG AA for every semantic pairing. Keep all semantic pairings AA in both themes.
- Mobile-first responsive utilities; wrap tables in `overflow-x-auto`; prefer CSS state over JS layout swaps to avoid layout shift.
