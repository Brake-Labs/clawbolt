# Memory

Clawbolt keeps durable business knowledge across conversations: rates, preferences, shorthand, and process rules. Current customer details, balances, schedules, and job status stay in their connected systems so Clawbolt can look them up instead of relying on a stale copy.

## How it works

Memory is exposed to the agent as a per-user `MEMORY.md` workspace document and stored in PostgreSQL. The agent manages it with workspace tools. When conversations get long, automatic compaction updates `MEMORY.md` and appends timestamped breadcrumbs to `HISTORY.md`. Compaction folds new facts into existing sections, replaces outdated ones (a changed rate, a finished job), and aims to keep `MEMORY.md` under a size target (`COMPACTION_MEMORY_BUDGET_CHARS`, 8000 characters by default), since the whole file is read on every message.

## Saving facts

Clawbolt automatically picks up on facts from your messages. You can also be explicit:

```
You: Remember that my hourly rate for electrical work is $95

Clawbolt: Got it! I've saved that your hourly rate for electrical
          work is $95.
```

## Recalling facts

Clawbolt's memory is always available in context. You can ask directly:

```
You: What's my standard plumbing rate?

Clawbolt: Your standard plumbing rate is $95 per hour.
```

## Forgetting facts

You can ask Clawbolt to forget outdated information:

```
You: Forget my old rate for plumbing, it's $90 now

Clawbolt: Done! I've updated your plumbing rate to $90/hour.
```
