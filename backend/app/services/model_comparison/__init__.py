"""Model comparison report.

Replays a user's own recent turns through a candidate model and lays what it
decided beside what production actually did, which is already in the record.
The output is evidence an operator reads: deterministic safety checks per
side, whether the candidate reached the same writes, what the candidate cost
and how long it took, and the handful of turns worth looking at.

It deliberately returns no verdict. The evaluator this replaced converted a
noisy comparison into a recommendation through sign tests, ceilings and
blocking tiers, and four rounds of review kept finding artifacts in that
machinery: identical candidates blocked, bad candidates approved, judge
blinding that leaked. There is no judge model here and nothing that reads as
a green light.

Entry points:

- :func:`~backend.app.services.model_comparison.runner.launch_run` starts a
  run on a background task; the admin route creates the row first and polls it.
- :data:`~backend.app.services.model_comparison.runner.interrupted_run_sweeper`
  closes out runs orphaned by a restart, periodically rather than at boot.
- :func:`~backend.app.services.model_comparison.runner.mark_interrupted_runs`
  is the sweep itself, also called at startup.

Only reachable in ``AUTH_MODE=multi_user``: the router that drives it is
mounted there, and the thing it exists to inform (moving one tenant to a
different model) has no meaning in a single-user deployment.
"""

from backend.app.services.model_comparison.runner import (
    interrupted_run_sweeper,
    launch_run,
    mark_interrupted_runs,
)
from backend.app.services.model_comparison.types import (
    Finding,
    RunStatus,
    Side,
    TurnOutcome,
    WriteOutcome,
)

__all__ = [
    "Finding",
    "RunStatus",
    "Side",
    "TurnOutcome",
    "WriteOutcome",
    "interrupted_run_sweeper",
    "launch_run",
    "mark_interrupted_runs",
]
