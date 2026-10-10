# Engine optimization: owned seven/discard transitions

Implemented 2026-10-10 in the main project. This increment prepares the existing
rules for efficient rollout execution. The subsequent [Stage 4 increment](stage-four-engine.md)
now runs complete minimal games; observations, parallel workers, policy inference,
and training remain unfinished. Measurements below describe the earlier discard-only milestone.

## API and ownership

| API | Accepted transition | Return / ownership |
| --- | --- | --- |
| `resolve_seven(state)` | An already-known seven at ROLL | Independent successor; source retained |
| `apply_action(state, action)` | A validated DiscardAction at DISCARD | Independent successor; source retained |
| `resolve_seven_inplace(state)` | Same seven rule | Same state, updated in place |
| `apply_action_inplace(state, action)` | Same manual discard rule | Same state, updated in place |

Both API families call shared validation and mutation helpers. In-place calls
retain the state, every array, and the complete RNG object; they neither clone
nor advance the RNG for these deterministic rules. Functional calls keep the
existing complete-copy contract, including cached Gaussian continuation.

An in-place caller must exclusively own valid state and its mutable arrays/RNG.
Retained references observe updates. Take `state.copy()` before branching or
saving a mutable snapshot, and give separate workers independent states and seed
streams. This API does not enforce exclusive ownership at runtime and is not a
transaction for arbitrary invalid storage or concurrent edits. Phase, player,
count, and availability checks finish before any mutation; rule-rejected calls
preserve state values, object identities, and RNG continuation. Full `validate()`
remains an explicit debug gate rather than a per-transition cost.

Manual choices, pending snapshots, roller ownership, clockwise scheduling, resource
conservation, and ROBBER_MOVE completion are unchanged. The browser continues to
use functional successors. Geometry IDs, no edge-adjacent 6/6, 6/8, or 8/8, and all
nine ports are unchanged; board/state-copy compatibility gates still apply.

## Temporary storage

The seven kernel sums hands directly into the existing int16 pending array,
then replaces each total with the required discard count. Valid totals are at
most 95, so narrowing the reduction output is safe. This removes a separate
totals array and the division/comparison/selection arrays from the former path.
Each discard converts its immutable resource tuple once into an int16 array,
then reuses that array for both hand subtraction and bank addition. This avoids
coercing the tuple separately for each update.

No new mutable state fields, global scratch storage, or persistent scratch API
were added. A five-count reusable-buffer prototype saved only about 0.7 us over
one conversion in the initial probe; the extra ownership/API complexity was not
justified. A Python scalar-update prototype was slower than the single-array
updates in that probe. Those probe timings include equal fixture resets and are
not the end-to-end measurements below.

## Example

Until full setup/dice actions exist, create a conserved test fixture:

```python
from catan.actions import DiscardAction
from catan.engine import apply_action_inplace, resolve_seven_inplace
from catan.state import Phase, new_game

worker = new_game(seed=42, n_players=3)
worker.phase = Phase.ROLL
worker.setup_index = len(worker.setup_order)
worker.hands[0] = (4, 3, 2, 0, 0)
worker.bank -= worker.hands.sum(axis=0)
worker.validate()

snapshot = worker.copy()  # Independent snapshot, if one is needed.
assert resolve_seven_inplace(worker) is worker
assert worker.discard_pending[0] == 4
assert apply_action_inplace(worker, DiscardAction(0, (2, 1, 1, 0, 0))) is worker
assert worker.phase == Phase.ROBBER_MOVE
assert snapshot.phase == Phase.ROLL
assert snapshot.hands[0].tolist() == [4, 3, 2, 0, 0]
worker.validate()
```

## Reproduce measurements

```powershell
docker build -t catan-checks .
docker run --rm catan-checks python -m scripts.benchmark_engine
docker run --rm catan-checks python -m scripts.benchmark_engine --players 3
```

The default is seven alternating warmed batches of 10,000 operations. Optional
`--iterations` and `--repeats` must be positive. Both paths visit distinct,
identically initialized states, avoiding repeated consumption of an already
mutated fixture. State pools, copies needed to prepare those pools, action
construction, reset, debug validation, and cleanup happen outside the timer.
The explicit copy operation measures copying inside its timer. Pool memory scales
with iteration count. Completed fixtures are checked against expected results
outside the timer; the suite tests broader equivalence and rejection behavior.
A round means resolving a seven plus two/three preselected manual discard actions
for three/four players. It stops at ROBBER_MOVE, not a complete game turn.

Measured 2026-10-10 with Python 3.14.7 / NumPy 2.4.6, four players, seven warmed
alternating batches of 10,000 operations, without competing test/stress jobs:

| Operation | Functional (us) | In-place (us) | Time ratio |
| --- | ---: | ---: | ---: |
| Resolve seven | 68.07 | 14.11 | 4.82x |
| One manual discard | 62.74 | 11.85 | 5.29x |
| Seven + three manual discards | 284.14 | 59.22 | 4.80x |

The explicit full-state copy measured 45.85 us in the same run. These comparisons
are between the current functional and in-place APIs, not a claimed speedup of
the current functional API over an earlier implementation.

These medians depend on machine load, runtime, fixture, and memory/cache behavior.
They exclude complete gameplay, rendering, observation encoding, policy inference,
and learning; the ratios are not claims about future training speed.

## Verification and remaining work

Initial owned-transition gate: 415 Python tests and 21 JavaScript tests passed.
The subsequent Stage 4 contract records the current expanded gate. New gates
compare both paths across 700 varied rounds for all seats and three/four players,
including no-discard cases, cached Gaussian/RNG continuation, conservation, all
array identities, functional source isolation, and rejected-action preservation.
Threshold and 95-card cases additionally forbid `GameState.copy()` in in-place
calls, proving the owned path does not silently clone state. Benchmark smoke tests
exercise both player counts and reject invalid measurement arguments.
Ruff lint and 30-file formatting, compilation, and dependency consistency pass.
The Python checks image builds. The Node target rebuild hit a Docker registry
metadata timeout; all current JavaScript tests pass using the existing Node checks
image with a read-only mount of the main source. No browser script/layout changed.

The subsequent Stage 4 increment completes the minimal gameplay loop and measures
headless baseline episodes, including capped stalemates. Next broaden Stage 5
validation, then implement observation/action/reward interfaces, preallocated
rollout buffers, parallel games, and batched policy inference. Network compilation/mixed precision remain
later, workload-dependent work. No ML dependencies or host packages were added.
