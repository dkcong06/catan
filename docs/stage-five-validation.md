# Stage 5: engine validation

Historical milestone record. The 2026-10-11 [expanded engine and performance review](expanded-engine.md)
supersedes earlier feature boundaries, array sizes, schema dimensions and current test counts below.
Historical measurements retain their original workload and date. Training remains deferred.


Stage 5 implements the continuous invariant and random-game gate from Section 5
of `Catan_RL_Guide.pdf`. It adds explicit validation tooling, not observations or
a trainer. The minimal rules from Stage 4 remain the supported game.

## APIs and checks

`catan.validation.assert_invariants(state)` first calls `state.validate()` to check
all eight arrays' shape, dtype, contiguity and writability, resource conservation,
piece supply, ownership/building agreement, building VP, distance, pending
manual discards, scalar ranges, setup bookkeeping, and the standard RNG type.
It additionally checks completed setup before normal play, roll bookkeeping for
setup/roll/main/robber phases, winning VP and the current terminal player, and
that each owned road component connects to an owned building.

Both independent setup buildings are valid roots. A later opponent settlement
can interrupt extension without invalidating existing roads; connectivity checks
therefore traverse existing roads through that vertex. The engine still rejects
new extension through an opponent building. These are state invariants, not a
proof that an arbitrary synthetic snapshot has a legal history. Seeded runs
exercise actual validated transitions to check reachable behavior.

```python
from catan.state import new_game
from catan.validation import assert_invariants, validate_game

state = new_game(42, n_players=4)
assert_invariants(state)
result = validate_game(0, n_players=4, max_actions=10_000)
assert result.winner is not None and not result.capped
print(result.actions, result.scores, result.fingerprint)
```

`validate_game` checks the initial state and every successor, including terminal
and capped states. The default uses an exclusively owned state and in-place
transitions. `inplace=False` exercises independent functional successors with the
same policy/actions and complete RNG continuation. `ValidatedGame` is an
immutable summary: seed, player count, actions, winner, scores, per-phase visit
counts, and a final SHA-256 fingerprint. No full trajectory is retained. The
fingerprint includes board composition, all scalar bookkeeping, all eight
arrays, and complete game/policy RNG states. It is reproducible with the same
implementation/runtime, not a portable serialized game or proof of correctness.

A `GameValidationError` includes seed, player count, first failing step, phase,
action and diagnostic, with the original error chained. Initialization failures retain the seed and use step zero / `INITIALIZE`. The command stops at its
first returned rule/check failure and exits nonzero. Infrastructure failures
also exit nonzero. No failure is reported as a passed gate.

## Bounded random-game runner

```powershell
docker build -t catan-checks .
docker run --rm catan-checks python -m scripts.validate_games --games 10000 --players both --workers 8

# Save a JSON report outside the disposable container.
New-Item -ItemType Directory -Force output/stage-five | Out-Null
docker run --rm --mount "type=bind,source=${PWD}/output/stage-five,target=/reports" catan-checks python -m scripts.validate_games --games 10000 --players both --workers 8 --report /reports/stage-five-validation.json

# Replay one case; use its explicit player count from the report.
docker run --rm catan-checks python -m scripts.validate_games --games 1 --seed 3 --players 4

# Exercise the copying path using the same case.
docker run --rm catan-checks python -m scripts.validate_games --games 1 --seed 3 --players 4 --functional
```

`--games` is the total episode count. Seeds are `--seed + index`; `--players both`
alternates three and four seats by index, so 10000 episodes cover 5000 each with
10000 distinct seeds. `--max-actions` defaults to 10000; `--workers` defaults to
one. Python 3.14's process-pool `buffersize=2*workers` with four-case chunks bounds
submitted work. Each process owns its state/RNG; no states or live mutable caches
are shared. Result ordering and the suite fingerprint are deterministic across
serial/parallel execution. Choose workers to suit available CPU/memory.

The JSON report records arguments/runtime, checked actions/states, actual wins,
caps, per-player-count seat wins, phase coverage, failures, capped-case replay
coordinates/fingerprints, elapsed time and an ordered suite fingerprint. Retained
summaries scale with capped episodes, never with every action. It prints progress
every 100 episodes and writes the final report after the batch.

**Termination boundary:** the minimal rules can stalemate, as demonstrated by the
existing four-player seed-3 regression. Capped episodes are truncations, not wins,
losses or invariant failures. A zero-error bounded run does not prove the guide's
assumption that every random game reaches ten VP. The runner makes that limitation
visible; the future RL environment must keep `terminated` and `truncated`
separate. It does not change the victory threshold or add deferred rules to hide
caps. The older `play_random_game` API continues to raise at its cap.

## Optimization and verification

Full validation remains opt-in and outside normal engine steps, copies and legal
queries. Static geometry/adjacency and cached action objects retain their IDs and
ordering. State layout, independent copies, RNG draws, no adjacent red tokens,
port generation/discounts and the optimized Stage 4 legality/production paths
retain their contracts. No mutable per-state cache, ML dependency, state field,
or persistent scratch buffer was added.

After the initial implementation passed all 563 tests, profiling showed repeated
tiny-array NumPy reductions/masks and NumPy-scalar road traversal dominated
continuous validation. The optimized checks convert small arrays to Python values
per call, tally placed pieces once, and traverse existing immutable adjacency
tables with integer ownership values. Storage validation preserves its diagnostic
ordering and messages. Transient Python lists/sets are bounded by the board size.

Five warmed alternating paired batches, without competing test/stress jobs,
compared the initial Stage 5 checker to its optimized version on Docker Linux,
Python 3.14.7 / NumPy 2.4.6. A pool of 100 states was sampled from real three/four
player games; each microbenchmark performed 4000 checks. Episode batches used
seeds 0, 1 and 3 for both player counts, capped at 1000 actions (5608 actions total).

| Work | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Storage check | 256.18 us | 79.07 us | 3.24x |
| Full invariant check | 416.71 us | 126.51 us | 3.29x |
| Six continuously checked episodes | 2.75194 s | 1.05479 s | 2.61x |

All episode summaries and final state/full RNG fingerprints matched. Storage
validator acceptance/errors matched on 15490 valid and corrupted fixtures. The
initial and optimized unit gates each passed 563 tests. The raw paired report is
`output/stage-five/stage-five-optimization.json`; throwaway source snapshots and
comparison code are removed after review. These timings measure validation,
including reset/legal queries/random policy in the episode batch. They exclude
rendering, neural inference and training and do not imply an RL training speedup.

Reproduce current profiling with:

```powershell
docker run --rm catan-checks python -m cProfile -s cumulative -m scripts.validate_games --games 1 --seed 0 --players 4 --max-actions 1000
```

New regressions cover all phases in completed games, exact functional/in-place
and serial/parallel summaries, caps, invalid bounds, state/RNG isolation, failure
coordinates, disconnected road components/cycles, independent setup roots,
later opponent blocking, temporal/winning-phase checks, and keeping checks out
of normal engine steps. Production is checked against an independent
building-first oracle across all non-seven totals and varied shortages, with an
explicit roll-eight/two-tiles/city multiplier case. Existing tests retain manual
seven choices, robber stealing, exact costs, distance, trade and copy coverage.

The continuous stress gate checked **10000 episodes / 17112427 actions** with
**zero crashes or invariant failures**, using seeds 0-9999, the default 10000-action
cap, and eight workers. Including initial states, it performed 17122427 full
checks. All eight phases were visited. Runtime was 520.40 seconds on this run;
this is a validation workload, not a training benchmark.

| Seats | Episodes | Real wins | Capped |
| --- | ---: | ---: | ---: |
| 3 | 5000 | 4941 | 59 |
| 4 | 5000 | 4689 | 311 |
| Total | 10000 | 9630 | 370 |

The ordered suite fingerprint is
`fcbdf601cab5d68f7c003f5f8b33ff0d4bf7cf38529d1f4f1a0a08710ee750ca`.
The complete report, including individual capped-case replay coordinates, is
`output/stage-five/stage-five-validation.json`. A cap's cause is not automatically
classified as a structural stalemate; that diagnosis is established for the
existing seed-3 regression, not every capped seed. The guide's universal
termination assumption remains unsupported for the minimal rules.

Final review added tests for actual CLI win counts, first-error report/exit
behavior and initialization failure coordinates. These affect diagnostics;
continuous rule/check bodies and successful seeded behavior remain unchanged.
Final gates pass: 566 Python tests (33 Stage 5 regressions), 21 JavaScript tests,
both Docker builds, Ruff lint / 36-file format check, compilation, dependency
consistency, guide-generator F-check, and whitespace. The expanded foundation
gate passes 100000 boards / all 19 desert positions and 5000 three/four-seat states
with two independent branches each. Five runnable Python examples in impacted
documentation pass. Report totals, all-phase coverage, unique capped seeds and
seat assignment were independently checked. Temporary comparison/example source
files were removed; result/profile artifacts remain. PDF/generator/assets,
dependencies, geometry IDs/setup RNG fingerprints and state layout were preserved.

## Next boundary

Representation design and concrete flat encoding (guide Sections 6-7) are now
implemented separately in `rl.encoding`; see the [representation contract](representation-encoding.md)
for the fixed relative schema, visibility, preparation and buffer ownership.
Graph neural network design (Section 8) is next, with a flat MLP baseline.
Keep action encoding/masks and training at their guide stages. The engine remains usable without PyTorch, and
capped episodes must stay distinguishable from wins when an RL wrapper is added.
