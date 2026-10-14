# Relative flat observations (guide Sections 6-7)

Current 2026-10-13 update: [the optional model/targets](policy-value-targets.md) and [bounded PPO updates/checkpoints](ppo-updates.md) are implemented. Earlier next-work statements below describe dated milestones. Long training/self-play and learned-strength comparisons remain future work; existing PDFs are unchanged.

The next implementation after Stage 5 establishes the representation design and
its concrete flat encoder. `rl.encoding` uses NumPy only, separately from the
engine. The guide recommends this baseline before adding a graph network.
The original step introduced observations only. The 2026-10-11 expanded engine adds
action lookup/masks and batched observations while training remains on hold; see
[the expanded feature/performance contract](expanded-engine.md). The 2026-10-12
[agent environment and outcome evaluation](agent-environment-evaluation.md) now
wrap these features. The 2026-10-13 [collector](rollout-collector.md) records
same-seat trajectories using these exact observations. Neural models, training
and spatial augmentation remain later work.

## API and ownership

```python
import numpy as np
from catan.state import new_game
from rl.encoding import OBS_DIM, FlatEncoder, decision_player, encode_flat

state = new_game(42, n_players=4)
encoder = FlatEncoder(state.board)  # Once per immutable board/reset.
observation = encoder.encode(state)  # Independent float32 array.
assert observation.shape == (OBS_DIM,) and OBS_DIM == 1505
assert decision_player(state) == 0

buffer = np.empty(OBS_DIM, dtype=np.float32)
assert encoder.encode(state, out=buffer) is buffer
branch = state.copy()
np.testing.assert_array_equal(encoder.encode(branch), observation)
np.testing.assert_array_equal(encode_flat(state), observation)
# A value/terminal query can choose a different observer explicitly.
assert encoder.encode(state, player=2).shape == (OBS_DIM,)
```

Prepare one `FlatEncoder` per board. Functional successors and state copies share
that board, so they reuse preparation. A new `new_game` call needs a new encoder,
even if its seed/layout equals the previous board: an identity check rejects stale
contexts in constant time. Encoders have no mutable rollout state or global board
cache. Their immutable static template is shared across their calls; discard the
context after a reset. Multiple workers can share a context if each owns its
states and outputs.

By default, the observer is the **decision maker**. During manual discards this is
`state.discard_player`, while `current_player` remains the roller. The actor's
exact hand is therefore available for selecting discarded resource counts. After
all choices, the perspective returns to the roller. `GAME_OVER` has no actor;
the default observer is the winning current player. Explicit integer observers
support all real seats, including terminal/value queries. Booleans, floats,
strings and out-of-range seats are rejected.

A returned observation owns independent contiguous writable float32 storage.
An optional `out` must be a writable contiguous float32 array with shape `(OBS_DIM,)`,
disjoint from all fourteen state arrays. A row of a preallocated batch is valid. Invalid
buffers/perspectives/board contexts are rejected before writing. Every element is
overwritten, including inactive seats and old robber/setup/phase flags. The output
never aliases static preparation. **Consume or copy reused buffers before the next
call**; retaining the same buffer for an entire rollout would overwrite history.
Default allocation is appropriate when observations must be retained.

Encoding assumes a valid `GameState`; it does not run full debug invariants.
Call `state.validate()` / `assert_invariants` explicitly during debugging.
No state fields, arrays, RNG draws or global/policy random state are changed.

## Current schema version 2

`SCHEMA_VERSION`, `OBS_DIM`, `FEATURE_SLICES` and `FEATURE_SHAPES` publish the
layout. Slice/shape mappings are immutable. Changes to feature order/meaning need
a new schema version before saved policies or rollout files are used.

| Group | Shape | Channels / values |
| --- | --- | --- |
| tiles | 19 x 9 | Resource one-hot 6; dice probability; high-number flag; live robber flag |
| vertices | 54 x 16 | Relative owner one-hot 5; building one-hot 3; port kind one-hot 7; setup anchor flag |
| edges | 72 x 5 | Relative owner one-hot 5 |
| my_hand | 5 | Exact own resource counts / 19 |
| opponent_hand_sizes | 3 | Clockwise opponent totals / 95, with zero padding |
| players | 4 x 9 | Present; own total or opponent public VP / 10; settlements / 5, cities / 4, roads / 15 remaining; pending discard / 47; card total / 25; knights / 14; longest trail / 15 |
| bank | 5 | Public remaining bank resources / 19 |
| phase | 9 | One-hot in existing `Phase.value - 1` order |
| last_roll | 12 | One-hot: None, then totals 2 through 12 |
| turn_owner | 4 | Current player's relative seat one-hot |
| acting_player | 4 | Decision maker's relative seat one-hot; all zero at GAME_OVER |
| setup_progress | 1 | setup_index / (2 * number of players) |
| my_development / my_development_new | 5 each | Own held and freshly bought counts / type supply (14,5,2,2,2) |
| development_deck_size | 1 | Remaining total / 25; composition hidden |
| development_played | 5 | Public spent cards / type supply; VP cards remain held |
| longest_road_owner / largest_army_owner | 5 each | Relative holder or unheld |
| development_used | 1 | One non-VP card already played this turn |
| free_roads_remaining | 1 | Remaining free choices / 2 |
| resume_phase | 3 | None, ROLL, MAIN continuation |

Resources keep wood, sheep, wheat, brick, ore, desert ordering. Port kinds are
wood through ore, generic, none. Buildings keep empty, settlement, city ordering.
Owner channels are me, next clockwise seat, next, next, unowned. In a three-seat
game the fourth owner channel is unused, the last player row is zero, and the
third opponent total is zero. The present flag distinguishes absent padding from
an actual seat with an empty hand. Public port locations/types are encoded even
when unowned; building ownership determines access without a mutable port cache.
The setup anchor marks the settlement that the pending setup road must touch.

Dice probability is pips / 36 (zero for desert). Probability alone loses number
identity: 6 and 8 have equal probability but produce on different rolls. One
`number > 7` flag distinguishes each mirrored pair while preserving all token
numbers without eleven extra one-hot channels. The last-roll total is encoded
separately. All tile/vertex/edge IDs and static adjacency remain unchanged in
`GEOMETRY`; the flat baseline uses stable positions without duplicating incidence
matrices in every observation. A future GNN can use those existing static tables.

Counts are scaled by physical maxima. Values are not clipped: a legitimate
winning city upgrade can give 11 VP, represented as 1.1. Player count is inferred
from present flags. Relative seat remapping applies consistently to owners,
hands, pieces, VP, pending choices, actor and turn owner. Absolute seat numbers
are absent. Cyclic seat relabeling yields identical encodings; this does not
implement board rotations/reflections or change the engine's fixed setup order.

Only the observer's exact hand is included. Opponents expose totals, never their
resource compositions; bank, board, pieces, public scores and pending counts are exposed. Opponent hidden
VP cards are subtracted from total scores before encoding; own total is visible.
Terminal observations reveal all scores. Opponent card composition/freshness and
deck composition are hidden; only card totals, played knights and public awards
are revealed. This is a policy choice for the observation contract, not full public
history or a belief-state implementation.
Bank plus own hand can reveal aggregate opponent resources by conservation, which
is inherent in those public features. Game/policy RNG internals are excluded.
The observation is therefore partially observed, not a complete Markov state
with hidden opponent hands. Swapping opponents' compositions while keeping public
facts and their totals fixed leaves the observer's vector unchanged.

## Batch API

`encode_batch(states, encoders, players=None, out=None)` requires a prepared encoder
per row and optionally an explicit observer per row. It supports mixed boards and
seat counts. All lengths/contexts/observers/output aliases are validated before
writing. Outputs have shape `(batch, OBS_DIM)`. Empty batches are supported.
Batches below eight use the scalar path; larger batches gather board channels in
chunks of at most 128 and share the global-feature writer. No shared state, global
board cache or persistent scratch buffer is added. Consume/copy buffers before
reuse. Mask/catalog contracts are in the expanded-engine document.

## Historical v1 optimization and cleanup

The measurements below describe the 1461-value / 5844-byte version-one milestone.
Current v2 has 1505 values / 6020 bytes; its measurements are recorded separately
in the expanded-engine review. Current globals are written in one operation and
road query ownership snapshots are transient Python lists. Historical rates are
not current expanded-rule throughput.

Preparation computes static tile/port features once. Read-only byte-backed NumPy
lookups remap owner/building categories; one template copy initializes each output.
There is no per-call board hash, port search, geometry rebuild, full state copy,
full invariant check, or mutable state/legality/port cache. Engine imports do not
load this module or any learning framework.

Profiling identified tiny NumPy reductions/ufuncs in the player/resource groups.
The final version converts these small groups to Python values, uses precomputed
float32 scale constants, and writes the 37 count features together. Vectorized
category gathers remain for the larger board arrays. Default actor indices avoid
redundant integer coercion; explicit user indices retain validation. Unneeded
import-time mutable aliases and intermediate table allocations were removed.

Five warmed alternating paired batches on 81 reached three/four-seat states,
covering all eight phases, compared the initial encoder with the final version.
Each batch performed 20000 reused-output calls; no competing stress/test jobs ran.
Docker Linux / Python 3.14.7 / NumPy 2.4.6 measured **88.75 -> 55.24 us**, a
**1.61x** speedup (about 38% less encoding time). Outputs matched exactly for every
observer in the pool. `output/encoding/optimization.json` retains the paired data.

An isolated final benchmark with five alternating batches of 12000 calls measured:

| Operation | Median |
| --- | ---: |
| Prepare one board | 44.61 us |
| One-off encoding with new preparation and output | 108.97 us |
| Prepared encoding with independent output | 46.59 us |
| Prepared encoding with checked reusable output | 56.32 us |

Prepared allocation was 2.34x faster than rebuilding preparation every call.
Buffer reuse avoids retaining new output allocations, but its eight state-array
alias checks make it slower in this small workload; it is an ownership/allocation
option, not a claimed latency win. Each observation and each prepared template
occupies **5844 payload bytes**, excluding headers/board objects. A 2120-byte
compact-template experiment reduced cache memory but increased latency (56.75
vs 54.34 us for the full template in the same alternative comparison); it was
removed. Throwaway snapshots/probes are removed after review; reports remain in
`output/encoding`. Empty future RL scaffolds carried no rollout work at this milestone; the later
cleanup removes them to make implemented scope clear.

```powershell
docker build -t catan-checks .
docker run --rm catan-checks python -m scripts.benchmark_encoding
# Optional: --iterations 12000 --repeats 5 --report /reports/final.json
# Mount a writable output directory at /reports to retain a report.
docker run --rm catan-checks python -m pytest tests/test_encoding.py -q
```

Pool creation and preflight checks are outside measured calls. These numbers
measure encoding, excluding game steps, inference, training, rendering and imports;
they are not RL training throughput. Run timing batches without competing jobs.

## Verification and next boundary

The encoding tests use an independent scalar oracle, completed seeded games in
both seat counts/all phases, varied reached states and every observer/seat rotation.
They check hidden-hand/RNG boundaries, manual-discard perspective, terminal/value
queries, all port locations, number identity, normalization, cleared reused flags,
immutable preparation, independent outputs and rejected aliases for all fourteen
state arrays. Existing rule/copy/seeded geometry fingerprints remain applicable.

The historical v1 gate passed **599 Python tests**, including 33 new encoding cases.
Both container images build, all 21 JavaScript checks pass, and Ruff lint,
38-file formatting, compilation, dependency consistency, guide-generator F-check,
whitespace and eight impacted documentation examples pass.
The extended check exercised **1000 distinct seeded games** (500 per seat count),
with a 2000-action cap and eight bounded workers. It encoded **1308055 states** /
1307055 actions, checked full invariants on every state and covered all eight
phases. All vectors had the required shape/dtype and finite nonnegative features.
On 17583 sampled states (every 127th step, first phase visits and terminal states),
outputs matched the independent scalar oracle and matched the initial encoder
exactly; source fields/array identities/full RNG were preserved. There were no
failures. The run recorded 832 true wins and 168 caps (three seats: 435/65; four
seats: 397/103). Caps remain truncations rather than encoding failures or invented
wins. `output/encoding/stress.json` retains counts and its ordered fingerprint;
this smaller-cap representation check supplements the earlier 10000-game Stage 5
gate. Stress elapsed time includes concurrent regression checks and is not a
performance benchmark. Temporary comparison/stress scripts were removed after
review; the maintained oracle tests and bounded encoding benchmark remain.

The next guide section covers graph neural network design (Section 8), with the
flat MLP as the pipeline baseline. Action encoding and legal masks are now implemented separately; rewards and
environment/training integration remain deferred. Version schema changes, prepare once per reset and keep debug validation
outside rollout calls. The eventual environment must still distinguish real wins
from capped truncations in the minimal rules.
