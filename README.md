# Catan

Phases 1-3 of `Catan_RL_Guide.pdf` are complete: package/container setup, board
geometry, legal seeded layouts with ports, and compact, reproducible game state
with independent copies.
Phase 4 now provides a playable minimal engine: setup, seeded dice/production,
manual discards, robber movement/stealing, building, bank/port trades, end turns,
and ten-point victory. The [Stage 4 contract](docs/stage-four-engine.md) covers
scope, ownership, lazy legal choices, random-game validation, and capped stalemates.
The [Stages 1-4 audit](docs/stages-1-4-audit.md) records gameplay optimizations,
cleanup and seeded compatibility.
Stage 5 adds [continuous invariant and random-game validation](docs/stage-five-validation.md),
with deterministic replay reports, bounded parallel workers and explicit win/cap counts.
Guide Sections 6-7 now provide [relative flat observations](docs/representation-encoding.md):
fixed float32 features for three/four seats, hidden opponent hands, and prepared
immutable board lookups with optional output buffers. The browser remains a board/discard demo. The optional model, collector and PPO updater are implemented; long training experiments remain future work.
The [discard rule contract](docs/manual-discards.md) documents manual choices. The
[Phase 3 plan and verification](docs/phase-three-plan.md) document the state contract.

The board generator and board-preview routes use only Python's standard library.
Game state and observation encoding use pinned NumPy 2.4.6; pytest/Ruff are development tools. PyTorch is available separately in the optional learning image. Dependencies are installed in Docker.


## PPO updates and checkpoints (2026-10-13)

The [PPO contract](docs/ppo-updates.md) describes `learning.ppo.PPOConfig` and
`PPOUpdater`: fresh stochastic rollout checks, fixed same-seat GAE, globally
normalized advantages, clipped joint-action/critic losses, Adam, finite/clipped
gradients, policy-drift early stopping and atomic resumable local checkpoints.
Recorded discard reevaluation combines all five known prefixes into one call;
actual discard sampling remains sequential. The engine and every `rl` module
remain NumPy-only. Only bounded update-mechanics tests ran, not a long learning
experiment or a playing-strength comparison.

```powershell
docker build --target learning-checks -t catan-learning-checks .
docker run --rm catan-learning-checks python -m scripts.benchmark_ppo --threads 1
docker run --rm catan-learning-checks python -m scripts.validate_ppo --seeds 12 --workers 4
```

The inspected laptop reports AMD Radeon integrated graphics; the verified image
is CPU-only. Benchmark a compatible dedicated GPU when repeated model arithmetic
becomes a large share of total runtime. GPU execution does not accelerate the
Python/NumPy simulator, and CUDA correctness/performance remains unverified.

This increment passed 911 Python tests, 823 core-only tests with two intentional
optional-module skips, and 21 JavaScript tests. The144-game audit checked 93272 actions
and 72 updates, each repeated exactly after checkpoint restoration. Fused recorded-
discard reevaluation reduced paired update time 16.6%; fused CPU Adam reduced it5.8%.
Four update threads were28.1% faster than one in a separate same-weight comparison.
A full rollout tensor cache saved only 2.9% while duplicating45MB and was removed.
The contract records the actual workloads, limits and next training boundary.
## Agent environment and evaluation (2026-10-12)

The [agent interface contract](docs/agent-environment-evaluation.md) covers
`rl.env.CatanEnv`, per-seat outcome rewards, observation-only reference policies
and reproducible seat-rotated evaluations. Training remains on hold. The environment
uses owned in-place engine steps, explicit acting seats and complete manual discard
vectors; real wins and external action-limit cutoffs are reported separately.

```python
from random import Random
from rl.env import CatanEnv
from rl.policies import random_policy

env = CatanEnv(max_actions=5000)
request = env.reset(42, n_players=4)
policy_rngs = [Random(100 + player) for player in range(4)]
while request is not None:
    result = env.step(random_policy(request, policy_rngs[request.player]),
                      player=request.player)
    request = env.decision()
assert result.terminated != result.truncated
print(result.winner, result.rewards, result.actions)
```

```powershell
docker run --rm catan-checks python -m scripts.evaluate_policies --seeds 10 --players 4
docker run --rm catan-checks python -m scripts.evaluate_policies --seeds 10 --players 3 --validate
docker run --rm catan-checks python -m scripts.benchmark_environment
```

Reference policy versions, schemas/seeds/caps, physical match/action counts,
seat-specific wins/losses/cutoffs and fingerprints accompany evaluation reports.
Same-policy controls reuse identical finished trajectories across role labels;
different-policy rotations are played individually. Reports and all measured
retained/rejected implementation comparisons are in `output/agent-interface`.
Per-seat collection is now implemented; see the collector section below.
The model/target layer is now implemented; training updates remain later work.
The browser remains the board/discard demo.

## Per-seat rollout collector (2026-10-13)

The [collector contract and comparison report](docs/rollout-collector.md) covers
`rl.collector.collect_rollouts`, batched observation-only callbacks, complete
per-seat trajectories, sparse manual discards and optional numeric NPZ archives.
The collector records actual behavior probabilities and optional critic estimates,
links every record to that seat's next view, and attributes true outcomes to each
seat's final record. External cutoffs keep permitted final observations and zero
rewards. Missing critic estimates are NaN. Training remains held.

```python
import numpy as np
from rl.collector import ReferencePolicy, collect_rollouts

batch = collect_rollouts(ReferencePolicy("random"), (40, 41),
                         n_players=3, batch_size=2, max_actions=5000)
assert len(batch) == sum(episode.actions for episode in batch.episodes)
rows = np.flatnonzero(batch.players == 0)[:32]
next_views = batch.successor_observations(rows)
assert next_views.shape == (len(rows), 1505)
```

```powershell
docker run --rm catan-checks python -m scripts.collect_rollouts --games 4 --validate
docker run --rm catan-checks python -m scripts.benchmark_collector
docker run --rm catan-checks python -m scripts.validate_collector --seeds 50 --workers 8
```

Columns are contiguous and independently owned. Growing private storage reduces
peak memory compared with list/stack or reserving every capped row. Same-seat
links avoid storing every successor observation twice. Boolean masks/float32
observations preserve existing schemas; no ML dependency was added. Synchronous
batching enables a future batched model callback; it does not parallelize Python.
Data memory is bounded by the caller's finite seeds/action caps; use repeated small
batches for large runs. Optional --output .npz/--report JSON files need a writable
output bind mount to persist outside Docker. See the contract for examples and
return/discount/cutoff boundaries.


## Policy, critic and learning targets (2026-10-13)

The [model and target contract](docs/policy-value-targets.md) covers the optional
`learning` package and NumPy-only `rl.returns`. A two-layer MLP scores legal flat
moves and estimates future reward; a shared prefix-conditioned head selects a
complete discard. The collector records real probabilities/value predictions.
Same-seat generalized advantage estimates preserve terminal/cutoff distinctions
and require explicit cutoff predictions. The next [PPO update stage](docs/ppo-updates.md) now changes the model in bounded mechanics checks; this dated model-stage report describes its earlier no-update verification.

```powershell
docker build --target learning-checks -t catan-learning-checks .
docker run --rm catan-learning-checks python -m scripts.collect_neural --games 4 --max-actions 500 --validate
docker run --rm catan-learning-checks python -m scripts.benchmark_learning --threads 1
docker run --rm catan-learning-checks python -m scripts.validate_learning --seeds 25 --workers 4
```

PyTorch2.14.1+cpu is installed only in the optional image; default engine/collector
checks remain NumPy-only. Source edits reuse dependency layers. Width128 is a
configurable baseline; no learned-strength comparison has been made. In the tested
6358-action workload, neural batching improved throughput about58%, while retained
target methods were1.5-4.9x faster than the first scalar implementation. These are
CPU execution measurements, not training-speed or game-strength guarantees.

```python
import numpy as np
from rl.collector import ReferencePolicy, collect_rollouts
from rl.returns import compute_targets

batch = collect_rollouts(ReferencePolicy(), (42,), max_actions=20)
# A future model supplies actual predictions; these explicit zeros are only a
# calculation example, not a collector/trainer default for missing predictions.
targets = compute_targets(batch, values=np.zeros(len(batch), dtype=np.float32),
                          bootstrap_values=np.zeros((1, 4), dtype=np.float32))
assert targets.advantages.shape == (20,)
```

The learning image passed 842 Python tests; the lightweight image passed 823 and
intentionally skipped the optional model test module. The124-episode neural audit
checked 92930 actions with independent engine replay, probabilities/values, cutoff
bootstraps and finite gradients while every model weight remained unchanged.
See the contract for exact formulas, ownership, benchmarks, CLI archive outputs
and the next PPO update boundary.

## Development cards, awards and lookup improvements (2026-10-11)

All five base development card types, buying, timing restrictions, Longest Road,
Largest Army and their scoring are implemented. Knights and other playable cards
can be used before rolling. Fresh cards cannot be played that turn; fresh VP can
win immediately. Road Building asks for legal free roads sequentially. Cards and
awards participate in copying, equality, conservation, debug checks and observations.
The 2026-10-11 milestone passed 664 Python tests and 21 JavaScript tests; 10000 expanded-rule
games complete with 9491529 checked actions and zero invariant failures.
See [the current expanded-engine contract](docs/expanded-engine.md) for details,
measured retained/rejected improvements and the latest verification results.

Observation schema v2 has 1505 float32 values (6020 payload bytes). It hides
opponents' exact development cards and private VP, exposes public card totals,
and adds own cards, timing, award holders, road lengths and played knights.
`encode_batch` handles mixed boards and three/four seats in bounded chunks;
use the normal prepared encoder for a single observation.

```python
from rl.actions import ACTION_DIM, action_mask, indexed_action
from rl.encoding import FlatEncoder, encode_batch
from catan.state import new_game

states = [new_game(seed, 3 + seed % 2) for seed in range(8)]
encoders = [FlatEncoder(state.board) for state in states]
observations = encode_batch(states, encoders)
assert observations.shape == (8, 1505)
mask = action_mask(states[0])
assert mask.shape == (ACTION_DIM,) and ACTION_DIM == 254
choice = indexed_action(states[0], int(mask.nonzero()[0][0]))
```

Flat masks agree with engine legal actions and use the same rule predicates.
DISCARD uses `discard_count_mask(state, prefix)` for five sequential resource
counts, avoiding a combinatorial flat space; submit the completed `DiscardAction`.
Terminal masks are empty by design. These utilities do not implement a trainer,
rewards, network, negotiation or full-game browser controls.

```powershell
docker run --rm catan-checks python -m scripts.benchmark_system
docker run --rm catan-checks python -m scripts.benchmark_lookup
```

These measure games/queries/encoding, excluding inference and training. Capped
games remain truncations. Empty unimplemented RL module files have been removed;
future components will be added when their behavior is implemented.

## Checks in Docker

```powershell
docker build -t catan-checks .
docker run --rm catan-checks
docker run --rm catan-checks python -m ruff check catan rl scripts tests
docker run --rm catan-checks python -m ruff format --check catan rl scripts tests
docker build --target viewer-checks -t catan-viewer-checks .
docker run --rm catan-viewer-checks
```

The image uses Python 3.14. Tests include 10, 000 seeded boards, exhaustive legal
red-token-set validation for all 19 desert positions, reciprocal geometry checks,
coastal port placement, reproducibility, immutability, and viewer HTTP routes.
State checks cover three/four players, every initial quantity, independent copies,
mixed RNG branches, nested deepcopy aliases, conservation, and invalid-state diagnostics.
The Stage 1-3 audit recorded 242 Python tests and nine JavaScript behavior tests.
The Stages 1-4 audit passed 533 Python tests and 21 JavaScript tests
(2026-10-10), including setup, all phases, production/shortages, connectivity,
port trades, full-game replay, selector behavior, and functional/in-place equivalence.
The Stage 5 gate passed 566 Python tests and the same 21 JavaScript tests;
its separate continuous validation run is documented below. With observation
encoding, the historical gate passed 599 Python tests (33 new encoding cases),
plus a 1000-game/1308055-observation check with no failures.
The separate Node test target uses built-in tools and adds no Python runtime or
host dependencies. Browser checks cover seed generation and the responsive layout.

For extended stress checks and the optimization harness:

```powershell
docker run --rm catan-checks python -m scripts.stress_foundations
docker run --rm catan-checks python -m scripts.benchmark_foundations
```

The stress harness checks 100, 000 boards and 5, 000 states with two independent
branches each. See the [Stage 1-3 audit](docs/stages-1-3-audit.md) for scope,
paired measurements, compatibility evidence, and remaining implementation boundaries.

## Generate a board

```python
from random import Random
from catan.board import GEOMETRY, make_board

rng = Random(42)
board = make_board(rng)
print(board.resources, board.numbers, board.robber_tile)
for port_id, port in enumerate(board.ports):
    print(port.name, port.ratio, GEOMETRY.port_vertices[port_id])
```

Pass the same RNG onward to future game state so dice and steals continue the
seeded stream. Identical seeds replay a layout with the same implementation and
Python version; reusing an RNG advances it. Seeds now include constrained tokens
and ports, so they produce different layouts from earlier unconstrained previews.

`Resource` indices 0-4 are wood, sheep, wheat, brick, ore; 5 is desert.
`board.numbers` uses 0 only on the desert, where the robber starts. No pair of
edge-adjacent tiles may both carry 6 or 8 (including 6/6 and 8/8).

All geometry is generated once as shared immutable tuples. The original location
IDs are preserved. Tables include `tile_neighbors`, `tile_vertices`,
`vertex_tiles`, `vertex_neighbors`, `vertex_edges`, `edge_vertices`, and 2D
tile/vertex/edge positions. `coast_edges` walks the 30-edge boundary in a stable
order. Nine `port_edges` have repeating cyclic gaps of 3, 3, and 4 coastal edges;
`port_vertices` maps each port to its two land vertices, and `vertex_ports` maps
vertices back to a port ID or -1. `port_positions` places labels outside the land.

Port locations stay fixed while their types are shuffled: four generic 3: 1 ports
and one 2: 1 port per resource. `PortType.resource` is `None` for generic ports;
specific types share their offered resource's integer index. These are setup
data; Stage 4 trades honor the best owned port ratio. Setup constraints and
port composition follow the requested rules and the
[official CATAN rules and almanac (2020)](https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf).

## Game state (Phase 3)

```python
from catan.state import new_game

state = new_game(seed=42, n_players=4)
state.validate()
branch = state.copy()
assert branch == state
assert branch.board is state.board
print(state.phase.name, state.setup_order, state.array_bytes)
print(state.port_access(0))  # Empty until player 0 owns a coastal building.
```

`GameState` holds resource hands, pending discard counts, VP, remaining settlement/city/road pieces,
vertex buildings/owners, road owners, bank supply, live robber location, turn
phase, last roll, and setup bookkeeping. New games have no placed pieces or
resources in hand; the bank has 19 of each resource. Three or four players are
supported. Ownership -1 means unowned; buildings are 0/1/2 for empty/settlement/city.

The same RNG first generates the board, then carries future dice/steal randomness.
`state.copy()`, `copy.copy(state)`, and `copy.deepcopy(state)` independently copy
mutable arrays and the complete RNG state, while sharing immutable board data.
Equality includes RNG state, bookkeeping, and array values/dtypes. `validate()`
provides explicit debug checks and is not called automatically on every copy.
Small integer arrays use shape/dtype-checked byte comparisons; noninteger debug
arrays retain NumPy's equality semantics. Port access checks the two endpoints
directly without generator allocations or a mutable cache.
It checks physical/bookkeeping invariants, including pending discards. Phase 4
implements `legal_actions`, both action-transition APIs, and terminal/winner queries.
Legal choices are yielded lazily; consume them before mutating the queried state.

```powershell
docker run --rm catan-checks python -m scripts.benchmark_state
```

Measured four-player medians: 13.88 microseconds to initialize from supplied
board/RNG, 53.86 to create a full seeded game, and 38.98 to copy including RNG.
Those timings were measured before the discard increment. The current expanded mutable arrays occupy 316 bytes (287 for three players), including cards, awards and pending counts and excluding
object/RNG overhead. Historical measurement details are in the Phase 3 document.

## Manual discards after a seven

`resolve_seven(state)` handles an already-known seven from `Phase.ROLL`. Each
player over seven cards chooses half, rounded down, through `DiscardAction(player,
resources)`, with resource counts ordered wood, sheep, wheat, brick, ore.
`apply_action` returns an independent state and returns those cards to the bank.
Exactly seven cards require no discard. Choices run clockwise from the roller;
`state.discard_player` identifies the decision maker while `current_player` keeps
the turn owner. Only after all choices does the engine enter `ROBBER_MOVE`.

Open **Try manual discards after a seven** in the board viewer, or visit
`http://127.0.0.1: 8000/discards`. Click **Resolve a 7**, select resource counts,
then **Confirm discard** for each pending player. The page uses sample hands and
the real engine transitions; full engine gameplay is described below. This page is
a discard-only demo. This feature needs the Docker image's NumPy dependency. See the
[discard documentation](docs/manual-discards.md) for a runnable engine example,
API behavior, validation, and remaining integration work.

## Play the engine

```python
from catan.engine import winner
from catan.players.random_agent import play_random_game

state = play_random_game(0, n_players=4, max_actions=10_000, validate=True)
print(winner(state), state.vp.tolist())
```

The explicit random agent uses a policy RNG separate from game randomness.
Manual users still choose discard resource counts themselves; the engine never
selects those cards. `play_random_game` raises at its action cap rather than
inventing a winner. The historical minimal rules could stalemate when expansion was exhausted;
the agent environment now distinguishes episode truncation from a real win.

```powershell
docker run --rm catan-checks python -m scripts.benchmark_games
docker run --rm catan-checks python -m scripts.benchmark_games --players 3
```

The gameplay benchmark compares both APIs on the same seeded random-policy games,
reports completed/capped episodes, and includes reset, legal queries, and policy
selection. Neural inference/training and rendering are excluded. Development
cards and both awards are now implemented; negotiated trades remain deferred.

## Continuous validation (Stage 5)

```powershell
docker run --rm catan-checks python -m scripts.validate_games --games 10000 --players both --workers 8
```

This checks the initial state and every action with `assert_invariants`, including
resource/piece/VP/distance bookkeeping, turn phases and rooted road components.
`--games` counts total episodes; `both` alternates three/four seats. Each worker
owns its state/RNG. Wins and capped truncations are reported separately, with
reproducible seed/player/action coordinates on failure. Checks remain opt-in;
normal RL rollout steps retain their existing fast lookup paths.
The Stage 5 run checked 10000 episodes and 17112427 actions with zero failures:
9630 actual wins and 370 caps. All eight phases were visited. The minimal rules
still do not guarantee that every game reaches a win.
See [Stage 5 validation](docs/stage-five-validation.md) for JSON reports, replay,
functional-mode checks, profiling, results and the minimal-game termination boundary.

## Relative observations (Sections 6-7)

```python
from catan.state import new_game
from rl.encoding import OBS_DIM, FlatEncoder

state = new_game(42, n_players=4)
encoder = FlatEncoder(state.board)  # Reuse until the board/reset changes.
observation = encoder.encode(state)
assert observation.shape == (OBS_DIM,) and OBS_DIM == 1505
```

The default perspective follows the decision maker, including manual discarders,
while exposing only that seat's exact resource hand. Other seats are relative
and reveal hand totals. Features include public board/ports, robber, ownership,
pieces/VP, bank, phase, roll, turn and setup bookkeeping. Three-player games use
zero padding with explicit seat presence. `out=` accepts disjoint reusable float32
storage; consume or copy it before reuse. Encoding does not copy or mutate game
state/RNG, run invariants, or import PyTorch. Prepare a new encoder after reset.

```powershell
docker run --rm catan-checks python -m scripts.benchmark_encoding
```

The paired optimization reduced reused-output encoding from 88.75 to 55.24 us
(1.61x), with exact float32 agreement. See the [representation contract](docs/representation-encoding.md)
for schema, ownership, visibility, measurements and verification. This measures
encoding, not training throughput.

## Engine transitions for owned rollout states

`resolve_seven_inplace(state)` and `apply_action_inplace(state, action)` update a
valid state exclusively owned by the caller and return the same object. The latter supports all expanded actions, including seeded dice, purchases and stealing. They
retain all array and RNG identities; the functional APIs continue to return
independent successor states. Both paths share rule validation and mutation
logic, including manual resource choices. The browser retains functional calls.
Hand totals reuse the pending-count array, and each discard converts its counts
once for both hand and bank updates. Full validation stays outside the hot path.

```powershell
docker run --rm catan-checks python -m scripts.benchmark_engine
```

This compares warmed, alternating batches using identical prebuilt fixtures.
Pool creation, resets, debug validation, and inference are excluded; the results
measure seven/discard transitions rather than complete-game or RL throughput.
See [engine optimization](docs/engine-optimization.md) for the ownership contract,
example, measured results, and remaining training work.

## Board viewer

Run in Docker, then open `http://127.0.0.1: 8000`:

```powershell
docker run --rm -p 127.0.0.1:8000:8000 catan-checks python -m scripts.board_viewer --host 0.0.0.0 --no-browser
```

With an existing Python installation, no third-party packages are needed:

```powershell
python -m scripts.board_viewer
```

The local Python command opens a browser automatically. Press Ctrl+C to stop it.
The viewer displays resources, tokens, robber, vertices, edges, and ports.
Enter a seed (0-4294967295) or use **Random seed** to generate another legal board.
`GET /api/board?seed=42` returns a reproducible JSON preview; seed 0 is cached,
other seeds are generated on demand without an accumulating per-seed cache.
Static geometry JSON is encoded once and combined with each new layout, keeping
the same flat API response. Valid numeric input such as `1e3` is normalized to
seed 1000; obsolete responses cannot replace a more recently requested board.
The original `/api/tiles` endpoint remains available. HTML is read on each request
so edits appear on refresh. `--host`, `--port`, and `--no-browser` configure startup.

## Detailed project decision report

`docs/project-decisions.md` explains 71 material choices and their rationale.
`docs/build_project_report.py` creates the separate detailed PDF under `output/pdf`,
including current data contracts, measured results and full historical records.
The generator uses the supplied document runtime with ReportLab/pypdf and retained
schema/result JSON in `output/full-review`; it is outside application dependencies
and Docker runtime imports. The original guide remains unchanged.

## Build guide

`Catan_RL_Guide.pdf` contains the roadmap. `generate_guide.py` and `assets/` are
its documentation sources, separate from the game runtime. Regenerating the
guide requires ReportLab and Matplotlib; they are not game dependencies.

## Setup efficiency (2026-10-09)

At import, cache every legal set of four mutually nonadjacent tiles, grouped by
desert position. Prune illegal prefixes during cache construction while preserving
lexicographic order, so existing seeded layouts and RNG continuation are unchanged.
Each board samples one set uniformly for its desert, shuffles
the two 6 s and two 8 s onto it, then shuffles the remaining tokens and ports.
This has bounded per-board work and avoids rejection loops or biased repair
swaps. The shared tuple cache occupies about 103 KB on the measured runtime.

```powershell
docker run --rm catan-checks python -m scripts.benchmark_board
```

On Docker Linux / Python 3.14.7, seven paired batches of 10, 000 boards measured
median **23.93 microseconds/board** (41, 784 boards/s) for the direct sampler,
versus **99.32 microseconds/board** for full-token rejection sampling: **4.15x**
faster. Both construct resources, legal numbers, and ports with a shared seeded
RNG. The benchmark warms up both samplers and alternates measurement order.
Timings depend on machine load and exclude imports, game state, and gameplay;
they do not measure future engine or RL throughput.
