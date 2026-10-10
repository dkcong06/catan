# Catan

Phases 1-3 of `Catan_RL_Guide.pdf` are complete: package/container setup, board
geometry, legal seeded layouts with ports, and compact, reproducible game state
with independent copies.
Phase 4 has started with player-selected discards after a seven, including browser
resource selectors. Full turns, dice production, robber movement/stealing, and RL
remain later increments. The [discard rule contract](docs/manual-discards.md)
documents this implementation. The
[Phase 3 plan and verification](docs/phase-three-plan.md) document the state contract.

The board generator and board-preview routes use only Python's standard library. Game state
uses pinned NumPy 2.4.6; pytest/Ruff are development tools. PyTorch is deferred
until learning code requires it. Dependencies are installed in Docker.

## Checks in Docker

```powershell
docker build -t catan-checks .
docker run --rm catan-checks
docker run --rm catan-checks python -m ruff check catan rl scripts tests
docker run --rm catan-checks python -m ruff format --check catan rl scripts tests
docker build --target viewer-checks -t catan-viewer-checks .
docker run --rm catan-viewer-checks
```

The image uses Python 3.14. Tests include 10,000 seeded boards, exhaustive legal
red-token-set validation for all 19 desert positions, reciprocal geometry checks,
coastal port placement, reproducibility, immutability, and viewer HTTP routes.
State checks cover three/four players, every initial quantity, independent copies,
mixed RNG branches, nested deepcopy aliases, conservation, and invalid-state diagnostics.
The Stage 1-3 audit recorded 242 Python tests and nine JavaScript behavior tests.
With exclusive-owner transitions added, the suite passes 415 Python tests and 21
JavaScript tests (2026-10-10), including rule/API transitions, selector behavior,
and functional/in-place equivalence.
The separate Node test target uses built-in tools and adds no Python runtime or
host dependencies. Browser checks cover seed generation and the responsive layout.

For extended stress checks and the optimization harness:

```powershell
docker run --rm catan-checks python -m scripts.stress_foundations
docker run --rm catan-checks python -m scripts.benchmark_foundations
```

The stress harness checks 100,000 boards and 5,000 states with two independent
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

Port locations stay fixed while their types are shuffled: four generic 3:1 ports
and one 2:1 port per resource. `PortType.resource` is `None` for generic ports;
specific types share their offered resource's integer index. These are setup
data; trade actions become part of the rules engine later. Setup constraints and
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
It checks physical/bookkeeping invariants, including pending discards. Phase 4's
`apply_action` currently handles `DiscardAction`; other actions and `legal_actions`
remain to be implemented.

```powershell
docker run --rm catan-checks python -m scripts.benchmark_state
```

Measured four-player medians: 13.88 microseconds to initialize from supplied
board/RNG, 53.86 to create a full seeded game, and 38.98 to copy including RNG.
Those timings were measured before the discard increment. Mutable array data now
occupies 258 bytes (241 for three players), including pending counts and excluding
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
`http://127.0.0.1:8000/discards`. Click **Resolve a 7**, select resource counts,
then **Confirm discard** for each pending player. The page uses sample hands and
the real engine transitions; it is a rule demo while complete gameplay is being
built. This feature needs the Docker image's NumPy dependency. See the
[discard documentation](docs/manual-discards.md) for a runnable engine example,
API behavior, validation, and remaining integration work.

## Engine transitions for owned rollout states

`resolve_seven_inplace(state)` and `apply_action_inplace(state, action)` update a
valid state exclusively owned by the caller and return the same object. They
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

Run in Docker, then open `http://127.0.0.1:8000`:

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

## Build guide

`Catan_RL_Guide.pdf` contains the roadmap. `generate_guide.py` and `assets/` are
its documentation sources, separate from the game runtime. Regenerating the
guide requires ReportLab and Matplotlib; they are not game dependencies.

## Setup efficiency (2026-10-09)

At import, cache every legal set of four mutually nonadjacent tiles, grouped by
desert position. Prune illegal prefixes during cache construction while preserving
lexicographic order, so existing seeded layouts and RNG continuation are unchanged.
Each board samples one set uniformly for its desert, shuffles
the two 6s and two 8s onto it, then shuffles the remaining tokens and ports.
This has bounded per-board work and avoids rejection loops or biased repair
swaps. The shared tuple cache occupies about 103 KB on the measured runtime.

```powershell
docker run --rm catan-checks python -m scripts.benchmark_board
```

On Docker Linux / Python 3.14.7, seven paired batches of 10,000 boards measured
median **23.93 microseconds/board** (41,784 boards/s) for the direct sampler,
versus **99.32 microseconds/board** for full-token rejection sampling: **4.15x**
faster. Both construct resources, legal numbers, and ports with a shared seeded
RNG. The benchmark warms up both samplers and alternates measurement order.
Timings depend on machine load and exclude imports, game state, and gameplay;
they do not measure future engine or RL throughput.
