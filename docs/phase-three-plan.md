# Phase 3: modeling game state

This plan paraphrases Section 3 of `Catan_RL_Guide.pdf` (pages 8-9), with
implementation details chosen for this repository. Phase 2 is the immutable
board foundation; Phase 3 introduces mutable state. Gameplay rules remain Phase 4.

Status: **implemented and verified**. `catan/state.py` contains the state layer;
`tests/test_state.py` checks initialization, copying, RNG branches, and invariants.

## First implementation increment

The implementation provides `Phase`, a slotted `GameState`, and
`new_game(seed, n_players=4)`. It supports three or four players and rejects
other counts and noninteger seeds. Integer-like NumPy inputs are accepted;
`seed=None` selects system entropy. One `random.Random(seed)` generates the
board and remains on the state, advanced to the point needed for future dice
and steals. The state layer imports no ML packages.

Small, contiguous NumPy arrays hold mutable quantities. NumPy 2.4.6 is pinned in
`requirements.txt` and installed/verified in the Python 3.14 container; its
[release notes](https://numpy.org/doc/stable/release/2.4.6-notes.html) confirm
Python 3.11-3.14 support. Immutable `Board` and `GEOMETRY` objects are shared
across copies, including ordinary Python deep copies, without rebuilding them.

| Field | Representation | New-game value |
| --- | --- | --- |
| `hands` | `int16[n_players, 5]` | all zero |
| `vp` | `int16[n_players]` | all zero |
| `pieces_left` | `int8[n_players, 3]`, settlement/city/road columns | `[5, 4, 15]` per seat |
| `vertex_owner` | `int8[54]` | `-1` (unowned) |
| `vertex_building` | `int8[54]` | `0` (empty); later 1/2 for settlement/city |
| `edge_owner` | `int8[72]` | `-1` (unowned) |
| `bank` | `int16[5]` | 19 of each resource |
| `discard_pending` (added in Phase 4) | `int16[n_players]` | all zero |
| `robber_tile` | integer tile ID | `board.robber_tile` |
| `current_player` | integer seat | 0 |
| `phase` | `Phase` enum | `SETUP_SETTLEMENT` |
| `last_roll` | integer or `None` | `None` |
| `setup_index` | integer | 0, indexing the derived snake order |
| `setup_vertex` | integer vertex ID or `None` | `None`; connects each setup road |
| `rng` | `random.Random` | advanced past board generation |

The phases are `SETUP_SETTLEMENT`, `SETUP_ROAD`, `ROLL`, `ROBBER_MOVE`,
`ROBBER_STEAL`, `MAIN`, and `GAME_OVER`. On 2026-10-10, the user selected manual
discards. Phase 4 now appends `DISCARD` (value 8, preserving existing values),
with pending counts and a derived decision player; the roller remains the turn
owner. See the [manual discard increment](manual-discards.md).

## Copying, equality, and port access

`state.copy()` copies every mutable array and restores the complete RNG state
with `getstate()` / `setstate()`. The RNG is allocated directly and restored
before use, avoiding unused reseeding; this includes the cached Gaussian value.
Copies bypass the initialization constructor and use direct field assignments.
`copy.copy` and `copy.deepcopy` also isolate mutable state; deep copies preserve
array/RNG aliases in nested containers. The immutable board is shared by identity.

Explicit equality compares the board, every bookkeeping scalar, array dtypes
and values, and RNG state. The Stage 1-3 audit optimizes small integer comparisons
with shape/dtype checks and C-order bytes, retaining NumPy comparison for other
debug dtypes. Mutable states are unhashable. `validate()` raises
`ValueError` for invalid storage, resource conservation, piece counts, ownership,
VP/building agreement, adjacent buildings, and turn/setup metadata. Validation
is explicit and excluded from the hot copy path; connectivity/action legality
will be handled in Phase 4.

`state.port_access(player)` returns sorted distinct `PortType` values granted by
that player's settlements or cities. It checks nine port endpoint pairs rather
than scanning 54 vertices. Roads do not grant access. Access is derived directly
from ownership and immutable board metadata, without a mutable ownership/ratio
cache. Trade legality and bank updates belong to Phase 4 or the later expansion.

Begin with the guide's fully observable training baseline. Keep true state
separate from the future observation encoder so realistic hidden hands can be
introduced without changing the simulator's underlying data.

## Verification and efficiency gate

1. Check every initial value, array shape, dtype, and sentinel for three and four
   players, plus invalid player counts.
2. Verify equal seeded games (including RNG continuation), different seeds,
   and shared immutable geometry.
3. Mutate every array and relevant scalar of a copied state; prove the original
   remains unchanged. Check independent RNG advancement and equality changes.
4. Assert initial resource conservation (`hands.sum(axis=0) + bank == 19`),
   remaining-piece counts, empty ownership, and desert robber placement.
5. Check snake order `(0, 1, 2, 3, 3, 2, 1, 0)` and the three-player equivalent,
   and port lookup for both endpoints and unported vertices.
6. Build the Docker image and run the full suite, lint, format, and compilation.
   Benchmark construction and copy separately with warmups and paired batches;
   report array bytes and time per state, without brittle timing test thresholds.

Phase 3 is complete when states initialize correctly, compare explicitly,
copy without shared mutable data, and replay independent seeded branches.
Keep functional copies for Phase 4 until profiling actual gameplay or search
shows copying is a bottleneck. Undo stacks, vectorized environments, GNNs,
PPO, and MCTS are later work.

## Verification and measurements (2026-10-09)

The initial Stage 3 suite passed 234 tests, including all prior geometry/board/viewer checks
and 110 state cases. State stress checks initialize and copy 1,000 seeded games
across three and four players; mixed random draws and Gaussian-cache tests verify
branch independence. Standard deep copies preserve aliases regardless of whether
the state or its arrays/RNG appear first in an outer container. Board and viewer
imports remain free of NumPy/PyTorch.

The subsequent [Stage 1-3 audit](stages-1-3-audit.md) expands this to 242 Python
tests (113 state cases) and nine viewer JavaScript tests, plus 100,000 boards and
5,000 state pairs in the stress harness. It adds populated-state branches,
strided/float equality regressions, concurrent previews, and seeded compatibility
fingerprints. Initial construction/copy measurements below remain historical;
the audit measures the newly optimized operations separately.

Run `docker run --rm catan-checks python -m scripts.benchmark_state` to reproduce
the performance harness. Seven warmed batches of 10,000 operations on Docker
Linux, Python 3.14.7 / NumPy 2.4.6 measured the following four-player medians:

| Operation | Time |
| --- | ---: |
| Initialize state with supplied board and RNG | 13.88 microseconds |
| New seeded game, including board and RNG construction | 53.86 microseconds |
| Copy state, including independent RNG | 38.98 microseconds |

An additional paired comparison measured copy cleanup from 51.88 to 39.73
microseconds, 23.4% less time, by avoiding unused RNG reseeding and dynamic
per-field attribute lookups. Both versions copy the same arrays and complete RNG
state. Separate runs can differ slightly; these are measurements, not pass gates.

At that measurement date, mutable array payload was 235 bytes for three players and 250 bytes for four,
excluding ndarray/Python headers, RNG memory, and the shared board. Timings
exclude imports, invariant checking, and gameplay. Profile actual engine moves
before introducing packed buffers, undo stacks, or a different RNG. Adding the
pending-discard array on 2026-10-10 increases these payloads to 241/258 bytes;
the timings above are historical and do not measure that increment.
