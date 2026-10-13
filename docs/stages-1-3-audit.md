# Stages 1-3 audit and optimization

Historical milestone record. The 2026-10-11 [expanded engine and performance review](expanded-engine.md)
supersedes earlier feature boundaries, array sizes, schema dimensions and current test counts below.
Historical measurements retain their original workload and date. Training remains deferred.


Date: 2026-10-09. Scope: everything implemented through the guide's first three
stages, including the package/container workflow, immutable geometry, constrained
board setup and ports, mutable state, local preview API/canvas, and benchmark tools.
Guide requirements were rechecked against Sections 1-3 in `generate_guide.py`.
Gameplay transitions and learning remain future stages.

Subsequent increment (2026-10-10): [manual discards](manual-discards.md) implement
the first Phase 4 transition. This audit's test counts, payload sizes, and timings
describe the earlier Stage 1-3 snapshot. The subsequent [Stage 4 engine](stage-four-engine.md)
adds the remaining minimal gameplay rules and current verification; this audit
retains its historical scope and measurements.

## Changes and measured effects

| Operation | Before | After | Less time |
| --- | ---: | ---: | ---: |
| Build the red-tile cache once | 6,482.11 us | 1,698.38 us | 73.8% |
| Compare equal four-player states, including RNG | 137.55 us | 93.71 us | 31.9% |
| Query ports on an empty board | 13.26 us | 5.53 us | 58.3% |
| Generate/encode a seed-42 preview | 438.84 us | 79.60 us | 81.9% |

The red-set search prunes a prefix as soon as it contains adjacent tiles, avoiding
the remaining illegal combinations. Ascending iteration retains the original
lexicographic cache order, uniform selection, board outputs, and RNG consumption.
This speeds cache construction at import; it does not change the per-board sampler.

State equality checks dtype and shape, then compares C-order bytes for small
integer arrays. It avoids repeated NumPy reductions and allocates only short-lived
byte strings; no mutable values are cached. Noninteger debug arrays still use
NumPy equality, including signed-zero and NaN behavior. Strided integer inputs
retain value equality. Copying, integer storage widths, and the RNG are unchanged.

Port lookup checks the two endpoints directly instead of allocating an inner
generator for every port. Settlements/cities still grant access, roads do not,
and duplicate port types are deduplicated. There is no ownership cache to update.

The API serializes static geometry once and joins it to each new layout while
retaining the same flat JSON fields. Compact separators reduce the seed-42 response
from 6,123 to 5,452 bytes (11.0%). Only the existing default seed is cached as a
complete board; other previews do not accumulate in memory.

Browser testing exposed a numeric-input bug: `1e3` was valid in the number field
but rejected by the digit-only API. The UI now submits numeric seed 1000. It also
checks obsolete responses before rendering or reporting HTTP errors, keeping the
latest requested board/status visible even when responses finish out of order.

Measurements use Docker Linux, Python 3.14.7 / NumPy 2.4.6. Before/after functions
were taken from the pre-audit workspace and compared in the same interpreter.
Seven warmed alternating batches used 500 cache constructions or 10,000 operations
per batch. Both versions were checked for matching results. Benchmarks vary with
load: a separate current-only run measured 1,609.32 / 93.33 / 5.54 / 72.83 us for
the four operations above. Timings exclude imports, HTTP transport, canvas drawing,
gameplay, and learning; cache construction occurs only once at board import.

The existing benchmark tools also run successfully. A four-player recheck measured
15.35 us to initialize from a supplied board/RNG, 57.71 us for a seeded new game,
and 41.83 us for an independent copy. Array payload remains 250 bytes (235 for
three players), excluding headers/RNG/shared board. These paths were not changed
in this audit. The direct board sampler measured 23.06 us versus 102.48 us for the
rejection reference in seven paired 10,000-board batches (4.44x ratio); the sampler
itself is unchanged and the difference from earlier timings reflects run variation.

## Verification

| Product | Evidence |
| --- | --- |
| Stage 1 packages | Fresh interpreter imports every `catan`/`rl` package and scaffold without PyTorch, PyG, or DGL. Board/viewer imports remain free of NumPy. |
| Geometry | Exact 19/54/72 counts, unit edges/hexes, reciprocal adjacency, connected graph, degree distribution, coastal cycle, and unchanged eleven-table fingerprint. |
| Setup and ports | Exhaustive red-set validation for every desert; cache ordering check; 10,000 seeded unit-test layouts; fingerprint of 1,000 sequential boards plus complete final RNG state. |
| Game state | 113 state cases cover initialization, all mutable fields, RNG/Gaussian-cache branches, deepcopy aliases, corrupted storage/metadata, and port ownership. 200 varied populated states exercise resources, settlements/cities, roads, live robber, and independent mutations. |
| HTTP API | HTML/legacy routes, seed limits and malformed inputs, deterministic data, 32 concurrent previews, response lengths, and geometry/seed isolation. |
| Viewer JavaScript | Nine tests execute the actual inline script with controlled DOM/canvas/fetch adapters: rendering counts, scientific notation, input limits, abort/error recovery, random seed, and late response/failure ordering. |
| Real browser | Seed 0, `1e3`, maximum seed, random seed, fractional input rejection, and restoration to seed 0. A 390x844 viewport showed all nine ports and no horizontal overflow; viewport reset afterward. No console errors/warnings on the updated page. |

The full Python suite passes **242 tests**, and the Node target passes **nine**.
Both Docker targets build. Ruff lint and formatting for 24 Python files, Python
compilation, dependency consistency, guide-generator static checks, JavaScript
syntax, and Git whitespace checks pass. No source errors or warnings remain from
these checks.
The additional stress harness passed **100,000 boards** covering all 19 desert
positions, then **5,000 three/four-player states with two branches each**. It
checked independent arrays, complete RNG continuation, invariants, and an unchanged
global RNG. The viewer is available at `http://127.0.0.1:8124/`; audit evidence is
saved locally in `output/stages-1-3-viewer.png` (ignored by Git).

## Reproduce

```powershell
docker build -t catan-checks .
docker run --rm --network none catan-checks
docker run --rm --network none catan-checks python -m ruff check --no-cache catan rl scripts tests
docker run --rm --network none catan-checks python -m ruff format --check --no-cache catan rl scripts tests
docker run --rm --network none catan-checks python -m compileall -q catan rl scripts tests
docker run --rm --network none catan-checks python -m pip check
docker build --target viewer-checks -t catan-viewer-checks .
docker run --rm --network none catan-viewer-checks
docker run --rm --network none catan-checks python -m scripts.stress_foundations
docker run --rm --network none catan-checks python -m scripts.benchmark_foundations
docker run --rm --network none catan-checks python -m scripts.benchmark_board
docker run --rm --network none catan-checks python -m scripts.benchmark_state
```

The dedicated Node 24 target uses the built-in
[test runner](https://nodejs.org/docs/latest-v24.x/api/test.html) and adds no host
packages or dependencies to the Python runtime image. Performance runs should be
sequential with competing test/stress jobs finished. The current-only harness is
retained; temporary pre-audit snapshots/probes are removed after verification.

## Remaining boundary

The first three stages meet their guide exit gates. The viewer displays starting
layouts, and state validation checks storage/physical bookkeeping. Neither
implements legal moves, payout, stealing, discards, trading, or terminal rules.
Stage 4 must implement/test `legal_actions` and functional `apply_action` before
gameplay throughput, rule correctness, or RL readiness can be assessed. Keep the
existing copy policy until real gameplay/search profiling supports a change.
