# Stages 1-4 audit before Stage 5

Historical milestone record. The 2026-10-11 [expanded engine and performance review](expanded-engine.md)
supersedes earlier feature boundaries, array sizes, schema dimensions and current test counts below.
Historical measurements retain their original workload and date. Training remains deferred.


Date: 2026-10-10. Main project: `C:\Users\david\Documents\Coding\Catan`.
Reviewed the package/container boundaries, immutable geometry and seeded setup,
compact state/copy invariants, viewer/API sessions, minimal rules, random policy,
and measurement tools. Stage 5 and learning code remain subsequent work.

## Retained optimizations

Gameplay profiling identified legal settlement queries, port trade queries, and
resource production as useful targets. The changes keep existing action objects,
rule predicates, array widths, copying, RNG draws, and seeded action ordering.

- Main-phase settlement queries examine only endpoints of owned roads. At most
  15 roads contribute 30 distinct vertices, versus scanning all 54 vertices.
  Candidates are transient and sorted; the existing placement predicate still
  checks distance/connectivity. Setup continues to examine all vertices.
- Trade action objects are grouped by offered resource, checking hand affordability
  five times instead of twenty. Receive-resource order and bank checks are retained.
- Internal trade ratios are derived directly from nine port endpoint pairs, avoiding
  a sorted/deduplicated public port result and repeated enum conversion. Generic
  discounts cannot overwrite specific 2:1 discounts. Public `port_access` remains
  available with its argument validation; no mutable cache is added. Valid state
  guarantees that an owned vertex has a building.
- Ordinary production uses five small Python count lists instead of a NumPy
  demand matrix, five NumPy reductions, and column views. Demands from all matching
  tiles are still aggregated before payment. Cities, robber blocking, and scarce
  bank supply retain the same rules. Scratch counts belong to one call only.

The owned path continues to retain state/array/RNG identities. The functional
path retains independent successors and rejection-before-copy behavior. Manual
choices remain explicit, and completed pending discards are not recomputed.

## Cleanup

Removed the unreachable discard-only type guard inside the typed helper; shared
public action dispatch already handles unsupported types. Renamed the shared
roll validator to reflect both dice and known-seven use. Removed obsolete temporary
source snapshots/probes from `tmp/optimization-review` and removed this audit's
comparison/example probes after verification. No duplicate engine is retained.

Corrected stale README, API/example documentation, and discard-demo copy that
still described setup, dice, trades, or robber rules as unfinished. The browser
continues to demonstrate sample hands and ends after discards. Historical stage
measurements remain clearly dated. Existing benchmark/reference samplers and
public endpoints are still used, so they remain; empty RL modules are the Stage 1
scaffold and contain no runtime learning code.

Geometry IDs, the 1,000-board/RNG fingerprint, nine separated ports, no adjacent
6/6, 6/8, or 8/8, state payload/copy contracts, dependencies, and guide sources
remain unchanged. No host package or ML dependency was installed.

## Paired gameplay measurements

Final results are recorded below after warmed alternating comparison batches.
The reference is the pre-audit engine at commit `e3a52b8`; both implementations
run in the same interpreter with the same unchanged state/actions/policy modules.
Each path uses five alternating batches after warmup, with the 10,000-action cap.
Final state/RNG equality is checked outside timing. Reset, legal queries, and
random-policy choices are timed; validation, rendering, and learning are excluded.

Python 3.14.7 / NumPy 2.4.6, with no competing tests/stress jobs:

| Players / seeds | API | Actions | Completed / capped | Before actions/s | After actions/s | Throughput ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Four / 0-9 | Functional | 38,973 | 7 / 3 | 9,797 | 12,288 | 1.254x |
| Four / 0-9 | In-place | 38,973 | 7 / 3 | 17,944 | 28,086 | 1.565x |
| Four / 0-2 | Functional | 3,183 | 3 / 0 | 9,121 | 11,384 | 1.248x |
| Four / 0-2 | In-place | 3,183 | 3 / 0 | 16,303 | 23,876 | 1.465x |
| Three / 0-9 | Functional | 12,968 | 10 / 0 | 10,184 | 12,781 | 1.255x |
| Three / 0-9 | In-place | 12,968 | 10 / 0 | 19,871 | 30,259 | 1.523x |

Functional gameplay gains about 25% throughput; owned in-place gameplay gains
47-57%, depending on this workload. These compare pre/post implementations of
each API, rather than comparing the two APIs with one another.

These are baseline gameplay rates, not training rates. Mixed capped-game work
is separate from fully completing workloads. The existing minimal-rule stalemate
and truncation semantics are unchanged. Hardware, seeds, runtime, and load affect
results. Reproduce current rates using the existing harness:

```powershell
docker build -t catan-checks .
docker run --rm catan-checks python -m scripts.benchmark_games
docker run --rm catan-checks python -m scripts.benchmark_games --games 3
docker run --rm catan-checks python -m scripts.benchmark_games --players 3
```

## Verification

- Both Docker targets build successfully.
- 533 Python tests pass; 21 viewer JavaScript tests pass.
- Ruff lint, 33-file formatting, compilation, dependency consistency, guide-generator
  static checks, six runnable Python documentation examples, and whitespace pass.
- The expanded foundation gate passes 100,000 boards spanning all 19 deserts and
  5,000 three/four-player states with two independent branches each.
- Exact pre/post actions, every state field, and complete game/policy RNG match
  across 51,941 steps in twenty seeded three/four-player episodes, including caps.
- Focused regressions compare pruned settlement queries with exhaustive direct
  placement validation on populated, funded states and verify combined generic/
  specific port precedence and stable trade order in either port ordering.
- Existing physical resource/piece invariants, production scarcity oracle,
  functional/in-place equivalence, rejected-action preservation, geometry/setup
  fingerprints, manual selectors, revision races, and seeded win/cap gates pass.
- Revised browser copy renders correctly in the live discard demo without console
  warnings/errors. No layout or interaction behavior changed.

Stage 5 remains next: broader rule/reachability validation and replay reporting.
Observation/reward/mask interfaces and parallel learning are later increments.
