# Expanded engine and holistic performance review

Later 2026-10-13 update: [the optional MLP policy/critic and same-seat learning targets](policy-value-targets.md) are implemented. Earlier model/target next-work statements below describe their dated milestone. Training/optimizer updates remain held; existing PDFs are unchanged.

2026-10-13 update: the [per-seat rollout collector](rollout-collector.md) is now implemented. It records same-seat successors, complete discards and outcome/cutoff data. The dated report and PDF remain historical; models/training are still held.

2026-10-12 update: [agent environment, outcome rewards and evaluation](agent-environment-evaluation.md) are now implemented. The dated milestones and next-work statements below describe the 2026-10-11 report; its PDF remains unchanged. Training remains held.

Date: 2026-10-11. Main checkout: C:/Users/david/Documents/Coding/Catan.
Training remains on hold. This contract supersedes earlier feature/schema boundaries.

## Rules and interfaces

All base development types and both awards are now implemented, alongside the
existing setup, production, manual discards, robber, building and bank/port trades.
Negotiated player trades and a full-game browser interface remain deferred.
Official rule sources were checked for this increment:
[base rules/almanac (2020)](https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf)
and [official basegame FAQ](https://www.catan.com/faq/basegame).
The following is a paraphrase of the implemented rules.

| Action | Legal phase | Result |
| --- | --- | --- |
| BuyDevelopmentAction | MAIN | Pay sheep+wheat+ore; draw one hidden card; repeated purchases allowed |
| KnightAction | ROLL or MAIN | Consume older knight, count army, choose robber move/victim; no discards |
| RoadBuildingAction | ROLL or MAIN | Consume older card; choose up to two free RoadAction placements sequentially |
| MonopolyAction(resource) | ROLL or MAIN | Take all opponents' holdings of chosen type; bank unchanged |
| YearOfPlentyAction(counts) | ROLL or MAIN | Select two available bank cards, or fewer only when bank total is below two |

One non-VP development card may be played per turn. A fresh card cannot be played
that turn, but an older copy can be played even after buying a fresh copy of the
same type. VP cards stay held, automatically contribute to true score and can win
immediately when bought. Fresh counts clear on end turn. A knight played before
the dice returns to ROLL after its robber choices; it never pretends dice rolled
seven. Road Building resumes the original phase after count/pieces/legal spaces
are exhausted. Its second choice sees its first placement. The current interface
requires both roads when both can be placed. No resource choices are automatic.

The 25-card supply is 14 knight, 5 VP, and 2 each Road Building, Year of Plenty and
Monopoly. Five remaining counters plus conditional weighted draws are equivalent
in distribution to a uniform shuffled deck without replacement. Initialization
draws nothing, preserving board/setup RNG fingerprints. Card trajectories differ
intentionally from the earlier engine with no development actions.

Longest Road uses an exact edge trail: no edge repeats, vertices can repeat,
loops count, branches are not summed, own buildings allow passage and opponent
buildings stop passage. A blocked vertex may be an endpoint. Threshold is five
roads. Largest Army needs three played knights. Each awards two VP; an incumbent
retains a maximum tie. A unique eligible leader takes an unheld award. If an old
road holder loses the lead and several others tie, the award becomes unheld.

Cached road lengths update on road placement (builder only) and settlement
blocking (incident opposing road owners). City upgrades and resource actions do
not change the graph. The debug invariant checker independently recomputes lengths.
Award holders preserve necessary tie history. Total score equals building points,
held VP cards and award bonuses. Only the current seat wins; a player reaching
ten through an off-turn award waits until its own turn. A pre-roll win is valid.

Both functional/owned APIs share validators and mutations. Functional calls copy
all mutable data/full RNG; owned calls retain every array/RNG identity. Rejections
precede copying, mutation and draws. All fourteen arrays/new scalars are included
in copy/equality/invariant/fingerprint gates. Raw arrays now total 287/316 bytes
for three/four players, excluding headers, RNG and shared board.

## Current state additions

| Field | Shape/dtype | Meaning |
| --- | --- | --- |
| development | n x 5 int8 | Held cards, including private VP |
| development_new | n x 5 int8 | Fresh subset of current turn's held cards |
| development_deck | 5 int8 | Remaining type supply |
| development_played | 5 int8 | Public spent types; VP column stays zero |
| knights_played | n int8 | Public per-seat spent knights |
| road_lengths | n int8 | Cached exact longest trails |
| longest_road_owner / largest_army_owner | scalar | -1 unheld or incumbent seat |
| development_used | bool | One non-VP card already played this turn |
| free_roads_remaining | scalar | Zero normally; one/two during ROAD_BUILDING |
| resume_phase | Phase or None | ROLL/MAIN continuation for multi-choice effects |

## Observations and lookup

Observation schema v2 has 1505 float32 values / 6020 payload bytes. It exposes own
exact cards, public opponent card totals, played knights, exact trails, holders,
spent types and effect timing. Opponents' private VP is removed from their public
score; own total is visible and terminal observations reveal final totals. Deck
composition and opposing card freshness/composition remain hidden. The updated
[representation contract](representation-encoding.md) lists every group.

`encode_batch(states, encoders, players=None, out=None)` supports mixed boards/seats
with one prepared context per row. All input/context/buffer checks finish before
writing. Fewer than eight rows use the scalar path. Larger batches gather spatial
channels in chunks of at most 128 and reuse the single global writer. No global
board cache, worker sharing or persistent scratch is introduced.

The fixed catalog is published in `rl.actions`, outside the engine:

| Inclusive indices | Count | Action |
| --- | ---: | --- |
| 0-71 | 72 | RoadAction, including setup/free roads |
| 72-125 | 54 | SettlementAction |
| 126-179 | 54 | CityAction |
| 180-198 | 19 | RobberMoveAction |
| 199-218 | 20 | Ordered bank give/receive pairs, excluding same resource |
| 219 | 1 | EndTurnAction |
| 220 | 1 | RollAction |
| 221-224 | 4 | Relative steal-victim slots; self/absent slots masked |
| 225 | 1 | BuyDevelopmentAction |
| 226 | 1 | KnightAction |
| 227 | 1 | RoadBuildingAction |
| 228-232 | 5 | Monopoly resource choices |
| 233-253 | 21 | Year of Plenty zero/one/two-count vectors in stable catalog order |

`action_index`/`indexed_action` convert membership, not legality. `action_mask`
shares engine predicates and exactly matches legal-object projection. Terminal
masks are all false. DISCARD instead requires `discard_count_mask(state, prefix)`:
twenty possible counts per resource, five conditional choices, no intermediate
mutation. The complete tuple is one existing DiscardAction. This avoids a large
flat discard-combination enumeration while retaining explicit manual selection.

## Experiment decisions

The original minimal-system baseline is retained as context, not a causal comparison
with expanded rules: six games/seeds 0-2 for both seat counts, cap 2000, 6497 actions,
6 wins, 0 caps; medians owned 21710 actions/s, copied 9413, owned+v1 observations
8095. Expanded rules have different trajectories and 1505 rather than 1461 values.
Raw reports are under `output/full-review`.

First prototypes on 144 reached states, five warmed alternating batches:

| Batch | Scalar encoding us/state | Prototype batch us/state | Decision |
| --- | ---: | ---: | --- |
| 1 | 68.71 | 130.33 | Reject vector batch for single observations |
| 8 | 62.24 | 54.95 | Keep batch option |
| 32 | 63.20 | 44.24 | Keep batch option |
| 128 | 59.60 | 43.10 | Keep batch option |

Prototype direct masks reduced 52.43 to 45.39 us versus legal-object conversion.
Production APIs were then added with exactness/isolation/rejection tests and rerun
under the expanded schema. Final production comparison:

| Work | Reference us/state | Retained us/state |
| --- | ---: | ---: |
| Batch 8 encoding | 74.89 | 67.08 |
| Batch 32 encoding | 72.17 | 55.19 |
| Batch 128 encoding | 74.73 | 69.50 |
| Direct masks vs legal projection | 47.71 | 41.68 |

The batch-one production API falls back to scalar encoding; callers can continue
using the existing prepared scalar API directly. Improvements depend on size and
load; no universal batch speedup is claimed. Raw prototype/final reports are
`improvement-prototypes.json` and `final-lookup.json`.

The holistic profile found legal queries, global observation writes and functional
copying prominent; trail updates were a smaller cost. We retained transient road
owner snapshots, early unusable-card exits, one bank snapshot per Plenty query,
and a single global-feature write. We did not add undo, packed state or mutable
legality/port caches. Paired five warmed alternating batches with identical
expanded rules, seeds, policy/game RNG and 7095 actions per six-game batch:

| Work | Before | After | Throughput ratio |
| --- | ---: | ---: | ---: |
| Owned gameplay | 18387 actions/s | 19871 actions/s | 1.081x |
| Functional gameplay | 7345 actions/s | 8218 actions/s | 1.119x |
| Owned game + observations | 5457 actions/s | 6139 actions/s | 1.125x |
| Prepared allocating encoding | 81.56 us | 72.41 us | 1.126x |

Functional timings had wider load variation; the full samples are retained.
Every action/state field/full RNG matched across 24199 additional replayed steps
and every observer in 78 reached-state encoding cases matched exactly. Reports:
`paired-optimization.json`, `expanded-before-optimization.json`, `expanded.profile`.
Rates include reset/legal queries/policy and optional encoding; they exclude debug
validation, rendering, neural inference and training. These are measured engine
and representation gains, not claims about ML learning speed or sample efficiency.

Historical rejected prototypes remain rejected: persistent discard scratch saved
only about 0.7 us, and a compact static encoder template saved memory but increased
latency. Their complexity was not justified. Future optimizations should compare
identical workloads and verify semantics before adopting native/GPU/undo machinery.

## Cleanup, checks and remaining work

Six empty future RL modules were removed (env, network, ppo, mcts, selfplay,
evaluate). They did not consume rollout time; removal clarifies implemented scope.
Task-specific before/expanded snapshots, probe and patch scripts are removed only
after comparison/review. Raw measurements, historical docs, original guide,
generator and diagram assets remain. No host packages or ML dependencies added.

Current gate: 664 Python tests and 21 viewer JavaScript checks pass. Both container
targets build. Foundation stress passes 100000 boards/all nineteen deserts and
5000 three/four-seat states with two independent branches each. The expanded continuous check completed 10000 distinct seeds (5000 games per seat
count), a 5000-action cap and eight bounded workers: 9491529 actions / 9501529 full
invariant checks, all nine phases, 10000 true wins, zero caps, zero crashes or
invariant failures. Ordered fingerprint:
`209065dd840c00500e580c92db0d62b52cc4e56c87829c16fac44040ce6f7fe5`.
Elapsed 1638.54 seconds includes debug checking and concurrent regression/document
work; it is not an engine performance benchmark. The finite gate does not prove
universal termination. Ruff lint/39-file formatting, compilation, dependency
consistency, generator F-checks, whitespace and eleven documentation Python
examples pass. Training remains on hold. No commit is requested or made.

```powershell
docker build -t catan-checks .
docker run --rm catan-checks
docker run --rm catan-checks python -m ruff check catan rl scripts tests
docker run --rm catan-checks python -m ruff format --check catan rl scripts tests
docker run --rm catan-checks python -m scripts.benchmark_system
docker run --rm catan-checks python -m scripts.benchmark_lookup
docker run --rm catan-checks python -m scripts.validate_games --games 10000 --players both --max-actions 5000 --workers 8
```

Next work is an environment/reward/actor/termination contract, followed by MLP
baseline and later GNN comparison when training is authorized. Negotiated trades,
full-game UI, symmetry, self-play/PPO and search remain explicit future increments.
The detailed project report is authored from [the decision register](project-decisions.md)
and milestone sources, separately from Catan_RL_Guide.pdf.
