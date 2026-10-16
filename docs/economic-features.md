# Economic features and fixed decision benchmarks

This implements Steps 1–3 of the architecture investigation: preserve a reference,
measure decisions at reproducible locations, and expose public economic
calculations to the existing MLP. This task launches no training experiment.
The new checkpoint is an **untrained migration**, not a stronger trained agent.

## 1. Frozen baseline and comparison contract

All work is in the main project, C:\Users\david\Documents\Coding\Catan.
The original selected checkpoint remains
output/training/run-20261014/checkpoints/u00000350.pt. Its SHA256 is
67a0ad501d00b5031e3e53f3f87383f96a21b1224ce812907bf6f29d8c1de8a2.
An exact independent copy, extracted configuration, earlier-result hashes and
the initial economic checkpoint are in
[baseline metadata](../output/economics-20261015/baseline/metadata.json).
[The protocol](../output/economics-20261015/protocol.json) separates this implementation
from a later training experiment.

The original has 244,955 parameters, hidden width 128, model1/observation2/action1,
350 committed updates and 54,864 optimizer steps. Its PPO settings are retained:
learning rate 0.0003, four epochs, minibatches 256, gamma/lambda 1, clip 0.2,
value coefficient 0.5, clipped values 0.2, entropy coefficient 0, global gradient
norm 0.5, sampled KL threshold 0.02, advantage normalization and fused Adam.
Shared actor/critic gradients remain enabled. No reward/curriculum/decoder change
is bundled into the feature implementation.

The future comparison uses the same migrated initial weights in both arms:
zero economic inputs versus real inputs, with the same parameter count and
interfaces. Practice boards, shuffle seeds, seat/player-count/opponent coverage
and checkpoint rule must match. Reserved replicate namespaces are practice
600,000,000 + replicate*1,000,000, development 1,011,000,000 + replicate*1,000
and final 2,011,000,000 + replicate*1,000. The common opponent slots are greedy,
frozen original, greedy and random, with balanced coverage and learner-only PPO
records. Final checkpoints are primary; final-test results cannot choose models.

These are frozen choices, **not permission to start training**. Runtime/cohort
allowances remain unset. Primary evidence will be paired whole-board greedy
win-rate differences with uncertainty. Random-opponent regression, real cutoffs,
own VP, decision scores, memory and speed are secondary. Correct calculations or
small diagnostic changes do not prove stronger play.

## 2. Decision benchmark

[scripts/benchmark_decisions.py](../scripts/benchmark_decisions.py) constructs a
bounded development suite:

| Category | Measurement | Interpretation |
| --- | --- | --- |
| Placement | Production relative to the largest legal settlement yield | Quantity only; diversity, expansion and blocking can justify another site |
| Affordable build | Settlement/city, trade and end-turn probabilities | Availability does not make a build strategically mandatory |
| Near purchase | Locations one card short of a settlement/city/development purchase | Material distance, not legality or a value target |
| Trade | Source hand gaps before/after a selected bank trade; model action probabilities | Closest-purchase distance is not a complete trade strategy |
| Discard | Full five-count likelihood and deterministic decoder gaps versus achievable per-purchase bounds | Bounds are attainable separately; they need not be attainable together |
| Verified win | Probability assigned to engine-proved deterministic winning moves | An exact check within an intentionally incomplete action set |

Sixteen new development boards start at 3,300,000,000. Each has all three/four-player
candidate seats against random and greedy: 14 roles per board, 224 fresh games.
The historical missed-win game (3,000,003,013, four players, candidate seat 2,
greedy opponents) is marked historical and excluded from fresh-game summaries.
It is a regression fixture, never fresh strength evidence. Per-seat policy
RNG 17 uses the existing catan-evaluation-v1 convention.

At most two occurrences of each category come from a game. Stable SHA256
priority retains at most 128 cases per category, independently of game/policy RNG.
Memory is bounded by 768 rows. This is a diagnostic sample, not a frequency-weighted
estimate of all decisions. Categories can share a location.

Each row stores the original observation, suffix, legal mask, submitted index or
complete discard counts, the observer's own hand, and the verified-win mask.
JSON records role, phase, pre-action step and descriptions. No opponent's private
hand or future hidden draw becomes a policy input. Hashes cover numeric cases,
descriptions and episode results. NPZ loading disables pickle. Future models
are scored at these fixed locations even if their natural games would diverge.

Winning labels fork the state and execute legal road, settlement, city and Knight
actions through the engine. They never read a future roll, steal or purchased
development-card identity to label a move. The set can miss other wins/sequences;
“no verified action” does not prove “no winning move.” Copies, snapshots and
oracles stay in offline diagnostics, outside normal collection/inference.

Discard bounds avoid enumerating all vectors. If K cards remain, purchase c can
retain at most min(K, sum_r min(hand_r,c_r)) useful cards. Its minimum shortfall
is its total cost minus that amount. Exhaustive bounded-hand tests check this.
The benchmark uses the existing complete prefix decoder to evaluate likelihood.
The deterministic-discard metric describes that separate decoder mode, not
the stochastic policy's expected return.

Using the optional learning image, from the main project:

~~~powershell
docker build --provenance=false --target learning-checks -t catan-economics-checks .
docker run --rm --network none --mount "type=bind,source=C:/Users/david/Documents/Coding/Catan/output/economics-20261015,target=/out" catan-economics-checks python -m scripts.benchmark_decisions build --checkpoint /out/baseline/original.pt --output /out/new-suite --boards 16
docker run --rm --network none --mount "type=bind,source=C:/Users/david/Documents/Coding/Catan/output/economics-20261015,target=/out" catan-economics-checks python -m scripts.benchmark_decisions score --checkpoint /out/baseline/initial-economic.pt --suite /out/decisions --output /out/new-scores.json
~~~

Build refuses an existing directory; scoring refuses an existing output file.
Never feed these development locations into PPO/imitation or final testing.

## 3. Public economic observation

[rl/economics.py](../rl/economics.py) is NumPy-only and opt-in. The original 1505
floats are an unchanged prefix, with 495 extra float32 values:

| Suffix offsets | Shape | Meaning and normalization |
| --- | --- | --- |
| 0:200 | 4 players ×5 resources ×10 rolls | Cards owed per roll /19; observer first, relative seats; unused fourth seat zero |
| 200:220 | 4 purchases ×5 resources | max(cost − own hand,0) /3 |
| 220:490 | 54 vertices ×5 resources | Expected cards per roll from one settlement at the vertex |
| 490:495 | 5 resources | Observer's best bank/port ratios /4 |

Resource order is wood, sheep, wheat, brick, ore. Purchases are road, settlement,
city, development. Rolls are 2–6 and 8–12; 7 has no production. Cities count twice.
Production and site yield respect the **current** robber. Production is gross
entitlement before finite-bank rationing, not guaranteed payment. The bank is
still present in the original observation. Tests remove scarcity in their
fixture before comparing all rolls with the engine; the shortage rule is unchanged.

Shortfalls use only the observer's exact hand. They do not assert affordability
when pieces are exhausted, routes blocked, sites occupied or the deck empty.
Site yields describe one hypothetical building unit; original inputs/masks
still determine ownership and legal moves. Port ratios derive from current
public ownership and the nine fixed slots, without a mutable port cache.

These summaries make board/resource/ownership joins explicit. Previously the
MLP had to learn those interactions from dispersed tile and vertex inputs.
They supply convenient calculations, not a strategy, hidden knowledge, future
dice or lookahead. They do not solve location bias or long-horizon planning.

~~~python
from learning.model import EconomicActorCritic
from learning.ppo import PPOUpdater
from rl.collector import collect_rollouts
from rl.economics import EconomicEnvFactory

source = PPOUpdater.load("original.pt")
model = EconomicActorCritic.from_base(source.model)
updater = PPOUpdater(model, source.config, shuffle_seed=source.shuffle_seed)
batch = collect_rollouts(
    updater.policy(), (670000100,), env_factory=EconomicEnvFactory(),
)
# Collection only. Training is a separate explicitly authorized step.
~~~

EconomicEnvFactory(False) gives the same 2000-wide profile with a zero suffix.
Ordinary CatanEnv/ActorCritic remain defaults. Prepared encoders are bound to
board identity and contain immutable lookups only. There are no new GameState
fields, mutable memoization, global board caches or invalidation work.
Reusable buffers are overwritten and must be writable contiguous float32,
disjoint from state arrays. Discards use the actual discard actor; terminal
and cutoff value observations support explicit observers.

For a match where only one seat needs these features, explicitly pass
EconomicEnvFactory(feature_player=seat). The selected observer gets the complete
suffix, including terminal views; other observers get zero suffixes and their
unchanged original inputs. This skips unused calculations for reference agents
or the zero-projection frozen original. All-seat features remain the default.
Do not use this option to deprive another trained economic policy of its inputs.
It is independent of record_players: that argument still controls storage only.
A caller rotating the learner seat must also rotate feature_player explicitly;
no existing trainer/evaluator behavior is silently changed.

## Compatibility and cleanup

EconomicActorCritic uses model3/observation4/action1, with a 495-to-hidden zero
projection before the first ReLU. Original trunk/output/discard parameters
retain names, values and shapes. Flat outputs and complete conditional discard
likelihoods start unchanged. The 128-wide branch adds 63,360 parameters:
308,315 total, about 25.9% more. Observations add 1,980 bytes each, about 32.9%
more observation payload; that is not a percentage of total training memory.

New PPO/checkpoint and collector profiles explicitly identify the schema.
Original model1/resource-history model2 remain readable. PPO rejects
observation/reward mismatches before optimizer mutation. Resume restores the
branch, Adam and local shuffle state exactly. Migration instead starts empty
Adam/counters and records the original hash; it does not reuse an optimizer with
a different parameter set. No unused imitation-data profile is implemented.

The knowledge and economic branches share one small internal implementation,
removing duplicate forward/migration code. Model2 retains knowledge.weight and
its contracts. There is no generic plugin architecture. Engine rules, geometry,
RNG ordering, state copying, masks, discards, awards and reward are unchanged.

## Measurements and verification

The saved suite contains 725 rows: 128 each for placement, affordable build, near
purchase, trade and discard, plus 85 verified-win rows. Independent reconstruction
of every suffix from its original permitted observation passed. The migrated
checkpoint produces exactly equal logits, values, hidden activations, scores and
joint action likelihoods on all 725 rows. All 85 winning locations were replayed,
with 200 legal winning actions executed on engine copies and invariant checked.
Fifteen matched original/migrated roles covered 12,416 invariant-checked actions
with exact full game and policy RNG matches. At historical step 530, road 26 is
still chosen while settlement vertices 43/48 are both directly verified wins.

The 224 fresh development roles finished with zero cutoffs. The original won 84/112
against random and 3/112 against greedy; own mean VP was 9.0446/3.7857 respectively.
These are new development diagnostics, not a trained-feature comparison.
The retained win cases show 62/85 source actions taking a verified win, with
mean verified-win probability 0.7472. Source placement production was 0.7367 of
the best legal production on average; expected production under its distribution
was 0.7679. Category sampling and the historical fixture prevent interpreting these
figures as population-level error rates or independent samples.

Kernel comparisons used 12 representative three/four-player states, seven
shuffled-order repeats and 12,000 calls per method per repeat on the same CPU
container, NumPy 2.4.6 and Torch 2.14.1. Median production-kernel microseconds:

| Method | Microseconds | Decision |
| --- | ---: | --- |
| Tile matrix, int16 | 47.58 | Replaced |
| Vertex matrix, int16 | 65.74 | Slower and more lookup storage |
| Vertex matrix, float32 | 51.16 | Faster than dense int16, still slower than sparse; 10,800-byte matrix |
| Python tile loops | 47.34 | Avoided matrices but repeated lookup work |
| Python occupied-vertex loops | 42.81 | Better sparse work, still decoded tile identities repeatedly |
| Prepared occupied-vertex loops | 34.21 | Kept; 19 immutable column indices plus site-yield table |

The selected kernel takes about 28.1% less time than the first tile method;
that is not a whole-training speedup. No candidate changes feature semantics.
Another comparison tested copying an allocated prefix versus validating a prefix
view. The medians were 217.62/218.77 microseconds, a negligible 0.5% difference with
one extra transient 1505-float allocation. It was rejected, and its prototype is
removed after preserving measurements.

Longer end-to-end checks used eight fixed boards at each player count, active
batch 8, learner seat 0 against greedy, seven shuffled-order repeats and one CPU
Torch thread. Each profile performed the same 5792 engine actions per repeat
with identical final game and policy RNG fingerprints:

| Collection path | Median actions/second |
| --- | ---: |
| Original 1505-wide profile | 2606.03 |
| New 2000-wide profile, zero suffix | 2219.88 |
| New profile, calculate features for all seats | 1793.41 |
| New profile, calculate features for learner only | 2158.86 |

Explicit learner scoping improved throughput about 20.4% over all-seat calculation.
It was still 17.2% below the original profile and 2.7% below the zero-suffix control.
Thus the extra calculations are inexpensive relative to the already-expanded
profile, but the complete profile has a material cost relative to the legacy bot.
This is CPU collection evidence for these roles, not a PPO/GPU timing or a
universal speed guarantee. Short two-board measurements were noisier, including
overlapping zero-suffix/learner-only samples; all raw samples are preserved rather
than selecting the most favorable run.

Final gates passed:1070 full learning tests,14 NumPy-only feature tests,
repository lint,86-file formatting and compile checks, dependency boundaries,
and main/image source-hash equality. Only three previously existing Python
sources changed (model, PPO profile dispatch, collector profile dispatch);
six focused files were added, no application sources deleted, all engine sources
unchanged. Four task-created runtime helpers were hash checked and removed.
The shared feature branch replaces duplicate migration/forward logic. Existing
experiments/checkpoints/results remain available; they were not declared obsolete
just because this task does not use them.

Evidence:
[independent audit](../output/economics-20261015/verification.json),
[final checks](../output/economics-20261015/final-checks.json),
[kernel comparisons](../output/economics-20261015/kernel-comparison.json),
[assembly alternative](../output/economics-20261015/assembly-comparison.json),
[short final timing](../output/economics-20261015/performance-final.json),
[longer collection timing](../output/economics-20261015/collection-stability.json),
[source inventory](../output/economics-20261015/source-final.json),
[cleanup receipt](../output/economics-20261015/cleanup.json).

## Next step

Implement the small matched training comparison using the frozen initial model
and settings. Persist the complete environment profile in resume/evaluation
identities and rotate any feature_player restriction with the actual learner
seat. Set a finite allowance before launching. Compare real held-out wins,
random-opponent regression, decision diagnostics and wall cost. The current
new checkpoint has zero learning updates and gives the original answers; this
implementation supplies no evidence of a learned strength improvement yet.

## Plain-language explanation

We kept the old bot and gave the new version a small calculator. It can see
what each roll produces, which cards it still needs, what settlement spots
produce, and its port rates. It still has to learn how to use those numbers.

We also saved game positions, like practice exam questions. Future versions can
answer exactly the same questions. Some have a provably correct answer, such as
building a settlement that wins immediately. Others reveal tendencies without
pretending that a simple heuristic always knows the best strategy.

The new version initially acts exactly like the old one. A later controlled
experiment can test whether learning with these calculations helps, while
keeping rewards, opponents and the rest of the architecture consistent.
