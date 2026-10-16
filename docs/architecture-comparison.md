# Economic features and a compact graph encoder

## Status

Completed and independently verified on 2026-10-15. Neither economic features
nor the compact graph demonstrated a reliable improvement against greedy in
this fixed-budget comparison. The original remains the default; no model was
promoted and no additional training was started.

The real run launched at 2026-10-15T19:14:33Z in owned container
catan-architecture-20261015, using immutable image
sha256:b2bb7931088f3554eafafaad66a3a53ac705de6a37550ae6a1b180b2b7b892ab.
The owned container exited successfully, without OOM, at
2026-10-15T21:28:49Z. Its wall interval was 8,055.63 seconds (2h14m15.63s).
The runner reports 8,078.72 seconds through comparison creation and 8,086.93
through its final bookkeeping. These independently recorded clocks differ;
the raw values are preserved rather than silently reconciled. Allowances were
fixed cohort counts, so this discrepancy did not grant additional training.
The three-hour figure was an estimate, not a requirement to spend unused time.
There is no remaining computer-awake or Docker requirement for this experiment.
[Launcher](../output/architecture-20261015/launcher.json),
[live state](../output/architecture-20261015/run/state.json),
[checks](../output/architecture-20261015/final-checks.json).

Verification passed 1,086 full learning tests, 107 NumPy-only environment/collector
tests, then 40 affected tests after final seed isolation. All 93 Python sources
pass lint/format/compile and match the frozen image. The independent audit
passed both capped CLI smoke outputs, including raw totals, paired intervals,
teacher-label regeneration and exact model/RNG/invariant replays. These are
engineering checks, not strength evidence for the real experiment.

## Actual final results

Each row below contains 1,344 games per opponent. All 16,128 final roles
completed with a real winner: zero action-cap cutoffs. The two original-reference
rows use different board families, so compare arms within their own stage.

| Comparison | Model | Greedy wins | Greedy rate | Random wins | Random rate | Mean own VP against greedy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Features | Original reference | 13 | 0.967% | 997 | 74.18% | 3.644 |
| Features | Continued MLP, zero suffix | 11 | 0.818% | 930 | 69.20% | 3.563 |
| Features | Continued MLP, economic suffix | 8 | 0.595% | 913 | 67.93% | 3.554 |
| Architecture | Original reference | 20 | 1.488% | 1,022 | 76.04% | 3.832 |
| Architecture | Fresh economic MLP | 5 | 0.372% | 754 | 56.10% | 3.437 |
| Architecture | Fresh graph | 5 | 0.372% | 733 | 54.54% | 3.563 |

The primary feature difference, economic minus zero suffix, was -0.223
percentage points. Its paired whole-board 95% interval was [-0.893, +0.521]
points; the multiplicity-adjusted 97.5% interval was [-0.967, +0.595].
This supports neither a reliable benefit nor a claim that all possible uses of
these features are harmful.

The primary graph difference, graph minus fresh MLP, was 0.000 points.
Both the 95% and adjusted 97.5% intervals were approximately
[-0.446, +0.446] points. With only five wins per architecture, a small effect
remains unresolved. Higher graph VP with identical wins is not a win improvement.

Per-replicate greedy wins were:

| Replicate | Zero suffix | Economic suffix | Fresh MLP | Fresh graph |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1 | 3 | 1 | 0 |
| 1 | 3 | 4 | 2 | 2 |
| 2 | 7 | 1 | 2 | 3 |

Random-opponent differences were also inconsistent across replicates. Economic
minus zero changed by -9.38, +9.60 and -4.02 points; graph minus fresh MLP by
-10.94, -3.79 and +10.04. The pooled random differences were -1.265 points
[95% interval -5.357, +2.679] and -1.563 [-6.994, +3.646], respectively.
Selecting the best replicate would reverse parts of the story and was not done.

Relative to the original on matched boards, both continued feature arms lost
random-opponent performance: zero -4.99 points and economic -6.25, with their
ordinary paired 95% intervals below zero. Both fresh architectures also remained
below the original. The original's longer training history makes it a reference,
not a fair architectural control; the fair architectural comparison is fresh
MLP versus fresh graph.

These intervals resample 192 board clusters per comparison. They condition on
the three trained models and do not estimate the full uncertainty across all
possible training runs. Random-opponent and original-reference comparisons are
secondary; the two greedy primary intervals received the declared adjustment.

## What the decision diagnostics show

The original, all 12 final models and six teaching checkpoints were scored on
the unchanged, hashed 725-position development suite. No diagnostic position
entered teaching, PPO or final evaluation. Its 85 immediate-win locations have
engine-proved winning actions; other heuristic scores remain descriptive.

| Model group, average across three replicates | Probability assigned to verified immediate wins | Expected placement production ratio |
| --- | ---: | ---: |
| Original | 74.72% | 0.768 |
| Continued zero suffix | 73.46% | 0.734 |
| Continued economic suffix | 71.21% | 0.737 |
| Fresh MLP after teaching | 40.40% | 0.738 |
| Fresh MLP after PPO | 34.22% | 0.772 |
| Graph after teaching | 41.51% | 0.738 |
| Graph after PPO | 35.32% | 0.721 |

The first column is probability mass on correct actions at saved positions,
not a game-win percentage. Production ratios compare expected gross yield
against the best legal yield at those positions, ignoring resource balance,
blocking and future plans. The suite contains states visited by the original,
which limits generalization to the new models' own trajectories.

Three findings matter:

1. The common teaching pass reduced held-out actor loss from about 1.991 to
   1.706-1.710 in both architectures, but it did not reproduce the original's
   immediate-win reliability. Lower teaching loss did not establish strong play.
2. PPO then reduced immediate-win mass in both fresh architectures. The fresh
   MLP's placement yield ratio rose while its greedy wins remained rare. This
   argues against treating placement yield alone as the explanation for failure.
3. Greedy practice supplied very little positive terminal feedback. Across
   1,008 greedy practice games per arm, learner wins were zero-suffix 9,
   economic 6, fresh MLP 1 and graph 1. Losses still produce PPO/value updates;
   these counts specifically describe scarce positive win feedback.

The economic branch was connected and learned: final weight norms were
1.878-1.936, whereas controls remained exactly zero. Blanking its inputs at
the saved public positions changed some flat-action choices and produced
mean total-variation changes of 0.0052-0.0092. That posthoc input ablation
confirms activity, not a causal estimate of training benefit.

These observations support investigating teaching coverage and retention of
winning decisions before increasing model complexity. They do not prove a
single cause, a broken graph implementation, or that features can never help.
No new reward, hyperparameter search or extra run was introduced after seeing
these results.

The predeclared allowance was 84 eight-game cohorts per case, three complete
28-cohort coverage cycles. There were 12 trained cases, 1,008 PPO cohort updates and 8,064
practice games across the two comparisons. The matched arms receive identical
opportunities, not identical wall time. Full balanced cost probes measured
2.84 seconds/cohort for the economic MLP and 4.55 for the graph. Batched 112-role
probes took 27.96 and 41.08 seconds. Including final/development checks, teaching
and headroom, this targeted about three hours. The counts remained fixed after
outcomes were visible. The inherited 86,400-second RunConfig field is
informational under fixed_cohorts_v1, not a 24-hour training allowance.

Actual reconciliation found 8,064 practice games, 4,079,701 engine actions,
1,120,459 retained learner rows, 27,747 learner discard rows and 19,374 PPO
optimizer steps across 1,008 cohort updates. There were no practice cutoffs.
Equal game/cohort opportunities do not mean identical minibatch counts:
different trajectories create different row counts and the unchanged KL guard
can stop an epoch early.

This continues [economic Steps 1-3](economic-features.md). The 725-row decision suite
and 495 public economic features already existed; these comparisons test whether
learning with them helps and whether a different board encoder helps.

## Step 4: isolate the information

Both arms start from the exact same migrated original checkpoint: model 3,
observation 4, no PPO updates and empty Adam. The control receives a zero 495-float
suffix; the candidate receives the public economic calculations. Both keep
308,315 parameters, flat 254-action output, complete conditional discard decoder,
terminal winner reward and original PPO settings.

Three replicates vary practice boards and minibatch shuffles, intentionally
starting from identical original weights. Each 28-cohort schedule covers every
3/4-player learner seat against two greedy slots, one frozen original slot and
one random slot. Opponents never learn. The original receives only the unchanged
1,505 observation entries through an explicit prefix adapter.

Both feature arms reset Adam and update counters together; this isolates
features conditional on that shared continuation procedure. It does not test
whether retaining the old Adam moments would prevent the observed regression.
PPO kept learning rate 0.0003, four epochs, minibatches of 256, clip 0.2,
value clip 0.2, value coefficient 0.5, entropy coefficient 0, gradient limit 0.5,
KL target 0.02, gamma 1 and lambda 1. Advantages were normalized and the critic
continued updating shared features. These settings were not tuned after results.

Only learner rows are retained. All seats still act with their own policy RNG.
Economic encoding is restricted to the actual learner, rotating with the
balanced schedule. Reference evaluation rotates it with the candidate seat.
This saves work without changing the candidate's information.

## Step 5: the compact graph

Optional GraphActorCritic is model 4/observation 4. It embeds 19 tiles,
54 intersections and 72 roads into width-12 vectors from their original 9/16/5
features. Two simultaneous rounds exchange neighbor means along tile/intersection
and intersection/road relations. Each update concatenates the node's old vector
and the relevant neighbor means, then applies a shared linear layer and ReLU.

The 145 node vectors flatten in existing geometry-ID order. They join the 605
remaining public/global/economic entries, then a 2,345-to-112-to-112 trunk feeds
the same action/value output and complete discard decoder. It has 309,167
parameters, only 0.28% more than the 128-wide economic MLP.

Width 112 was chosen to match capacity, rather than adding a much larger graph
and attributing its results to connectivity alone. Mean aggregation keeps
message scale comparable for coastal and interior nodes. Sum aggregation,
attention and deeper encoders were not tested. The implementation constructs
only its own layers, without allocating an unused legacy trunk. Initialization
uses scoped orthogonal weights, small policy-output gain and an independently
scaled value row, following the existing model's conventions.

The board encoder is equivariant when inputs and incidence are relabeled
together. The ordered readout means the whole policy is not permutation
invariant. Two message rounds do not implement arbitrary-depth road planning.
Global readout can combine distant information, but only test results can
establish whether the inductive bias helps.

Normalized incidence buffers are built once and checked against canonical
geometry on checkpoint loading. No dynamic graph objects, graph dependency,
new GameState fields, hidden opponent cards, search, recurrence, shaped reward
or new action heads are introduced. Scoped initialization preserves the caller's
Torch RNG. Old/default model and environment profiles remain readable.

## Step 6: fresh learners on equal opportunities

Fresh economic MLP and graph models use initialization seeds 1701, 1702 and 1703.
Both receive the same bounded greedy demonstration shards and two fixed passes,
with phase-balanced actor likelihood, minibatches of 256, learning rate 0.0001 and
shuffle seeds 1901+replicate. Critic values are not teacher labels. Each then
starts PPO with fresh Adam and the same fixed cohort allowance. The final
warm-up pass is used without validation-dependent selection.

The common dataset has 128 training games and 16 separate validation games,
half each 3/4 players. Each numeric shard contains at most 8 complete games;
one shard resides in memory at a time. Forced flat labels are omitted only from
imitation; complete discards are retained. Demonstrations never become PPO
behavior data. Shard-boundary persistence restores exact weights, Adam,
shuffle RNG and cursor after a pause.

The actual 18 shards contain 22,644 non-forced training labels and 2,708
validation labels. Every fresh model received two identical passes and exactly
194 actor-imitation optimizer steps. All six recipes, cold initializations,
teaching weights, fresh PPO Adam states and shuffle identities were verified.

| Purpose | Feature comparison | Fresh architecture comparison |
| --- | --- | --- |
| Practice |603000000+rep*1000000 |703000000+rep*1000000 |
| Development |1014000000+rep*1000 |1015000000+rep*1000 |
| Final |2014000000+rep*1000 |2015000000+rep*1000 |
| Teacher training | none |853000000, plus10000 for4p |
| Teacher validation | none |853100000, plus10000 for4p |

The frozen production seed_shift is 3,000,000; engineering unit/smoke runs use
40,000,000. Earlier capped smoke checks used shift 0 before this isolation was
added. All those test models/results are separate, never selected or reused.
The real final boards are disjoint from both engineering namespaces.

Development uses 4 boards per replicate. Final evaluation uses 64 distinct boards
per replicate, all 3/4-player seats, against greedy and random. Each trained
arm gets 1,344 games per opponent. The original is an additional matched reference
in both comparisons. Together these schedule 16,128 final game roles.

The last fixed PPO checkpoint is primary. Early checkpoints/development curves
show learning onset or collapse without selecting a final winner. Primary
outcomes are real greedy-win differences with paired whole-board bootstrap
intervals: ordinary 95% and 97.5% adjusted for the two primary comparisons.
Per-replicate results accompany pooled estimates. These intervals condition
on three trained models; they do not estimate all training-seed uncertainty.

Real wins, losses and action-cap cutoffs stay distinct. Own final victory
points are diagnostic reporting, not invented wins or extra rewards.
Random-opponent regression and immediate-win mistakes also qualify conclusions.
Rare wins whose intervals cover zero remain inconclusive.

## Efficiency and rejected alternatives

The shared fixed-cohort runner was extracted from the critic experiment instead
of duplicating its checkpoint/clock/pause code. Collection encodes economic
features only for the learner; no validation sweep or graph construction
enters the normal engine path.

An isolated comparison tested dense incidence multiplication against padded
neighbor gathering/weighted reduction, including output and gradient agreement.
Dense multiplication was faster in all 12 relation/batch combinations, so it
is retained. There is no runtime alternative flag or unused graph dependency.
The benchmark remains as a reproducible measurement tool.

One-thread inference medians at batches 1/8/256 were approximately:

| Model |1 |8 |256 |
| --- | ---: | ---: | ---: |
| Fresh economic MLP |0.199ms |0.430ms |4.705ms |
| Graph |0.745ms |1.513ms |22.204ms |

The two greedy-only 8-game probes took 1.53/1.52s collecting and 0.40/0.28s
updating the MLP, versus 2.13/1.60s collecting and 1.16/0.75s updating the graph.
These are cost probes on different resulting trajectories, not strength or
universal throughput claims. Peak process RSS reached 402,548 KiB with all probes
sequentially resident; that is not an isolated per-model memory estimate.
Raw results: [benchmark](../output/architecture-20261015/benchmark.json).

Actual collection/update totals across the three replicates were:

| Arm | Collection seconds | PPO update seconds | Largest retained rollout payload |
| --- | ---: | ---: | ---: |
| Zero suffix | 388.32 | 109.55 | 19.76 MiB |
| Economic suffix | 419.76 | 106.48 | 18.73 MiB |
| Fresh economic MLP | 478.61 | 121.15 | 25.32 MiB |
| Fresh graph | 563.71 | 322.44 | 23.31 MiB |

Graph collection plus PPO updating used about 1.48 times the fresh MLP's
time on the actual resulting trajectories. Its PPO updates took about 2.66
times as long. These are complete-workload costs, not identical-state kernel
comparisons. Development evaluation and checkpoint I/O are additional and
are recorded separately in the raw review.

Retained rollout payload excludes Python objects, models, Adam, activations
and allocator caches. The real training process's peak RSS was not recorded;
the 4 GiB container limit is a bound, not a measured peak. The isolated probe's
402,548 KiB process peak includes sequential models and caches. No OOM occurred.
Dense aggregation was 1.96-6.64 times faster than padded gathering in these
probes, with maximum output error 2.38e-7 and gradient error 1.91e-6.

The graph is more expensive despite similar parameter count. A bigger graph,
attention, recurrent memory, new rewards or action heads would add cost and
make this comparison harder to interpret, so they are excluded. Teacher loading
and checkpoint serialization stay outside inference.

## Verification and operation

Tests cover scalar graph updates/gradients, joint node/incidence relabeling,
strict geometry loading, complete discard likelihoods, profile rejection,
scoped RNG, exact PPO next-update resume, rotating learner scopes, exact original
prefix actions/RNG, common labels, shard-boundary warm-up resume, outer
cooperative pause and tiny end-to-end final accounting. Container lint,
format, compile, full tests and dependency boundaries gate launch.

Use the learning-checks target. The entry point is python -m
scripts.compare_architecture with --output, --original-checkpoint and
--economic-checkpoint. --resume uses stored parameters and rejects overrides.
A pause.request asks the owned writer to commit and stop at a bounded
cohort/shard/evaluation boundary. Remove it only after explicit user resume.
Exclusive locks reject competing writers. Original checkpoints and raw results
are preserved; there is no automatic model promotion.

The independent audit checked all 138 saved checkpoint files (132 PPO files and
six imitation-resume states), final versions/hashes/counters/configs, three full
coverage cycles per case, all milestones and all 16,128 final roles. It rebuilt
18 matched greedy roles with invariants and exact policy samples/RNG advancement,
checked unchanged model/learner RNG, and regenerated bounded training/validation
teacher shards against their numeric labels.

A separate review checked per-replicate and original-reference intervals,
exact fresh initializations and warm-up recipes, all 93 main/frozen source
hashes, the unchanged original source checkpoint, actual timing totals and
the public development decision suite. Practice, teacher, development, final
and engineering-test seeds were reconciled. Interpreting final identity
economic_enabled flags is relevant to observation4 arms; the original reference
uses observation2 and CatanEnv with no economic suffix.

Evidence:
[full final comparison](../output/architecture-20261015/run/comparison.json),
[independent audit](../output/architecture-20261015/run/verification.json),
[separate review and curves](../output/architecture-20261015/review.json),
[summary tables](../output/architecture-20261015/summary-tables.json),
[decision diagnostics](../output/architecture-20261015/decision-diagnostics.json),
[feature sensitivity](../output/architecture-20261015/feature-sensitivity.json),
[job log](../output/architecture-20261015/job.log),
[container receipt](../output/architecture-20261015/container-final.json).

Only verified task-created runtime helpers are removed. All saved checkpoints,
teacher shards, raw metrics, smoke checks, probes and earlier project results
are preserved. The optional graph remains available for inspection/reproduction
and adds no graph computation to the default model. The shared fixed-cohort
implementation replaces duplicated runner code. No engine, default bot, old
checkpoint, website or AWS resource was changed by this comparison.
The completion follow-up is paused. The hash-verified helper removals are
recorded in the [cleanup receipt](../output/architecture-20261015/cleanup.json).

## Recommendation

Keep the original as the baseline. Do not adopt this graph for training speed
or claim a strength benefit from the economic features. Keep both experimental
profiles available for controlled follow-up, rather than discarding evidence
or switching the default based on a favorable replicate.

The next proposed task is to investigate why the bounded teaching pass leaves
immediate-win behavior weak and why PPO does not retain it, then predeclare a
small teaching/retention experiment. Stronger teaching coverage for critical
decisions is a plausible direction, not an implemented or proven remedy.
More graph capacity or a longer repeat is not authorized by this report.

## Plain-language explanation

We gave two copies of the old bot the same practice. One got a calculator
showing resource production and building shortfalls; the other saw zeros in
those extra boxes. This tested whether the calculator helped learning.

Then we taught two new bots with identical examples. One read the board as a
long list. The other passed messages between connected tiles, intersections
and roads. They had almost the same number of adjustable settings and received
the same practice opportunities.

Both faced the same fresh test boards. The calculator did not produce a
reliable improvement, and the message-passing bot tied the fresh list-reading
bot against greedy while costing more to run.

The new bots still often failed to choose an immediate winning action in saved
game positions. Their brief teaching helped some, but ordinary practice did
not preserve that skill reliably. This points us toward improving how they
learn and keep important decisions before making their brains more elaborate.
The old bot stays in place. Everything needed to inspect the experiment has
been saved, and no further training is running.
