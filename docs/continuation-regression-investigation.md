# Why continued training lost useful behavior

Completed and independently verified on2026-10-15. The strongest localization
evidence is a substantial later-game component to the random-opponent regression:
fixing openings leaves a deficit, while replacing MAIN decisions removes the
observed pooled deficit. Sparse winning feedback, cumulative policy concentration
and shared critic correction are measured warning signs and plausible mechanisms.
Their causal contribution still needs a controlled training ablation.
No optimizer steps, new training, model promotion or production-source changes
were performed. The exact historical onset cannot be recovered from the retained
checkpoints.

## Question and evidence boundary

The [completed four-arm comparison](training-hypotheses.md) found that ordinary
continuation B fell from the original's77.31% random wins to70.91%, without a
clear greedy gain. This investigation asks what changed, whether the last update
caused the loss, and whether openings or subsequent economic decisions explain
the regression.

An important limit appeared immediately: the experiment retained each initial
checkpoint and only its final two updates. The available B weights are
0/236/237,0/229/230 and0/157/158. Intermediate checkpoints were removed by the
documented bounded retention policy during training. Full per-update metrics
survive, but do not contain model weights or historical evaluation curves.
Consequently an exact first bad update cannot be recovered without another
controlled training experiment. We do not invent such a timeline from entropy
or training wins.

## Frozen method

The [protocol](../output/continuation-review-20261015/protocol.json) fixes32new
development boards3100600000..3100600031, both player counts/all seven learner
seats, random and greedy opponents, policy seed17 and5000physical actions.
Each of seven saved models has448roles: the common original plus final and
penultimate B checkpoints for each replicate,3136games total. These are
development diagnostics and are not a new untouched final test or a checkpoint
selection exercise. The earlier final reports remain unchanged.

We also inspect identical positions sampled only from the original's trajectories.
Numeric shards retain up to eight nonforced decisions per phase/category/chunk,
including complete discards. Every model sees the same observation, legal mask
and actual original action/prefix. This prevents a comparison of policy
probabilities from confusing changed positions with changed choices. The sample
is stratified, not an estimate of the natural frequency of every game phase.
PPO targets and rewards do not enter these likelihood-only probe shards, and
the shards are never fitted.

A separate phase-control diagnostic uses the first16development boards and
all seven seats against random opponents. For original and each final B model,
it compares normal play, replacing only setup decisions with greedy, and
replacing only MAIN decisions with greedy. This is1344games in total. Controls
consume each neural proposal before replacing the selected action, preserving
the established diagnostic RNG convention. These are hybrid evaluation
controllers, not trained models or eligible PPO trajectories. They can alter
all subsequent chance streams, so effects are conditional on the controller.

All win comparisons use paired whole-board bootstrap intervals,10000draws/RNG71.
The intervals are exploratory95% intervals without multiplicity correction.
The three B models share initialization and the same32development boards; they
are not independent initializations or96independent boards. Rare greedy wins
cannot establish a small strength difference.

The frozen local learning image is
da75ab63a33e6864062d41f76ea9255d4c3df2dc0d53234b8e92b5054986cb5e,
with network disabled/four CPUs/4GiB. Main-project inputs and every model are
mounted read-only. Offline helper code is temporary; measurements and raw
artifacts are preserved.

## What all625recorded continuation updates show

The complete B logs contain313greedy cohorts,2504games and28learner wins.
Of those cohorts287(91.69%) contained no learner win. All three replicates
have this sparse-success problem. With undiscounted Monte Carlo targets,
every action of a completed losing game has target0; actions of a winning
game have target1. There is no intermediate information saying which purchase,
discard or placement was good. The state baseline can still supply advantages,
but a small all-loss sample provides weak evidence for a winning strategy.

The last20greedy cohorts in each replicate still won only1,1and2games out
of160, compared with2,4and3in the first20. These are different, changing
training games and seats, not paired fixed-policy evaluations. They do not
identify a change point. Random-cohort wins similarly vary rather than forming
a universally monotone decline.

Mean logged entropy in the first20 versus last20greedy cohorts changed:

| Replicate | First20 | Last20 |
| --- | ---: | ---: |
| B-r0 | 0.3112 | 0.1789 |
| B-r1 | 0.3180 | 0.1635 |
| B-r2 | 0.3222 | 0.1767 |

Logged entropy includes forced decisions and changing positions; it is evidence
of concentration on sampled trajectories, not by itself proof of harmful
overconfidence. The fixed-position probe below is the more direct comparison.

Mean per-update sampled KL is approximately0.0049-0.0074 across replicate/
opponent groups. Maximum recorded full-rollout KL is0.01777, below the0.02
threshold, and only8of625updates stopped early. A minibatch can trigger a stop
even when the final aggregate KL is smaller. PPO's local clipping/KL checks
do not constrain cumulative change across hundreds of updates. These metrics
do not show an obvious numerical blow-up; they also do not establish good play.

Practice is unbalanced within player counts: the3/4alternation aligns with
greedy/original/greedy/random, so three-player cohorts always face greedy and
four-player cohorts face original/random. This shared limitation was established
in the previous report. The critic receives the public/own state, not an explicit
opponent-policy identifier. Learning under a mixed opponent distribution is
legitimate, but the present schedule offers few greedy successes and incomplete
coverage rather than a smoothly balanced curriculum.

## Exact first-batch replay and shared-network gradients

The three first greedy cohorts were reproduced from their saved initial weights,
board/policy seeds and learner seats:24games/7290actions, with every game and
policy RNG fingerprint matching the training log and invariants checked.
No PPO update was run.

All24games were losses. The initial critic's mean predicted rewards were
0.5517,0.2741and0.5233. Values are unconstrained reward estimates, not calibrated
win probabilities. Their actual target on these trajectories is0. Recomputed
same-seat targets and advantage means/standard deviations matched the recorded
statistics.

We used autograd only to measure the actor and weighted critic gradients on
the four shared trunk parameters, with no optimizer step or parameter.grad
accumulation. The first minibatch uses the exact saved local shuffle state.

| Replicate | Forced flat rows in cohort | Weighted critic/actor gradient norm, first minibatch | Gradient cosine |
| --- | ---: | ---: | ---: |
| B-r0 | 65.66% | 6.46 | +0.097 |
| B-r1 | 66.17% | 2.41 | -0.112 |
| B-r2 | 61.11% | 4.78 | -0.008 |

Across all rows, the corresponding norm ratios are7.68,3.57and7.42.
The critic coefficient0.5 is included in these ratios. Forced actions have no
actor-choice gradient but still have useful value targets. They are retained
for temporal links and critic learning; dropping them outright would change
the contract.

This is direct evidence that initial shared-feature gradients are dominated
in norm by critic correction, often pointing largely independently of the
actor gradient. It is a plausible interference mechanism when adapting a
self-play-trained model to mostly losing greedy games. It is not proof that
critic interference causes the win regression: Adam preconditions gradients,
the directions evolve, and no alternative update rule was trained here.
Fresh-Adam continuation is another experimental choice that was not ablated.

Evidence: [telemetry](../output/continuation-review-20261015/telemetry.json),
[first-batch replay and gradients](../output/continuation-review-20261015/initial-gradient-review.json).

The three last cohorts also replayed exactly from their penultimate checkpoints:
another24games/12957actions. Recomputed returns and final-versus-previous sampled
KL/clip fractions matched the saved logs. B-r0's last cohort faced greedy and
won0/8; B-r1/B-r2 faced the frozen original and each won2/8. Last-cohort shared
critic/actor gradient ratios were1.34/1.80/1.95, smaller than at startup. Initial
critic dominance is not a claim that the same ratio persists throughout training.
See [last-cohort checks](../output/continuation-review-20261015/last-gradient-review.json).

## Saved checkpoints on fresh development boards

Each entry below has224random and224greedy games on the same32boards/all seats.
Update numbers are continuation-run counters; original means the preserved
original update350 weights migrated without changing predictions.

| Checkpoint | Continuation update | Random wins | Random rate | Greedy wins |
| --- | ---: | ---: | ---: | ---: |
| Original | 0 | 173 | 77.23% | 1 |
| B-r0 previous | 236 | 161 | 71.88% | 0 |
| B-r0 final | 237 | 159 | 70.98% | 0 |
| B-r1 previous | 229 | 177 | 79.02% | 4 |
| B-r1 final | 230 | 168 | 75.00% | 2 |
| B-r2 previous | 157 | 170 | 75.89% | 1 |
| B-r2 final | 158 | 170 | 75.89% | 1 |

All3136games completed with no cutoff. All final B models have a lower observed
random rate than the original here, but B-r1's penultimate model is slightly
above it. The last update changes random wins by-2,-9and0. It is therefore
incorrect to say the last update always causes the decline, or that every
intermediate checkpoint must have been worse. We do not select or promote
B-r1's penultimate checkpoint from this exploratory sample.

The average final B random rate is73.96%, versus77.23% original. This small
development panel has wider uncertainty than the earlier192-board final
comparison: pooled change-3.27percentage points,95% interval[-8.78,+2.53].
This panel alone does not establish a nonzero average decline. Last-update
changes are-0.89[-5.36,+3.57],-4.02[-10.27,+2.23]and0.00[-6.70,+7.14]
percentage points. None establishes a single destructive final step.
The earlier final-test regression (-6.40points,95% interval[-10.04,-2.75])
is not replaced by this new exploratory panel. Conditional32-board greedy
change averages0.00points, interval[-0.45,+0.45]; these rare counts do not show
an improved greedy policy.

## Changes on identical positions

The56numeric probe shards contain3607stratified decision rows:448each setup
settlement/setup road/robber move/other MAIN,432each MAIN with a legal settlement
or city/discard,435nonforced ROLL,337steal and179free-road choices. Some initial
positions can repeat across opponent roles; these are rows, not3607independent
games. All extra resource-memory features are zero, as required for B.

| Model | Opening-settlement entropy | Opening KL from original | Probability of settlement/city when legal |
| --- | ---: | ---: | ---: |
| Original | 1.371 | 0 | 82.09% |
| B-r0 final | 0.389 | 2.229 | 78.17% |
| B-r1 final | 1.164 | 1.176 | 79.00% |
| B-r2 final | 0.600 | 2.718 | 80.66% |

These are masked distributions on exactly the same observations. Entropy and
KL use natural logarithms. Mean total-variation distance for opening settlements
is0.481/0.429/0.572: substantial probability mass has moved to different choices,
even though each individual PPO update stayed close to its immediately previous
policy. Lower entropy is not automatically bad; the whole-game controls below
test consequences.

There is no single identical economic mistake across all replicates. On the
448common MAIN positions without an immediate settlement/city, original assigns
9.07%probability to buying development. B-r0 assigns1.10%, B-r1 20.52%, B-r2
9.17%. End-turn probability is13.72%original versus20.87%/0.23%/11.67%.
Thus B-r0 largely suppresses card buying on these states, whereas B-r1 follows
a different economic pattern. These means include states where each action is
unavailable; they compare identical masks and are not purchase success rates.

Robber-move distributions also shift substantially (mean total variation
0.720/0.748/0.825). Discard choices change: original chosen-vector mean log
likelihood falls by5.31/0.95/8.26nats under the final B models. This latter
quantity is likelihood of original choices, not exact current joint KL.
Discard entropy is the sum of conditional entropies on recorded original
prefixes, not an enumerated full joint entropy. Drift alone does not show that
every changed robber/discard action is worse. These phases are not replaced
by the setup/MAIN controls.

See [common-state metrics](../output/continuation-review-20261015/fixed-state.json)
and [probe identities](../output/continuation-review-20261015/probes/manifest.json).

## The critic fits losses but underestimates random-game returns

On fresh games, all completed-row targets were independently checked against
the actual learner winner: target1if it won,0otherwise, with correct same-seat
closures. No reward/return bookkeeping mismatch appeared in these checks.
The model and optimizer were not updated.

| Model | Mean V,3p random | Mean V,4p random | MSE,3p random | MSE,4p random |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.649 | 0.443 | 0.280 | 0.312 |
| B-r0 | 0.065 | 0.122 | 0.605 | 0.517 |
| B-r1 | 0.138 | 0.146 | 0.531 | 0.472 |
| B-r2 | 0.204 | 0.190 | 0.492 | 0.458 |

Actual row-weighted winning returns in these random groups range about0.657
to0.774, well above the continued models' predictions. In contrast, greedy
target errors fall substantially, because nearly every greedy game is lost.
The critic has adapted toward low values without becoming a useful random-game
return predictor. Negative values are permitted by the model and are not by
themselves a software error.

These are decision-weighted errors on each model's own trajectories, not
identical-state causal comparisons or game-level calibrated win rates. Long
games contribute more rows. Also, the acting policy samples actor logits; it
does not choose actions by consulting V. The concern is the critic's influence
on training advantages and shared features, rather than a pessimistic value
directly instructing the bot to end its turn during evaluation.
See [return diagnostics](../output/continuation-review-20261015/calibration.json).

## Opening versus later-game controls

Each cell is wins out of112games against random opponents on16common development
boards/all seats. There were no cutoffs in any of the1344control games.

| Model | Normal neural play | Greedy setup only | Greedy MAIN only |
| --- | ---: | ---: | ---: |
| Original | 89 | 103 | 107 |
| B-r0 | 79 | 90 | 103 |
| B-r1 | 79 | 95 | 109 |
| B-r2 | 84 | 92 | 110 |

Average B-minus-original gaps, with16whole-board95% descriptive intervals:

| Controller | Gap in percentage points [interval] |
| --- | ---: |
| Normal play | -7.44 [-12.50,-1.79] |
| Same greedy openings | -9.52 [-15.18,-4.46] |
| Greedy MAIN decisions | +0.30 [-4.76,+5.65] |

The opening intervention improves absolute wins for every model, including
the original, but does not remove the continued models' deficit. We independently
replayed all448setup segments (four models times112roles) and verified identical
complete post-setup states and all engine/per-seat policy RNG streams for each
matched role. The residual deficit therefore provides evidence for a post-setup
component, rather than merely different starting positions.

MAIN replacement raises each continued model substantially and removes the
observed pooled deficit relative to the original under that same controller.
MAIN covers building, trades and relevant card choices; it leaves neural
openings, discards, robber choices and pre-roll card choices in place.
These experiments localize a useful intervention rather than identify one bad
action type or prove that every other phase is harmless. Hybrid win rates near
the ceiling can hide remaining differences. They also test random-opponent
regression, not a solution to the original greedy-opponent weakness.

Observed normal-play purchase decisions support that localization. Original
selected a settlement/city on472of602MAIN decisions where one was legal (78.4%).
B-r0/B-r1/B-r2 did so on563/943(59.7%),331/480(69.0%)and485/742(65.4%).
Road-over-build counts were72original versus131/94/106. These are decision
opportunities, often repeated within a turn/game; they are not counts of lost
games or proofs that every declined purchase was wrong. On identical positions
the purchase-mass shift is smaller, showing why trajectory and fixed-position
metrics should both be reported.

Simple material-deficit-worsening trades account for26.4%oforiginal trades and
about29.0%/29.0%/30.3%ofB trades. The heuristic ignores reachability and long-term
plans. Discard deficit comparisons are similar across models; strong likelihood
drift is not evidence that discard quality uniformly worsened. At9VP the
diagnostic also counts declined legal point-building decisions, but many such
games later win. No claim is made that each flagged decision caused a loss.

Greedy setup raises production pips from about18to24, while its oreless/wheatless
start counts can be higher than the original's. This is another reason not to
treat resource coverage or pips alone as an opening-quality oracle. Total cards
produced per game also depend on game length: a faster winner may produce fewer
total cards. We do not interpret a lower whole-game total as a lower production
rate without normalization by opportunities.

Evidence: [phase controls](../output/continuation-review-20261015/controls.json),
[paired intervals](../output/continuation-review-20261015/paired.json),
[identical setup boundaries](../output/continuation-review-20261015/setup-boundaries.json).
Twelve annotated candidate-view traces are saved alongside these artifacts.

## What this establishes, and what it does not

The start weights/zero extension are correct. First and last recorded training
cohorts replay exactly. Winner/return assignment, same-seat links, checkpoint
identity and logged final policy changes reconcile on the checked data. We found
no such bookkeeping mismatch to fix. This does not prove the absence of every
possible learning or engine defect.

The recipe receives very few successful greedy trajectories, becomes more
concentrated and moves far from its original policy over many modest updates.
Its critic becomes a worse random-return predictor while correcting the initial
overestimates on losing greedy trajectories. Startup shared critic gradients
are substantially larger in norm than actor gradients. Together with the
phase controls, these facts motivate protecting useful policy features and
improving the practice distribution before another long run.

We have not trained lower learning rates, entropy bonuses, a reference-policy
KL anchor, separate actor/critic trunks, critic warm-up, or different batch sizes.
Nor did we test keeping original Adam moments. These remain hypotheses, not
proven fixes. Changing several of them together would prevent attribution.

## Recommended next controlled experiment

First add bounded early checkpoint retention and a fixed development evaluation
curve: initial, the first few updates, then logarithmic milestones plus final.
Retaining only the last two generations saved disk space but prevents this
investigation from locating onset. Bound the milestone count and charge evaluation
overhead explicitly; continue using new untouched boards only for final testing.

Make the common practice schedule cover every player-count/opponent/learner-seat
combination, with explicit coverage tests. Changing just one modulo expression
can create another seat/opponent alignment. The existing unbalanced frozen run
and all results should remain intact.

The first targeted training ablation should compare the same balanced control
with a variant that prevents the critic loss from updating shared actor features.
The actor can still update its shared features and the critic can still learn
its value head. This directly tests the measured interference hypothesis, has
bounded extra computation, and does not require a larger model or extra engine
work. It may reduce value representation learning; that tradeoff must be measured,
not assumed beneficial. Compare development win curves, random retention,
critic error, shared gradients and fixed-state drift under the same budget/seeds.
Keep the reward, actor learning rate and entropy setting fixed in that ablation.

Only after that comparison should we decide whether to test production/action-local
features, an exploration term or an original-policy anchor. No such implementation
or training was started by this investigation. The original remains the reference.

## Verification, efficiency and cleanup

The main analysis executed3136saved-model games and1344phase-control games in
1127.25seconds (about18.8minutes), plus bounded independent checks. This is an
offline diagnostic runtime, not a training throughput benchmark. Rollout arrays
were released after each eight-seed chunk. The common-state corpus is56numeric
NPZ shards/about23.24MB,3607rows, without arbitrary pickle or hidden-opponent
inputs. No full-game history was added to engine or learner hot paths.

All3136roles/summaries, model/probe hashes and exploratory paired intervals were
independently reconciled. Full invariant/RNG replay reproduced28additional
evaluation games/10470actions, and all48first/last training games/20247actions.
The448normal diagnostic games also matched their batched evaluation fingerprints.
Invariants ran on the first board of every controller/role and at every independently
checked setup boundary. We do not claim invariant re-execution of all4480games.

All78production/test Python sources match the previously tested frozen image,
and original checkpoint SHA remains
67a0ad501d00b5031e3e53f3f87383f96a21b1224ce812907bf6f29d8c1de8a2.
Four temporary offline helpers passed F/E9 lint and compilation, then were removed
after final receipts. Existing full-suite results remain the prelaunch1028tests
plus9final guard tests; this docs/offline investigation did not rerun that suite.
All raw diagnostic results, probe shards, traces, logs and old checkpoints are
preserved; production code, training parameters, prior reports and monitoring
state were not changed. No ongoing training or awake-computer requirement remains.

See [independent verification](../output/continuation-review-20261015/verification.json)
and [completion/cleanup receipt](../output/continuation-review-20261015/completion.json).

## Plain-language explanation

The bot did not simply get a worse starting position. Even when we gave the
original and continued bots exactly the same opening and random-number state,
the continued bots still did worse. When a simple greedy controller made their
later building/trading decisions, that observed gap disappeared.

During practice, the bot almost never sees a win against greedy players. Its
score predictor learns to expect little success, and changing that predictor
also changes the shared features used to choose moves. Meanwhile the bot becomes
more committed to different habits, without learning to beat greedy. Those are
warning signs; the next comparison should test whether shielding move-selection
features from score-predictor corrections helps.

We cannot say exactly when the damage began because the earlier run did not keep
enough intermediate models. Future runs should keep a small, planned set and check
the same development games along the way. This investigation changed no model:
it provides a more specific, testable next experiment instead of another long
run with several guesses changed at once.

