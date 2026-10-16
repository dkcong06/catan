# Teaching quality and retention of winning decisions

Investigation completed 2026-10-15. This is a diagnostic study of existing artifacts, not a new training experiment or a new playing-strength evaluation. No optimizer steps, architecture changes, reward changes, model promotion, or engine changes were made.

The evidence supports fixing the teaching pipeline and testing retention before adding model complexity. There are three separate problems: the teacher sometimes demonstrates an avoidable mistake, the students only partially learn its behavior, and some learned winning-decision skill declines during subsequent PPO. The imitation-to-PPO value-function handoff also has a measurable problem, but this investigation does not prove that it causes the later decline.

## What was inspected and verified

All work used the main project and the previous frozen image, sha256:b2bb7931088f3554eafafaad66a3a53ac705de6a37550ae6a1b180b2b7b892ab. Previous results were mounted read-only. New evidence is under [output/teaching-retention-20261015](../output/teaching-retention-20261015/).

The investigation:

- Replayed every saved teacher game: 128 training games and 16 validation games. All 51,750 engine actions ran with invariants enabled. Every saved demonstration column, including complete discard counts and conditional logs, matched its regenerated data exactly. Every episode fingerprint, including game and policy RNG state, matched.
- Scored 115 existing checkpoints: the original, both continuation arms across three replicates and nine saved milestones, and fresh MLP/graph models across three replicates, their cold states and nine PPO milestones. Milestones were 0, 1, 2, 4, 8, 16, 28, 56 and 84. Update 0 means after imitation for the fresh models.
- Reused the frozen 725-location development suite, including 85 verified immediate-win locations. Fresh models were also evaluated on all 2,708 held-out teacher labels at every milestone; cold, taught and final states additionally used the 22,644 training labels.
- Replayed 18 existing eight-game practice cohorts exactly: each fresh model immediately before updates 1, 29 and 84. Their 144 episode fingerprints, row counts, actions, discard counts, payload sizes and PPO target statistics matched the saved run. The replay inspected gradients without applying them.
- Independently checked 2,635 probability calculations with a scalar float64 NumPy softmax; reproduced all six original cold/warm validation-loss receipts; verified that their value-head weights were unchanged during imitation; and replayed five concrete teacher mistakes with independently enumerated deterministic counterfactuals and exact final RNG checks.
- Verified all 587 previous JSON/JSONL/NPZ/checkpoint inputs remained unchanged. All 93 frozen Python sources matched the image; the main project's corresponding source hashes also matched.

The temporary helpers disabled Adam steps and PPO updates. Gradient calculations used autograd.grad, which did not populate parameter gradient buffers. Each inspected updater's weights, Adam state, shuffle RNG and counters were hashed before and after and stayed identical. The clone used to inspect the first minibatch consumed only a copied shuffle generator.

The core evidence is [teacher-summary.json](../output/teaching-retention-20261015/teacher-summary.json), [checkpoint-scores.json](../output/teaching-retention-20261015/checkpoint-scores.json), [cohort-gradient-diagnostics.json](../output/teaching-retention-20261015/cohort-gradient-diagnostics.json), [summary.json](../output/teaching-retention-20261015/summary.json), and the three verification receipts in that directory.

## 1. Teacher quality: valid data can still teach the wrong move

The teacher is the project's deterministic greedy heuristic, reference:greedy-v1. These are engine-generated demonstrations of that heuristic, not top-level human play and not optimal-play labels. Its choices are legal and reproducible; that does not make every choice strategically correct.

The immediate-win audit tries each legal deterministic road, settlement, city or knight action on an isolated engine copy and checks whether it ends the game with the acting player winning. It does this in MAIN or ROLL when the player has at least eight VP. It never looks ahead into a dice roll, steal or development-card draw. It does not cover every possible win, particularly free-road phases and uncertain card purchases. Counts below are opportunities within this deliberately limited, verified scope.

| Dataset | Non-forced/complete-discard labels | Verified win opportunities | Teacher takes a winning move | Teacher misses a winning move |
|---|---:|---:|---:|---:|
| Training, 128 games | 22,644 | 143 | 104 (72.73%) | 39 (27.27%) |
| Validation, 16 games | 2,708 | 14 | 12 | 2 |

All 39 missed training opportunities had a winning road available. Their taught choices were 13 other roads, 10 settlements, 10 trades, five development-card purchases and one city. There were 140 winning road alternatives across those 39 locations; that is an action count, not 140 separate decision opportunities. Repeated opportunities within one game are not independent samples.

This is a specific consequence of the current greedy implementation. During MAIN it checks settlement and city purchases before considering roads. Its road score estimates access to attractive future settlement sites; it does not first prioritize taking Longest Road to win immediately. Consequently, imitation can penalize a student for selecting an available guaranteed win when the teacher selected something else.

One independently replayed example is board seed 853000011, action offset 172, player 0:

| Available choice | Own VP afterward | Own longest path | Immediate result |
|---|---:|---:|---|
| Teacher's settlement, catalog 75 | 9 | 4 | Game continues |
| Any of roads 1, 3, 26 or 28 | 10 | 5 | Player wins |

The teacher later won that particular game, but delaying a guaranteed win is still an inferior instruction under the current terminal-win objective. A second example, seed 853000050, has the teacher trading at offset 170 and buying a development card at 171 while several roads would win immediately. Counterfactual card draws were not inspected. The road wins alone establish the avoidable delay.

The [five complete examples](../output/teaching-retention-20261015/counterfactual-examples.json) cover every observed missed-choice family. Their chosen actions, exact catalog alternatives, own scores, road lengths and eventual winners are recorded. The source states and policy RNGs were unchanged by the hypothetical actions.

On the separate 85-location development win suite, greedy chooses a verified winning action at 76 locations and misses nine. Thus its win-priority weakness also appears outside its own demonstration trajectories.

This is instructional error, not corrupted storage: all saved labels exactly reproduce the intended teacher.

## 2. Critical decisions receive very little nominal teaching weight

Only 143/22,644 training labels, 0.632%, contain a verified immediate-win opportunity. Two fixed imitation passes expose a model to those rows 286 times, including 78 presentations of the 39 bad teacher choices.

The current imitation loss balances phases separately within each shard. Every shard represents eight phases, so each phase receives approximately one eighth of the nominal sample weight.

| Training phase | Rows | Share of labels | Nominal phase weight |
|---|---:|---:|---:|
| MAIN | 11,734 | 51.82% | 12.5% |
| ROBBER_MOVE | 3,497 | 15.44% | 12.5% |
| ROBBER_STEAL | 2,092 | 9.24% | 12.5% |
| ROLL with a real choice | 2,006 | 8.86% | 12.5% |
| DISCARD, complete choices | 1,127 | 4.98% | 12.5% |
| SETUP_SETTLEMENT | 896 | 3.96% | 12.5% |
| SETUP_ROAD | 896 | 3.96% | 12.5% |
| ROAD_BUILDING | 396 | 1.75% | 12.5% |

Phase balancing prevents common phases from dominating, but it does not distinguish a routine MAIN action from a game-winning MAIN action. The verified-win rows receive only 0.226% of nominal weighted sample mass after balancing, about 36% of their unweighted share. This includes the incorrect teacher choices. It is a weight calculation, not a measured fraction of gradient magnitude or Adam movement; minibatch tails and optimizer history also matter.

Simply repeating this same dataset longer would repeat both its correct and incorrect instructions. Adding a tactical bucket is useful only after correcting its labels and accepting every verified winning alternative, rather than insisting on one arbitrary winning road.

Forced flat actions were correctly removed from imitation, since their actor likelihood has zero gradient. Complete discard labels were retained and reproduced exactly. There is no evidence that forced-row filtering or incomplete discard likelihood caused these findings.

## 3. Students learn only part of the teacher before PPO starts

Each fresh model received exactly two shard passes, 194 imitation Adam steps, learning rate 0.0001 and phase-balanced joint-action likelihood. The teacher data and order recipe were shared across the MLP/graph comparison. No validation-based checkpoint selection occurred.

The mean held-out highest-probability flat-action agreement was:

| Model family, mean of three runs | Cold | After imitation | After 84 PPO cohorts |
|---|---:|---:|---:|
| MLP | 23.86% | 45.69% | 34.38% |
| Graph | 22.38% | 45.68% | 37.15% |

This agreement excludes the 116 complete-discard validation rows; its denominator is 2,592 non-forced flat decisions. It is deterministic argmax agreement with the teacher, not the probability that a stochastic rollout chooses its label and not a game win rate.

Warm MLP training agreement is approximately 46.5%, close to its validation agreement. Poor agreement already exists on the taught boards, so ordinary train-versus-validation overfitting does not explain it. This does not prove that extra passes, a new loss, better features or more capacity would solve it. Those alternatives require a controlled learning-curve experiment after fixing the teacher.

The independently reproduced phase-weighted held-out imitation losses improved from roughly 1.991 to 1.706–1.710 in all six runs. Lower loss proves some learning occurred; it does not demonstrate that the teacher was fully learned or that the resulting policy is strong.

## 4. Winning-decision retention declines during PPO

The main diagnostic below is the sum of the policy probabilities assigned to all verified winning actions at each location, averaged over the same 85 fixed locations and three model replicates. It answers: if this exact location were presented, how likely is the stochastic policy to take a known immediate win?

It is not a game win rate.

| Saved stage | MLP winning-action probability | Graph winning-action probability |
|---|---:|---:|
| Cold | 28.48% | 28.48% |
| Taught, before PPO | 40.40% | 41.51% |
| PPO 1 | 40.59% | 42.89% |
| PPO 2 | 39.29% | 42.81% |
| PPO 4 | 39.59% | 41.36% |
| PPO 8 | 38.23% | 36.92% |
| PPO 16 | 35.30% | 36.70% |
| PPO 28 | 35.99% | 35.14% |
| PPO 56 | 34.37% | 34.24% |
| PPO 84 | 34.22% | 35.32% |

Final winning-action probability was below the taught checkpoint in all six individual runs:

| Run | Taught | Final | Change |
|---|---:|---:|---:|
| MLP r0 | 40.39% | 36.65% | -3.74 percentage points |
| MLP r1 | 40.16% | 27.83% | -12.33 points |
| MLP r2 | 40.66% | 38.18% | -2.48 points |
| Graph r0 | 41.42% | 36.78% | -4.65 points |
| Graph r1 | 41.72% | 35.86% | -5.86 points |
| Graph r2 | 41.38% | 33.31% | -8.08 points |

The finding also appears on teacher-origin validation positions, although there are only 14 verified-win locations there. Their mean winning probability falls from 44.81% to 36.31% for MLP and from 46.43% to 28.01% for graph. These are separate state origins from the original-policy decision suite, which reduces one source-distribution concern but does not create a large independent test.

This is not a uniform collapse immediately after imitation. Mean winning probability rises at the first PPO checkpoint. Some runs improve at intermediate stages; later behavior is uneven. The retained checkpoints localize a decline that was previously visible only at endpoints.

The continued original-policy controls are also not uniformly degrading on this diagnostic: zero-suffix r0 ends above its initial winning probability, while other continuation runs differ. Therefore “PPO always forgets everything” is not supported.

There is no behavior-cloning replay, explicit teacher-policy anchor or tactical retention objective in the subsequent PPO loss. It optimizes current rollout returns and value fitting, not continued agreement with the demonstration policy. The KL safeguard compares against the current cohort's behavior policy on collected states, not against the original taught policy on previously seen winning locations. A succession of locally limited changes can therefore move away from taught behavior.

## 5. The imitation-to-PPO critic handoff needs attention

The actor and critic share features. Imitation has no critic target. All six value-head weight/bias rows stayed bit-for-bit identical between cold and taught checkpoints, while the shared features changed. Consequently, an unchanged value head no longer receives the same inputs.

Mean value estimates on the same 2,708 teacher validation positions changed as follows:

| Run | Cold mean value | Taught mean value | First PPO critic/actor gradient-norm ratio |
|---|---:|---:|---:|
| MLP r0 | -0.260 | -3.467 | 73.12x |
| MLP r1 | -0.092 | -1.825 | 27.64x |
| MLP r2 | -0.149 | 0.447 | 8.36x |
| Graph r0 | 0.005 | -1.446 | 20.54x |
| Graph r1 | 0.425 | 7.525 | 156.06x |
| Graph r2 | 0.250 | 3.363 | 81.17x |

The linear critic is unbounded; an estimate outside [0,1] is not a serialization bug or proof of illegal policy inputs. But true completed-game targets in this experiment are 0 or 1, and these estimates show that actor-only warm-up does not prepare a calibrated value function for PPO.

The ratios are measured on each saved first PPO minibatch. “Critic” includes the actual value coefficient 0.5, so the scaling already used by the trainer is included. Combined global gradient clipping multiplies these first gradients by approximately 0.00217–0.06921. Both direct shared-feature critic gradients and global clipping are relevant; actor gradient magnitude alone does not describe the actual update.

All six first eight-game cohorts were against greedy. The exact replays contained 48 losses, zero learner wins and zero verified learner win opportunities. With gamma=lambda=1 and true terminals, the recomputed value targets were exactly zero for those learner rows. Advantages still varied because they subtract the model's value estimates; global normalization then scales that variation. Thus an all-loss batch can change the actor even without a successful learner trajectory in that batch. Finite-sample baseline variation is not successful tactical instruction.

This does not establish the critic as the cause of later forgetting. The first update actually improves mean winning probability, and the inspected first-minibatch gradient directions are not uniformly harmful to the fixed win probe.

Across all 18 inspected cohorts, the combined gradient has a locally aligned direction with the probe's winning-action loss eight times and an opposed direction ten times. These are local gradient inner products under an infinitesimal SGD step, not applied Adam updates, later minibatches, or a controlled causal intervention. The exact ratios and cosines are in the raw gradient report. No such diagnostic gradient was applied.

The earlier critic-shielding comparison did not establish a strength improvement and should not be reinterpreted as having solved this issue. Its continued-original-policy starting point differs from this fresh actor-only imitation handoff.

## 6. The problem occurs in actual saved games too

The 18 exact historical cohort replays give a bounded illustration of opportunities actually encountered:

| Model state used for replay | Opponent | Games across six runs | Learner wins | Verified immediate-win opportunities | Winning choices taken |
|---|---|---:|---:|---:|---:|
| Before PPO update 1 | Greedy | 48 | 0 | 0 | 0 |
| Before update 29 | Frozen original | 48 | 3 | 27 | 3 |
| Before update 84 | Random | 48 | 22 | 56 | 18 |

A learner can miss an immediate win repeatedly and still win later, including through an uncertain development-card purchase or an unlabelled free-road phase. Winning choices here therefore do not equal total game wins.

These are selected, predeclared milestone cohorts from the existing practice schedule, not new held-out evaluation games. Their opponents and player/seat roles differ by schedule; the three rows are not a fair before/after strength comparison. They establish that the retained models sometimes pass up guaranteed wins in real trajectories, not only on synthetic or off-policy fixtures.

The previous fresh-model final result remains five wins against greedy out of 1,344 roles for each family. This investigation does not change that result, establish a new improvement, or select a different checkpoint.

## What we can conclude, and what remains uncertain

Confirmed:

- The teacher's priority order produces avoidable immediate-win mistakes, concentrated here in Longest Road opportunities.
- The stored labels, RNGs, full discard likelihoods, episode accounting and replayed PPO targets are consistent with the implementation. Fixing teacher quality does not mean repairing corrupted data.
- Correct tactical examples are scarce, phase balancing does not protect their importance, and the current short warm-up only partly imitates even the training labels.
- Fixed-location winning probability declines from taught to final models in every fresh replicate, with corroboration on a small teacher-origin validation set.
- Actor-only warm-up can substantially move critic predictions through shared features, creating a large corrective critic gradient at the handoff.

Not established:

- Which single change would improve held-out game wins, how much each mechanism contributes, or whether retention loss is the main cause of low greedy win rates.
- That more imitation epochs, replay anchoring, critic calibration, intermediate rewards, a hierarchical decoder or a larger graph will solve the problem.
- That all deviations from greedy are bad. Greedy itself makes mistakes; preserving its raw choices without correction can preserve bad play.
- That these development probabilities are general game win rates or statistically independent observations. Neither the 85 nor the 14 win locations are a new final test; repeated decisions can share a board and game.

## Recommended next implementation and comparison

Keep the existing MLP and decoder for the first repair. The immediate problem has concrete teaching and retention defects that do not require a larger network to investigate.

1. Version and fix the teacher's immediate-win priority. Compute deterministic tactical outcomes from an explicitly sanitized public board/ownership view plus the acting player's own information. Prioritize guaranteed terminal wins before ordinary build/frontier preferences. Cover Longest Road acquisition and transfer, blocked paths, settlements/cities, Largest Army, pre-roll knights and free-road phases with engine-oracle tests. Never peek at future dice, development draws, steals or opponents' private hands.
2. Create new, separately seeded demonstration and validation data. Preserve the old teacher tag, shards and comparison. Reject known incorrect tactical labels. In a tactical target, credit the whole set of verified winning actions rather than one preferred road. Keep the existing development suite and these inspected validation positions out of training. Explicitly measure ordinary-phase coverage and realistic high-VP opportunities.
3. Run a bounded warm-up learning-curve check with the MLP. Track both training and untouched validation agreement, per-phase joint likelihood, and tactical argmax correctness plus total winning probability. Predeclare a tactical acceptance gate before launching; do not stop merely because average imitation loss decreases. Decide whether additional passes or different sampling help using that learning curve, rather than assuming the old two-pass budget was sufficient or that capacity is the bottleneck.
4. Test a stable value-function handoff without moving the actor. A candidate is a small head-only value fit on completed on-policy practice returns with actor features and actor outputs frozen; check actor probabilities remain exact. It is a proposed experiment, not an implemented fix or calibrated-probability guarantee. Treat it separately from the retention ablation, or apply the same verified handoff to both retention arms so it cannot confound their comparison.
5. From one identical corrected taught checkpoint, compare ordinary PPO with PPO plus a bounded, fixed-weight replay/teacher-anchor term using training-only data. Keep opponent schedule, rollout seeds, reward, decoder, cohorts and compute accounting matched. Preserve early checkpoints and examine retention across all phases. Do not anchor to the flawed old teacher, choose the weight on final results, or train on diagnostic fixtures.
6. Judge success on fresh complete games against fixed greedy and stronger reference opponents, all seats and both player counts, with multiple training seeds and paired whole-board comparisons. Keep immediate-win retention as a diagnostic alongside real wins/losses/cutoffs, own VP and collection/update costs. Only reconsider architecture if a clean teacher cannot be learned or retained despite these controlled checks.

No new training allowance, recipe or experiment was launched by this investigation.

## Plain-language explanation

The bot has a teacher, but that teacher sometimes overlooks a move that would win immediately. For example, it builds a settlement to reach nine points even though a different road would give it Longest Road and ten points. We currently ask the student to copy that mistake.

The student also leaves its lessons only partly learned. It gets better at spotting winning moves after teaching, but still chooses poorly on many examples. During later practice, some of that improvement fades.

There is another rough transition: the part estimating how promising a position is changes indirectly during teaching without being taught the right answers. Early practice spends a lot of effort correcting those estimates. That is worth fixing and testing, but it is not yet proven to explain the lost winning skills.

The practical next move is to improve the teacher, give enough attention to decisive positions, verify the student has learned them, and test a way to preserve those lessons during practice. A bigger model would currently receive the same imperfect lessons.

## Evidence and cleanup

Detailed per-shard results: [teacher-audit.json](../output/teaching-retention-20261015/teacher-audit.json). Per-position numeric annotations are stored separately from the unchanged teacher shards and are diagnostic-only, not training data.

Verification receipts: [verification.json](../output/teaching-retention-20261015/verification.json), [gradient-verification.json](../output/teaching-retention-20261015/gradient-verification.json), [independent-verification.json](../output/teaching-retention-20261015/independent-verification.json). Their helper hashes identify the exact temporary programs used. The first replay helper needed a NumPy Boolean JSON-serialization correction before its successful complete pass; no original data or training process was affected.

Three owned temporary Python helpers were removed only after checking their final hashes against these receipts. All raw diagnostic JSON/NPZ evidence, previous checkpoints, teacher shards and original results were preserved. Application Python code and dependencies were unchanged. Container Ruff/format/compile checks passed for the temporary helpers; execution, exact replay, invariants and independent numerical checks were the relevant verification gates. The full application suite was not rerun for this documentation/diagnostic-only task.

All owned diagnostic containers exited and were removed. No new heartbeat or continuing training/computer-awake requirement was introduced. [completion.json](../output/teaching-retention-20261015/completion.json) records final artifact hashes and cleanup.
