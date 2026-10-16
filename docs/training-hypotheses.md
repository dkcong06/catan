# Placement, point rewards and resource memory

Completed and independently verified on 2026-10-15. None of the three recipes
reliably improved wins against greedy opponents. Placement teaching and point
rewards reduced wins against random opponents relative to ordinary continuation.
Resource memory had a small, uncertain random-win gain over continuation, with
no extra greedy wins. Every continued model had a lower observed random win rate
than the preserved original. No checkpoint was promoted and no further training
was started after the comparison.

The user authorized this approximately three-hour comparison after reviewing
[the design](training-hypotheses-plan.md). The owned job ran from
2026-10-15T02:51:18Z to05:16:56Z (about145.6minutes), then exited0/noOOM.
Verification and explanation followed within the original overall target.
The [frozen protocol](../output/hypotheses/run-20261014/protocol.json) is the
authority for actual settings. This report describes measured outcomes and
distinguishes them from proposed follow-ups.

## What is compared

| Arm | Change from control | Training objective |
| --- | --- | --- |
| B | Ordinary continuation from the original selected weights | Real win |
| P | Gentle lessons on initial settlements and roads before continuation | Real win |
| R | Small credit for net progress from two to ten victory points | Real win plus net point progress |
| K | Memory of publicly inferable opponent resource counts, including uncertainty | Real win |

All use the same model2 architecture and original update-350 weights. B/P/R have
zero memory inputs; K has active memory. Original files are read-only to the job.
P changes its weights through setup-only actor imitation, with no critic labels.
No greedy action replacement is hidden inside a learned rollout. No model is
automatically promoted based on this experiment.

## Final results

Each row has1344games against greedy opponents and1344against random opponents:
64fresh boards per replicate, three replicates, all three/four-player seats.
The original is evaluated on exactly the same role/board schedule.

| Model | Greedy wins | Greedy win rate | Random wins | Random win rate | Mean own final VP vs greedy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preserved original | 15 | 1.12% | 1039 | 77.31% | 3.775 |
| B: ordinary continuation | 12 | 0.89% | 953 | 70.91% | 3.588 |
| P: placement lessons | 6 | 0.45% | 837 | 62.28% | 3.293 |
| R: point-progress reward | 13 | 0.97% | 870 | 64.73% | 3.487 |
| K: public resource memory | 12 | 0.89% | 987 | 73.44% | 3.769 |

There are13440evaluation roles in total. All greedy games completed. K-r1 has
one random-opponent cutoff at5000actions: seed2000201042, four players, candidate
seat0, own7VP. Across K's random games there are987real wins,356completed losses
and1cutoff. The win-rate denominator includes that cutoff; it is neither a win
nor a completed loss. Every other row/opponent has1344completed games.

Primary changes below are versus B, in **percentage points**, with paired
whole-board intervals. Greedy intervals use the predeclared98.33% confidence
allocation across three hypotheses; random intervals are95% regression checks.

| Recipe | Greedy change [adjusted interval] | Random change [95% interval] | Meets declared gate? |
| --- | ---: | ---: | --- |
| P | -0.45 [-1.26, +0.30] | -8.63 [-13.02, -4.32] | No |
| R | +0.07 [-0.82, +0.97] | -6.18 [-11.16, -1.26] | No |
| K | 0.00 [-0.89, +0.89] | +2.53 [-1.49, +6.70] | No |

R's single extra greedy win is inconclusive. K's random interval includes both
a loss and a gain. These data do not establish an improved bot. A separate,
descriptive95% comparison to the original gives random changes B -6.40points
[-10.04,-2.75], P -15.03[-19.35,-10.79], R -12.57[-17.11,-8.18], and K
-3.87[-7.59,-0.22]. These additional intervals are posthoc context, not more
primary tests. The control regression means we must investigate the continuation
recipe as well as the individual features before allocating more practice time.

Per-replicate wins (each cell greedy/random,448roles per opponent):

| Model | Replicate0 | Replicate1 | Replicate2 |
| --- | ---: | ---: | ---: |
| Original | 5/340 | 3/343 | 7/356 |
| B | 3/316 | 6/318 | 3/319 |
| P | 1/295 | 3/245 | 2/297 |
| R | 4/260 | 5/296 | 4/314 |
| K | 5/335 | 3/323 | 4/329 |

The three training replicates change practice/shuffle seeds and case order;
they share the original initialization. They do not represent three independently
initialized networks. Intervals resample192whole board clusters with all seven
roles together,10000draws/RNG71. They are conditional on these trained models
and this budget; they do not cover all possible training variability. Pairing
fixes initial boards and seat RNG identities, not identical later chance events
after policies choose different actions.

Raw results: [comparison](../output/hypotheses/run-20261014/comparison.json),
[independent reconciliation](../output/hypotheses/reconciliation.json),
[descriptive original/VP comparisons](../output/hypotheses/statistical-context.json).

## Implementation choices

rl/hypotheses.py extends the environment through an optional factory. It appends
30 normalized lower/upper bounds in relative opponent order, padding absent
seats. Default CatanEnv still provides its original 1505 floats and win rewards.
The experimental factory explicitly advertises observation3 and its reward
identity, so incompatible data cannot silently enter PPO.

KnowledgeActorCritic retains the original first matrix dimensions and every old
parameter, adds a bias-free 30-to-128 projection, and initializes that projection
to zero. This adds 3840 parameters, taking the standard model to 248795. Keeping
the original matrix shape preserves its numeric behavior during conversion;
initialization preserves the caller's Torch RNG. Original checkpoint loading
uses model1; experimental checkpoints use model2/PPO2 with explicit observation
and reward identities. All restore paths use restricted weights-only loading.

The tracker owns one small int16 interval array per active learner game, together
with public totals and bank stocks. It retains no full game history or hidden-hand
enumeration. Public receipts, payments, trades and card effects update intervals.
Production reuses the engine's scarcity rule on public-only inputs, without
reading real opponent hand composition. The thief and victim can identify their
own transferred resource; other observers widen bounds. Exact public bank stocks
mean discarded resource types are inferable here. This is the existing project's
information profile, rather than an assertion about every online Catan platform.

Feature queries tighten a copy against hand totals and aggregate supply and do
not mutate the tracker. Different query schedules therefore cannot change its
future behavior. The observer's exact hand pins its own bounds; other observers
receive zero extra features. Test-only truth checks compare bounds against hidden
hands, and never run in normal training. The engine itself has no new fields,
random draws, whole-state copies, ML imports or geometry changes.

The R reward is actual_win + 0.1 * delta clip((own_VP-2)/8,0,1). A point between
two and ten contributes 0.0125; losing points subtracts it. Setup earns no bonus,
and award gain/loss cycles cancel. This intentionally adds a small final-point
objective: it is not a proof that the win-optimal policy is unchanged. With
gamma=lambda=1, intermediate progress telescopes to net progress. A cap remains a
real cutoff with critic bootstrap, never a fabricated win.

The collector adds rewards from every physical action to the latest recorded row
of each affected learner seat. That retains off-turn award changes until the next
personal decision. Default terminal-only collection retains its original branch.
Experimental row storage uses the advertised width; successor observations,
demonstrations, policy value calls, updater preflight and persistence all carry
the corresponding profile. Recorded opponent rows remain excluded from PPO.

## Fixed practice and teaching

Default design: B/P/R/K each run three practice/shuffle seeds, 480 charged active
seconds per case. The 96-minute total includes each case's setup costs. One
complete cohort may finish just after its allowance; the overrun is reported.
P pays the full shared teacher-generation charge in each replicate plus its own
fitting/validation cost, so reusing data does not grant free preparation time.

Eight games form a cohort. Player counts alternate between three and four;
learner seats rotate. The opponent cycle is greedy, frozen original, greedy,
random. The frozen opponent is common to every arm, including P, rather than
P's own taught checkpoint. Weights remain fixed during each collection call.
The cap is 5000 physical actions. PPO uses four epochs, minibatches of 256,
learning rate 0.0003, actor/value clipping 0.2, gradient norm 0.5, KL threshold
0.02, no entropy bonus, gamma=lambda=1. Collection/evaluation use one Torch CPU
thread and updates use four. There is no compilation or GPU claim.

P uses 512 setup segments, 64 independent setup-validation segments and 32
separate main-turn validation games. Half use each player count. Setup segments
stop after 12/16 actions and are not counted as completed games. Only non-forced
actual teacher actions become imitation labels. Bounded numeric NPZ shards retain
hashes, seeds and original episode records. P uses two passes, learning rate
0.0001, minibatches 256 and phase weighting. Separate Adam/shuffle state serves
imitation; PPO starts with fresh optimizer state for every arm. Main-turn
validation measures collateral shared-network changes, without training on it.

Reserved seeds are disjoint: teacher 260000000/260010000, validation
260100000/260110000, main validation 260200000/260210000; practice begins
400000000+1000000*replicate with eight seeds per committed update; development
probes use the 420000000 region. Final boards begin 2000200000+1000*replicate.

## Recovery, evaluation and interpretation

The whole experiment and each case have exclusive Linux file locks. Atomic
checkpoint/state commits occur after every cohort; stored remaining active time
is the recovery authority. A pause.request file asks the process to finish the
current cohort or evaluation chunk, save, close and exit. Removing that file and
starting --resume restores the same config, optimizer, shuffle and seed schedule.
Paused downtime does not consume the new charged_active_v1 budget. The inherited
legacy deadline is explicitly informational in this runner. An unexpected kill
can replay only uncommitted work; do not run two writers or silently increase
the allowance. Teacher preparation is bounded and precedes the practice clock.

Evaluation commits chunk cursors with checkpoint digest, seed/role schedule,
policy settings and factory identity. Resume checks the completed role prefix
before continuing. Final checkpoint is primary; no tuning or test win selects
another checkpoint. Each replicate uses 64 fresh boards with all seven learner
roles across three/four players, against both greedy and random opponents. This
is 2688 games per arm, plus 2688 for the frozen original: 13440 final games.
True wins, losses and caps are reported separately.

Compare P/R/K to B with paired whole-board bootstrap intervals. The three primary
greedy comparisons use approximately 98.33% individual intervals (Bonferroni
alpha allocation). A useful candidate needs at least two percentage points of
greedy gain, a positive change in at least two replicates, adjusted lower bound
above zero, and random-play 95% lower bound above minus five points. These are
exploratory, conditional comparisons of the actual trained models; three practice
seeds do not establish universal training stability. Small rare-win differences
with intervals spanning zero remain inconclusive.

## Checks and measured tradeoffs

The implementation has tests for zero-projection numeric/RNG behavior, old/new
checkpoint profiles, private steal observations, public production scarcity,
complete-game reward telescoping, off-turn reward assignment, read-only feature
queries, exact resumed updates, evaluation cursors and a capped four-arm smoke.
The smoke validates mechanics, not strength. Final gate logs and measurements
are saved under output/hypotheses; these gates passed before launch.

The first isolated eight-game probe measured about 45% extra collection time for
K. The initial implementation unnecessarily read bank/hand snapshots on every
move and tightened already-exact intervals. The retained optimization skips
resource-neutral actions, uses existing public totals/bank as the previous view,
uses numeric-array tolist rather than repeated scalar conversions, and skips
constraint tightening until an unobserved steal introduces uncertainty. Monopoly
must still update even with unchanged bank; tests caught that edge case. VP
snapshots are owned tuples so in-place engine updates cannot erase the delta.
Final remeasurement on the same 2635-action/566-learner-row trajectories measured
median B 0.90394s, R 0.93063s and K 1.17817s across three alternating repeats.
R adds 2.95%; K adds 30.34%, down from the initial 45.19%. K does not meet the
aspirational 5% target. Its retained code is small and bounded; its actual extra
cost is charged to K's allowance, so the comparison tests benefit per fixed
compute budget. A profile attributes the remaining cost primarily to interval
features and event updates; more elaborate caching risks query-dependent state
and is deferred. The identical trajectory checks passed and all arms used
3835230 bytes of raw retained rollout arrays in this probe, excluding Python,
Torch, Adam and transient allocations.

The isolated 112-role K evaluation completed all games with no caps in 25.8304s.
Extrapolating that rate to 13440 roles gives about 51.7 evaluation minutes, or
147.7 minutes with the 96-minute practice allowance, before audit/overruns. This
is an estimate on unchanged weights and fresh development boards, not a runtime
guarantee for trained policies. The full Python suite passed 1028 tests in288.58s.
A final resume-configuration check was then added; all9 affected tests passed in
the final frozen image in13.84s, alongside lint/78-file formatting and source
compilation. The original checkpoint SHA and every catan source file are
unchanged. Measurements: output/hypotheses/preflight-initial.json, preflight.json,
tracker-profile.log and test logs. The task-created timing helper was removed;
cleanup.json records its verified path/hash. All raw results/models are retained.

The owned container catan-hypotheses-20261014 used networknone/four CPUs/4GiB
and frozen image da75ab63a33e6864062d41f76ea9255d4c3df2dc0d53234b8e92b5054986cb5e.
It exited successfully after final evaluation. The historical pause procedure
is to create run-20261014/pause.request, wait for committed paused status/exit,
and resume the identical image/config only after authorization. Docker-freezing
does not implement the new cooperative pause protocol. This experiment needed
no pause or crash recovery. Its completion follow-up is paused; no local training
or requirement to keep the computer awake remains for this task.

## Actual work and compute accounting

All12cases exhausted their480charged-second allowances; no extension or extra
training was used. The last complete cohorts exceeded their allowances by
0.206-3.727seconds each,17.212seconds total. Preparation/conversion and each
case's own fitting costs were charged before practice. P pays the13.181seconds
of shared teacher generation in every replicate, plus6.761/6.347/10.831seconds
of its own fit/validation and2.562seconds conversion. Total P preparation charges
are22.503/22.089/26.573seconds; other arms pay2.562seconds each.

| Arm | Updates by replicate | Total practice games | Learner rows | Physical actions |
| --- | --- | ---: | ---: | ---: |
| B | 237,230,158 | 5000 | 655678 | 2414724 |
| P | 230,230,148 | 4864 | 638303 | 2382582 |
| R | 239,236,223 | 5584 | 732409 | 2715420 |
| K | 196,194,151 | 4328 | 564541 | 2071117 |

The totals are2472PPOupdates,45080optimizersteps,19776completed practice games,
9583843physical actions and2590931learner rows. There were52837learner discard
decisions and4556learner wins against the scheduled mixture, with zero practice
cutoffs. The metric named wins counts a game's real winner, regardless of seat;
19776is therefore completed games, not the bot's win count. Maximum retained
raw rollout payload was19549002bytes, about18.64MiB, excluding Python/Torch/Adam.

The job's internal active counter is8750.20seconds (145.84minutes), versus
about145.63minutes from container UTC timestamps. We retain both clocks rather
than force equality. Individual wall timings also vary: several later cases
completed fewer updates in their equal allowances. Thermal or competing-work
effects are UNCONFIRMED. Consequently the isolated matching-trajectory probe,
not the unequal trained trajectories, establishes the measured feature overhead.
Equal time deliberately measures usefulness per compute budget, not per update.

## What the decisions show

An explanatory replay used the first three final boards in each replicate and
all seven seats against greedy:63games per policy. It compared original/B/P/R/K
and P-warm, the already-saved placement checkpoint before PPO. This added no
training and did not select a checkpoint. All315final-model outcomes/fingerprints
matched saved evaluations; P-warm adds63diagnostic-only games. These are
posthoc explanations on a bounded subset, not additional strength claims.

| Policy | Start pips | Starts with no ore | Starts with no wheat | Cards produced/game | Development buys total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 15.92 | 25/63 | 6/63 | 48.46 | 89 |
| B | 16.49 | 22/63 | 20/63 | 44.68 | 75 |
| P-warm | 15.08 | 30/63 | 22/63 | 39.44 | 132 |
| P final | 16.62 | 29/63 | 20/63 | 43.48 | 41 |
| R | 15.25 | 39/63 | 5/63 | 41.79 | 61 |
| K | 14.65 | 32/63 | 21/63 | 40.65 | 114 |

P's teacher likelihood improved: independent setup-validation loss4.9505 fell
to2.5940/2.5936/2.5967, and main-turn validation loss3.4713 fell to about2.74.
There were7168training labels,896setup-validation labels and5231main-validation
labels. Those losses measure agreement with the teacher, not wins or critic
quality. The fresh diagnostic openings already have worse pip/resource coverage
immediately after teaching; it would be inaccurate to say teaching first improved
the opening and PPO merely erased the gain. Later P recovered pip totals, while
many starts still lacked ore/wheat and realized production remained below the
original. This gentle512-segment/two-pass recipe did not reliably learn useful
general opening play. It does not establish that better placements cannot help.

Teaching did broaden locations and increase port access. Original's first
settlement was vertex19 in35/63starts, B43/63 and P-warm7/63. P-warm averaged
0.365initial ports versus original0 and final P0.143. But its crude count of
two-edge expansion sites was5.89 versus original6.16 and final P5.41. These
counts ignore site quality, rival races and material availability. More diverse
actions or more ports alone do not establish a better start. P's final mean
greedy score was0.295VP below B, descriptive95% interval[-0.451,-0.136].

R did not raise final points versus B: mean change-0.100VP, descriptive95%
interval[-0.250,+0.047]. On the small decision subset it often started without
ore (39/63), despite usually having wheat. The data show failure of this fixed
coefficient/budget, not proof that intermediate credit is useless or that ore
coverage alone caused the result. We did not change the coefficient after seeing
test outcomes. Reward telescoping and off-turn bookkeeping passed separately.

K's bounds covered the actual opponent hands at every checked candidate query:
5222queries/66295resource cells,65.21%exact bounds and mean interval width0.501
cards. Hidden truth was used only in offline assertions, never policy features.
All three K projection matrices changed (norms0.7695/0.7375/0.7012); B/P/R's
matrices remain exactly zero. Thus the added branch learned nonzero weights,
but that alone does not establish useful decisions or causal benefit. K's final
greedy score is0.182VP above B, descriptive95% interval[+0.035,+0.332], yet
still approximately the original score and with no extra greedy wins. Its
30.34%isolated collection cost matters before adopting it as default.

Economic problems remain across recipes. For example, B selected a settlement
or city on50of88main decisions where one was legal, versus original71of84;
it chose a road instead30times. In seed2000200000/3p/seat0, B had9VP and legal
settlements48/52 at actions273/274, but chose roads19/40; road probability mass
was97.76%/95.92% and settlement mass0.20%/0.46%. That game eventually won at10VP,
so these two declined immediate builds must not be counted as two lost games.
In a separate original trace on that board, action213 traded four sheep for
one brick while increasing its simple purchase deficit from2to3. Such metrics
flag resource choices for investigation; they ignore longer-term strategy and
are not proofs that every flagged trade/road is wrong.

Detailed records and six annotated traces are in
[decision-diagnostics.json](../output/hypotheses/decision-diagnostics.json) and
output/hypotheses/decisions-{original,B,P-warm,P,R,K}.jsonl. Policies received
only their normal own/public observations. Offline reports also inspect actual
states to explain outcomes; they are diagnostic artifacts, not imitation data.

## Independent verification and cleanup

The independent audit reconciled all12case checkpoint digests/versions, original
digest and conversion weights, observation/reward/config identities, optimizer
finiteness/counters, every metric total, practice schedules, charged allowances
and overruns. It checked every teacher shard hash and replayed all608teacher/
validation segments (19631actions), including exact labels, masks, observations,
discard choices and full game/policy RNG fingerprints. Teacher, practice and
final seed families are disjoint; all13440evaluation roles, win/loss/cutoff/own-VP
summaries, per-replicate deltas and adjusted primary intervals reconciled.

A separate invariant-enabled neural replay exactly reproduced210final roles
and104807actions across all15model/replicate jobs, including game/policy RNG
fingerprints and unchanged model tensors. The explanatory replay added378games/
128246actions, of which315games matched existing final greedy records. The one
random cutoff also exactly reproduced alone and in its original eight-seed
chunk. An initial extra cutoff-check command omitted policy_seed and used the
collector default0; its mismatch was an audit invocation error, corrected to
stored17. Both attempts' logs are retained. No application or raw-result change
was needed. These are bounded live replays, not a claim that all13440games were
re-executed with invariants.

All78main Python sources still match the frozen image and all10engine sources
match their pre-experiment manifest. Original checkpoint SHA256 remains
67a0ad501d00b5031e3e53f3f87383f96a21b1224ce812907bf6f29d8c1de8a2.
This completion work added documentation and temporary offline audits, with no
production-source changes after the tested freeze. We retained the small optional
experimental implementations, legacy compatibility, tests, all checkpoints,
teacher shards, raw metrics/results and job logs. Removing these optional
features now would break reproducibility without speeding the default engine.
Only verified task-created timing/audit helpers were removed; cleanup receipts
record exact paths and hashes. No unrelated code, output, site or AWS work was
removed. PDF documentation was preserved.

Evidence: [launch/check receipt](../output/hypotheses/verification.json),
[job log](../output/hypotheses/job.log),
[replay checks](../output/hypotheses/replay-verification.json),
[cutoff check](../output/hypotheses/cutoff-verification.json),
[completion receipt](../output/hypotheses/completion.json).

## What to try next, without launching more training

First isolate why ordinary continuation regresses: compare already-saved earlier
practice checkpoints on reserved development boards, inspect policy drift and
economic action coverage, and verify that any proposed change preserves the
original's random strength. Final boards used here are no longer fresh for tuning.
The experiment did not compare PPO step sizes, entropy schedules, alternate
reward coefficients, independent network starts or larger practice budgets.
Those remain untested choices, not explanations established by this run.

There is also a concrete schedule-coverage limitation: player counts alternate
3,4,3,4 while opponent types cycle greedy,original,greedy,random. Their alignment
therefore gives three-player practice only greedy opponents and four-player
practice only original/random opponents. All four arms share this rule, so the
controlled recipe comparisons remain valid under the frozen protocol, but the
practice distribution is not a balanced opponent mixture within each player
count. Greedy test wins split3p/4p as original8/7, B7/5, P5/1, R4/9 and K9/3
(denominators576/768). These rare counts do not establish the schedule as the
cause of poor play. Decoupling the two schedules to cover every player-count/
opponent combination is a specific next controlled change. We did not alter
the frozen run or start another experiment after discovering this limitation.

Then test explicit resource-by-roll production and action-local affordability/
connectivity features in one small, controlled representation experiment. The
opening learner currently has to infer these relationships through a flat
observation and shared network. A separate setup head or constrained teaching
can also be compared if it reduces interference; neither has been implemented.
For economics, build a small legal-action diagnostic set around immediate wins,
productive purchases, trades and discards, and measure accuracy before a long
run. Any teacher should include realistic states and remain subject to held-out
play, rather than treating lower supervised loss as success.

Keep public resource memory available as an optional ablation. Its point result
is a reason to study it, not to combine all three recipes or claim it works.
Profile the remaining interval-query cost before making it a default. After a
focused development comparison produces a credible gain without regression,
freeze a new protocol and evaluate once on newly reserved final boards. No such
follow-up or hyperparameter search is authorized by completing this run.

## Plain-language explanation

We gave copies of the same bot comparable practice time: ordinary practice,
starting-position lessons, small point rewards, or memory of visible card events.
We then tested them on new boards. None became reliably better at beating greedy.
The original also remained better against random players than every practiced
version. More practice with this recipe is therefore not a reliable solution.

The starting lessons made the bot imitate its teacher more closely, but did not
teach consistently stronger starts. Small point rewards did not produce higher
finishing scores. Card memory tracked uncertainty correctly and showed a small
point-score signal, but cost more computing time and did not add greedy wins.

The next useful work is to make resource production and good purchases easier
for the network to recognize, and understand why continued practice worsens
existing skills. The engine and measurement system passed their checks; that is
different from having trained a strong player. We kept the original and every
experimental checkpoint so future comparisons can be honest and reproducible.
