# Direct critic-gradient comparison

2026-10-15. Implementation, bounded pilot and independent verification are complete.
The gradient shield did not improve the primary greedy-opponent result. The
original bot remains the baseline; no new checkpoint was promoted.
The main project is C:\Users\david\Documents\Coding\Catan.

## Question and scope

The [continuation investigation](continuation-regression-investigation.md) found
that continued training could damage later play against random opponents, even
with identical strong openings. Greedy wins were sparse. At the first actual
minibatches, weighted critic gradients in the shared network were larger than
actor gradients. These observations motivate an experiment; they are not causal
proof, and gradient norms do not measure Adam parameter changes.

The new comparison asks whether blocking the critic's direct gradient into
shared features reduces that regression. Both arms also receive a corrected,
balanced practice schedule and early retained checkpoints. Because those two
changes are common to both arms, this experiment does not separately estimate
their benefit against the historical schedule.

## Implementation choices

The actor scores legal moves; the critic estimates the eventual return. They
currently share two feature layers and a fused final actor/value matrix.
PPOConfig.critic_updates_trunk defaults to true, preserving existing behavior.
The head_only arm sets it to false. Its actor still updates shared features and
its critic still updates the final value row and bias. Only the critic's direct
backward path through shared features is removed. The optional knowledge branch
obeys the same rule, although this pilot uses the original model1 without it.

The shield computes the value head from detached shared features, then uses
the original fused value for the numerical forward result:
values.detach() + (head - head.detach()). This preserves forward values,
log probabilities and entropy exactly. A direct detached-head alternative is
benchmarked separately. There are no new parameters, observations, actions or
reward definitions, and inference still uses the original forward method.

This is not full actor/critic independence. Global gradient clipping uses the
combined norm; value-head gradients can therefore affect the common clipping
scale. The same Adam optimizer contains both heads. Actor feature updates can
also change critic predictions. Separate optimizers, separate trunks and
separate clipping are different experiments and are deliberately deferred.

The balanced scheduler enumerates seven player-count/learner-seat roles against
four opponent slots: greedy, frozen original, greedy, random. Every 28 complete
cohorts covers each role once per slot. Duplicate greedy slots intentionally give
it half the practice games. A cycle has 12 three-player and 16 four-player cohorts;
this weights individual seat roles equally rather than player counts equally.
Successive cycles rotate the order, and seed streams derive from the committed
update. The legacy scheduler remains the default.

Milestones preserve and evaluate updates 0, 1, 2, 4, 8, 16, 32 and 56. Checkpoint_every=1
also retains the current/previous/best generations under the existing bounded
retention rule. Up to 32 explicitly configured milestones can be retained by
TrainingRun. Each curve records its checkpoint SHA256. Checkpoints save the
training flag, Adam state and shuffle RNG. Legacy checkpoints missing new
settings load with historical defaults.
New checkpoints include the optional training-config field and should be read
with the updated code; unchanged tensor layouts do not promise that older
application versions can read newly extended configuration dictionaries.

Evaluation now retains only the candidate seat's temporary rollout rows. Every
opponent still acts, consumes the same RNG and produces the same episode
results. No opponent storage enters learning. This removes unused observations,
masks and final views from evaluation batches, while preserving raw result
records and fingerprints. Debug invariants remain outside ordinary training.

The Docker context excludes sites and .cursor as well as existing output/tmp
exclusions. The independent website checkout and editor configuration need not
be copied into a learning image. Neither is modified by this task.

## Predeclared pilot

The pilot starts each of six cases from the preserved u350 model:
67a0ad501d00b5031e3e53f3f87383f96a21b1224ce812907bf6f29d8c1de8a2.
Warm starts reset Adam and update counters identically. The frozen-original
opponent is also that policy. There is no imitation, resource-memory input,
reward shaping, entropy change or model promotion.

| Setting | Fixed value |
| --- | --- |
| Arms | coupled, head_only |
| Replicates | 3 matched practice/shuffle seeds |
| Training allowance | 56 cohorts per case, 8 games per cohort |
| Physical practice games | 2,688 across 6 cases |
| Opponent slots | greedy, original, greedy, random |
| Development | 4 boards per replicate, 14 roles per board, 8 milestone points |
| Final evaluation | 16 fresh boards per replicate, 14 roles per board, 3 policies |
| Final roles | 2,016 total; 336 per opponent per policy |
| Primary checkpoint | Last committed u56, regardless of development curve |
| Primary contrast | head_only minus coupled greedy win rate |
| Regressions/context | Random wins; each arm versus frozen original |
| Inference/update CPU threads | 1/4 |
| Container | 4 CPU limit, 4 GiB limit, network disabled |

The seven roles per opponent are all seats in both three- and four-player games.
Each arm's final model faces the same test boards and roles within a replicate.
All three replicates have disjoint practice/development/test streams:

| Purpose | Seed schedule |
| --- | --- |
| Practice | 500000000 + replicate*1000000 + 8*committed_update + game_index |
| Development | 1007000000 + replicate*1000 + [0,4) |
| Final | 2007000000 + replicate*1000 + [0,16) |
| Policy/shuffle | policy 17; shuffle 901 + replicate |
| Preflight only | 560000000..560000007 and 560100000..560100001 |

This comparison holds the number of training cohorts constant to isolate the
gradient mechanism. It does not hold rows, optimizer steps or wall time equal:
different policies can create different game lengths and early KL stops. Actual
costs must be reported. RunConfig's inherited 86400-second field is informational
in this runner; its clock is explicitly fixed_cohorts_v1, and progress reports
remaining cohorts. It does not authorize a 24-hour run. No extra training starts
after u56 or after inspecting test results.

The single greedy primary contrast uses a 95% paired whole-board bootstrap,
resampling 48 board clusters with all seats retained together. Per-replicate
contrasts and random/context intervals are descriptive. The interval conditions
on these trained models and does not replace uncertainty over many independent
training runs. With only 336 greedy games per arm and few expected wins, a small
count difference is likely inconclusive. Development curves reuse four boards
to locate possible timing; they are neither a checkpoint-selection rule nor a
large independent strength test. No automatic promotion is allowed.

## Recovery and ownership

scripts.compare_critic owns an exclusive root writer lock and each TrainingRun's
case lock. Protocol/config/source hashes are immutable on resume. The runner
alternates arm order by replicate, completes all training before final tests,
and uses resumable bounded final evaluation chunks. A pause.request or handled
SIGTERM/SIGINT asks it to finish the active cohort/milestone evaluation, commit
and exit. A human pause must not be resumed without explicit authorization.
Resume changes no cohort allowance, hyperparameters or final selection.

Original checkpoints are mounted read-only. Every retained case checkpoint is
hashed in the completed comparison. Cached completion rejects changed retained
models. Raw metrics, curves, final records and original results are preserved.
The machine must remain awake with Docker running while a local job is active.
There is no cloud bill or external API call in this experiment.

## Reproduction

Use a new empty output subdirectory. The supplied paths refer to the main project.
Review protocol.json before resuming a stopped run.

~~~powershell
docker build --provenance=false --target learning-checks -t catan-critic-checks .
docker run --rm --network none --cpus 4 --memory 4g --mount 'type=bind,source=C:\Users\david\Documents\Coding\Catan\output\critic-ablation-20261015,target=/results' --mount 'type=bind,source=C:\Users\david\Documents\Coding\Catan\output\training\run-20261014\checkpoints,target=/baseline,readonly' catan-critic-checks python -m scripts.compare_critic --output /results/new-pilot --initial-checkpoint /baseline/u00000350.pt
~~~

Add --resume with the same stored output and source after an authorized resume.
Do not change CLI settings on resume, remove a human pause implicitly, or run a
competing writer. TrainingRun's ordinary wall-deadline runner and the earlier
charged-active-time hypothesis runner retain their separate clock contracts.

## Verification and results

The owned container ran from 2026-10-15T07:26:13.686Z to 07:50:43.101Z:
24 minutes 29.42 seconds, exited 0, no OOM. The runner's recorded active duration
was 1,465.02 seconds; container startup/import overhead explains the different
clock boundary. The pre-final-reporting image passed 1,045 tests in 293.12 seconds.
After clarifying fixed-cohort progress reporting, all 53 affected tests passed
in 66.49 seconds. All 80 Python sources pass lint, format, compilation and the
Torch-free rl import gate. A focused
run initially found one tuple/list JSON report mismatch; it was fixed by
canonicalizing the milestone list before writing/returning final reports.
Tests cover exact actor/forward equality, critic gradient isolation, complete
balanced role coverage, strict config flags, old-format loading, checkpoint
retention, resumable matched Adam/shuffle/model state, identical filtered
evaluation records and tamper/config rejection.

The isolated benchmark used an actual 8-game greedy cohort, 582 learner rows and
the first 256-row shuffled minibatch, including 2 discards. Nine rotated batches of
30 forward/backward iterations at 4 threads had medians 9.433 ms coupled, 9.673 ms
shielded and 9.552 ms direct-head prototype. The shield's measured cost was 2.54%.
The direct prototype was only 1.24% cheaper than the shield and changed value
predictions by up to 2.98e-7 (gradients 5.96e-8); exact forward preservation was
retained. These timings exclude collection/gathering/optimizer work and do not
measure overall training speed. No optimizer steps occurred in the benchmark.

Three alternating evaluation pairs over 28 complete roles reproduced every
result/fingerprint exactly. Candidate-only storage reduced peak numeric payload
from 11.77 MB to 3.53 MB (70.0%) and total retained rows from 13,665 to 3,953. Median time was
5.613 s versus 5.452 s for all-row storage, about 2.96% slower in this small test.
The memory reduction is useful for future batching; no speed improvement is
claimed. Array payload excludes Python/Torch/allocator overhead. Raw samples
are retained in output/critic-ablation-20261015/benchmark.json.

Independent verification reconciled all 54 retained checkpoints, their update
versions/configs, all 48 milestone curves, initial tensor equality with u350,
fresh Adam states, shuffle seeds, final u56 selection and source hashes.
Every one of the 336 logged update events and 2,016 final roles was accounted for.
Practice, development and final seed families are separated; matched arms use
identical practice seeds and initial development results within each replicate.
Each 28-cohort cycle covers all 7 seat roles against all 4 opponent slots.

The audit independently recomputed every pooled/per-replicate/versus-original
paired interval. It then replayed 28 matched final-test games, covering every
trained model plus the original, both player counts and both opponents.
All 13,193 actions and full game/policy RNG fingerprints reproduced exactly
through direct engine replay with invariants after every action. Observations
matched exactly; log probabilities and values matched within the established
float32 reevaluation tolerance across batch sizes. Model tensors and local
shuffle states stayed unchanged. No audit optimizer steps occurred.

All 80 current Python sources match the frozen image. Of 78 pre-existing Python
files, only learning/evaluation.py, model.py, policy.py, ppo.py and training.py
changed. The 10 engine files and every rl file are unchanged. The new permanent
Python files are scripts/compare_critic.py and tests/test_critic_comparison.py.
Temporary benchmark/audit helpers are removed only after path/hash verification;
trained checkpoints, metrics, curves, raw final results and logs are retained.

## Held-out results and interpretation

All final games terminated normally: zero cutoffs. Each table entry has 336
matched games against that opponent, covering 48 fresh boards and every seat.

| Policy | Greedy wins | Greedy rate | Random wins | Random rate |
| --- | ---: | ---: | ---: | ---: |
| Preserved original u350 | 5/336 | 1.49% | 250/336 | 74.40% |
| Standard shared critic, u56 | 4/336 | 1.19% | 233/336 | 69.35% |
| Direct critic shield, u56 | 1/336 | 0.30% | 254/336 | 75.60% |

Changes below are percentage points, with paired whole-board 95% intervals.
Only shield versus standard greedy wins was the primary contrast; the others
are descriptive secondary comparisons.

| Contrast | Greedy change and interval | Random change and interval |
| --- | --- | --- |
| Shield minus standard | -0.89 [-2.08, 0.00] | +6.25 [-1.19, +13.69] |
| Standard minus original | -0.30 [-1.79, +1.19] | -5.06 [-12.80, +2.98] |
| Shield minus original | -1.19 [-2.38, -0.30] | +1.19 [-4.76, +7.44] |

The shield did not help the primary objective. Its random score is encouraging
as a retention signal, but the interval includes no difference, and its advantage
over the original is also inconclusive. The negative greedy secondary comparison
against the original is a warning. With so few wins, these percentile intervals
can be discrete or degenerate: replicate 2 has zero greedy wins for both new
policies and a [0,0] empirical contrast interval. That does not establish
equivalence or rule out unseen rare wins. Do not promote the shield or claim it
has solved critic interference on this evidence.

| Practice replicate | Standard/shield greedy wins | Standard/shield random wins |
| --- | ---: | ---: |
| 0 | 2/1 of 112 each | 77/84 of 112 each |
| 1 | 2/0 of 112 each | 75/82 of 112 each |
| 2 | 0/0 of 112 each | 81/88 of 112 each |

Random wins move in the same direction across the three replicates, but three
practice seeds do not establish a broad training-method improvement. The
bootstrap conditions on the trained models; it does not create more independent
training replicates.

## What early checkpoints revealed

The following development counts use 84 roles per opponent at each update:
12 boards across three replicates. They repeatedly reuse the same small board
set and were not used to replace the primary final checkpoint.

| Update | Standard greedy | Shield greedy | Standard random | Shield random |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 53 | 53 |
| 1 | 0 | 0 | 53 | 53 |
| 2 | 1 | 1 | 61 | 60 |
| 4 | 0 | 0 | 56 | 53 |
| 8 | 0 | 1 | 52 | 55 |
| 16 | 0 | 1 | 51 | 53 |
| 32 | 0 | 0 | 51 | 58 |
| 56 | 1 | 0 | 52 | 55 |

These curves do not show a simple immediate collapse or sustained greedy gain.
The apparent update-2 peak is a small reused development-set observation, not a
new selected winner. Development and final rates use different boards and should
not be compared as though they were a single learning curve. Retention now makes
bounded follow-up replays possible without reconstructing deleted weights.

Greedy feedback remains extremely sparse: standard practice won 7/672 greedy
games; shielded practice won 6/672. In 155 of 168 greedy cohorts (92.3%), the
learner received no winning example. This supports investigating the lack of
useful economic/action information and positive feedback; it does not establish
that an untested shaping reward or teacher will improve play.

## Actual work and performance costs

Across all cases: 336 updates, 6,101 optimizer steps, 2,688 completed practice
games, 725 learner wins, 1,267,736 physical actions, 351,297 retained learner
rows and 7,258 learner discard records. Every practice game ended in a real
winner; the 2,688 completion count includes opponent wins and is not the bot's
win count. Seven updates hit the existing KL early-stop condition.

| Measured phase across six cases | Seconds |
| --- | ---: |
| Collection | 470.96 |
| PPO updates | 131.91 |
| Development evaluation | 467.04 |
| Checkpointing | 32.46 |

The case phase counters exclude final evaluation and some setup/bookkeeping;
they are not an exhaustive decomposition of the container's 1,469.42 seconds.
Final evaluation followed all training and is included in the overall runtime.
The pilot also executed 2,688 development roles. Frequent measurement deliberately
costs time here. Routine future training can use fewer development points and a
larger checkpoint interval while retaining explicitly selected milestones.

Equal cohort allowances produced different trajectories: standard/shield runs
used 640,710/627,026 actions, 176,700/174,597 learner rows and 3,061/3,040 optimizer
steps. Their wall times therefore cannot isolate the shield's arithmetic cost;
the isolated benchmark serves that purpose. Collection and evaluation dominate
these measured phases. Their exact simulator-versus-inference split and GPU
performance were not measured by this task.

## Retained choices, rejected alternatives and next step

Retain balanced scheduling, bounded early checkpoint retention, the exact-forward
opt-in gradient control and candidate-only evaluation storage. Keep the ordinary
gradient default and the original playing model. The experimental switch remains
a small reusable research control; no diagnostics or replay sweeps enter inference.
The comparison runner and tests provide reproducible evaluation/recovery rather
than adding work to ordinary engine transitions.

The direct-head prototype's small timing difference did not justify its forward
rounding changes. Separating the networks/optimizers or changing clipping at the
same time was rejected for this experiment because it would change the question
and add parameters or controls. A longer blind repetition, selecting update 2
after seeing the curve, and promoting the larger random score were rejected:
the greedy objective did not improve and the relevant uncertainty is substantial.
The historical schedule was not rerun as a third arm, so this pilot cannot
attribute an independent benefit to balanced scheduling.

The recommended next implementation is a compact public economic representation:
resource production by dice roll, production gaps and the resource shortfall
for legal builds. These summarize existing legal observations instead of exposing
opponent private cards. Benchmark the encoding, verify every feature against the
engine and retain old checkpoint compatibility through an explicit versioned
profile. Then predeclare a small matched experiment; preserve the original and
measure random retention alongside greedy wins. A targeted legal-winning-action
check is also useful, but broad greedy imitation and reward shaping should not
be repeated indiscriminately given the earlier negative results. No such feature
change or additional training was launched by this task.

Evidence: [protocol](../output/critic-ablation-20261015/pilot/protocol.json),
[raw comparison](../output/critic-ablation-20261015/pilot/comparison.json),
[independent verification](../output/critic-ablation-20261015/verification.json),
[benchmark](../output/critic-ablation-20261015/benchmark.json),
[development curves](../output/critic-ablation-20261015/development-curves.json),
[practice totals](../output/critic-ablation-20261015/metrics-totals.json), and
[job log](../output/critic-ablation-20261015/job.log).
The [final checks](../output/critic-ablation-20261015/final-checks.json) verify all
12 actual practice cycles and unchanged main sources. The
[cleanup receipt](../output/critic-ablation-20261015/cleanup.json) records removal
of the two task-created helpers, with their exact paths and hashes. No job remains
active and no computer/Docker requirement remains for this completed task.

## Simple explanation

The bot has a move chooser and a position scorer, which share an internal
understanding of the board. Earlier results suggested that training the scorer
might be changing that shared understanding too strongly. We now have a switch
that lets the chooser learn that understanding while the scorer learns only how
to read it. We compare the switch on and off from the same starting bot.

Both versions practice a fairer mixture of seats and opponents. We save early
versions so we can see when learning helps or hurts. The final test uses new
boards and counts real wins. A short pilot can reveal a clear regression or
promising mechanism; it cannot make a tiny rare-win difference convincing.

The test is finished. Protecting the chooser from this part of the scorer's
training did not make it better against greedy opponents. It scored somewhat
better against random players, but that result is too uncertain to call a
reliable improvement. We kept the original bot. The next useful direction is
to help the bot see what a move produces or makes affordable, then test whether
that information helps it learn winning decisions.
