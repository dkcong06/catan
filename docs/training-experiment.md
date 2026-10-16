# Budgeted Catan training experiment

The first real experiment uses the optional CPU learner in the main checkout.
The engine and every rl module remain unchanged and NumPy-only. New orchestration
lives in learning.training, neural/reference evaluation in learning.evaluation,
and the command-line application in scripts.train_ppo.

## Verified results (2026-10-14)

The experiment finished successfully. The model selected using tuning boards won
188 of 448 held-out games (41.96%), compared with 65 of 448 (14.51%) before training.
The improvement was 27.46 percentage points; an independently recomputed paired
board-seed bootstrap interval was +24.33 to +30.36 points. This is evidence of
improvement on the declared protocol. Most of the gain came against random play:
the selected model won 187/224 games (83.48%) against random opponents, but only
1/224 (0.45%) against greedy opponents. It has learned useful play and remains
weak against the stronger existing reference policy.

| Held-out matchup | Games per model | Initial model | Selected update 350 | Latest update 1112 |
| --- | ---: | ---: | ---: | ---: |
| Three players, random opponents | 96 | 31 wins / 32.29% | 80 / 83.33% | 82 / 85.42% |
| Three players, greedy opponents | 96 | 0 / 0.00% | 1 / 1.04% | 1 / 1.04% |
| Four players, random opponents | 128 | 34 / 26.56% | 107 / 83.59% | 101 / 78.91% |
| Four players, greedy opponents | 128 | 0 / 0.00% | 0 / 0.00% | 1 / 0.78% |
| All matchups | 448 | 65 / 14.51% | 188 / 41.96% | 185 / 41.29% |

Every held-out game completed, so there were no evaluation cutoffs and the
all-game and completed-game win rates coincide. Each model's 448 games come from
32 board seeds, with fourteen seat/opponent roles per seed. The confidence interval
resamples the 32 complete clusters with 10,000 draws and RNG seed 71. It does not
treat 448 correlated roles as 448 independent boards, and does not measure
variation across independently initialized training runs. The latest model's
predeclared secondary comparison was +26.79 points, interval +21.43 to +31.92.
The three-win difference between selected and latest alone does not establish a
reliable difference in their strength.

Selection was fixed before final evaluation. Update 350 first reached 49/112
tuning wins (43.75%). Updates 425 and 575 later tied that result; the earlier model
was retained under the declared tie rule. The last tuning evaluation, update 1100,
won 43/112. Forty-four trained checkpoints plus the initial model were considered.
The eight tuning seeds are a small, repeatedly examined selection set; the separate
32-seed final result provides the actual reported strength comparison. Future
experiments should use fresh final-test seeds when testing new design choices.

The run started at 2026-10-14T02:47:07.148420Z and finished at
2026-10-14T04:48:49.876716Z: 2 hours, 1 minute, 42.73 seconds. The original two-hour
wall deadline remained 2026-10-14T04:47:07.148420Z. Complete current operations and
final evaluation account for the finish after the deadline; the budget was not
extended. The owned container exited with code 0 and no out-of-memory failure.
No recovery or training-parameter changes were required during the run.

| Actual training work | Count |
| --- | ---: |
| PPO update cycles | 1,112 |
| Optimizer minibatch steps | 145,911 |
| Complete self-play games | 17,792 |
| Games with three / four players | 8,896 / 8,896 |
| Accepted engine actions | 9,701,155 |
| Complete discard decisions | 235,559 |
| Training cutoffs | 0 |
| Largest rollout array payload | 104,655,532 bytes / 99.81 MiB |
| Original experiment evaluation games | 6,384 |
| Original experiment evaluation actions | 3,286,652 |

Self-play completion is not a 100% agent win rate: every completed game has one
winner and every seat uses the learner. The table's array payload excludes Python
objects, active environments, Torch intermediates, Adam state and container
runtime. The four-GiB limit therefore cannot be compared directly with that payload.
Evaluation totals include initial baselines, all 44 tuning rounds and both final
models. The additional independent completion replay is excluded from run totals.

Measured phase totals were 3,279.19 seconds collecting games, 2,929.12 seconds
updating the model, 1,048.48 seconds evaluating and 18.88 seconds checkpointing.
These represent approximately 44.9%, 40.1%, 14.4% and 0.26% of total elapsed time;
remaining time includes orchestration and reporting. Self-play collection averaged
2,958 actions/second; collection plus updates averaged 1,563 actions/second;
including evaluation and other run work, training actions divided by elapsed time
were 1,328/second. These are observed rates for this evolving policy and hardware,
not isolated engine rates or before/after speedup claims. The completed state
includes a final small bookkeeping save after the immutable final-report snapshot,
so their checkpoint timing differs by about twelve microseconds.

Raw evidence is in [final-report.json](../output/training/run-20261014/final-report.json),
[final-verification.json](../output/training/run-20261014/final-verification.json),
[metrics.jsonl](../output/training/run-20261014/metrics.jsonl),
[performance-summary.json](../output/training/run-20261014/performance-summary.json)
and [container.log](../output/training/run-20261014/container.log).
Use [u00000350.pt](../output/training/run-20261014/checkpoints/u00000350.pt) as the
primary selected learner. The latest resumable learner is
[u00001112.pt](../output/training/run-20261014/checkpoints/u00001112.pt).
Both retain model weights, optimizer state, configurations and shuffle state;
state.json identifies the completed run and latest checkpoint.

## Experiment protocol

Target: 120 minutes of elapsed experiment time, including baseline evaluation,
collection, updates, tuning evaluation and checkpoint writes. Final test evaluation
finishes after that budget; a cycle already underway completes rather than being
interrupted halfway through an optimizer update. Setup and final review bring the
requested task close to 2-3 hours. The original UTC deadline persists after resume.

Configuration: float32 MLP width 128, 16 complete games per update, at most16 active
games, alternating3/4 players, 5000 accepted engine-action cap per game, four PPO
epochs, minibatch 256, fused Adam, learning_rate 0.0003, actor/value clips0.2,
value_coefficient 0.5, gradient norm 0.5, sampled KL stop 0.02, entropy coefficient 0.
The application uses one Torch thread during collection/evaluation and four during
updates. CPU/memory limits are set on the actual Docker run, not by installing host
packages or changing library-global defaults on import.

This experiment uses gamma 1/lambda 1: complete-game Monte Carlo feedback reaches
setup and earlier decisions immediately under the sparse final-win reward. That
tradeoff raises variance; it is an explicit first-baseline choice, not evidence
that lambda 1 is universally better than the existing configurable0.95 default.
Caps use the current unchanged critic's real final own-seat observations. Cap rates
are reported; they are not invented losses or guaranteed outcomes.

Training is shared-policy self-play: all seats use the same fixed model during each
collection call, then the learner updates. No opponent pool or asynchronous stale
policy correction is introduced. Fresh boards/policy streams use a deterministic
schedule derived from update counter. Self-play win counts measure game completion,
not candidate strength, because every completed game necessarily has one winner.

Seed partitions: training starts at 1, 000, 000 and advances16 seeds per update;
tuning uses 8 seeds starting at 1, 000, 001, 000; final testing uses 32 seeds starting
at 2, 000, 001, 000. The long run applies evaluation_seed_offset=1000 to avoid the pilot/benchmark boards; the stored configuration preserves this offset on resume. Configuration rejects overlapping ranges. Both player counts,
all candidate seats, and random/greedy reference opponents are evaluated. Tuning
makes112 physical games per checkpoint; final testing makes448 games per model.

The untrained model is evaluated before learning. Every25 updates, evaluate the
current model on tuning seeds. Select the checkpoint with the greatest number of
real tuning wins; retain the earlier checkpoint on ties. Test seeds never choose
models. At completion, compare the selected checkpoint and the predeclared latest
trained checkpoint against the untrained baseline on the same final-test protocol.
If the selected checkpoint remains the initial model, state that plainly.

Report wins over all games, wins over completed games, real losses, cutoffs, group
results for each player count/opponent, and actual work. Paired improvement intervals
bootstrap whole board seeds across all roles rather than treating seat rotations as
independent observations. These intervals describe this finite seed set/protocol;
seed equality fixes initial boards, not identical later game/chance trajectories.

## Persistence and ownership

The runner holds an exclusive Linux-container file lock. It refuses to overwrite an
existing run directory unless explicitly resumed. Checkpoints use the existing
restricted, validated PPO serializer. Atomic state.json references the exact learner
file/hash, update/optimizer counters, complete configurations, accumulated work,
UTC deadline and best-tuning selection. The external seed schedule is recovered
from the stored update counter. A killed process releases its file lock.

Every5 updates, commit learner and run state. After a process failure, resume from
the last complete commit; uncommitted metrics are trimmed and the corresponding
seeded work is replayed. Partial JSONL tails are discarded. The deadline is not reset.
SIGTERM/SIGINT requests a pause after the current operation; the latest valid state
is saved. Numerical failures do not save the quarantined model; use the last valid
commit after resolving the cause. Configuration/deadline overrides on resume are
rejected. The initial, current, previous and best checkpoints are retained; obsolete
owned generation files are removed after the manifest commit, bounding model storage.

Only one training rollout is retained. Evaluation reuses the existing collector in
small chunks, mixes the candidate with reference policies, then releases its numeric
columns. Reference critic values remain missing; evaluation batches never enter PPO.
This avoids maintaining a second simulator/scheduler/rule implementation. Validation
checks precede the long run; debug engine audits stay out of the normal training loop.

## Artifacts

Each run directory contains:

- state.json: last committed checkpoint/run state.
- progress.json: live status, actual/latest and committed update counts, deadline,
  cumulative actions/games/timing and most recent update statistics.
- metrics.jsonl: finite update records, episode seed/RNG fingerprints, losses,
  entropy, KL, gradient norm, payload and phase timing; tuning summaries/groups.
- checkpoints/uXXXXXXXX.pt: bounded learner generations, including untrained/best.
- evaluation/: initial, latest/best tuning and initial/selected/latest final-test reports.
- final-report.json: actual duration/work/configuration and paired performance changes.

Raw rollouts are not automatically archived. Checkpoint and JSON artifacts are local;
no paid GPU/cloud service or external publishing is used. The container uses a fixed
built source snapshot, so editing Cursor files cannot silently change an in-flight run.

## Commands

```powershell
docker build --provenance=false --target learning-checks -t catan-learning-checks .
docker run --rm --cpus 4 --memory 4g --mount 'type=bind,source=C:\Users\david\Documents\Coding\Catan\output\training,target=/results' catan-learning-checks python -m scripts.train_ppo --output /results/my-run --minutes 120
```

Resume the same directory with `--output /results/my-run --resume`; pass no training
configuration overrides. The actual long run uses a named detached container so it
continues independently of a terminal tool call, with a follow-up in this chat to
check completion and deliver the detailed/plain-language findings. Keep Docker and
the desktop app running; local scheduled checks require the machine/app to be on.

## Implementation and verification

Tests cover serial-engine equality of batched evaluation, role rotation, physical
RNG fingerprints, no model/global RNG mutation, rejection of mixed evaluation data
for PPO, cluster pairing and cap accounting, invalid budget/memory/seed bounds,
real 3/4-player updates, exact next-update resume, orphan/partial metric recovery,
locking, checkpoint digests/configurations/retention, pause/deadline behavior,
interrupted baseline selection recovery, immutable final reports and CLI execution.

The paired evaluation comparison ran112 physical games/72, 441 actions with exactly
matching episode/RNG fingerprints across widths1/8. Three alternating repeats took
median 27.010 s serial versus 21.910 s batched, an 18.9% reduction. Keep batching; its
collector storage remains bounded by a seed chunk. This is evaluation speed, not
proof of learned playing strength. Prior PPO optimizations remain in use.

## Pilot calibration

Before the long run, a two-update full-game pilot completed 32/32 games with
32,259 actions, 615 discards and 508 optimizer steps in 45.46 seconds. It checked
mechanics and calibrated cost; its two final-test seeds were insufficient for a
strength conclusion. The long run started from a fresh initial model and used
evaluation offset 1000 to separate its held-out boards from pilot evaluation.
Pilot artifacts remain under output/training/pilot.

## Implementation choices and their reasons

1. **A thin optional training layer.** Added learning.training, learning.evaluation,
   scripts.train_ppo and tests.test_training. All 64 prior Python sources stayed
   byte-identical. Existing geometry, rule engine, NumPy observations, masks,
   collector, targets and PPO updater remain the common path. This reduces the
   chance that a second simulator or encoder disagrees with real play. The model
   has 244,955 trainable parameters, hidden width 128 and the existing shared
   policy/value trunk. No new engine dependency or unused future-model scaffold
   was added.
2. **Fixed weights during complete-game collection.** Sixteen fresh games form a
   cohort. The model does not change while those games produce data, so recorded
   behavior probabilities refer to one known model version. After collection,
   PPO updates that same learner and increments its version. This avoids stale
   asynchronous data and keeps the existing PPO preflight meaningful. Complete
   games simplify sparse final rewards and same-seat return links; fixed-horizon
   fragments and additional worker processes remain separate future designs.
3. **Alternating three- and four-player cohorts.** This gives equal game counts
   to both supported seat counts without maintaining two learners. Relative seat
   encoding and empty-seat padding allow the same model to play both. Evaluation
   rotates every candidate seat to reduce dependence on starting position.
4. **Unchanged legal-action and discard contracts.** Flat choices use the existing
   254-entry mask. Discard sampling chooses five feasible resource counts in
   order, then sends one complete engine action. PPO uses the joint probability
   of all five choices. This keeps explicit card selection and avoids enumerating
   every possible discard combination. The neural agent makes its own choices;
   existing browser selectors still give a human manual control.
5. **Sparse final-win feedback with gamma/lambda both 1 in this experiment.** All
   early choices, including setup, immediately receive complete-game outcome
   credit. This increases variance. The configurable target library default
   lambda 0.95 was preserved; the run records its explicit override. No matched
   learning-quality experiment establishes which lambda is best.
6. **Existing PPO safeguards and zero entropy bonus.** Actor/value clipping,
   finite checks, normalized advantages, gradient norm clipping and sampled
   policy-drift stops stay active. Maximum raw gradient norms may exceed 0.5
   because the metric is measured before clipping. 180 of 1,112 cycles stopped
   early under the drift rule; that is expected bounded-update behavior, not a
   job failure. The existing zero entropy coefficient avoids implying exact
   joint-discard entropy from a behavior-prefix regularizer. Mean recorded
   entropy fell from 0.690 over the first 50 cycles to 0.218 over the last 50.
   This shows choices became more concentrated; it does not by itself prove
   overfitting or explain the greedy-opponent weakness.
7. **Phase-specific CPU threading.** One Torch thread handles collection and
   evaluation; four handle optimization. Earlier paired pipeline tests found
   extra threads slightly slower for small inference batches, while four threads
   reduced update time by 28.1% in the tested update workload. Thread counts are
   application settings, not hidden library import side effects. No GPU execution
   or GPU speed claim is introduced by this CPU experiment.
8. **Reuse the collector for batched evaluation.** Each bounded eight-seed chunk
   runs one neural candidate seat and reference policies in the other seats.
   Missing reference critic estimates stay missing. These rows are evaluation
   data and never enter PPO. Sharing the collector avoids another simulator and
   preserves exact game/policy RNG behavior. Serial and batched evaluation were
   explicitly compared before launch, with identical outcomes and fingerprints.
9. **Distinct practice, selection and final-test boards.** Training counters
   generate fresh seeds deterministically. Reserved ranges and offset 1000
   separate tuning/test boards from practice and pilot evaluation. The initial
   baseline is measured on the final protocol before learning, but its results
   never choose a checkpoint or modify run settings. New design tuning after
   inspecting this completed report should reserve a new final test set.
10. **Select on tuning only, retain earlier ties.** The maximum real tuning wins
    chooses the primary saved learner. This avoids treating the most recent
    update as automatically strongest. The last learner is a declared secondary
    comparison, not a candidate selected after looking at final results. A tiny
    three-win final difference does not justify a stronger ordering claim.
11. **Paired whole-seed uncertainty.** The same initial boards/roles are paired
    before and after learning. Entire seed clusters, including all fourteen
    roles, are resampled. Matching seeds fixes initial boards but policies can
    change later chance streams; there is no claim of identical post-action dice
    trajectories between different models. One training initialization and two
    reference opponents limit the generality of the result.
12. **Transactional recovery with a fixed deadline.** An exclusive container file
    lock prevents competing writers. Atomic checkpoints and manifests bind
    versions, hashes, counters, configs and the original deadline. Interrupted
    uncommitted metrics are rolled back and their deterministic seed blocks
    replayed. Restoring optimizer and shuffle state was tested exactly. The
    actual job finished without needing recovery; it never received extra time.
13. **Bounded data and checkpoint retention.** A training rollout is released
    after its update. Evaluation chunks are released after their episode
    summaries are extracted. Only initial, selected, latest and preceding
    checkpoints remain: updates 0, 350, 1110 and 1112, about 9.86 million bytes
    total. There is no automatic raw rollout archive, duplicated tensor cache,
    or model saved after every minibatch. Metrics and final evaluation records
    remain because they make results inspectable and reproducible.
14. **A frozen, owned Docker job.** The named container used a fixed source image,
    four CPUs, four GiB and no network. Main-project edits could not alter the
    running source. All 68 current Python sources matched that frozen image at
    completion. A temporary Windows awake request and quiet same-chat checks
    protected the local run without changing a power plan. The awake request
    released, the helper exited, and task runtime helpers were removed. The
    follow-up is paused after completion; no continuing training task remains.

## Compared, rejected and deferred methods

The new direct comparison tested evaluation scheduling widths 1 and 8 over the
same 112 games and 72,441 actions. Three alternating paired repeats gave medians
27.010 and 21.910 seconds: batching used 18.9% less time with exactly identical
physical outcomes and game/policy RNG fingerprints. The slower serial path remains
a supported configuration and test oracle, but is not the long-run setting.
Raw measurements are in [evaluation-methods.json](../output/training/evaluation-methods.json).

Earlier measured optimizations were retained without silently claiming another
speedup during this experiment. Fusing the five recorded discard-prefix evaluations
reduced update time 16.6%; fused CPU Adam reduced it 5.8% in their respective paired
workloads. A full rollout tensor cache saved only 2.9% while duplicating about
45.2 MB per 7,190-row batch and was rejected. Reusable NumPy minibatch gathers were
slower in the earlier test. Wider minibatches reduced arithmetic overhead but also
changed the number of optimizer steps; faster timings alone did not establish
better learning, so this run kept 256. Details and raw comparisons remain in
[ppo-updates.md](ppo-updates.md) and output/ppo.

Earlier collector comparisons also rejected packed masks because access/conversion
was slower than Boolean masks, and float16 observations because they changed
148,074 feature values in the tested data. Huge fixed allocation and list/stack
storage used substantially more peak memory for marginal timing gains. The retained
owned growing columns and same-seat links avoid unnecessary full observation
copies. These are established collector choices, not new methods retested here;
see [rollout-collector.md](rollout-collector.md).

Four inference threads did not help the earlier complete collection workload;
the retained single-thread path was slightly faster. Compilation was unavailable
in the earlier container without a C++ compiler, and this machine's verified image
is CPU-only. Compilation, CUDA, GNNs, opponent pools and asynchronous actors were
not tested as learning-quality alternatives in this experiment. They are deferred
or unavailable options, not demonstrated failed algorithms. No host tooling was
installed to pursue them.

The model's weak greedy-opponent result is an observed limitation of this learning
baseline. This experiment does not isolate its cause: pure shared-policy self-play,
reward variance, exploration, representation and amount of practice could all
matter. The current evidence supports keeping the working trainer and learned
baseline while improving the training experiment, rather than claiming that a
larger network or GPU alone will fix playing strength.

## Independent completion checks and cleanup

The completion audit independently summed all 1,112 metric rows and 17,792 episode
records, validated sequential update/optimizer counters, policy versions, alternating
seat counts, unique practice seeds, wins/caps and discard/action/payload totals.
It checked all retained evaluation role keys, summaries, group accounting and
fingerprints; tuning selection was reconstructed from the entire 44-round history.
Training, tuning, final-test and pilot evaluation separation was verified.

All four checkpoints loaded with restricted weights-only deserialization and
matched architecture, schemas, optimizer configuration, finite weights/moments,
shuffle state and cumulative optimizer counters. The latest checkpoint hash
matched state.json. The selected weights differ from initial weights; all model
parameters remained finite. The writer lock was available after the job exited.
Container log update records exactly matched persisted metrics and launch settings.
The current main sources and original decision PDF hashes were also verified.

An independent NumPy calculation reproduced both paired confidence intervals.
Then the selected checkpoint replayed all 448 final games and 217,581 actions in
69.45 seconds, matching the entire stored evaluation, including every game/policy
RNG fingerprint. Replay work is additional verification and is not training.
The final audit is [final-verification.json](../output/training/run-20261014/final-verification.json).

Prelaunch checks passed 935 full learning Python tests, 823 core tests with three
intentional optional-module skips, and 21 viewer JavaScript tests. After the final
seed-offset addition, all 27 training cases passed. Lint, formatting, compilation,
both dependency-consistency checks and the guide-generator F check passed. These
are the actual gates; a full 938-test rerun is not claimed. The two-cycle pilot
verified real updates, completed games and checkpoint recovery but was too small
for a strength conclusion. Neither PDFs nor the browser/engine rules were changed.

Five earlier implementation comparison helpers and the final temporary audit
script were removed after preserving results. The exited awake helper and its
runtime logs and the audit diagnostic log were removed after recording a
cleanup manifest. Trained checkpoints, raw metrics, baselines, final evaluations,
pilot evidence, build/test logs and owned container logs were retained. The stopped
owned container remains available for debugging; other containers were untouched.
No commit, publishing action or new training run was performed.

## Next steps

1. Diagnose the greedy-opponent gap with a small reproducible sample: compare
   setup choices, useful purchases, trades, turn-ending behavior and development
   card use. Keep diagnostic privileged snapshots outside policy inputs and the
   normal fast training path.
2. Design a controlled opponent mix or curriculum, including stronger references
   and older learner snapshots. Reference decisions must never be mislabeled as
   fresh neural PPO data. Define learner-seat filtering and behavior/version
   semantics before changing the collector/training callback.
3. Compare learning settings under equal budgets and fresh final-test seeds:
   more than one model initialization, lambda 1 versus 0.95, and carefully defined
   exploration choices. Eight tuning seeds and one training run do not establish
   optimal hyperparameters or robust competitive strength.
4. Profile collection, optimization and evaluation separately before adding
   parallel workers or a GPU. Updating consumed about 40% of elapsed time, so a
   compatible GPU benchmark may now be useful when available. Python simulation
   still requires CPU work. More complexity should require measured total-run
   benefits and preserved probability/RNG/checkpoint correctness.

These are plans only. The requested bounded CPU training experiment, final
evaluation, independent verification and cleanup are complete.

## Results in simpler terms

The program now has a complete practice-and-review loop. It plays sixteen games,
looks back at which choices led to wins, changes its model, and repeats. Separate
boards decide which saved version to keep. Another set gives that version its
final exam against the two existing opponents, from every player seat.

It practiced 17,792 complete games. On the exam, the saved model's win rate rose
from about 15% to about 42%. Against the random player it won about 83% of games;
against the greedy player it won only one game out of 224. The first training
attempt therefore worked, but the model still needs to learn much stronger play.

The saved version comes from update 350 because that was the earliest version to
reach the best score on selection boards. We did not simply keep the last version
or choose after seeing the final exam. I reopened that saved model and repeated
all 448 exam games exactly to check that the files contain the model we reported.

For speed, the system handles several games together when asking the model for
choices, keeps only the current practice records, and saves a small number of
recoverable models. The temporary tools used to measure and monitor it have been
cleaned up. The next priority is stronger practice opponents and learning
experiments, while preserving the fast engine and trustworthy comparisons.
