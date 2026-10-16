# Greedy imitation and mixed-opponent training

## Purpose and scope

The existing learner beat random opponents much more often than greedy opponents.
The decision audit found weak setup production, resource spending mistakes and an
occasional missed immediate win. This stage adds a reproducible way to teach the
existing model from our own greedy policy and then train against varied opponents.
The online human dataset is not an input: its raw games remain unverified.

The engine, board generation, geometry IDs, ports, manual discard rules and
observation/action/model/PPO schemas remain unchanged. No neural architecture or
reward shaping was introduced. The new training-run schema is 2 and still reads
schema-1 manifests/checkpoints. Existing completed experiments and weights remain
preserved.

## Teacher examples

learning.imitation.Demonstrations.from_rollouts consumes the existing collector's
ReferencePolicy("greedy") records. It retains float32 observations, Boolean legality
masks, int16 action IDs, int8 phases and sparse complete discard labels. Each
observation is from the acting player's perspective, including that player's
private hand and cards and only the existing public information about opponents.

Forced flat decisions are removed because their actor log probability is zero for
every model and they provide no actor gradient. PPO collection still retains its
forced learner decisions: their critic targets and temporal links remain useful.
Discard records keep the whole five-count choice and its physical hand; no
combinatorial enumeration or partial engine actions are added.

Demonstrations are intentionally a different type from RolloutBatch. They contain
no critic estimates, terminal rewards or future observations and cannot enter
PPOUpdater.update. Numeric NPZ persistence uses exclusive creation, schema checks,
allow_pickle=False, finite/shape/dtype/legality/complete-discard checks and a bounded
uncompressed size before array allocation. The comparison adds immutable shard
hashes. No arbitrary game objects, foreign checkpoints or pickle datasets are used.

The pilot generates 64 training games across three/four players plus eight separate
teacher-validation games. Collection shards contain at most eight games. Training
loads one shard at a time; teacher generation, archive work and validation stay
outside normal engine steps and inference.

## Imitation objective

ImitationTrainer fits only the actor. It evaluates the teacher's masked flat action
or joint autoregressive discard likelihood using the same action-gathering and
decoder used by PPO. All five conditional discard logs contribute to one complete
decision. Future chosen resources never appear in an earlier prefix.

Mean-one inverse-frequency weights give every represented phase equal total mass
within a shard. This prevents abundant main-turn records from overwhelming setup,
robber or card-timing phases. Every tail minibatch is included. The initial version
uses a three-pass corpus traversal, a separate fused Adam at learning rate 0.001,
minibatches of 256 and gradient clipping at 0.5. Shard order alternates between passes,
with an explicitly seeded local shuffle generator.

The critic is not trained to treat a teacher's future result as a calibrated
probability for the learner's future behavior. It still shares the trunk, so actor
pretraining can change its predictions; subsequent PPO fits actual learner returns.
Imitation Adam moments are not transferred into PPO. Every comparison arm resets
PPO Adam and version counters from the same initial weights to keep that choice
consistent across methods.

Before mutation, the trainer validates all labels and evaluates their likelihood.
Runtime failures quarantine that trainer; malformed input rejection preserves
parameters, Adam and shuffle RNG. The caller owns CPU thread counts.

## Learner-only rollout collection

collect_rollouts(record_players=(seat,)) changes storage, not gameplay. Every
opponent still receives its own Decision and policy RNG and submits an ordinary
engine action. Only recorded seats append observations, choices, probabilities,
values, rewards and same-seat links. Excluded seats have last_rows=-1 and zero
final-view slots, so their private final observations are not retained.

The same-seat elapsed-action gap includes intervening opponent actions. Actual
terminal rewards and legitimate cutoff bootstraps remain attached to the learner's
closing rows. If a cutoff excludes a seat before it acts, the collector creates no
invented record. Mixed training requires an action cap of at least eight so each
scheduled learner seat can act during setup.

The default record_players=None path preserves full-seat collection. The existing
EvaluationPolicy router also accepts a frozen NeuralPolicy opponent. The combined
policy tag deliberately rejects PPO; training supplies the learner version tag
only while recording that learner seat. Existing numerical preflight checks still
verify every stored probability and value against the unchanged current learner.
Merely relabelling a mixed batch does not make opponent records acceptable.

## Opponent schedule and persistence

The pilot schedule is greedy, self, previous, greedy, random: two greedy cohorts
and one of each other kind per five updates. Player counts alternate three/four,
and learner seats rotate. Each mixed cohort records one learner seat. A default
pure-self run still records every seat, so existing self-play behavior is preserved.

Self cohorts use one batched neural call for all players. Other cohorts batch the
learner and opponent requests separately. The previous-policy option alternates
the owned initial checkpoint and an owned retained previous generation. At most
two frozen CPU opponent models are cached. No unbounded league, duplicate game
simulator or persistent engine state is added.

TrainingRun retains its existing exclusive writer lock, atomic checkpoint/manifest
commits, replay of uncommitted update tails, seed schedule and immutable deadline.
Caches reconstruct from retained checkpoint files on resume. Original/previous/
latest/tuning-best checkpoint retention stays bounded. A new-run initial_checkpoint
copies weights through the existing restricted loader, records their source hash
and explicitly resets Adam/counters; it does not modify or resume the source run.

Totals distinguish physical engine actions from retained learner_rows and learner
wins. Completion counts remain game completion, not a claim of bot strength.

## Comparison protocol

scripts.compare_training runs self-play, mixed opponents and imitation plus mixed
opponents from the original selected update-350 weights. Three training-seed
replicates vary practice boards and local shuffles while preserving the starting
weights. They are not three independently initialized architectures.

Each arm has a 90-second training-time target in this pilot. The entire shared
teacher-generation cost and that replicate's pretraining time are charged to each
imitation arm, rather than giving reusable teacher data free extra compute. Initial
and periodic tuning evaluation consume the remaining training budget. Final
evaluation is separate. An already-started complete cohort can overrun a deadline;
reports retain actual elapsed times. No budget was increased after seeing results.

The comparison uses 12 fresh final board seeds and three separate tuning seeds per
replicate, rotating every candidate seat in both player counts against random and
greedy. Primary checkpoint selection counts greedy tuning wins only and retains
earlier ties. Final test results never select a checkpoint. Selected/latest reports
keep genuine wins, losses and cutoffs separate.

Teacher training uses 250000000 and 250010000; teacher validation uses 250100000 and
250110000. Practice ranges start at 100000000 plus 1000000 per replicate. Tuning/
final ranges start at 1000050000/2000050000 plus 128 per replicate. These do not
overlap the original run or diagnostic boards.

Paired intervals resample whole board seeds, including all seats on a board.
Same seeds pair initial boards and roles, not identical future chance trajectories.
Pooled intervals are conditional on these selected models and board sets and do
not estimate all training-seed uncertainty. This is an exploratory pilot, not a
universal algorithm ranking or automatic promotion of a deployed model.

## Measured implementation choices

An isolated eight-board, four-player benchmark preserved all 3,079 actions and
full game/policy RNG fingerprints. Retaining every seat used 19,618,228 array bytes
for 3,079 rows; learner-only retention used 4,779,398 bytes for 727 rows: 75.638%
less payload. Median collection time was 1.788% higher in this small measurement,
so this is a memory improvement, not a demonstrated collection speed improvement.
Payload excludes Python, Torch and optimizer overhead.

A five-repeat alternating forward/backward benchmark on 256 rows, including 16
complete discards, compared normal entropy work with likelihood-only work at four
CPU threads. Skipping unused entropy reduced median time by 6.445%, with bit-equal
joint logs and zero maximum gradient difference. ActorCritic.evaluate_actions now
accepts compute_entropy=False for imitation; returned entropies are zero
placeholders. PPO and inference defaults retain their normal behavior and entropy
reporting. This option changes no parameter shape or checkpoint field.

The implementation reuses action_inputs for both objectives, the existing collector
for teachers and learner games, the existing seat router and TrainingRun persistence.
These avoid duplicate simulation, action decoding, discard sampling or checkpoint
machinery. NumPy/stdlib engine and rl imports remain independent of PyTorch.

Rejected/deferred choices: importing approximate human-derived CSV labels; silently
overriding learner choices while recording neural PPO probabilities; discarding
forced PPO rows; keeping every opponent observation as training data; building an
unbounded opponent pool; transferring old Adam moments inconsistently between
comparison arms; and changing architecture, reward shaping, precision or GPU runtime
during the same strength comparison. Phase-balanced teacher fitting was chosen to
protect sparse decision phases; a broader supervised-objective ablation remains
separate work rather than an established strength benefit.

## Results and verification

The frozen implementation passed 1,004 Python tests, repository lint/format checks
and compilation of all 73 Python sources. These include exact learner-only
trajectory/link/privacy tests, independent joint-discard likelihood checks,
likelihood-only gradient equality, malformed teacher rejection, mixed-batch
rejection even after tag spoofing, warm-start isolation, schema-1 compatibility,
exact mixed/history resume and a full capped three-arm smoke test.

The pilot completed all nine cases. Its owned container exited successfully with
no out-of-memory failure at 2026-10-15T01:13:43Z (October 14, 9:13 p.m. EDT).
Across the arms it performed 119 PPO updates, 952 completed training games,
488,896 physical actions and 264,064 retained learner decisions, with zero training
cutoffs. These count practice, not evaluation or teacher-generation games.

The teacher corpus contains 11,033 training labels from 64 games and 1,454
validation labels from eight other games, in ten bounded shards. Each imitation
replicate made 144 supervised optimizer steps over three corpus passes. Held-out
phase-weighted likelihood loss fell from 3.9404 to 1.7302, 1.7425 and 1.7396.
This confirms learning of the labels; it does not demonstrate stronger gameplay.

Primary final results pool 36 distinct test boards and all candidate seats across
three training-seed replicates. Every method has 252 greedy roles and 252 random
roles; all games completed without cutoffs.

| Selected policy | Against greedy | Against random |
| --- | ---: | ---: |
| Original update-350 weights | 1/252 (0.40%) | 202/252 (80.16%) |
| Self-play continuation | 1/252 (0.40%) | 202/252 (80.16%) |
| Mixed-opponent continuation | 1/252 (0.40%) | 202/252 (80.16%) |
| Imitation then mixed opponents | 2/252 (0.79%) | 162/252 (64.29%) |

Imitation plus mixed opponents improved the observed greedy rate by only 0.40
percentage points. The paired whole-board 95% interval is [-0.79, +1.98] points,
which includes no improvement. Its random-opponent change is -15.87 points, with
an exploratory interval [-23.81, -7.54]. The evidence does not justify replacing
the original policy. All trained candidates and raw outcomes are preserved; no
checkpoint was automatically promoted.

Self-play and mixed primary results match the original exactly because their
greedy-tuning selection retained update zero in all three replicates. This does
not mean PPO made no updates. Three tuning boards provide only 21 greedy roles,
so rare wins give a sparse selection signal; earlier ties were deliberately kept.
The selected imitation versions were update 10, 0 and 0, where its update zero
already includes supervised pretraining.

| Replicate | Self-play updates / selected | Mixed updates / selected | Imitation-mixed updates / selected |
| --- | ---: | ---: | ---: |
| 0 | 13 / 0 | 19 / 0 | 10 / 10 |
| 1 | 15 / 0 | 20 / 0 | 8 / 0 |
| 2 | 12 / 0 | 17 / 0 | 5 / 0 |

The predeclared latest-checkpoint secondary results were 2/252 greedy and 199/252
random for self-play, 4/252 and 192/252 for mixed, and 1/252 and 153/252 for
imitation-mixed. These do not select a new winner after looking at final tests.
Imitation alone, before PPO, had 1/252 greedy wins and 161/252 random wins. The
random regression therefore was already present after pretraining; this is an
observation about this procedure, not proof that imitation generally fails.

The supervised generation/pretraining charges were 18.8464, 20.3206 and 24.1445
seconds per imitation arm, leaving 71.1536, 69.6794 and 65.8555 seconds of the
original 90-second target for its run. Baseline and tuning evaluation consumed a
large share of these small budgets. Only five to ten imitation-mixed PPO updates
fit; this is a mechanics and preliminary-strength pilot, not an adequate test of
eventual convergence. No extra training budget was granted after seeing results.

The user paused the container at 2026-10-15T00:04:52Z and explicitly resumed it at
01:10:33Z. The eighth case had committed its final update before the pause and
was evaluating, so direct Docker unpause preserved live evaluation/game/RNG state.
All training deadlines and budgets stayed unchanged. Approximately 3,940.872
seconds of suspension are included in raw mixed-r2 elapsed/evaluation counters;
the UTC wall duration minus that suspension is approximately 202.492 seconds.
Wall and performance clocks did not track identically across the suspension, so
the affected evaluation timer is not used to claim active evaluation throughput.
Raw timings remain unchanged. General mid-training suspension accounting is not
implemented; the harness does not support restarting the whole comparison after
process loss, although individual committed training runs can resume.

The independent audit verified all nine saved configs, source hashes, fresh-Adam
initializations, counter/metric totals, bounded checkpoint versions and finite
weights/Adam state. It reconstructed earliest-best greedy-tuning selection,
checked every test role, win/loss/cutoff summary and seed-family separation, and
recomputed all primary paired intervals without calling the comparison function.
It replayed all 72 teacher games with invariants on every action and matched every
label array and game/policy RNG fingerprint. A full selected-model evaluation
replay reproduced all 168 games and 95,567 actions exactly, including every RNG
fingerprint, with invariants on every action. This audit took 156.90 seconds;
119.58 seconds were the instrumented evaluation replay, not a normal throughput
benchmark. All 73 Python sources matched the frozen tested image; all ten core
engine sources and the original selected checkpoint were unchanged.

The task-created benchmark and audit scripts were removed after preserving their
results and verified file hashes. Teacher shards, trained checkpoints and raw
reports/logs remain intact; [cleanup evidence](../output/curriculum/cleanup.json)
records the two exact removed paths. No unrelated files were removed.

Raw evidence: [comparison](../output/curriculum/pilot-20261014/comparison.json),
[independent verification](../output/curriculum/verification.json),
[audit log](../output/curriculum/verification.log),
[container log](../output/curriculum/pilot-container.log),
[optimization measurements](../output/curriculum/optimization-preflight.json),
[full test log](../output/curriculum/tests-full.log) and
[pause/resume accounting](../output/curriculum/resume.json).
The original checkpoint SHA256 remains
67a0ad501d00b5031e3e53f3f87383f96a21b1224ce812907bf6f29d8c1de8a2.

## What to investigate next

1. Diagnose the warm-started policy on fresh diagnostic boards using the existing
   decision tracer. Measure teacher agreement by phase and check setup production,
   useful spending and immediate wins. Determine which habits pretraining lost
   before increasing the training budget.
2. Compare bounded supervised settings using separate teacher-validation boards:
   smaller learning rate, phase weighting versus uniform weighting and stronger
   coverage of economic decisions. The current three-pass phase-balanced setting
   is a working initial choice, not an experimentally proven best objective.
3. Once the warm start retains basic playing strength, predeclare a larger matched
   experiment with more greedy-tuning boards and enough practice time after setup
   and evaluation. Keep the final test boards fresh and inaccessible to selection.

Economic observation features and the more expressive discard head from the
earlier diagnosis remain separate candidate changes. GPU, a larger opponent
league and unverified human-game imports were not implemented. More expensive
training should follow evidence about learning quality rather than this pilot's
lower teacher loss alone.

## Reproduction

Run from the main project in Docker. Mount only the new output folder writable and
the existing checkpoint folder read-only:

    docker build --provenance=false --target learning-checks -t catan-curriculum-checks .
    docker run --rm --network none --cpus 4 --memory 4g --mount 'type=bind,source=C:\Users\david\Documents\Coding\Catan\output\training\run-20261014\checkpoints,target=/baseline,readonly' --mount 'type=bind,source=C:\Users\david\Documents\Coding\Catan\output\curriculum,target=/results' catan-curriculum-checks python -m scripts.compare_training --output /results/new-comparison --initial-checkpoint /baseline/u00000350.pt --seconds-per-arm 90 --replicates 3 --teacher-games 64 --teacher-epochs 3 --games 8 --test-seeds 12 --tune-seeds 3

Existing train_ppo also supports --initial-checkpoint, --opponents and
--tuning-opponent greedy. Its --resume mode accepts only the saved output and
--resume; it restores the committed configuration/deadline. Teacher examples and
candidate checkpoints remain explicit, separate artifacts.

## Simple explanation

We now have a teacher and a student. The teacher is our existing greedy bot. We
record the situations it sees and its legal moves, then show those examples to
the neural student. That provides basic habits before the student practices again.

During practice, the student faces greedy bots, copies of its current policy,
older policies and random opponents. We keep only the student's decisions as PPO
training records. Opponent moves still happen normally.

We tested the student on new boards. It learned to copy the teacher more closely,
but it still almost never beat greedy opponents and became worse against random
bots. One extra greedy win is too little to establish an improvement.

The teaching and practice machinery is built and tested. The next useful step is
to inspect which decisions the taught student still gets wrong and adjust the
teaching method before paying for a much longer training run. The original bot
and every experimental model remain available for comparison.

