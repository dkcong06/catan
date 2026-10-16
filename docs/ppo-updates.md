# PPO update contract (2026-10-13)

2026-10-15 optional profile: [public economic features](economic-features.md)
add model3/observation4/PPO3/collector3 with the original reward and loss.
Legacy defaults remain unchanged. Strict new-profile loading and exact Adam/local
shuffle resumption are tested; the new feature checkpoint has no training updates.

2026-10-14 update: [imitation and mixed-opponent training](imitation-curriculum.md)
reuses this updater. Supervised Demonstrations are a separate type and never enter
PPO. Mixed games retain only the current learner's rows, own version tag and actual
probabilities/values; the same numerical preflight rejects opponent records even
after a tag is relabelled. No PPO schema, target or loss change was introduced.

Later experiment update: [time-budgeted real training](training-experiment.md) is complete and independently verified. The selected model won 188/448 held-out games versus 65/448 initially, with most improvement against random opponents and very little success against greedy opponents. Earlier no-long-training boundaries below describe this dated PPO mechanics milestone.

This stage adds `learning.ppo`: a bounded update mechanism for the existing optional
PyTorch policy/critic. It changes model parameters during requested mechanics tests.
It does not launch a long training experiment, manage opponent pools, stream horizon
fragments, add a GNN, or establish learned playing strength. Earlier engine,
observation, action, environment, collector and target schemas remain unchanged.

## The full path

1. Obtain `updater.policy()` and hold its model fixed while collecting a finite batch.
2. `collect_rollouts` records legal choices, actual stochastic behavior probabilities,
   critic guesses, same-seat successors and real win/cutoff boundaries.
3. `updater.update(batch)` validates behavior and all choices before updating anything,
   evaluates genuine cutoff futures using the unchanged critic, and computes fixed GAE.
4. It normalizes advantages across the complete rollout, shuffles indices with a local
   CPU Torch generator, and visits every row in finite minibatches, including the tail.
5. Adam updates the shared model with clipped actor/value losses and clipped gradients.
   Policy-drift limits can stop further steps early. Statistics report actual work.
6. The updater advances its version; a new collection must use the new policy version.
   Local checkpoints permit exact continuation of tested CPU updates.

The engine stays CPU NumPy/Python, and `catan` and every `rl` module remain free of
PyTorch imports. Optional learning dependencies and container layers are unchanged.

## API example

Run this in the optional learning container, rather than installing host packages:

```python
import torch
from learning.model import ActorCritic
from learning.ppo import PPOConfig, PPOUpdater
from rl.collector import collect_rollouts

torch.set_num_threads(1)  # The application owns its thread configuration.
updater = PPOUpdater(ActorCritic(128, seed=0), PPOConfig(epochs=2), shuffle_seed=1)
batch = collect_rollouts(updater.policy(), (42, 43), max_actions=500, batch_size=2)
result = updater.update(batch)
assert result.optimizer_steps > 0 and result.update == 1
updater.save('/results/checkpoint.pt')
restored = PPOUpdater.load('/results/checkpoint.pt')
assert restored.policy().tag == updater.policy().tag
# Collect again before another update; reusing batch now raises ValueError.
```

The caller owns collection bounds and device placement. `PPOUpdater` accepts an
existing float32 `ActorCritic`; its constructor never changes Torch thread settings.
Its policy references that model, so do not collect concurrently with an update.
A saved old policy object is not an immutable opponent snapshot.

## Objective and fixed feedback

For each complete recorded action, ratio = exp(new_log_probability - old_log_probability).
The actor loss is the negative mean of the smaller of ratio*advantage and
clip(ratio, 1-epsilon, 1+epsilon)*advantage. Both positive and negative advantages
have independent analytic gradient tests. Old logs/values, advantages and value
references are detached; feedback is computed once before the first update step.

A discard is one engine action. Its log probability is the sum of five feasible
conditional resource-count log probabilities. PPO uses the complete joint ratio,
not five independent PPO losses, averaged resource probabilities, or physical-card
sampling probabilities from the reference random agent.

The critic loss is half the mean squared error against fixed value targets. With
value clipping enabled, take the larger squared error from the current value and
old_value + clip(current_value-old_value, -value_clip, value_clip). Clipping can be
disabled explicitly with `value_clip=None`; the reference defaults are configurable
mechanics choices, not Catan hyperparameters proven best through training.

Defaults: learning_rate=0.0003, epochs=4, minibatch_size=256, clip_ratio=0.2,
value_clip=0.2, value_coefficient=0.5, entropy_coefficient=0, max_grad_norm=0.5,
target_kl=0.02, gamma=1, gae_lambda=0.95, decision discount, global advantage
normalization, adam_mode=fused. Adam uses epsilon 0.00001; scalar and foreach modes remain selectable. Explicit config validation rejects invalid
bounds, nonfinite coefficients, wrong Boolean types and ambiguous discount names.

Normalization uses float64 mean/population standard deviation over all rows and a
minimum divisor of 1e-8, followed by float32 storage. A constant/single-row rollout
has zero centered actor advantage; its critic still has a valid update. A final
minibatch containing one row does not recompute a sample variance or produce NaN.
Disabling normalization leaves the computed advantages intact.

True terminal futures remain zero. Capped games require the current unchanged
critic's permitted final own-seat observations; a cap is not a fabricated loss.
Absent/unacted seats receive no invented training records or bootstrap estimates.
The existing NumPy `compute_targets` enforces seat/episode/closure/link semantics.

Flat entropy is exact. Discard entropy sums conditional entropies along recorded
behavior prefixes, rather than enumerating the current policy's complete joint
entropy. That optional term is a behavior-prefix regularizer and has no claim of
being an unbiased current joint-entropy gradient after parameters change. It is
disabled by default until a training experiment deliberately chooses its semantics.

The PPO clipping/early-stop rationale follows the primary algorithm documentation:
[OpenAI Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
The implementation is project-specific and shares the existing masked model.

## Freshness and failure behavior

Only the current updater's stochastic policy tag is accepted. Deterministic
point-mass evaluation logs, reference agents, old versions and missing predictions
are rejected. Preflight checks actual per-row model log probabilities and values
within float32 tolerances, so a manually modified model or incorrectly labeled
batch cannot silently serve as an on-policy batch. Schema/storage, finite input,
legal catalog choices, complete physical discards and their conditional/joint
logs are checked before any optimizer or shuffle mutation.

This is numerical/version validation, not a cryptographic provenance system.
Callers must not spoof tags or change parameters during collection. It is not
an asynchronous stale-policy correction or off-policy replay algorithm.

Gradients are reset with `set_to_none=True`, then clipped by their global norm.
Nonfinite objectives, gradients, parameters and diagnostics fail loudly. Gradient
clipping is not a guarantee of a bounded parameter or probability change.

Approximate KL is mean(expm1(log_ratio)-log_ratio), a sampled nonnegative drift
estimate. Before each proposed step, excessive minibatch KL stops that step.
After every visited epoch (including a partially stopped one), all rollout rows
are reevaluated for global KL and clip fraction. Excessive global KL stops later
epochs. Earlier steps remain applied; these checks do not retroactively enforce a
hard KL bound. Actor and critic share features, so critic updates can also alter
policy probabilities. Returned KL/clip metrics describe the final model across all
rows; actor/value/entropy losses average accepted minibatches by their row counts.
`completed_epochs` excludes a partially stopped epoch. The un-clipped gradient norm
is reported separately from the configured clipping threshold.

Preflight rejection preserves parameters, Adam and shuffle state. A numerical/runtime
failure after update processing starts marks the updater failed and disables further
policy/update/save calls; recover from a valid checkpoint. This avoids copying model
weights and Adam moments for rollback on every minibatch. Gradients are cleared and
the model returns to evaluation mode on both successful and failed update exits.

## Checkpoints

`save(path)` writes a temporary local file in the destination directory and atomically
replaces the named checkpoint. Failure preserves an existing checkpoint and removes
the incomplete temporary file. Only requested checkpoint paths are written; there
is no automatic expanding archive or background retention process.

Stored: model architecture/seed/state, all schema versions/reward identity, complete
PPO config, Adam moments/settings, completed update and actual optimizer-step counters,
local shuffle seed/state. `load(path, device='cpu')` uses `weights_only=True` and creates
a new updater; incompatible schemas, nonfinite weights/moments, incorrect shapes/
dtypes, altered fixed buffers, invalid counters and mismatched optimizer settings
are rejected. Engine states/hidden player hands are not checkpointed here.

Exact CPU resumption is tested across two real updates, including identical new
rollouts, next shuffles, model parameters and optimizer states. Checkpoints capture
the learner between collection calls, not an in-flight simulator or policy Random
stream. Reproducing a whole experiment also requires its external seed/opponent/
collection schedule. Cross-device or future-library bit equality is not promised.
Load only local checkpoint artifacts from trusted sources; restricted deserialization
is not a complete defense against arbitrary hostile files.

## Measured method review

The raw method comparisons live under `output/ppo`. The benchmark includes complete
preflight, cutoff values, targets, row gathering, masked forward/backward, clipped
Adam steps and final diagnostics. Construction, collection and checkpoint I/O are
measured separately or excluded as stated. Performance batches run without stress
or test jobs competing for CPU. Repeats alternate method order and use medians.

Recorded-discard teacher forcing was tested as five separate calls and one fused
call. All prefixes are known during reevaluation, so one tensor of five conditional
feature vectors preserves the same probability distribution and gradients within
float32 tolerance. Seven paired full updates fell from median 4.010 s to 3.345 s, a16.6% reduction; maximum final parameter difference was5.96e-8. Keep fused reevaluation. Sampling still chooses counts sequentially with unchanged
per-seat Random streams and feasible masks. No parameter shape, model schema,
state_dict key or engine RNG behavior changes.

Owned NumPy gathers, a full Torch tensor cache and reusable NumPy `take` buffers were
compared. The 7190-row cache duplicated 45, 236, 188 bytes. Gather-only times were 0.379 ms owned, 0.274 ms cached, and 0.448 ms reusable. In nine paired full updates, caching changed 3.373 s to 3.275 s (about 2.9%) with identical parameters. That small gain did not justify duplicating all observations/masks; retain owned bounded minibatch gathers. Reusable storage was slower and would require additional backward-lifetime ownership code; its prototype was removed.

Four minibatch sizes (128/256/512/1024) and scalar/foreach Adam are compared with
fixed four-epoch work and no KL early stop. Optimizer variants use identical data,
shuffle/objective and parameters within documented float32 tolerance. Minibatch-size
changes have different optimization trajectories, so higher row throughput does
not establish better learning or justify changing all training defaults. Fused CPU Adam was also tested: five paired updates fell from 3.472 s scalar to 3.270 s fused (5.8%), with maximum parameter difference 5.31e-8. Keep fused Adam as the verified CPU default, retaining explicit scalar/foreach alternatives. Thread count is caller-owned; benchmark it separately for the actual deployment.

The final safety-enabled default update (three repeats, 7190 rows, 128 hidden,
256minibatch, four epochs, fused Adam, one CPU thread) took median 3.056 s,
116 optimizer steps, final sampled KL0.004818. Collection took 2.239 s in one
separate sample, with 3 real wins and 5deliberate1000-action caps. Its complete
episode/RNG fingerprint matches the earlier measurement. Collection is not a
repeated median here, and these eight games do not measure playing strength.
The final scalar/foreach sweep ranged from 4.754/4.505 s at 128minibatch to
2.325/2.286 s at 1024. The larger setting makes32 steps instead of228 at 128;
keep256 as a provisional configurable learning baseline until outcome-vs-budget
experiments can compare update quality, rather than treating fewer steps as an
algorithm-preserving optimization.
With the same initial weights/data/shuffle/default update and five alternating
repeats, one/two/four CPU threads took 2.994/2.290/2.152 s median. Four threads
reduced update time 28.1%; final parameters matched one thread exactly in this
workload (two-thread maximum difference 2.98e-8). Earlier collection measurements
favored one thread, so the application may configure one for collection and four
for isolated updates. The library leaves thread ownership with the caller;
parallel collectors/audit workers should avoid multiplying per-worker thread
counts beyond the available CPU budget. Thread settings are deployment choices,
not global side effects of model construction.
A separate cProfile run recorded3.473 s for the one-thread default update: linear
layers0.864 s, backward0.806 s, total gather0.173 s, preflight0.249 s and four full-rollout
KL diagnostic passes0.885 s cumulative. These categories overlap and must not be
summed as independent buckets. Model arithmetic is substantial; CPU gathering
is not the main bottleneck. Keep global diagnostics rather than dropping safeguards
for a misleading speed claim. Profile instrumentation changes timings; use the
unprofiled paired measurements for performance claims. The example checkpoint is
2, 956, 641 bytes; single sampled save/load times0.069/0.086 s, excluded from updates.
## GPU decision

Hardware inspection found AMD Radeon integrated graphics. The verified container
uses torch 2.14.1+cpu; no NVIDIA GPU was reported. No GPU benchmark, ROCm installation,
cloud rental or host driver/package change was performed.

A GPU is worth benchmarking when repeated model updates become a large part of
end-to-end runtime, after collecting enough rows for substantial minibatches, or
when a larger architecture makes forward/backward matrix operations dominant.
The existing Python/NumPy simulator remains CPU work. Small batches and five
sequential discard sampling calls can incur transfer/synchronization overhead.

Measure CPU collection, transfer, update and evaluation separately; compare equal
work end-to-end on a compatible GPU container before committing to GPU capacity.
If fraction f of current runtime is acceleratable model work, even infinitely
fast model execution cannot beat 1/(1-f) overall speedup while the rest is fixed.
Use that bound to decide whether CPU collection improvements should come first.
GPU timing must account for asynchronous execution and transfers, as described by
[PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html) and
[the PyTorch transfer guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).
Ordinary tensor device arguments are portable; CUDA correctness/performance remain
unverified, and the updater's validation currently performs host-visible checks.

## Verification and next boundary

Reproduce:

```powershell
docker build --target learning-checks -t catan-learning-checks .
docker run --rm catan-learning-checks
docker run --rm catan-learning-checks python -m ruff check catan rl learning scripts tests
docker run --rm catan-learning-checks python -m ruff format --check catan rl learning scripts tests
docker run --rm catan-learning-checks python -m scripts.benchmark_ppo --threads 1
docker run --rm catan-learning-checks python -m scripts.validate_ppo --seeds 12 --workers 4
```

The bounded audit collects from both player counts before and after actual updates,
replays actions in an independent engine, checks private observations/legality/RNG
fingerprints, verifies finite unchanged-model behavior/GAE/gradients, and repeats each
update from a saved checkpoint. Deliberate cutoffs cover 1/25/65 actions as well as
longer games. These are optimizer mechanics checks, not a playing-strength study.

Final gates: 911 Python tests in the optional image (69 new PPO cases), 823 core tests
plus 2 intentional optional-learning-module skips, and 21 viewer JavaScript tests.
All 64 Python sources pass Ruff lint/format and compilation; both dependency checks,
the guide-generator F gate, ten current documentation examples, Markdown link checks,
and diff whitespace pass. Learning/core/viewer images all build. A local Docker
cached-snapshot export error was resolved with `docker build --provenance=false`;
no Dockerfile/dependency change, host installation or cache/data deletion was needed.

The bounded audit passed 144 episodes (72 per player count): 96 real wins and 48 deliberate
short cutoffs, 93272 actions, 186832 invariant checks, 93776 exact observation comparisons,
1591 complete discards, 72 actual updates and 818 optimizer steps. All 72 updates repeated
bit-exactly after restoring checkpoints. Ordinary audit cases needed no KL stop;
a separate aggressive-learning-rate/tiny-KL test verifies that stop behavior.
Fingerprint: `64022a8c882fc5eeff91b9640311450581687144fdd0c2c122f6e15e2237ba8e`.
Cutoffs split equally across 3/4 players, with 1/25/65-action caps; all longer audit games
ended in real wins. These bounded results do not prove termination for all policies
or that the updated policy is stronger.

Source manifests verify 59 of 60 prior Python files byte-unchanged. The intentional
change is `learning/model.py` for fused recorded-discard reevaluation; four new
Python files implement the updater, benchmark, audit and tests. The original PDFs,
guide generator and document assets remain intact; the 2026-10-11 decision PDF hash
remains `9a08738c531abfef71bcdf92f4c92c37e3622c88258aa32277046b0b4dec6e1a`.
Temporary comparison implementations are removed after preserving raw measurements,
a profile, source verification, optimization diff and a usable example checkpoint.
Next: a small reproducible training experiment with fixed budgets, held-out board
seeds, seat-rotated reference opponents, checkpoints and evaluation metrics that
separate wins from caps. Compare learning quality and total time before changing
width, learning rate, clipping, entropy or minibatch defaults. Then consider parallel
collection/GPU profiling where measured, opponent snapshots/self-play scheduling,
fragments, GNNs or symmetry when a baseline exposes their value.

## Optional direct critic-gradient control (2026-10-15)

PPOConfig.critic_updates_trunk defaults to true. False trains the value head
from detached shared features while keeping the fused forward values, actor
gradients and inference unchanged. This blocks direct critic backpropagation
into shared features; combined gradient clipping remains an indirect coupling.
The flag is saved with PPO config, and older checkpoints load with true.
Balanced training schedules and explicitly bounded milestone retention are
separate opt-in TrainingRun settings. See the
[controlled comparison](critic-gradient-comparison.md) for mechanism tests,
costs, seed schedules, recovery and the final-checkpoint-primary pilot.

## Simple explanation

The collector records what a player saw and did. The model makes move scores and a
rough guess about how promising a position is. PPO now uses the recorded feedback
to adjust that model: promising choices tend to become more likely, and its position
guesses move toward the feedback targets. Those guesses are not reliable yet.

The update has brakes. It limits the incentive for drastic probability changes,
clips unusually large gradients, checks for invalid numbers and stops taking more
steps when the policy has moved too far on the sampled data. A checkpoint saves the
model plus its learning bookkeeping, so the next tested update can continue exactly.

For discards, we still select a legal complete set of cards. While reviewing past
choices, all counts are already known, so we can process them together. A GPU would
speed up large batches of model arithmetic; it would not make the game-rule code
itself run on the GPU. The next stage is to measure whether the completed learning
mechanism actually produces a stronger Catan player on games it has not trained on.
