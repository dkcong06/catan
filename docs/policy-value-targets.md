# MLP policy, critic and learning targets

Current 2026-10-13 update: [the optional model/targets](policy-value-targets.md) and [bounded PPO updates/checkpoints](ppo-updates.md) are implemented. Earlier next-work statements below describe dated milestones. Long training/self-play and learned-strength comparisons remain future work; existing PDFs are unchanged.

Implemented 2026-10-13 in C:/Users/david/Documents/Coding/Catan. This increment
adds an untrained actor/critic, collector adapter, differentiable action
reevaluation and same-seat advantage/value targets. No optimizer updates, PPO,
self-play training or model-strength comparison runs. Earlier engine/collector
sources and observation 2/action 1/environment 1/collector 1 schemas are unchanged.
Model and target schemas start at 1. The original PDFs remain historical artifacts.

## Architecture and package boundary

The engine remains the referee; the collector remains the record keeper.
learning.model.ActorCritic interprets the existing 1505-float32 observations.
learning.policy.NeuralPolicy supplies PolicySamples to the unchanged collector.
rl.returns.compute_targets uses only numeric records and explicit value estimates.

The optional learning package uses pinned torch 2.14.1+cpu. catan and every rl module
remain NumPy/stdlib-only, including targets. The default catan-checks image retains
its prior runtime dependencies. learning-checks installs the CPU model dependency
separately; host packages, CUDA/vision/audio/graph packages and compiler toolchains
are not installed. Dependency layers precede source COPY, so editing application
code does not reinstall PyTorch. Versions were verified on 2026-10-13 against the
[official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/) and
[PyTorch installation guidance](https://pytorch.org/get-started/previous-versions/).
Benchmarks use Python 3.14.7/NumPy 2.4.6/PyTorch 2.14.1+cpu in Docker Linux.

```powershell
docker build -t catan-checks .
docker run --rm catan-checks
docker build --target learning-checks -t catan-learning-checks .
docker run --rm catan-learning-checks python -m pytest -q -p no:cacheprovider -W error
```

The default suite deliberately skips the optional Torch test module; the learning
suite runs it. The unchanged fresh-process package-import test proves catan/rl do
not import Torch even when it is installed. learning.__init__ also imports no model.

## Model and outputs

ActorCritic(hidden_size=128, seed=0) is a standard torch.nn.Module. Two shared
Linear/ReLU layers process observations, followed by one fused 254-policy-logit /
1-value projection. Default width 128 has 244955 trainable parameters, about 980 KB
of float32 parameter payload, excluding gradients/optimizer/intermediate storage.
Widths 64/128/256 are supported; more positive widths can be explicitly selected.
Width 128 is a modest baseline, not an empirically strongest architecture.

Construction uses explicit CPU/float32 allocation and a scoped CPU generator seed.
It restores process Torch RNG state and leaves thread settings unchanged. Hidden
layers use orthogonal gain sqrt(2), policy/count projections gain 0.01, value row
gain 1 and zero biases. This is an initialization choice, not evidence of learning
quality. Constructor seed, model/policy version, width and inference mode accompany
the policy tag. Future callers must keep weights fixed throughout a collection
call and update the version tag when weights change; tags do not inspect changes.

forward(observations) requires float32(N, 1505), returning NetworkOutput(logits,
values,hidden). logits has shape(N, 254), values(N,), hidden(N,H). Values are
unconstrained future-reward estimates, not calibrated winning probabilities or
Catan victory points. They are initially untrained and may be inaccurate.
The current verified runtime is CPU; CUDA/GPU throughput and transfers are not
validated, even though tensors/modules use ordinary PyTorch device operations.

## Masks, sampling and complete discards

Flat distributions mask illegal logits to negative infinity, then use stable
log-softmax. All-illegal masks and nonfinite logits are rejected. Illegal
probabilities/gradients are zero. Forced actions have probability 1, log 0 and
entropy 0. Entropy calculations avoid 0 times negative-infinity NaNs.

Stochastic NeuralPolicy samples with the collector's per-seat Python Random
streams. It uses stable double-precision exponent/CDF arithmetic on model logits;
forced choices draw no randomness. Game RNG and global Torch RNG are untouched.
The recorded probability is the behavior probability of the submitted action.
Float 32 teacher-forced reevaluation agrees within documented roundoff tolerances,
not an across-hardware bit-identical guarantee. Same-seed repeat and tested batching
configurations have identical trajectories and full RNG fingerprints.

A discard is still one engine action with five resource counts. The trunk is
computed once, then a shared 20-count head receives hidden features, the previous
five counts normalized by 19, and a five-component resource-position code. Future
counts stay zero. Each resource is masked using the existing Decision.discard_mask
before sampling. Thus later logits can respond to earlier counts, beyond merely
restricting their feasibility. The head is shared across resources to keep it small.
Count logs sum to the joint action log. No count-vector enumeration, partial engine
mutations, hidden opponent hand or automatic human/browser discard is introduced.

Deterministic mode selects the highest legal score, breaking ties by lower index,
and records point-mass behavior logs 0. It draws no policy randomness. It is for
assessment/demonstration; its logs are not the stochastic softmax behavior logs
needed for future PPO actor updates. Differentiable evaluation always describes
the stochastic softmax model. The CLI uses stochastic behavior.

NeuralPolicy(model,deterministic=False,policy_version="untrained-v 1") runs one
shared forward pass per callback under inference_mode. It sets the supplied model
to eval mode and never changes parameters or CPU thread settings. It stacks fresh
writable observation storage before Torch wrapping, respecting read-only Decisions.
NeuralPolicy.values(observations) provides finite critic predictions using owned
writable inputs. bootstrap_values(batch) evaluates only acted truncated seats and
returns(E, 4) values; unused/terminal/absent/unacted entries remain NaN. True terminal
future values are never queried. Off-turn cutoff perspectives are explicitly
included; a future trained critic must learn to handle those observations.

## Differentiable action reevaluation and minibatches

```python
import numpy as np
import torch
from learning.model import ActorCritic
from learning.policy import NeuralPolicy, action_batch
from rl.collector import collect_rollouts
from rl.returns import compute_targets

torch.set_num_threads(1)  # Explicit application configuration, not model side effect.
model = ActorCritic(128, seed=0)
policy = NeuralPolicy(model)
batch = collect_rollouts(policy, (42, 43), max_actions=100, batch_size=2)
bootstrap = policy.bootstrap_values(batch)
targets = compute_targets(batch, bootstrap_values=bootstrap)
rows = np.arange(min(64, len(batch)))
inputs = action_batch(batch, rows)
evaluated = inputs.evaluate(model)
assert evaluated.log_probs.shape == (len(rows),)
assert np.isfinite(targets.advantages).all()
# No optimizer step or parameter change is performed.
```

action_batch(batch,integer_rows,device="cpu") gathers only requested observations,
masks, actions, sparse discard hands/counts and old behavior logs/value estimates.
Arrays are writable/owned before zero-copy CPU tensor wrapping; no caller's
read-only dataset is wrapped unsafely. Sparse global rows become ascending local
minibatch positions, including when source rows are repeated. Empty batches work.
Old outputs are detached data. This does not normalize advantages, move entire
datasets to GPU or introduce a trainer/minibatch scheduler.

ActorCritic.evaluate_actions(observations,masks,actions,discard_rows,discard_hands,
discard_counts) validates physical/legal choices, returns differentiable joint
log probabilities, entropy measures and critic estimates. Training discards use
recorded prefixes (teacher forcing), so all five head contributions have gradients.
Flat entropy is exact. Discard entropy is the sum of conditional entropies along
the supplied prefix trajectory; it is not an enumeration of full joint entropy.
It is an estimator/regularizer whose future off-policy use must be specified.

Tests verify full probability normalization for an enumerated small discard hand,
maximum 95-card/47-discard choices, finite gradients, illegal-gradient zeros, hidden
opponent-hand invariance, seed/RNG/thread ownership and state_dict round trips with
weights_only=True. Backward checks calculate derivatives but never update weights.
No checkpoint manager was added; ordinary PyTorch state_dict is sufficient at this
stage. A future trainer must store architecture/version, optimizer and RNG state
for actual resumable training.

## Advantage and value-target contract

compute_targets(batch,values=None,bootstrap_values=None,gamma=1.0,
gae_lambda=0.95,discount="decision") is NumPy-only. values defaults to the
collector's per-decision predictions; missing/NaN predictions raise, rather than
being treated as zero. Actual finite cutoff bootstrap predictions must be supplied.
Only used entries of(E, 4) bootstrap_values need to be finite. Terminal futures are
zero even if unused entries are NaN. Inputs are not mutated and output float32
advantages/value_targets are independently owned, contiguous and read-only.

Let next_i be the next row for the same player/game, or the final permitted view.
The definition is delta_i = reward_i + d_i * value_next - value_i;
advantage_i = delta_i + d_i * l_i * advantage_next for internal links.
Closing rows have no future advantage continuation. value_target_i = value_i +
advantage_i is a lambda-return target, not necessarily an undiscounted Monte Carlo
outcome. gamma/lambda must be finite numbers in[0, 1].

- decision: d=gamma and l=lambda once per personal decision, including closure.
- engine: d=gamma**elapsed_actions and l=lambda**elapsed_actions.
- Defaults use gamma 1 so the objective stays final winning reward without a
  preference for earlier wins; lambda 0.95 trades the extent of critic corrections.
- lambda 1 gives bootstrapped Monte Carlo targets; lambda 0 gives one-step TD targets.
- Actual terminal target futures are zero. Cutoffs use that same seat's supplied
  estimate, retaining privacy and never fabricating a terminal loss.

Validation rejects out-of-range/backward/cross-seat/cross-game/forked links,
nonpositive durations, ambiguous closing rows, absent players, missing/nonfinite
critic predictions and incompatible shapes/dtypes. It assumes the valid collector's
step/duration history; it does not re-run engine checks or invent reward shaping.
Float 64 intermediate arithmetic limits long-trace accumulation error; stored
outputs are checked for finite float32 representability. Advantage normalization,
PPO clipping, policy/value losses, optimizer updates and learning schedules remain
future work.

## Benchmarks and implementation choices

Performance runs had no competing tests/stress jobs. Same-workload pairs were
warmed up and alternated; forward size/mode/thread sweeps were exploratory median
microbenchmarks. None measures learned playing strength, sample efficiency, GPU
performance or optimizer throughput. Raw reports are in output/learning.

A same-weight 128-row head comparison measured fused 260.25 us versus separate
267.32 us. The small difference is not a broad speed claim; one fused projection
also reduces module/state handling. A manual masked log-softmax implementation and
Categorical measured 724.92/723.06 us, effectively similar here. Manual operations
remain because selected logs/illegal-gradient/entropy behavior are explicit and no
distribution object is needed. Full-gradient/no-grad/inference-mode forward medians
were 1.811/1.719/1.737 ms at batch 128. No-grad and inference-mode were close; use
inference-mode for the explicitly read-only sampling contract, and normal gradients
for differentiable reevaluation. No universal timing advantage is claimed.

| Width | Parameters | Batch 1 forward | Batch 128 forward |
| --- | ---: | ---: | ---: |
| 64 | 118619 | 100.98 us | 896.17 us |
| 128 | 244955 | 124.76 us | 1691.66 us |
| 256 | 522203 | 171.99 us | 3547.20 us |

Width 64 was faster and 256 slower; speed alone cannot establish useful policy
capacity. Keep 128 as a configurable baseline and compare learning quality later.
No GNN dependency/implementation was added before that baseline can be evaluated.

Full neural collection: eight games, seeds 0-3 for both player counts, cap 1000,
6358 accepted actions, seven warmed alternating batches. Width 8 is an upper bound;
there are four available games per player count, so at most four are active.
One thread: serial 4.00714 s/1587 actions/s versus batched 2.54345 s/2500 actions/s,
about 57.5% more throughput. Four threads: serial 4.23119 s/1503 versus batched
2.64461 s/2404. End states/all-policy/game RNG fingerprints match within every
comparison and across these thread configurations. Five actual wins and three
cap 1000 cutoffs are properly distinguished; this is not a policy-strength test.

Four threads roughly doubled raw 128-row forward throughput (74079->151740
observations/s), yet the full collector was about 4% slower than one thread. Keep
one thread as the CLI/worker default; let applications choose after measuring their
actual inference/update pipeline. Constructing/importing models never sets threads.
One-thread target preparation including cutoff predictions was 5.34 ms for 6358 rows;
no archive saving or debug validation is included in collection timings.

Target scans compared NumPy scalar indexing, Python-float lists and NumPy
predecessor waves. Few/long trajectories favor lists; many independent trajectories
favor vector waves. Retain a bounded hybrid: waves when N>=4096 and at least 64
closing trajectories, lists otherwise. Validation guarantees a unique predecessor.
Nine alternating complete-target comparisons, with exactly matching float32 outputs:

| Layout | Rows | Initial scalar path | Retained path |
| --- | ---: | ---: | ---: |
| 2 games x 64 personal decisions x 4 seats | 512 | 0.829 ms | 0.565 ms |
| 8 games x 256 x 4 | 8192 | 8.182 ms | 4.584 ms |
| 128 games x 64 x 4 | 32768 | 31.658 ms | 6.442 ms |
| 8 games x 1024 x 4 | 32768 | 32.996 ms | 19.230 ms |
| 16 games x 1024 x 4 | 65536 | 66.661 ms | 29.332 ms |

Observed target speedups were 1.5-4.9 x. These are target calculations, not full RL
training speedups. List paths need extra temporary Python allocations: at 32768 rows
traced peak grew 2.75->6.64 MB in the few-long-trajectory case. Vector-wave peaks
were essentially unchanged (2.76 MB at 32768 rows). The extra list memory is small
relative to recorded 6020-byte observations but remains an explicit tradeoff.
Thresholds are measured starting points, not an optimum for every trajectory mix.

Torch compile was attempted and could not be measured: the minimal runtime has
no working C++ compiler for Inductor. No compiler packages were added just for an
unproven optimization; eager execution remains the tested path. Compilation/GPU
sampling/worker orchestration should be revisited with an actual training workload.
No mixed precision or cached stale legal masks are introduced.

## Tools, verification and next work

```powershell
docker run --rm catan-learning-checks python -m scripts.collect_neural --games 4 --max-actions 500 --validate
docker run --rm catan-learning-checks python -m scripts.benchmark_learning --threads 1
docker run --rm catan-learning-checks python -m scripts.validate_learning --seeds 25 --workers 4
```

collect_neural optionally writes --output records.npz, --targets targets.npz and
--report report.json. JSON records model/target schemas, seed/width, discount/
lambda/thread settings and zero optimizer updates. NPZ is numeric plus JSON and
loads with allow_pickle=False. Disposable containers need a writable output bind
mount to retain artifacts. Model initialization and archive I/O are outside timed
collection. Model schema and source/runtime versions should accompany real future
checkpoints; example data is untrained and is not evidence of learned strength.

Final gates: 842 Python tests in the learning image;823 applicable tests plus one
intentional optional-module skip in the default image;21 JavaScript tests; all
three Docker targets built; Ruff lint/60-file formatting, compilation and both
pip consistency checks passed. Warnings were treated as errors during tests.
The 67 new cases include analytical target boundaries, independent trajectory
oracles, large vector waves, hidden information, forced/deterministic RNG draws,
conditional probability normalization, gradient ownership and numeric CLI archives.
All 51 earlier Python source hashes remain unchanged. The expanded
neural audit covers 124 episodes, 92930 actions, 186108 full invariant checks,
93364 exact observation comparisons, 1805 discards,all eight decision phases and
100 real wins/24 deliberate cutoffs. Direct engine replay reproduces chosen actions
and game/RNG state; per-seat policy streams are independently advanced according
to the specified nonforced-choice draw counts. Different reevaluation minibatches
match probabilities/values within float32 tolerances. Lambda 1 targets match true
outcomes or explicit cutoff bootstraps. Backward derivatives are finite and every
weight remains byte-identical. Audit fingerprint:
`a1a245b953ca4891978080cd812cae791105cd5266811898bfc43ced0bae34e7`.
This bounded gate is not a universal correctness or game-termination proof.

The next increment is a minimal PPO update: stochastic-action ratio checks,
clipped actor loss, critic loss, detached/normalized advantages, finite/clipped
updates and checkpoint/version handling. First verify it on analytic toy cases,
then measure real update/inference workloads. Training itself, self-play/opponent
scheduling, GNN comparisons, symmetry, recurrent/horizon fragments and search
remain future authorized work.

## Plain-language explanation

The collector is the notebook. The new model is an untrained decision maker that
reads a page and scores the allowed moves. A second output guesses how promising
the situation is. For discards, it chooses how many cards of each resource to give
up while checking that the complete selection is allowed.

The target calculator compares those guesses with what happened later. It can
send the final outcome back through that player's earlier decisions, without
mixing in another player's private view or pretending a stopped game was a loss.
Those calculations prepare feedback for learning; they do not change the model.

We measured several ways to run the calculations. Processing several games'
decisions together was faster. Different target-calculation methods suit different
numbers/lengths of player histories, so a small tested selection chooses between
them. The model has not learned Catan yet: the next step is the rule that actually
updates its weights from this feedback, followed by authorized training experiments.
