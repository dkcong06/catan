# Per-seat rollout collector

Later 2026-10-13 update: [the optional MLP policy/critic and same-seat learning targets](policy-value-targets.md) are implemented. Earlier model/target next-work statements below describe their dated milestone. Training/optimizer updates remain held; existing PDFs are unchanged.

Implemented 2026-10-13 in C:/Users/david/Documents/Coding/Catan. Training remains
held. This increment adds NumPy-only complete-episode data collection; it does
not add a model, trainer, return/advantage computation or reward shaping.
Observation v2, action v1, environment v1 and existing game/RNG behavior are unchanged.
Collector schema starts at v1. The browser remains the board/manual-discard demo.

## Purpose and decisions

The engine is the referee; the environment provides decisions; the collector
records each player's permitted view, submitted action, behavior probability,
optional value estimate, next personal decision and outcome. A policy callback
can eventually run one batched neural inference for several independent games.
It receives no game state, hidden opponent data, game RNG or evaluation summaries.

Use complete episodes first because outcome attribution and seat continuity can
be tested without partial-fragment ownership or unresolved rewards crossing calls.
The caller chooses a finite seed set and per-game action cap. At most batch_size
games are live; finished slots refill in input-seed order. There is no implicit
thread/process parallelism in collection and no global mutable cache. Reference
policies are stateless; arbitrary stateful callbacks must manage their own state.
A policy should remain at one version throughout a call; policy_tag records the
version supplied by the caller but cannot verify neural weights.

Games may end at different lengths. Rows are interleaved by the synchronous
scheduler; episode_indices and absolute players identify them. Same-seed episodes
with reference policies have identical actions/outcomes/all-RNG fingerprints at
widths 1, 2, 4, 8 and larger-than-game-count widths. Row ordering/indices can differ
with scheduling, so compare each episode's ordered records rather than raw row
numbers. There is one player count per call (three or four), with a fixed feature
shape for both. There is no automatic seat rotation or duplicated control reuse:
every requested episode is actually played and contributes its own records.

## API

```python
import numpy as np
from rl.collector import ReferencePolicy, collect_rollouts

batch = collect_rollouts(
    ReferencePolicy(("greedy", "random", "random", "random")),
    seeds=(10, 11, 12, 13),
    n_players=4,
    policy_seed=7,
    batch_size=4,
    max_actions=5000,
)
assert len(batch) == sum(episode.actions for episode in batch.episodes)
rows = np.flatnonzero(batch.players == 0)[:32]
next_views = batch.successor_observations(rows)
assert next_views.shape == (len(rows), 1505)
assert (batch.elapsed_actions > 0).all()
```

collect_rollouts(policy, seeds=(0,), *, n_players=4, policy_seed=0,
max_actions=5000, batch_size=8, policy_tag=None, validate=False) validates integer
bounds, distinct nonempty seeds, three/four players, callable policy, Boolean
validation and the int32 row bound before playing. ReferencePolicy accepts one
known policy name for all seats or one per seat. Repeated collection calls own
independent outputs. No user/environment instance is reused or implicitly reset.

The callback contract is policy(tuple_of_decisions, tuple_of_policy_rngs), returning
exactly one PolicySample per request in that order. Inputs have read-only array
flags to catch accidental mutation; this is an ownership convention, not protection
against hostile Python callbacks. The Random objects belong to the policy streams.
The game owns separate randomness; stream seeds retain the existing evaluation
formula catan-evaluation-v1:policy_seed:seed:seat. Probabilities add no RNG draws.

PolicySample(action, log_prob, value=None, discard_log_probs=None) has:

- action: a legal flat index or the complete five-count discard tuple.
- log_prob: the log of the probability the behavior policy assigned to that action;
  finite, nonpositive and representable in float32. Deterministic choices have zero.
- value: an optional finite float32-representable critic prediction for this view.
  None is stored as NaN, distinguishing missing predictions from predicted zero.
- discard_log_probs: five conditional count-choice log probabilities for a discard;
  their sum must match log_prob. Catalog actions must not supply this field.

The collector checks all outputs in a callback batch before stepping any of that
batch's games. It checks legality, physical discard counts, metadata and cardinality.
Initialization, callback, malformed-output and engine/debug failures retain replay
context and original exceptions. Failure returns no partial success batch. Policy
RNGs may have advanced while computing rejected outputs; the collector does not
promise rollback/retry of arbitrary callback side effects. Full state invariants
are opt-in and remain outside normal collection.

## Columns and ownership

All result columns are C-contiguous, independently owned by the returned batch and
have read-only flags. They cannot be overwritten by later environment decisions or
collections. Internal growing buffers remain private until they are shrunk to the
actual row count. No policy receives a view into those buffers. A future tensor adapter should
use writable gathered minibatches (for example observations[integer_rows]); direct
wrapping of read-only dataset arrays needs an explicit ownership decision. Frozen result
containers do not make malicious array flag changes impossible; callers should
copy columns before editing their own data.

N is the number of accepted atomic actions; E is the requested episode count; D
is the number of complete discards. steps counts accepted actions before this row,
so the first action has step zero. One turn can produce several records.

| Column | Shape / dtype | Meaning |
| --- | --- | --- |
| observations | N x 1505 / float32 | Acting seat's permitted pre-action view |
| masks | N x 254 / bool | Flat legality; all false on discard rows |
| actions | N / int16 | Flat index, or -1 marking a sparse complete discard |
| log_probs | N / float32 | Behavior joint action log probability |
| values | N / float32 | Critic estimate, NaN when absent |
| rewards | N / float32 | Zero except true outcome on the seat's final row |
| next_rows | N / int32 | Next row for the same episode/seat, or -1 at closure |
| elapsed_actions | N / int32 | Positive engine-action gap to next view/closure |
| episode_indices | N / int32 | Input seed's local episode index |
| players / phases | N / int8 | Absolute acting seat / existing phase code |
| steps | N / int32 | Pre-action accepted-action count |
| terminated / truncated | N / bool | Closure flags on each seat's last row only |
| last_rows | E x 4 / int32 | Final row per seat, -1 if absent or never acted |
| final_observations | E x 4 x 1505 / float32 | Permitted final view per seat; absent seat zero |

Episode summaries contain seed, policy_seed, players, actions, actual winner,
terminated/truncated and the existing complete state/all-policy-RNG fingerprint.
Fingerprints are diagnostic summaries, not portable save states or correctness
proofs. Privileged state/RNG data and true diagnostic scores are not stored as
training observations. policy_tag identifies the caller/reference policy version.

Raw payload is 6308*N + 34*D + 24096*E bytes. This excludes Python objects,
allocator/RNG/environment overhead and temporary growth/inference/archive memory.
The finite worst row bound is len(seeds)*max_actions; keep seed sets appropriately
small and call repeatedly for large-scale use. The collector retains the requested
batch in RAM; it is not an unbounded stream-to-disk service. Growth starts at at
most 256 rows and doubles up to the finite bound, with exact-size final outputs.
Do not concatenate batches' index columns blindly: next_rows, discard rows,
episode_indices and last_rows need index remapping, or keep batches separate.

## Seat continuity, rewards and cutoffs

If Alice acts at engine step 20, Bob and Carol act, and Alice next acts at step 25,
Alice's row points to her step-25 row with elapsed_actions=5. It does not use Bob's
observation as Alice's successor. Consecutive decisions by Alice have gap one.
Manual discards use the discarder as actor while the roller remains turn owner.

At a genuine win, every seat's last recorded row is closed and receives its own
terminal_win_v1 reward: one for the actual winner, zero for others. This includes
seats that did not make the final engine action. Earlier rewards remain zero;
a later trainer must propagate outcomes backward. If a seat never acted, no fake
record is created; its episode outcome/final perspective still exists. True wins
on the last allowed action take precedence over the cap.

An action-limit cutoff closes each acted seat with truncated=True, reward zero,
no winner and its own final nonterminal observation. Hidden information stays
hidden. A future critic may evaluate these final observations; no value is
invented here. They can be from an off-turn state, so the future critic/return
contract must explicitly support those perspectives. True terminal rows must not
be bootstrapped with a future value. Internal rows have neither closure flag.

successor_observations(integer_rows) gathers only a requested minibatch. Internal
successors come from existing recorded rows; closure successors come from the same
seat's final observation. The output is new writable float32 storage. Empty
minibatches work. Storing links avoids duplicating 6020 bytes per row: the measured
3117-row dataset used 12468 bytes of links instead of 18764340 bytes of duplicate
successor observations. Discounting is deliberately undecided: downstream code
may discount once per personal decision or use gamma**elapsed_actions for engine
step discounting; it must choose and document one convention.

## Discards and behavior probabilities

Discards are one engine action, rather than five partial game mutations. The
sparse Discards table stores rows (D int32), hands and counts (D x 5 int8) and
log_probs (D x 5 float32). The hand is only the discarding player's permitted
hand snapshot. Counts use resource enum order. Required count is their sum; no
redundant mutable required-count column is kept. Discards.masks() reconstructs
D x 5 x 20 feasible prefix masks on demand, including empty D. Every chosen count
must be true under the corresponding mask.

The random reference player samples physical cards without replacement. Its
resource-count vectors are not uniformly distributed. For a count vector c from
hand h with k cards discarded, probability is product(comb(h[r],c[r])) / comb(sum(h),k).
Conditional count probabilities follow the same hypergeometric law: each prefix
updates the remaining cards and required selections. Their logged probabilities
sum to the joint log probability without extra random draws. Tests enumerate a
small hand and verify total probability one, exact combinations, feasible prefixes,
the 95-card/47-discard maximum, and unchanged policy RNG states.

The greedy reference player is deterministic, so its joint and conditional log
probabilities are zero. Both adapters preserve existing policy choices/version1.
Custom learned adapters must report their own actual behavior probabilities;
legality/sum validation cannot prove that arbitrary reported probabilities match
a neural distribution. No behavior-cloning, policy-gradient or critic fitting runs.

## Persistence and commands

```python
import json
import tempfile
from pathlib import Path
import numpy as np
from rl.collector import ReferencePolicy, collect_rollouts, save_rollouts

batch = collect_rollouts(ReferencePolicy(), seeds=(42,), max_actions=20)
with tempfile.TemporaryDirectory() as directory:
    destination = Path(directory) / "batch.npz"
    save_rollouts(batch, destination)
    with np.load(destination, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        assert metadata["collector_schema"] == 1
        assert archive["observations"].shape == (20, 1505)
```

save_rollouts(batch,path,compressed=False) writes numeric NPZ members and Unicode
JSON metadata, without Python-object pickles. Paths require .npz. Metadata includes
collector/environment/observation/action schemas, reward scheme, policy_tag and
all Episode summaries. Consumers must check schema versions. Archive saving is
optional and occurs after collection, outside its timed loop; no automatic
persistence, checkpoint restoration or generic load/replay framework is added.

```powershell
docker run --rm catan-checks python -m scripts.collect_rollouts --games 4 --players 4 --policies greedy,random,random,random --batch-size 4 --validate
docker run --rm catan-checks python -m scripts.benchmark_collector
docker run --rm catan-checks python -m scripts.validate_collector --seeds 50 --workers 8
```

CLI --output writes .npz, --compressed enables archive compression, and --report
writes summary JSON. Disposable containers need a writable output bind mount to
retain files, for example from the main project in PowerShell:

```powershell
New-Item -ItemType Directory -Force output/collector | Out-Null
docker run --rm --mount "type=bind,source=${PWD}/output/collector,target=/results" catan-checks python -m scripts.collect_rollouts --games 4 --output /results/reference.npz --report /results/collection.json
```

## Implementation comparisons and retained tradeoffs

Runtime was Docker Linux Python3.14.7/NumPy2.4.6. Performance experiments ran
without competing tests/stress jobs, warmed up and used alternating order for
paired comparisons. Eight four-seat games, seeds 0-7, greedy seat0 and random
others produced 3117 actions and identical stored-column/episode/RNG digests.
Seven paired full-loop batches gave:

| Storage method | Median seconds | Traced peak bytes | Decision |
| --- | ---: | ---: | --- |
| Geometric private resize | 0.70069 | 26478593 | Retained: lower peak memory |
| Allocate/copy on growth | 0.68824 | 38113090 | Rejected: small speed difference, larger peak |
| Preallocate every maximum row | 0.67978 | 253939218 | Rejected: about 10x peak |
| Retain row objects then stack | 0.67971 | 40857553 | Rejected: about 54% higher peak |
| Fixed chunks then concatenate | 0.71746 | 41127235 | Rejected: slower and larger peak |

The retained method was about 3.1% longer than list/stack in this workload, while
using about 35.2% less traced peak memory. Tracemalloc reports traced allocation
peaks, not process RSS; these are workload-dependent tradeoffs, not a universal
fastest-storage claim. Arrays remain contiguous and exact float32. Private resize
uses refcheck=False only because no internal buffer views escape before finish.

Packed-mask storage/selected-row unpacking microtiming was 0.04750s versus 0.00965s
for Boolean copies/access, despite reducing that mask payload from 791718 to 99744
bytes. Masks are only about 4% of dense row payload. Keep Boolean masks to avoid
added conversion/API work; optional future storage needs can revisit this.
Half precision changed 148074 observation values (max absolute error0.000234365);
it breaks exact observation compatibility and needs learning-quality evidence.
A bulk-default-initialization prototype changed paired full-loop medians from
0.72100 to 0.71623s (about0.7%): not compelling enough to retain added complexity.
It was reverted, including its extra storage initialization rules.

Exploratory five-repeat scheduling medians for widths 1/4/8/16 were
0.66996/0.64565/0.62087/0.62875s for three seats and
0.77103/0.72784/0.71793/0.71620s for four. Only eight games are available, so width 16
has at most eight active requests. This was a fixed-order exploratory test;
no causal speedup or neural-batching claim is made. Width 8 is a configurable
starting point, not an optimum for hardware/model workloads that do not yet exist.

Final benchmark: sixteen games (seeds 0-7, both player counts), 5834 actions,
seven warmed alternating batches, full state/all-policy-RNG agreement. No-record
agent-loop median 1.11729s (5222 actions/s); collector 1.34774s (4329 actions/s), about
20.6% extra time for records, probabilities, batch validation and seat bookkeeping.
Payload 37191270 bytes, 143 discards. Both include reset/observations/masks/policies/
steps/final fingerprints; neither includes debug validation, inference, training,
rendering or persistence. Profile shows observations/masks dominate total work;
there is no justification here for rewriting the engine or adding GPU/native state.

Optional archive comparison for3117 rows: uncompressed 19870090 bytes/write 0.03888s;
compressed 743675 bytes/write 0.43260s. Observation/mask load medians0.02189/0.03684s.
Keep in-memory collection and uncompressed saving as defaults; compressed output
is useful when disk/transfer size matters, but runs outside the collection loop.
Raw JSON reports/profiles are retained in output/collector. Temporary prototype
sources and rejected implementations are removed after verification.

## Verification and next work

Final gate: 775 Python tests (57 new collector cases), 21 viewer JavaScript tests,
both Docker builds, Ruff lint/51-file formatting, compilation and dependency checks
pass. Tests run with Python warnings treated as errors.

The expanded independent audit collects/replays 636 episodes: 600 normal complete
games and 36 deliberately short cutoffs. It checks 313132 actions, 627536 full initial/
successor invariants across collector and direct replay, 315358 oracle observations,
5908 discards and all eight decision phases plus true terminal states. No crashes,
incorrect links/probabilities/observations/RNGs or unexpected cutoffs occur.
Ordered fingerprint: 45e3d6777ef0cdc71a3c9d7a7ad85a05d1fce957f9b0f9ce3444c3b2a47cab9f.
The finite audit does not prove correctness for all possible games. Its elapsed
140.83s includes debug work concurrent with tests; it is not a speed benchmark.

The tests cover independent direct-engine/policy replay, same-seat successors/durations,
all-seat outcomes, actual off-turn winner versus submitting actor, last-action
win precedence, short caps/unacted seats, all phases, sparse discard probabilities,
readonly/independent/contiguous outputs, overflow/invalid metadata, batched rejection,
replay error coordinates, empty minibatches and pickle-free archive/CLI round trips.
Earlier sources, geometry/setup fingerprints, ML-free imports and browser tests
remain regression gates; no earlier source or dependencies are changed.

Next, design the MLP baseline's policy/critic callback and the return/discount
contract, including five conditional discard heads and nonterminal cutoff values.
Then connect and measure actual batched inference before adding workers or GPU
transfer optimizations. Training/PPO/self-play, recurrent state, horizon fragments,
GNN comparison, symmetry, negotiation and search remain future authorized work.

## Plain-language explanation

The collector is a notebook for computer players. Each time someone acts it writes
what that player could see, what they chose, how likely their strategy was to
choose it, and where their next personal decision appears. Other players can act
in between; their private views never become that player's next page.

When the game ends it adds each player's result to that player's final page.
A stopped game gets a different label from a completed game. Discard pages record
exactly which cards were selected. Several games can supply pages in one callback,
so a future computer-player model can process several decisions together.

The notebook stores ordinary numbers in compact arrays. It points to existing
pages instead of copying every page twice. Temporary comparisons tested which
storage methods were quick and which used less memory. We kept a practical
memory/speed balance and preserved the exact information the game already exposes.
This produces tested learning examples; it does not teach a player by itself.
