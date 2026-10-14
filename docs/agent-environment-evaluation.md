# Agent environment, rewards and evaluation

Current 2026-10-13 update: [the optional model/targets](policy-value-targets.md) and [bounded PPO updates/checkpoints](ppo-updates.md) are implemented. Earlier next-work statements below describe dated milestones. Long training/self-play and learned-strength comparisons remain future work; existing PDFs are unchanged.

Implemented 2026-10-12 in the main project. Training remains on hold.
2026-10-13 update: the [per-seat collector](rollout-collector.md) is implemented.
The dated performance and next-work descriptions below refer to this earlier increment.
This increment completes the agent environment and outcome/evaluation steps.
Observation schema v2 (1505 float32 values) and action schema v1 (254 flat slots
plus conditional discards) retain their meanings. Environment and evaluation
schemas start at v1. NumPy remains the existing runtime dependency.

## Environment contract

`rl.env.CatanEnv` owns one mutable GameState and one prepared FlatEncoder.
It wraps existing validators/transitions; it does not implement a second ruleset.
A caller exclusively owns the environment and submits synchronous decisions.

| Operation | Behavior |
| --- | --- |
| CatanEnv(max_actions=5000) | Validate a positive integer external action cap |
| reset(seed=None, n_players=4) | Create seeded game/encoder; reset counter; return first setup Decision |
| decision() | Return a fresh acting-player request, or None when episode is closed |
| observe(player=None, out=None) | Encode a permitted perspective; available after wins/cutoffs |
| step(action, player=actor) | Apply one index or complete discard vector; return lightweight Transition |
| snapshot() | Explicit independent full state/RNG copy for privileged diagnostics/replays |

`n_players` and `player`/`out` arguments shown above are keyword arguments where
required by the source signature. Player is mandatory on step. Normal wrong-seat,
malformed or illegal choices are rejected before mutation/draws/counter increments.
A rejected reset leaves the previous episode intact. Calls requiring a game reject
use before reset. A real win or external cutoff closes submissions until reset.

A Decision carries absolute player, phase, independent observation, boolean mask,
and (during DISCARD only) a snapshot of that player's five integer hand counts
and required discard total. Policies receive these permitted values, not GameState
or its RNG. The full snapshot method is for diagnostics, not policy inputs.
Each normal observation/mask has 6020 + 254 array payload bytes; Python object
headers are additional. Retaining old requests does not let later steps overwrite
them. Optional observe(out=...) follows the encoder's explicit buffer ownership
and alias checks; consume/copy the output before reusing it.

A turn contains multiple atomic actions. The environment preserves setup, pre-roll
cards, rolls, discards, robber decisions, main play and free roads as separate
choices. Dice/card draws and theft remain engine chance events after validation.
It does not choose opponents' moves or manual resources implicitly.

The actor is the next pending discarder during DISCARD; current_player remains the
roller. Otherwise the actor is the turn owner. A complete five-count discard is one
step, not five partial engine mutations. Decision.discard_mask(prefix) uses the
same feasibility helper as live-state discard_count_mask; each accepted prefix
admits a full physical discard. It can be queried from the old request without
referencing a changed live hand. Engine validation remains the submission authority.

Transition is an immutable named tuple with player (the submitting actor),
next_player (underlying next actor), rewards, terminated, truncated, winner and
actions. No arrays are forced into the result. Request the next Decision when it
will be consumed. This keeps replay/manual stepping from building unused features.
No per-step state copy, mutable legality/port cache or full invariant pass is added.

## Rewards and episode boundaries

The reward scheme is `terminal_win_v1`: every ordinary action returns an immutable
zero tuple, reused safely; a true win returns one for the winner and zero for other
seats. No point/resource/turn-length shaping is applied. Rewards are in absolute
seat order, independent of observation-relative seats. For example seat 2 winning
four-player play yields (0,0,1,0). A transition can be submitted by seat 0 while an
end-turn check declares seat 1 the winner; the reward still belongs to seat 1.

Termination comes from the real engine GAME_OVER. Truncation comes from the
external accepted-action limit, without changing phase, VP or RNG or inventing a
winner. A true win on the final allowed action takes precedence over truncation.
Only accepted actions count. At cutoff, decision() returns None, but next_player
still names the underlying actor and observe(player) supplies each seat's final
nonterminal perspective. Opponent private VP stays hidden at cutoff; actual engine
terminal observations follow the existing final-score visibility contract.

A cutoff's zero immediate reward is not a fabricated terminal loss. The [collector](rollout-collector.md) now closes each seat's pending trajectory.
A future trainer must distinguish terminal
outcomes from external cutoffs and decide its discount/bootstrapping convention.
There is no rollout buffer, value model, return/advantage calculation or training
in this increment. The distinction follows the concepts in
[Gymnasium time-limit documentation (1.4.0)](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
The choice-by-choice actor organization also matches the principle described in
[PettingZoo's AEC documentation](https://pettingzoo.farama.org/api/aec/).
These references describe concepts; this is a standalone NumPy/stdlib interface.

## Running an agent

```python
from random import Random
from rl.env import CatanEnv
from rl.policies import random_policy

env = CatanEnv(max_actions=5000)
request = env.reset(42, n_players=4)
policy_rngs = [Random(100 + seat) for seat in range(4)]
while request is not None:
    choice = random_policy(request, policy_rngs[request.player])
    result = env.step(choice, player=request.player)
    request = env.decision()
assert result.terminated != result.truncated
assert sum(result.rewards) == int(result.terminated)
print(result.winner, result.actions, result.truncated)
```

Game RNG and policy RNGs are separate. No request/query draws game randomness.
The engine still supports functional copies for shared/branching uses; the owned
wrapper uses its in-place path. Prepared encoding remains once per immutable board.

## Reference policies and evaluation

`rl.policies` publishes stateless, versioned random and greedy references:

- Random chooses uniformly from true flat mask indices. Manual discards sample
  physical cards without replacement; uniformly choosing feasible count vectors
  would be a different policy. The catalog order differs from the legacy engine
  random agent's object ordering, so seed trajectories need not match that agent.
- Greedy is deterministic: prefer productive settlement/city choices, available
  free-road/knight cards, resource choices toward a build/purchase goal, development
  purchases, trades that strictly reduce a goal deficit, and frontier roads before
  ending. Robber choices weigh publicly visible production and opponent totals.
  Discards drop from the most abundant remaining type. It is a transparent reference,
  not an optimal Catan strategy or a trained agent.

Both use observations, legal masks and own discard counts. They do not read hidden
opponent resource/card composition, fresh cards or remaining deck composition.
Same-visible-facts hidden-state swaps produce identical choices and policy RNGs.
Policy version 1 retains exact choices through the measured optimizations.
Static road frontiers derive once from immutable GEOMETRY. Heuristic potential
caching is local to one decision; there is no accumulating board/state cache.

`rl.evaluate.play_match(policies, seed, ...)` accepts one callable per seat and
returns a frozen MatchResult. Each callable receives Decision and its own Random.
Policy streams use the explicit versioned string
`catan-evaluation-v1:{policy_seed}:{seed}:{seat}`. Full game and every policy RNG
enter the final diagnostic fingerprint. Invalid policies fail with seed, player
count, policy seed, actor, phase and action coordinates rather than a success result.
Optional validate=True checks the initial state and every successor.

`evaluate(candidate='greedy', opponent='random', seeds=range(10), n_players=4, ...)`
places one candidate against opponents, rotating every seat per distinct seed.
Seeds must be nonempty distinct integers. Both player counts are supported.
The same board seed fixes the initial layout; policy-dependent actions can alter
subsequent dice/steal/card RNG consumption. Comparison does not assert identical
future chance sequences across different policies.

Reports include schemas/reward/policy versions, runtime, seeds, limit, validation
mode, seat-specific and aggregate wins/losses/caps, two win-rate denominators,
actions/completed turns/scores, ordered fingerprints and individual role results.
Final true scores are privileged diagnostic labels, not extra observation features.
`win_rate_all` is wins divided by all reported assignments, with caps separately
listed; `win_rate_completed` excludes cutoffs and is None when none complete.
Cutoffs are never included in candidate_losses. These finite seeded results are
not confidence intervals or proof of general playing strength.

For identical built-in policies, a seed produces one deterministic match regardless
of the candidate label. The evaluator executes it once and emits independent role
records for all seats. `matches_played` and `executed_actions` report physical work;
`actions` sums over reported role assignments. actions_per_second uses physical
executed actions. Same-policy rotations are not independent samples. Different
candidate/opponent pairings execute every rotation. The reuse is local per seed,
retains only an immutable finished summary and introduces no persistent state cache.

```python
from rl.evaluate import evaluate

report = evaluate(seeds=(0, 1), n_players=3, max_actions=5000)
assert report['summary']['games'] == 6
assert report['matches_played'] == 6
assert report['executed_actions'] == report['actions']
print(report['summary'])
```

```powershell
docker build -t catan-checks .
docker run --rm catan-checks python -m scripts.evaluate_policies --seeds 10 --players 4
docker run --rm catan-checks python -m scripts.evaluate_policies --seeds 10 --players 3 --validate
docker run --rm catan-checks python -m scripts.benchmark_environment
```

For retained JSON, create an output directory and mount it at /reports, then pass
`--report /reports/name.json`. Without a mount the disposable container's file is
not retained. The CLI accepts candidate/opponent random or greedy, first seed,
number of distinct seeds, policy seed, cap, validation and report path.

## Measured implementation choices

Measurements used Docker Linux, Python 3.14.7/NumPy 2.4.6, warmed alternating
orders and no competing test/stress jobs. Raw results are in output/agent-interface.
No neural inference, learning or browser rendering was measured.

| Comparison | Median before/reference | Median retained/alternative | Decision |
| --- | ---: | ---: | --- |
| Known-action replay, no policy/feature query | 0.05849 s owned/lazy | 0.13451 s copying every step | Keep exclusive ownership |
| Same replay with unused eager queries | 0.05849 s lazy | 0.25131 s eager | Keep explicit lazy requests |
| 12000 complete decision queries | 2.21632 s independent outputs | 2.58083 s checked reuse | Keep independent defaults |
| 300000 transition records | 0.52111 s frozen dataclass | 0.20974 s named tuple | Keep cheaper immutable tuple |
| Full six-game agent loop, 1458 actions | 0.33749 s first implementation | 0.31196 s retained | 1.082x throughput |
| Same loop with record change alone | 0.33749 s | 0.33438 s | Modest 1.009x, not a large overall claim |
| Three-seat same-policy control, three seeds | 1.37023 s repeated rotations | 0.45860 s reuse | Keep exact result reuse |
| Four-seat same-policy control, three seeds | 2.99104 s repeated rotations | 0.74404 s reuse | Keep exact result reuse |

The query/copy/record experiments use five batches; full-loop/control comparisons
use seven. Microbenchmarks describe their named operations, not end-to-end training
speed. Reusable outputs remain an explicit encoder/observe option, but repeated
alias checks outweighed allocation savings here and shared scratch would weaken
old-request ownership. Eager features add cost when the caller already has actions;
a normal policy loop still requests each needed observation/mask.

The profile showed observation encoding and legality queries dominating; heuristic
work and wrapper bookkeeping were smaller. Retained changes reduce repeated guard
calls, use immutable tuple results, skip feature decoding for forced choices, defer
resource goals, convert tiny public features to Python values once, use static road
frontiers and compute potentials only once per request. No engine rules changed.
Every action, field, request, transition and full game/policy RNG matched across
25066 additional steps / 48 random, greedy and mixed games covering all nine phases.

The maintained final benchmark compares the same direct agent frontend with the
wrapper: 4923 vs 4684 actions/s (0.29615 vs 0.31126 s), about 5.1% extra time for
actor checks, counters, rewards and cutoff handling. Both have identical final
state/all-RNG fingerprints. These short-workload rates depend on load and are
not comparable to prior engine-only or different-policy trajectory rates.

We did not add undo, packed/native/GPU state, a global legality/port cache, framework
adapters or a collector/network. The measured profile does not justify those in
this interface increment. Full invariant checks remain explicit debug work.

## Verification and remaining work

The final gate passes 718 Python tests, including 54 new environment/evaluation
cases. It covers direct functional/owned engine replay, every phase, caller
ownership, rejected choices/invalid resets, cached RNG/array isolation through
existing copy gates, actor/roller differences, discard prefix snapshots, pre-roll
card continuations, private facts, bank shortages, cap/win precedence, wrong-seat
rejection, all-seat rewards, known-control accounting and repeatable reports/CLI and exact initialization/post-action failure coordinates.
Both Docker targets build; 21 viewer checks, Ruff lint/46-file formatting,
compilation, dependency consistency, both documentation-generator F-checks and
fourteen Python documentation examples pass.

The expanded gate plays 1800 distinct policy/seed/seat lineups: seeds 0-99 for both
player counts; all-random, all-greedy, one greedy at every seat, and one random at
every seat. It checks 691144 actions / 692944 initial/successor full invariants,
all nine phases, 1800 real wins, zero caps or failures. Ordered fingerprint:
`6b2f844f084ed9247685f8c4cde63630e89acc8074258b79b4d3b7225b10149b`.
This gate is finite; no universal-termination proof is claimed. Elapsed validation
work runs concurrently with tests and is not a performance benchmark. Raw cases
and summaries are retained in stress.json. The evaluation CLI can reproduce each
policy pairing and seat rotation with --seeds 100, --players 3/4 and --validate.

Temporary passing-source snapshots, prototype/patch/comparison and stress drivers
are removed after review; raw reports/profiles and the maintained benchmark remain.
Earlier engine/geometry/state code and source seed fingerprints remain unchanged;
only the discard feasibility helper is shared within RL. The dated 2026-10-11 PDF
is preserved rather than regenerated. Empty future network/PPO/search modules stay
absent; env/evaluate now exist because they contain implemented behavior.

The subsequent [collector increment](rollout-collector.md) implements the
per-seat rollout/trajectory contract: attribute actions
and delayed outcome rewards, save discard prefixes/joint choice probabilities,
handle terminal versus truncated final values, own buffers and batch independent
games. Tests can use these reference policies while training remains held. MLP,
GNN comparisons, PPO/self-play, negotiation, symmetry and search remain later work.
