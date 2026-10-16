# Settlers-RL: methods, evidence and lessons for this project

Inspected 2026-10-15 UTC. Sources: Henry Charlesworth's [project article](https://settlers-rl.github.io/) and [implementation](https://github.com/henrycharlesworth/settlers_of_catan_RL). Code is pinned to [`62a1c04ec26c2dacc3bbe16cb05975569560b9d2`](https://github.com/henrycharlesworth/settlers_of_catan_RL/tree/62a1c04ec26c2dacc3bbe16cb05975569560b9d2), committed 2022-04-06. This is historical work; the inspection date is not its publication date.

This is a more direct methodological reference for us than Monte Catano: PPO, structured observations/actions, opponent-resource memory, historical opponents and learned-policy forward planning. It offers specific experiments to consider, alongside article/code differences and rule mismatches that prevent direct strength comparisons.

## Evidence boundary

Cached the article and 45 relevant source/documentation files as inert text in [output/settlers-rl-investigation](../output/settlers-rl-investigation/). All downloaded repository files match their Git blob hashes; [manifest](../output/settlers-rl-investigation/downloads.json). Inspected the central model, observation, training, opponent selection, search, evaluation and relevant engine paths. Foreign Python, installation scripts and checkpoints were not executed or loaded.

Our side was checked against current encoding/model/PPO/resource-memory code, the [hypotheses experiment document](training-hypotheses.md), completed [training report](training-experiment.md), [policy diagnosis](greedy-policy-diagnosis.md) and prior [Monte Catano study](monte-catano-investigation.md). The new B/P/R/K experiment is separate ongoing work; this investigation does not change its protocol or establish a result for it.

## What they implemented

### Structured representation and conditional actions

The model separates board, own-player and opponent information. Its board encoder applies two attention blocks to tile representations with local building ownership, followed by a flattened projection. Player modules encode economic/award information and use attention over development-card lists. The default model uses a 512-wide observation output and no LSTM. References: [model construction](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/RL/models/build_agent_model.py), [observation module](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/RL/models/observation_module.py), [tile encoder](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/RL/models/tile_encoder.py).

There is also an explicit production table per player: resource by dice result, with settlements contributing one and cities two. This is especially relevant to our economic diagnosis. Our encoding contains the underlying tile/building data and dice probabilities, but does not supply that aggregate production table directly. Giving a network an easily computed public summary may reduce what it must learn from scratch; whether it improves wins per compute budget needs a controlled test. Attention itself is not demonstrated superior by a matched architecture ablation in this inspection.

Their action architecture chooses among 13 action types, then conditional heads choose locations, cards, players or resources. Recurrent trade heads build give/receive bundles, with at most four resources on each side. Only relevant heads contribute to the joint action log probability. Source: [action heads](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/RL/models/action_heads_module.py).

Our 254-slot catalog is manageable because negotiated trading is excluded, and complete discards already have an autoregressive count decoder and joint likelihood. Conditional type/location heads could be compared separately, but adding all of their trading machinery is not necessary to obtain the representation or planning lessons. Their tile inputs also do not explicitly encode edge ownership; our existing road information should be preserved in any architectural experiment.

### Public resource memory

Each observer stores lower/upper counts for every opponent resource. Announced transfers update those bounds; an uninvolved observer widens uncertainty for steals. Observations encode the bounds rather than exact opponent resource compositions. This is implemented in [resource-estimate updates](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/game/game.py#L921-L1010) and [player features](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/env/wrapper.py#L526-L709).

This closely matches the principle of our K arm. Our implementation instead supplies 30 normalized relative-seat bounds, tightens them against public hand totals and exact public bank stocks, and keeps explicit observer/privacy/replay tests. The external implementation is evidence that this is a practical representation, not an isolated measurement of its strength benefit.

### PPO, opponent history and parallel collection

One randomly assigned seat is the active learner; the other seats use sampled historical policies. Only the learner's records enter its PPO batch. Historical selection mixes a uniform component with a preference for recent policies. Defaults retain up to 500 versions, adding a version every four updates. References: [opponent sampling](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/RL/ppo/update_opponent_policies.py), [training runner](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/RL/robust_train.py).

Workers each own games and local policies. They gather chunks independently, then synchronize at an update boundary. Defaults are 128 processes, five games per process and 200 learner decisions per game: 128,000 decisions per update batch. These are fragment collections, not necessarily complete games. PPO defaults include gamma 0.999, lambda 0.95, ten epochs, 64 minibatches and clipping 0.2. Checked-in entropy scheduling declines from 0.04 to 0.005; this differs from the article's stated fixed 0.01. Sources: [arguments](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/RL/ppo/arguments.py), [worker collection](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/RL/ppo/vec_gather_experience.py).

The learner uses fixed value normalization, and recomputes critic-based advantages each epoch. Our current updater instead computes fixed behavior-model targets once, with explicit numerical/version checks. That difference is an experimental choice to evaluate, not a reason to remove our safeguards. Sources: [PPO update](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/RL/ppo/ppo.py), [value normalizer](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/RL/models/utils.py).

We already have learner-only collection, per-seat decision links, mixed opponents and bounded history. Their larger pool and process collection offer scaling ideas. A pool does not automatically prevent all exploitable behavior, and fragment collection would require intentional continuation/bootstrap/RNG contracts beyond our current complete-episode collector.

### Root-only planning using the learned policy

The planner is simpler than a full MCTS tree. It proposes candidate root actions and allocates repeated simulations among them with an upper-confidence score. After that first action, the learned policy controls all seats; no additional decision tree is expanded. Short rollouts use reward and critic estimates, so they need not reach a terminal game. Defaults count depth in the controlling player's decisions, rather than all engine actions. References: [planner](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/RL/forward_search_policy/policy.py), [rollout worker](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/RL/forward_search_policy/worker.py).

Candidate generation deliberately covers legal action types, then samples additional choices. There is an option to consider every initial settlement. This is an important response to confident policy errors: searching only its highest-probability moves can omit the move that fixes the mistake. Parallel allocation includes simulations still in progress, reducing duplicate worker assignments. The planner chooses the highest estimated mean value at completion. Source: [candidate generation](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/RL/forward_search_policy/sample_actions_fn.py).

Before each simulation, `randomise_uncertainty` redistributes concealed opponent cards/resources, respecting resource bounds, hand totals and bank conservation in its sampling procedure. This attempts to address the information problem that Monte Catano leaves unresolved. It is a constrained sampling heuristic, not a demonstrated posterior or proof of information-set-correct planning. Its resource rejection loop is also unbounded. Source: [uncertainty sampling](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/game/game.py#L1207-L1282).

## Results and reproducibility limits

The article reports roughly one month of training on a 32-core machine with 128 GB RAM and a 3090 GPU, approximately 450 million learner decisions. Evaluation compared newer policies with older versions; the author still judged human play substantially stronger. The reported planning experiment won 47 of 100 four-player games against three copies of the final neural policy, using 32 search workers and ten seconds base thinking time per decision. This supports improvement within that experiment, not superiority over our agent, an external benchmark, or a particular feature's causal benefit. Source: [reported results and planning test](https://settlers-rl.github.io/#results).

The [evaluation script](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/evaluation/run_forward_search_evaluation.py) supports that matchup design, but the repository tree inspected here does not contain the corresponding raw 100-game result archive. The provided checkpoint is update 3825; the article's rounded training duration/update count is not a complete frozen run configuration. We did not load the checkpoint.

Important implementation qualifications:

- **Reward configuration differs between description and defaults.** The article describes dense shaping. The wrapper implements a +500 win reward and optional point/event rewards, but the pinned training manager creates `EnvWrapper()` with its `dense_reward=False` default. Search workers explicitly enable dense rewards. It is therefore unsafe to infer that the published checkpoint was trained with shaping, or that shaping caused its reported gains. Event bonuses are also applied inside a loop over all seats, rather than explicitly only to the actor. References: [reward code](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/env/wrapper.py#L85-L112), [training manager](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/RL/ppo/game_manager.py#L11-L17).
- **Discard rules differ.** Validation and mutation require discarding down to seven, instead of snapshotting half the original hand when above seven. With eight cards, that removes one rather than our four. Their agent selects individual discarded cards, whereas ours submits one complete vector. Reference: [discard mutation](https://github.com/henrycharlesworth/settlers_of_catan_RL/blob/62a1c04ec26c2dacc3bbe16cb05975569560b9d2/game/game.py#L785-L800).
- **Persistence is not our exact-resume contract.** The checked-in training runner writes a five-item current checkpoint while its load path expects seven; it does not save Adam or full RNG state in that tuple. This is a source observation, not a reproduced execution failure. Preserve our atomic optimizer/config/RNG recovery guarantees when borrowing architecture ideas.
- **Planner actor routing needs review.** Its worker's turn helper handles trade responders but omits the discarder handling that exists in its training/evaluation managers. This requires validation before reusing search code.

None of these issues negate the useful design ideas. They mean the repository is a research reference, not a validated replacement engine or plug-in opponent.

## Three-way comparison

| Aspect | Our project | Settlers-RL | Monte Catano |
| --- | --- | --- | --- |
| Learned policy | Small MLP/PPO | Structured attention/PPO | None |
| Search | Proposed separately | Root candidates plus learned rollouts/critic | Multi-depth UCT plus heuristic terminal rollouts |
| Resource memory | Optional observer-specific bounds under current experiment | Per-observer bounds | Exact private hands supplied |
| Observation structure | Flat board/edge/seat features; optional memory branch | Tile attention, player modules, explicit production | Full engine state |
| Opponents | Random/greedy/frozen policies; bounded history available | Large sampled historical pool | Primarily version comparison |
| Trading | Bank and ports | Also negotiated proposals/acceptance | Bank and ports |
| Strength evidence | Saved external-reference evaluations with known weaknesses | Historical-policy gains and author-reported planning test | Relative-version Elo claim |

## Recommended lessons

1. **Keep the existing B/P/R/K comparison interpretable.** This work strongly motivates studying public memory and economic representations, but supplies no clean ablation establishing their gains. Do not add attention, new rewards or search halfway through our frozen comparison.
2. **Prioritize a production-feature experiment after the current comparison.** Start with public per-resource production summaries and candidate settlement yields. Use the same MLP, opponents and budgets before testing an attention model. This directly targets our diagnosed placement/economic errors at lower implementation cost.
3. **Make root-only planning the first search baseline.** It is simpler than full MCTS and can reuse our simulator and frozen policies. Cover legal action families and all setup choices; explicitly include immediate winning moves even when neural probability is tiny. Compare greedy versus neural rollouts and short versus longer horizons under measured latency budgets.
4. **Integrate uncertainty sampling with public bounds carefully.** Consume a sanitized observer view, sample feasible hidden allocations with bounded work and a separate RNG, and test conservation/privacy. Do not let a planner receive actual hidden hands simply because our debug snapshot API can expose them. Root sampling still needs checks for unrealistic opponent information and strategy fusion.
5. **Evaluate exploration and scaling separately.** Nonzero entropy, historical diversity and process collection are reasonable experiments. They are not established fixes for our weaknesses. Measure current collection/update costs before increasing workers, minibatches or model size; their workload and hardware differ markedly from ours.
6. **Keep win rate and action quality as separate evidence.** Use our exact missed-win/trade/setup cases to explain behavior, then test fresh boards against fixed references and historical agents. Report wins/caps and planning time. Self-play progress or a shorter average game is insufficient evidence of general strength.

This investigation adds documentation and cached evidence. It does not implement any of these proposals, run a matchup, or alter the ongoing training experiment.
