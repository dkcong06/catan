# Monte Catano investigation and comparison

Inspected 2026-10-15 UTC. Public repository: [Algorhythm-sxv/monte-catano](https://github.com/Algorhythm-sxv/monte-catano). Pinned source: [`64edc0a074b97500e505157d0cafc9706d45f140`](https://github.com/Algorhythm-sxv/monte-catano/tree/64edc0a074b97500e505157d0cafc9706d45f140), committed 2026-06-12. This is a static source investigation, not a measured head-to-head strength or performance comparison.

Monte Catano offers a useful alternative to our current learned agent: search at decision time, supported by a simple Catan heuristic and compact Rust simulation. Its public snapshot does not establish strong human-level play. Search has privileged information, the game rules differ from ours, and the inspected code has concrete correctness concerns. The useful lesson is to test planning and better decision evaluation on our validated engine, with our information boundaries.

## Evidence and scope

Downloaded all 18 repository blobs as inert `.txt` files under [the ignored research folder](../output/monte-catano-investigation/). Every downloaded file matches its Git blob SHA1; see [verification](../output/monte-catano-investigation/verification.json). Preserved the source tree, metadata and all 18 commit messages. The releases and issues API responses were empty at inspection. Foreign source, build scripts and binaries were not executed. No packages were installed and no training was launched.

Core search, simulation, state, rules, command interface and match-runner files were inspected directly. Our comparison uses current `learning/model.py`, `learning/policy.py`, `rl/env.py`, `rl/policies.py`, encoding and award/engine code, together with the saved [training report](training-experiment.md), [decision diagnosis](greedy-policy-diagnosis.md) and [imitation pilot](imitation-curriculum.md). Existing experimental knowledge/reward additions in the working tree are separate from the checkpoint used in those completed experiments; their presence is not evidence of a new verified strength result.

The upstream repository identifies its license as AGPL-3.0. This investigation does not incorporate upstream implementation code into our engine.

## Their approach

### Monte Carlo tree search

[`src/mcts.rs`](https://github.com/Algorhythm-sxv/monte-catano/blob/64edc0a074b97500e505157d0cafc9706d45f140/src/mcts.rs#L159-L222) implements the usual selection, expansion, simulation and backpropagation loop:

1. Follow explored moves using UCT, balancing the acting player's observed win fraction against an exploration bonus.
2. Expand a randomly selected untried legal move.
3. Simulate the resulting position to a winner.
4. Increment visits and the winning seat's counter along the ancestor path.

Each node has four win counters. At a decision node the selection score uses the player acting there, so opponents pursue their own simulated wins. This is a multiplayer search heuristic, not a proof of equilibrium or optimal play. The final root move is selected by most visits, rather than the highest raw observed win percentage. See [`Node::uct`](https://github.com/Algorhythm-sxv/monte-catano/blob/64edc0a074b97500e505157d0cafc9706d45f140/src/graph.rs#L490-L497) and [`best_move`](https://github.com/Algorhythm-sxv/monte-catano/blob/64edc0a074b97500e505157d0cafc9706d45f140/src/mcts.rs#L255-L273).

The implementation attempts to represent dice and development-card purchases as chance nodes. Development-card selection uses remaining deck counts. Dice handling has the issues described below, so this is its intended architecture rather than an endorsement of its correctness.

### Heuristic-guided rollouts

The simulations are more directed than uniform random play. [`select_best_of_n`](https://github.com/Algorhythm-sxv/monte-catano/blob/64edc0a074b97500e505157d0cafc9706d45f140/src/game/game_data.rs#L219-L258) samples up to ten distinct legal moves, applies each to a temporary state and keeps the highest-scoring successor. `BEST_OF_N` is ten. Most rollout decisions use this procedure; some automatic robber handling in `simulate` instead selects randomly.

The handcrafted successor score is:

`10000 * own VP + 100 * production pips + longest-road length + 50 * has-any-port`.

Production accounts for cities and the robber. A proposed extra heuristic function currently returns zero. There is no neural policy, learned value function, PPO optimizer or training dataset in the inspected project. These facts come from [`heuristic_score`](https://github.com/Algorhythm-sxv/monte-catano/blob/64edc0a074b97500e505157d0cafc9706d45f140/src/game/game_state.rs#L635-L684), the simulation code and `Cargo.toml`.

This gives simulated players basic priorities immediately: improve production, acquire points and recognize ports. Search then estimates which candidate moves win more often under those simulated continuations. The rollout score is not a training reward, and does not establish that adding the same score to PPO would improve our agent.

### Simulation and interface design

Players' settlements, cities and roads use integer bitboards. Legal-action collections use `u128` flags. `GameState` is a fixed-size copyable Rust value, and the search stores nodes contiguously in a `Vec` arena with integer references. Immutable adjacency tables are precomputed. These choices are designed to make repeated successor generation cheap; no measured speed ratio against our Python/NumPy engine was established.

Equivalent states reached by different move sequences are not merged; the node documentation explicitly says transpositions are not identified. Each command constructs a fresh search. The command interface supports fixed playout counts or a time budget, plus move statistics. Search itself is single-threaded; the match runner parallelizes independent matches.

Our engine already shares immutable geometry, supports isolated copies and owned in-place transitions, and has batch inference. The transferable performance question is whether measured forward-simulation costs justify an additional native path, while preserving those contracts.

## Differences from our project

| Area | Monte Catano public snapshot | Our completed learned-agent baseline |
| --- | --- | --- |
| Decision method | UCT search and terminal simulations for each move | Masked neural policy inference; no online tree search |
| Knowledge | Handwritten rollout score | Two-layer width-128 MLP trained with PPO; greedy reference and imitation machinery also available |
| Information | Full `GameState`, including all hands and cards | Seat-specific `Decision`; opponent resource/card compositions hidden |
| Compute cost | Repeated simulations at move time | Training pays most learning cost; inference is a small forward pass |
| Main learning signal | No learning; terminal winners backpropagate search statistics | Actual winner reward in the completed baseline experiments |
| Discards | Engine automatically removes most-held resource cards | Explicit complete discard vector selected by the acting seat |
| Resource supply | No finite bank represented | Finite bank, legal supply checks and shortage handling |
| Evaluation | Built-in two-player, swapped-seat version comparison | Three/four players, all candidate seats, random/greedy opponents and fresh tuning/test boards |

Both projects include development cards, road/army awards and bank/port trades. Neither implements negotiated player-to-player trading in the inspected paths. Their random-board sampler also forbids adjacent identical numbers and adjacent 2/12, in addition to adjacent red tokens; that differs from our board distribution. Equal numerical seeds across engines therefore do not define comparable boards.

Our selected original PPO checkpoint won 187/224 games against random and 1/224 against greedy in its original final evaluation. The later small imitation/mixed pilot did not establish a reliable greedy improvement and regressed against random. These are documented limitations, not a prediction of a Monte Catano matchup. Its heuristic score already emphasizes production and points, the same areas our decision diagnosis found weak. Our existing greedy policy already evaluates all legal setup locations and explicitly prioritizes economic actions; best-of-ten sampling alone is not an established improvement over that reference.

## How much evidence supports strength?

The [README](https://github.com/Algorhythm-sxv/monte-catano/blob/64edc0a074b97500e505157d0cafc9706d45f140/README.md) advertises MCTS and an SPRT match runner, but supplies no external rating, human benchmark, opponent tournament or reproducible performance table.

A [2026-06-10 commit](https://github.com/Algorhythm-sxv/monte-catano/commit/acf04b018881daf1996acf06135b501c7594543d) reports approximately +73.3 Elo with a 30.9 margin from actually enabling best-of-N rollout selection. That is an author-reported relative improvement, without raw games or a complete benchmark configuration in the repository. It is not an absolute Catan rating and does not demonstrate a win rate against us. The latest heuristic-extension commit reports a neutral result.

The match runner uses two-player games, repeats each board with the engines swapping seats and runs independent pairs in worker threads. Those are useful testing ideas. Its SPRT thresholds target 0 versus +10 Elo with alpha/beta 0.05. However, its likelihood calculation uses a normal approximation and aggregate Bernoulli variance, even though it records paired outcomes; it does not use the pair distribution's variance. The code also returns zero evidence when every result is a win or every result is a loss. This testing implementation should not be copied as a calibrated statistical procedure without further validation. Source: [`src/sprt.rs`](https://github.com/Algorhythm-sxv/monte-catano/blob/64edc0a074b97500e505157d0cafc9706d45f140/src/sprt.rs#L46-L75).

Our whole-board paired intervals address a different, multiplayer evaluation design. We can add a properly designed sequential test later without abandoning fresh final tests or treating a two-player Elo gain as multiplayer strength.

## Correctness and information concerns

These are source-derived findings. I did not execute the foreign engine to measure their frequency or reproduce runtime failures.

**Privileged information.** `Mcts::new` receives a complete `Game`; nodes copy full `GameState`, which contains all four `PlayerState` values, exact resource hands, development cards and remaining deck counts. Opponent legal moves and simulations use those exact hands. The match protocol sends determined development-card purchases and steals to both engines. No observer boundary, belief-state sampler or hidden-state reconstruction was found. This is a perfect-information search interface, unlike our policy interface. The word determinized in their action comments concerns resolving random outcomes, not sampling plausible concealed opponent hands.

**Dice-node recognition.** `Node::is_roll` checks whether `next_sibling` equals `NodeRef::ROLL`, but `Node::roll` sets that field to `INVALID`. No assignment of the roll marker exists in the inspected source. Ordinary roll nodes can therefore fall through to decision-style UCT selection over outcomes, instead of weighted dice sampling. Separately, the final-setup chance node falls through `choose_child` to its first child, the roll-two branch. References: [`graph.rs:344`](https://github.com/Algorhythm-sxv/monte-catano/blob/64edc0a074b97500e505157d0cafc9706d45f140/src/graph.rs#L344-L355), [`graph.rs:499`](https://github.com/Algorhythm-sxv/monte-catano/blob/64edc0a074b97500e505157d0cafc9706d45f140/src/graph.rs#L499-L525), [`mcts.rs:116`](https://github.com/Algorhythm-sxv/monte-catano/blob/64edc0a074b97500e505157d0cafc9706d45f140/src/mcts.rs#L116-L143).

**Dice expansion applies a roll before constructing outcomes.** `expand` first applies `Roll(0)`, resolving a random roll and modifying resources or discards. It then passes that resulting state into `create_roll_nodes`, which applies individual dice outcomes again. The resulting branches can include effects from both rolls. This follows from [`mcts.rs:56`](https://github.com/Algorhythm-sxv/monte-catano/blob/64edc0a074b97500e505157d0cafc9706d45f140/src/mcts.rs#L56-L93) and `GameState::apply_action`'s roll branch.

**Largest Army threshold.** The award starts at `(PNone, 0)` through its default tuple. Playing a knight awards two points whenever the player's count exceeds that value, permitting the first knight to receive Largest Army. Our engine requires three. References: [`game_state.rs:532`](https://github.com/Algorhythm-sxv/monte-catano/blob/64edc0a074b97500e505157d0cafc9706d45f140/src/game/game_state.rs#L532-L545), [`game_state.rs:687`](https://github.com/Algorhythm-sxv/monte-catano/blob/64edc0a074b97500e505157d0cafc9706d45f140/src/game/game_state.rs#L687-L696).

**Other rule differences.** Production and resource-granting cards have no finite-bank shortage accounting. Road extension checks adjacent owned edges without checking an intervening opponent building; longest-road calculation receives only that player's state and settlement placement does not recompute blocked roads. Terminal detection checks any player's VP rather than requiring victory on the winning player's turn. These affect what the agent learns to favor in simulations. Relevant code is in `game_state.rs` and [`player_state.rs`](https://github.com/Algorhythm-sxv/monte-catano/blob/64edc0a074b97500e505157d0cafc9706d45f140/src/game/player_state.rs#L73-L135).

The position-import command also ends in `todo!`, so a robust external-engine adapter would require work. These limitations do not invalidate MCTS as an approach; they prevent treating this snapshot as a verified drop-in strong opponent.

## What we can learn and test

1. **Add a planning baseline on our engine.** Start with bounded setup search, then small main-turn spending sequences. Compare immediate scoring, short rollouts and deeper search under fixed decision-time budgets. Use our greedy and frozen neural agents as explicit rollout controls. Setup is an attractive first case because the board is public and there are no accumulated private opponent hands yet; it still requires honest stochastic future simulation.

2. **Use the existing tactical failures as regression cases.** Check whether planning takes the verified immediate winning settlement, avoids the harmful four-ore trade and produces better starts on unfamiliar boards. Those diagnostics can explain a change; fresh multiplayer wins remain the strength criterion. An immediate-win rule or search override used for play must be represented honestly if later included in training behavior.

3. **Keep hidden information correct.** For later-game search, generate plausible concealed hands/cards from the observer's public history and private hand, with a separate search RNG. Never pass actual opponent arrays or the live game RNG into the planner. Public resource bounds are useful constraints, but bounds alone are not a complete posterior over hands. A first root-sampling planner would still need checks for information leakage and inconsistent simulated opponent knowledge.

4. **Combine search with learning in stages.** First test search using greedy rollouts. Then test a frozen neural policy/value to guide or shorten rollouts. If search improves held-out decisions, collect its actual actions or visit distributions as separate imitation targets. This can move expensive planning into teacher-data generation while keeping the deployed network fast. Our recent imitation pilot shows why lower teacher loss cannot substitute for win-rate validation. Search-generated behavior must not be mislabeled as fresh on-policy neural PPO data.

5. **Separate our three proposed hypotheses.** Their heuristic supports investigating placement/economic evaluation. It provides no experimental evidence for PPO partial-VP rewards: rollout scoring and learner rewards are different mechanisms. It also provides no validated solution to opponent-resource inference, because the current search is supplied exact hands. Preserve separate controlled tests for placement, reward shaping and public-history knowledge before combining them with search.

6. **Measure before changing the simulator language.** Profile successor copying, rule queries, simulation and policy inference at identical workloads. Their bitboards and contiguous storage are useful design references, but Rust alone does not establish better strength per second. If native simulation becomes worthwhile, retain our IDs, seeded behavior, information boundary and independent invariant/replay gates. Report search latency alongside wins; a fast neural move and thousands of simulations have different deployment costs.

The most useful next experiment from this repository is a small, information-correct planning baseline on our engine. It would tell us whether explicit lookahead fixes the known economic mistakes and whether stronger search-generated teaching is worth its cost. No implementation, matchup or additional training was performed in this investigation.
