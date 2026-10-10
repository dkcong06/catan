# Stage 4: minimal playable engine

Implemented 2026-10-10 in the main Catan project. This follows the minimal rules
in Section 4 of [the build guide](../Catan_RL_Guide.pdf), pages 10-11, with the
existing user-selected manual discards and port setup retained. Bank trades also
honor owned ports because their immutable metadata/access query already exists.
Development cards, longest road, largest army, and negotiated player trades remain
the guide's later expansions. No observation encoder, trainer, or ML dependency
was added. The browser remains the board/manual-discard demo.

## Engine interface

- `legal_actions(state)` yields legal actions in stable order as an iterator.
- `apply_action(state, action)` validates then returns an independent successor,
  including its complete RNG; it preserves the source on accepted/rejected choices.
- `apply_action_inplace(state, action)` performs the same validated mutation on an
  exclusively owned state, returning that object and retaining array/RNG identities.
- `is_terminal(state)` tests GAME_OVER; `winner(state)` returns its seat or None.
- Known-seven helpers remain available without dice draws for the existing demo.

Callers provide valid state. Full `state.validate()` checks storage/physical
bookkeeping explicitly and remains outside the hot path; it does not certify all
historical network connectivity. The engine enforces legality at each action.
Both APIs share validation, connectivity predicates, production, and mutations.
Rule checks finish before copies, mutation, or RNG draws. In-place ownership is a
caller contract; retained aliases observe accepted updates. Use independent copies
for snapshots/branches and separate states/RNG streams for workers.

The guide sketches a list of legal choices. The implementation deliberately
returns an iterator to avoid needless lists and eagerly expanding manual discard
combinations. Consume a query before changing the state. Scalar actions reuse
shared frozen/slotted objects; no mutable legality/port cache is maintained.
A direct DiscardAction is checked against its five counts without iteration.
Future action masks/structured policy heads belong in the RL layer.

| Action | Fields | Valid phase / behavior |
| --- | --- | --- |
| SettlementAction | vertex 0-53 | SETUP_SETTLEMENT or MAIN |
| RoadAction | edge 0-71 | SETUP_ROAD or MAIN |
| CityAction | vertex 0-53 | MAIN; upgrade own settlement |
| RollAction | none | ROLL; draw two seeded dice |
| DiscardAction | player, five resource counts | DISCARD; exact pending manual choice |
| RobberMoveAction | tile 0-18 | ROBBER_MOVE; different tile |
| StealAction | victim seat | ROBBER_STEAL; nonempty adjacent opponent |
| BankTradeAction | give, receive resource 0-4 | MAIN; receive one card at best ratio |
| EndTurnAction | none | MAIN; next player starts ROLL |

Indices/counts reject booleans, floats, strings, and out-of-range values; NumPy
integers normalize to Python integers. Trade resources must differ. Acting seats
come from state, except the explicit pending discard seat; stealing names a victim.

## Implemented rules

Setup uses the derived three/four-player snake order. Each free settlement obeys
the distance rule; its road must touch that settlement. Only the second placement
round grants one card per incident non-desert tile. Pieces, VP, bank, setup index,
and next seat update together. The first normal roll belongs to seat zero.

Dice and theft use only the state RNG. Ordinary rolls aggregate all matching,
unblocked tiles before payment; settlement/city codes yield one/two cards.
If bank supply is short for multiple recipients, none receives that resource;
a sole recipient receives the available remainder. Other resources still pay.
These rules are paraphrased from the [official rules/almanac (2020)](https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf),
resource-production, road, city, maritime-trade, and robber sections.

Seven snapshots manual obligations before any robber action. All choices must
finish first. The robber moves to any different tile, including the desert.
Victims are distinct adjacent opponents holding cards; no eligible victim skips
stealing without an RNG draw. The roller chooses the victim, then the engine draws
one physical card uniformly (resource probabilities proportional to held counts).

Paid roads require an empty edge connected through an own building or own road.
An opponent's building interrupts extension through that vertex. Settlements
require an empty spaced vertex on an own road. Cities upgrade own settlements,
return the settlement piece, and add one net VP. Costs are wood+brick for roads,
wood+sheep+wheat+brick for settlements, and two wheat+three ore for cities.
Payments return cards to the bank; piece limits remain 5/4/15.

BankTradeAction receives one available card, paying the lowest applicable ratio:
4:1 normally, 3:1 through a generic port, 2:1 for the offered resource's specific
port. Access is derived from buildings at the fixed port endpoints. End turn
rotates the seat, clears last_roll, and enters ROLL. Ten building VP on the current
player's turn immediately enter GAME_OVER; winner is derived, not a new state field.

## Baseline and reproducibility

```python
from catan.engine import winner
from catan.players.random_agent import play_random_game

state = play_random_game(0, n_players=4, max_actions=10_000, validate=True)
assert winner(state) == 3
assert state.vp[3] == 10
```

`choose_action(state, policy_rng)` is an explicit random agent. It uses a policy
RNG separate from the game RNG, choosing among scalar legal actions and sampling
physical discard cards directly. The engine and browser never automatically pick
manual discards. `play_random_game` defaults to in-place steps; `inplace=False`
uses independent successors. Validation is opt-in and the action cap is explicit.

The minimal rules can stalemate. Seed 3 with four uniformly random players reached
scores [8,8,9,9] with every road and city used; even funded probes had no legal
building action. Scores stayed unchanged through 100,000 actions. Increasing the
cap cannot resolve that exhausted expansion. A capped baseline raises RuntimeError;
it never claims GAME_OVER or a winner. The benchmark reports these as capped runs.
Later RL environments must distinguish truncation from terminal wins. This is a
limitation of the guide's minimal rules, not permission to alter victory scoring.

## Efficiency and measurements

No state fields or array sizes changed: raw payload remains 241/258 bytes for
three/four players. Geometry/board IDs and seeded setup are unchanged; ports and
no adjacent 6/6, 6/8, or 8/8 retain their existing gates. The owned path avoids
full-state/RNG copying. Existing pending storage is reused on sevens, selected
counts are converted once, costs touch only relevant resources, and action objects
are shared. Production allocates one small demand matrix only on ordinary rolls.
No global scratch buffer, undo framework, network dependency, or host package was
introduced.

```powershell
docker build -t catan-checks .
docker run --rm catan-checks python -m scripts.benchmark_games
docker run --rm catan-checks python -m scripts.benchmark_games --players 3
```

Defaults are ten seeded games, five warmed alternating batches, and a 10,000-action
cap per game. Both APIs use identical seeds and random decisions; step counts,
final states, and complete RNG match outside the timer. The timer includes game
reset, legal-choice generation, and policy selection, excluding per-step invariant
checks, rendering, and neural inference/training. Capped runs remain in elapsed
work but are counted separately from completions. Flags allow games/repeats/player
count/start seed/cap changes. Reported actions/s are baseline gameplay throughput,
not future RL throughput.

Measured 2026-10-10 on Python 3.14.7 / NumPy 2.4.6, five warmed alternating
batches without competing check/stress jobs:

| Players / seeds | Actions per batch | Completed / capped | Functional actions/s | In-place actions/s | Time ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four / 0-9 | 38,973 | 7 / 3 | 9,644 | 17,817 | 1.85x |
| Four / 0-2 (`--games 3`) | 3,183 | 3 / 0 | 9,390 | 16,528 | 1.76x |
| Three / 0-9 (`--players 3`) | 12,968 | 10 / 0 | 10,080 | 19,469 | 1.93x |

The default four-player workload contains substantial capped-game work. The
separate fully completing seed batch prevents presenting that mixed workload
as a rate for completed games alone. These rates depend on hardware, runtime,
policy, seed mix, and load; they do not measure neural inference or learning.

The discard-only [optimization measurements](engine-optimization.md) describe
an earlier milestone and should not be treated as complete-game throughput.

## Verification and next stage

529 Python tests and 21 JavaScript tests pass. New tests cover 100 seeded setup
sequences; all action-phase rejection combinations; distance/blocked connectivity;
building costs and reclaimed pieces; production, city multipliers, and bank
shortages; manual seven scheduling; distinct victims and weighted seeded theft;
port ratios/supply; ten-point victory; lazy complete discard choices; bounded
stalemates; and functional/in-place full-game replay with all array/RNG identities.
Existing board/setup fingerprints, copy/Gaussian-cache tests, and browser HTTP/
selector tests remain green. Ruff lint/33-file format, compilation, dependency
checks, and the checks image build pass. Current browser scripts were tested with
a read-only main-source mount into the existing Node image; no UI change was made.

Stage 5 is next: broader randomized rule validation and reachability/connectivity
checks, replay/stalemate reporting across many seeds, and further profiling if it
shows a bottleneck. Observation/reward/mask interfaces and parallel training come
after that gate. PDF/generator/assets and the state schema remain unchanged.
