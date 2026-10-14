# Catan engine: architecture, decisions and performance

Current 2026-10-13 update: [the optional model/targets](policy-value-targets.md) and [bounded PPO updates/checkpoints](ppo-updates.md) are implemented. Earlier next-work statements below describe dated milestones. Long training/self-play and learned-strength comparisons remain future work; existing PDFs are unchanged.

2026-10-13 update: the [per-seat rollout collector](rollout-collector.md) is now implemented. It records same-seat successors, complete discards and outcome/cutoff data. The dated report and PDF remain historical; models/training are still held.

2026-10-12 update: [agent environment, outcome rewards and evaluation](agent-environment-evaluation.md) are now implemented. The dated milestones and next-work statements below describe the 2026-10-11 report; its PDF remains unchanged. Training remains held.

Version 2026-10-11. Main project: C:/Users/david/Documents/Coding/Catan.

## 1. Purpose and scope

This report explains every material implementation choice recorded in source, tests and project documentation: what was chosen, why, how it works, and the tradeoffs. It includes an architectural explanation, a numbered decision register, the current rules/performance contract, exact field/interface tables, reproducibility commands and the full milestone records. The original Catan_RL_Guide.pdf is unchanged. It is a roadmap, not evidence that every described component is implemented.

Undocumented historical motivations cannot be reconstructed with certainty. Reasons based on design consequences are architectural rationale, while measured optimization reasons are explicitly tied to reports. The report does not claim to recover every keystroke or speculative idea. Historical appendix statements such as eight arrays, 1461 features, no cards or the old seed-3 stall describe earlier implementations. Current state has fourteen arrays and schema-v2 observations have 1505 values.

Current scope includes generated geometry, constrained seeded layouts and ports, independent state copies, a playable engine, explicit manual discards, all five development card types, Longest Road, Largest Army, validation tools, relative observations, batching and action masks. Negotiation, full-game browser interaction, rewards/environment wrappers, neural models, PPO, symmetry augmentation and search remain deferred. Training was explicitly held back by the user.

## 2. How the system works

The data flow is immutable GEOMETRY -> seeded Board -> owned GameState -> typed action validation -> shared mutation -> next decision. Optional checks inspect states. Optional RL utilities create restricted observations and masks without changing the true state. A caller chooses an action, then the engine validates it again. The engine never imports the RL layer.

The catan package contains board, state, actions, cards, awards, engine, validation and the random agent. The rl package contains encoding and action lookup. Scripts provide benchmarks, bounded validation and viewer/demo servers. Tests exercise layers independently and together. Documentation generation is separate from application imports and dependencies.

A headless loop creates a seeded game and separate policy Random, prepares one encoder for its board, chooses legal actions and advances the exclusively owned state in place. A functional transition instead makes an independent branch. Both stop at a real winner or an explicit cap. The board/discard browser consumes stdlib previews and bounded sample-hand sessions; it is not a full-game UI. New cards are available through Python APIs, not automatically through browser controls.

## 3. Foundations: packages and board geometry

### D01 - Separate rules and learning

catan imports NumPy where state needs it, but never torch, graph libraries or rl. RL reads the engine in the opposite direction. The guide suggested torch early; it was deferred because no implemented behavior needs it. This keeps testing, bots and the viewer useful without a training stack. A fresh-interpreter test imports every implemented module and checks that no ML framework was loaded.

### D02 - Container-first dependencies

The Python checks image installs pinned NumPy and development tools; a separate Node target tests viewer scripts with built-in adapters. This follows the user's working agreement and avoids host installs. Interpreter versions accompany timings. Docker is a reproducible tooling boundary, not a promise that future builds or host loads have identical performance. PDF tools stay separate.

### D03 - Generate geometry once

Hex-coordinate construction deduplicates corners/sides into 19 tiles, 54 vertices and 72 edges, with real drawing positions. Immutable reciprocal adjacency supports rules, rendering and future graph features. Generation replaces hand-maintained tables and runs only at import. Counts, degrees, connectedness, incidences and a fingerprint are verified before gameplay relies on them.

### D04 - Preserve IDs when optimizing

Every ownership array, spatial action, observation and viewer label uses stable location IDs. Reordering would silently change stored meanings. Geometry optimizations therefore preserve the eleven-table fingerprint and original ordering. An intentional future migration must provide a mapping/version rather than replacing expected hashes merely to pass tests.

### D05 - Separate static Board and live state

Resources, number tokens, ports and geometry are immutable during a game. Copies share Board; live robber location belongs to GameState. Board.robber_tile is the starting desert, not the current block. This prevents branch mutation of shared data and permits safe per-board encoder preparation without copying the board each move.

### D06 - Enforce all adjacent red-token prohibitions

Any edge-adjacent pair in {6,8} is forbidden, including 6/6 and 8/8. The user required this refinement even though the guide allowed postponing it. Desert has zero token and no production. Exhaustive legal-set checks across nineteen deserts, seeded-board fingerprints and large sampling gates supplement visual inspection.

### D07 - Directly sample legal four-red sets

A shared immutable cache groups legal four-tile independent sets by desert. The caller's RNG samples a set uniformly, assigns two 6s/two 8s and shuffles other tokens. Bounded setup avoids rejection loops and biased repair swaps. Historical paired setup tests found roughly fourfold improvement over full-token rejection. This measures board setup, not games or learning.

### D08 - Prune prefixes without seeded drift

The cache builder stops expanding a partial set when adjacency is already violated. Ascending enumeration preserves lexicographic cache order, sampler indices and RNG continuation. The historical audit reduced one-time construction from 6.48 ms to 1.70 ms. The 1000-board/RNG fingerprint stayed identical, establishing compatibility rather than a new layout distribution.

### D09 - Fix nine separated port locations

The thirty-edge coast is traversed stably. Nine port gaps repeat 3,3,4; endpoints, reverse vertex-port IDs and label positions are immutable. Types shuffle as four generic 3:1 plus one per resource at 2:1. Fixed slots avoid additional placement search, retain clear rendering and use the caller's RNG for type randomness. Port generation is separate from owned access.

## 4. State, copying and ownership

### D10 - Small contiguous arrays

Resources, VP and pending discards use int16; owners, buildings, pieces, cards, knights and cached trail lengths use int8. Physical bounds fit. Ownership -1 and building 0/1/2 codes remain consistent. Current array payload is 287/316 bytes for three/four seats; Python/ndarray headers, RNG and shared board are excluded. Compact payload does not imply equally small total object memory.

### D11 - Exactly three or four players

Base geometry and supplies support these seat counts. Fixed snake orders and padded four-seat observations keep rules and future model dimensions predictable. Other counts are rejected instead of suggesting an unsupported variant. Presence flags distinguish padding from an actual seat with empty hands or no pieces placed.

### D12 - Strict input normalization

Indices accept integer-like NumPy values but reject booleans, floats and strings. Resource vectors contain exactly five bounded integer counts; trade resources differ. Constructors normalize frozen values, while engine validators handle state-dependent legality. This avoids turning malformed UI/policy choices into unrelated valid actions through silent coercion.

### D13 - Complete per-state randomness

One Random generates the board then continues into dice, theft and purchases. A separate policy RNG chooses random-agent actions. No game outcome uses global random state. Replay requires the same code/runtime and action sequence; the same seed alone does not preserve games after intentional rule/action changes. Copy tests include Gaussian-cache continuation.

### D14 - Independent copies sharing immutable data

copy() bypasses initialization, directly copies every mutable array and restores the full RNG state into a separate object. Board/GEOMETRY are shared. copy.copy and deepcopy retain isolation; deepcopy also preserves nested alias relationships. A shallow assignment would let one branch corrupt its parent or sibling. All added fields participate in this contract.

### D15 - Complete equality, no mutable hash

Equality compares Board, every scalar, array shape/dtype/values and full RNG. Small integer arrays use checked byte comparison; noninteger debug arrays retain NumPy semantics. State is unhashable. No cached equality hash can become stale after mutation. Fingerprints serve replay diagnostics, not a mutable dictionary-key contract.

### D16 - Functional and exclusively owned APIs

apply_action validates then creates an independent successor. apply_action_inplace validates then updates the same object, retaining arrays and RNG identities. Shared mutation avoids duplicate rule implementations. Owned rollouts avoid copies; branching/UI callers retain the functional API. Exclusive ownership is the caller's responsibility, not concurrency enforcement or an undo transaction.

### D17 - Reject before copy, mutation and draws

All rule rejection occurs before side effects. Rejected calls preserve values, identities and RNG; functional rejection also avoids copying. Valid storage is assumed in fast APIs. Corrupt arrays, races and arbitrary exceptions are not transactional rollback cases; use explicit validation and independent worker ownership to prevent them.

### D18 - Derive cheap facts, store necessary history

Winner derives from current seat at GAME_OVER; discard player from clockwise pending counts; setup order from seat count; port access from owned endpoints. Redundant cheap caches invite drift. Award holder history is necessary for ties, and exact trail length is expensive enough to cache between graph changes. These added caches are independently checked and fully copied.

## 5. Turns, resources and manual choices

### D19 - Explicit phases, stable enum values

The original seven phases remain unchanged. DISCARD was appended as eight and ROAD_BUILDING as nine. Continuation records ROLL or MAIN for card effects. Explicit states prevent building during robber selection or ending a pre-roll card turn without dice. New phase values intentionally expand observation schema but preserve old numeric addresses.

### D20 - Paired snake setup

Each free settlement obeys distance; its road touches that anchor. Only the second round grants resources from incident non-desert tiles. Bank, pieces, VP, index, anchor and next player move together. Supply is validated first. Both initial buildings can root later road components, and initial road placement needs no earlier connected network.

### D21 - Shared placement rules

Roads use an empty edge connected through an own building or an unblocked own road. Opponent buildings stop extension. Paid settlements need distance and an owned incident road. Cities upgrade own settlements, return the settlement piece and add one VP. Queries, masks and action validation share predicates instead of maintaining separate legality implementations.

### D22 - Aggregate production before shortages

Every matching unblocked tile contributes demand before bank payment. Settlements yield one, cities two. Enough supply pays all; insufficient supply pays none for multiple recipients or the remainder to a sole recipient. Other resource payouts remain independent. Small per-call Python lists replaced tiny NumPy matrices after profiling; a building-first oracle tests behavior independently.

### D23 - Best bank/port ratio without cache

A trade takes one available bank card for four offered, three through generic access or two through a specific port. Generic access never replaces a better specific discount. Nine endpoint checks and offered-resource action groups avoid repeated allocation/affordability work. Cards paid return to the bank, preserving nineteen per resource.

### D24 - Snapshot seven obligations once

Only hands strictly above seven owe floor(total/2); exactly seven owes zero. Pending counts snapshot before choices so a large remainder is not discarded twice. The existing int16 array is reused as a reduction output; valid totals are at most ninety-five. Known-seven helpers draw no RNG; RollAction still draws two actual dice.

### D25 - Explicit counts and separate turn owner

DiscardAction names the next pending seat and five selected counts, exactly equal to its obligation and no larger than its hand. The engine never chooses automatically. current_player remains the roller while discard_player derives clockwise from pending counts. Completed obligations clear once; all choices finish before robber movement. The explicit random agent samples physical cards as policy behavior only.

### D26 - Two robber decisions and physical-card theft

The robber moves to a different tile; eligible victims are distinct adjacent opponents with cards. No victim skips stealing without a draw. A selected victim loses one uniformly sampled physical card, making resource probability proportional to count. Separate move/victim actions avoid a tile-by-victim Cartesian catalog and preserve the user's control.

### D27 - Wins are different from caps

Ten true VP on the current player's turn enters GAME_OVER. Caps are truncations, never invented wins. The historical minimal seed-3 exhaustion stall is documented; expanded cards/awards intentionally change its trajectory and it now finishes. Finite stress gates do not prove universal termination. Future reward bootstrapping must preserve the terminal/truncated distinction.

## 6. Development cards

### D28 - Five-count conditional deck

The deck starts at 14 knights, 5 VP and 2 each Road Building, Year of Plenty and Monopoly. Buying draws a weighted integer and decrements one remaining type. This matches uniformly shuffled sampling without replacement in distribution, while avoiding an ordered deck/cursor and eager RNG shuffle. Initial board/RNG fingerprints remain compatible. It does not promise the same individual sequence as an eager-shuffle implementation.

### D29 - Held, fresh and public spent counts

Held counts include private VP; fresh counts are a subset held by the current turn owner; spent counts conserve types. Knights also have per-seat played counts for army scoring. Fresh clears at end turn, making those cards eligible on the next own turn. Counts, timing and holders enter copies, equality, corruption checks, array size, fingerprints and restricted observations.

### D30 - Unknown purchase outcome inside engine

BuyDevelopmentAction cannot request a type. It pays sheep+wheat+ore and samples remaining supply. Repeated affordable purchases are legal while the deck remains. Rejected purchases consume no draw. VP updates score immediately, including a fresh winning VP. Moving outcome selection into a policy would control hidden randomness instead of playing the game.

### D31 - One older non-VP card per turn

Playing requires held minus fresh to be positive and development_used false. It consumes one card, increments spent count and sets the turn flag. Buying remains independent: a player can buy after playing, or play an older copy after buying a new card of that type. VP has no explicit play action, remains held and is exempt from the one-card restriction.

### D32 - Cards may precede dice

Playable cards are legal in ROLL and MAIN. Multi-choice effects remember which phase to resume. A knight before rolling does not set last_roll=7 or bypass the dice. Immediate Year of Plenty/Monopoly leave their original phase. This follows official rules and avoids the convenient but incorrect restriction of all cards to MAIN.

### D33 - Knights never trigger discards

A played knight increments its public count, updates army and enters robber movement. It never calls seven resolution. Existing movement/steal rules finish then resume the saved phase. Rich hands remain unchanged unless actual dice roll seven later. If the army bonus immediately wins, the game ends without forcing irrelevant robber choices.

### D34 - Sequential free RoadAction choices

Road Building enters its own phase with up to two pending free roads. Each choice uses ordinary placement without cost. The second sees the first, enabling extension. The effect finishes when pending count, available pieces or legal spaces are exhausted. This avoids bundled two-road combinations and a second placement implementation. The current interface requires both roads when both can be placed.

### D35 - Exact feasible Year of Plenty counts

The player explicitly takes two cards, duplicates allowed, or all available when total bank supply is below two. Each chosen resource must exist. The transfer has no draw. Twenty-one catalog vectors cover totals zero, one and two; masks select only feasible choices. The engine does not choose resources or permit taking unavailable supply.

### D36 - Monopoly transfers opponents, not bank

One chosen type transfers every opponent's full holding to the current player, leaving all other types and bank unchanged. Zero gain is legal and consumes the card. Bounded seat iteration needs no random draw. Opponent composition is true simulator state and is not revealed to the policy simply to simplify choosing a type.

## 7. Longest Road, Largest Army and victory

### D37 - Exact edge trails, not connected-size approximation

Longest Road cannot repeat edges but may revisit vertices, so loops count and branches cannot simply be summed. Only the player's at-most-fifteen edges enter search. A bit mask tracks used edges and per-call memoization uses vertex+mask. A slow independent used-edge-set oracle covers sampled graphs and explicit loop/block cases. Search exits early when all owned edges fit one trail.

### D38 - Opponent buildings stop passage

Arriving at an opposing building ends a trail; starting there as an endpoint is allowed. Own buildings do not block. A loop with one opposing vertex can use the whole loop with that vertex at both ends; two blocks split it. Rooted-component validation deliberately traverses existing roads through later blockers: an interrupted old road is not thereby illegally disconnected from its historical root.

### D39 - Recompute only on graph changes

Road placement refreshes the builder's length. Settlement placement refreshes incident opposing owners whose traversal may be blocked. City upgrades, rolls, trades and resource effects change no blocking graph. Per-seat int8 cached lengths avoid repeating trail search for ordinary actions. The debug gate independently recomputes them; copying and equality include them.

### D40 - History-sensitive holders and ties

Longest Road needs five, army needs three played knights; each grants two VP. Incumbent retains a tie at the maximum. A strictly higher eligible unique leader takes the award. If a road holder falls below top and other leaders tie, the award becomes unheld. Scores remove/add the bonus on transfer. Counts alone cannot reconstruct incumbent history, so holders are necessary state.

### D41 - True score with restricted observations

state.vp totals settlements, twice cities, held VP cards and award bonuses. Fast winner checks use that total. Observations show own true score and opponent public score minus hidden VP; terminal observations reveal all scores. Debug validation checks the exact equation. Treating true state as a public feature would leak card types and change the learning problem.

### D42 - Own-turn winning and continuation cleanup

An award can put another seat at ten off turn; that seat waits until its turn. EndTurn checks the newly current seat and can produce a pre-roll win with no last dice. Debug checks permit off-turn winning totals and terminal last_roll=None. Immediate card/free-road wins clear irrelevant continuation and pending free roads. Only current_player identifies the actual winner.

## 8. Observation and action representation

### D43 - Flat normalized relative baseline

The initial MLP-friendly encoder creates independent contiguous float32 features. Small categorical IDs become one-hot channels, not ordinal magnitudes. Physical maxima normalize counts and VP divides by ten. Values are not clipped; eleven VP is 1.1. Implementing encoding does not establish that a neural policy learns better or that any training pipeline exists.

### D44 - Probability plus high-number bit

Tile features include resource one-hot, pips/36, number>7 and robber flag. Probability alone conflates 6 and 8 and loses shared-roll correlations; one bit recovers mirrored token identity without eleven extra flags. Vertices encode owner/building/port/anchor, edges encode owner. Stable spatial IDs remain the same as geometry/actions.

### D45 - Decision-maker perspective

Seats and owners remap clockwise relative to the observer. Default is the manual discarder in DISCARD, otherwise turn owner; GAME_OVER has no actor and defaults to winner. Explicit observers support terminal/value queries. Tests rotate seat arrays and holders consistently. This is cyclic seat relabeling, not implemented board rotations/reflections.

### D46 - Explicit hidden-information boundary

Own resource/card composition is visible. Opponent resource/card totals are visible, exact composition and fresh types are hidden. Deck exposes size, not remaining type counts. Board/bank, pieces, pending counts, spent cards, knights, road lengths and awards are exposed. RNG is hidden. Conservation can reveal aggregate facts; no public-history memory or belief state is implemented.

### D47 - Input schema versioning

Version one had 1461 floats/5844 payload bytes. Version two has 1505/6020: larger player rows, a new phase and card/award/timing groups. Shape/slice maps define layout once. Future saved models/replays must reject incompatible schemas instead of silently using the wrong input. Old numbers remain only in dated measurements.

### D48 - Immutable preparation and output ownership

FlatEncoder prepares a byte-backed read-only template per Board identity. Copies share that context; resets need a new context even when layouts compare equal. There is no accumulating global board cache. Each output is fully overwritten. Optional out storage must be writable contiguous float32, exact shape and disjoint from all fourteen state arrays. Consume/copy it before reuse; reuse is not automatically faster.

### D49 - Bounded mixed-board batching

encode_batch accepts per-row contexts and optional observers, three/four seats and empty input. All lengths, contexts, observers and aliases validate before writing. Below eight rows uses scalar encoding; larger batches gather spatial channels in chunks of at most 128 and share global construction. Scratch stays local/bounded. This supports future batched inference without sharing game state or importing a model.

### D50 - A fixed 254-slot catalog

The guide's first 220 road/settlement/city/robber/trade/end addresses remain. New blocks add roll, relative steal victims, purchase, knight, free-road card, Monopoly and Year of Plenty vectors. Missing fourth-seat victims and self targets are masked. Catalog membership/decoding does not mean legal. Immutable cached objects and a single mapping module prevent output-contract drift.

### D51 - Direct masks with shared predicates

The reference projects legal objects to slots. The retained mask avoids that conversion for common choices and shares engine placement, affordability, trade and victim predicates. Small card choices use the same iterator. Tests compare every reached mask with the reference and roundtrip actions. No draw/copy occurs; terminal all-false masks are intentional rather than an invented pass.

### D52 - Five conditional discard masks

A ninety-five-card hand creates too many complete count vectors for a flat catalog. discard_count_mask instead offers twenty counts for the next resource given a zero-to-four-count prefix. Feasibility bounds ensure the rest of the hand can satisfy the obligation. Five selections create one DiscardAction, with no intermediate state mutations. Flat action_mask rejects DISCARD with the structured API instruction.

## 9. Validation and optimization

### D53 - Explicit layered debug checks

state.validate handles physical/storage bookkeeping; assert_invariants adds temporal, winning, rooted-component and recomputed trail/award checks. Neither runs implicitly in transitions/encoding. Continuous random validation opts in every step. This retains efficient rollouts and detailed diagnostics without weakening either API's stated contract.

### D54 - Independent oracles and full replay

Production uses a building-first oracle, features scalar loops, trails used-edge-set traversal and masks legal-object projection. Functional/in-place episodes compare all fields and complete RNG. Corruption and rejection tests inspect failures as well as successes. Hashes catch drift but do not prove rule correctness; finite examples also do not prove all histories.

### D55 - Bounded parallel validation summaries

Every worker owns state/RNG. Ordered results and bounded process-map submissions preserve deterministic report fingerprints without unbounded queued trajectories. First failure includes seed, seat count, step, phase and action and exits nonzero. Wins/caps are counted separately. This is verification parallelism, not a learned rollout collector.

### D56 - Profile complete work before adding complexity

The holistic profile includes reset, queries/policy, transitions and optional features. Queries, observations and functional copies dominate; trail recomputation is smaller. Cold import/compiler work is not steady-state overhead. The chosen changes reduce transient reads/writes instead of adding undo, packed storage, shared legality caches or GPU machinery without evidence.

### D57 - One global-feature write

Expanded encoding originally used multiple tiny ufuncs and slice writes. The retained writer converts bounded groups to Python values, uses precomputed float32 scales/flags and writes all globals together. Spatial categories remain vectorized. Paired every-observer checks preserve exact feature values. This changes allocation/dispatch work, not visibility or schema meaning.

### D58 - Transient query ownership and card snapshots

Road queries convert owner arrays to Python lists once, then call the shared predicate. Direct action validation still reads arrays. No mutable legality cache is stored. Card queries return early if unusable; Year of Plenty snapshots bank counts once instead of repeated reductions per candidate. Sorted settlement pruning and grouped trades from earlier audits remain.

### D59 - Retain useful improvements with qualified measurements

Prototype single-observation batching was slower, so scalar preparation remains. Larger batches and direct masks retained useful measured gains. Paired expanded-rule replay preserved 24199 steps and medians improved owned gameplay about 8.1%, gameplay with observations about 12.5%. Hardware/load matter; no neural inference or training is measured. The current contract and raw JSON show exact workloads and values.

### D60 - Reject insufficient benefit and defer unmeasured systems

A historical persistent discard scratch saved only about 0.7 us and was rejected for ownership complexity. A compact encoder template saved storage but was slower and was rejected. Undo, native compilation, GPU simulation, global board caches and learning-quality comparisons remain deferred. Their absence reflects current evidence/scope, not proof they could never help.

### D61 - Remove obsolete experiments and empty future modules

Verified task-specific snapshots/patch/probe scripts are removed after comparison; raw JSON/profile evidence remains. Six truly empty future RL modules were removed. Empty files did not slow gameplay; cleanup clarifies implemented scope. Useful benchmarks, tests, guide source/assets and historical records remain. No unrelated files are removed speculatively.

### D62 - Documentation tools stay outside engine

This Markdown report and its ReportLab generator are editable documentation sources. PDF authoring uses the supplied runtime and does not enter application requirements/imports. The new PDF is separate from the original guide. Source hashes record inspected artifacts. Application container checks remain independent of report generation.

## 10. Viewer and API

### D63 - Stdlib board previews, lazy state import

Board/GET routes need no NumPy or ML. NumPy-backed discard demo code loads lazily on POST only. This keeps visual inspection lightweight and avoids treating the board viewer as an RL frontend. Package-import and HTTP tests enforce the dependency boundary separately from later learning code.

### D64 - Encode immutable JSON geometry once

The API combines preencoded geometry with each seeded layout, preserving flat fields. Only the existing default preview is cached; other seeds do not accumulate. Compact separators historically reduced response size. Per-request state is bounded and legacy tiles access stays useful. Geometry reserialization is outside the rollout engine entirely.

### D65 - Numeric normalization and stale-response protection

Valid scientific-notation input is submitted as its integer value. A request-order guard prevents older success/failure responses from overwriting newer requests. Boundary/fractional inputs, aborts and recovery are tested with actual inline JavaScript. Separate browser inspection established appearance; adapters alone do not prove visual correctness.

### D66 - Bounded revision-checked sessions

The sample-hand demo keeps at most thirty-two in-memory sessions. A lock covers revision checks and functional replacement so duplicate/concurrent choices cannot discard twice. Stale/expired sessions have explicit errors. Next-hand forms reset; server/network failures retain selections for retry. This does not imply persistence or a multiplayer service.

### D67 - Keep manual decisions explicit across layers

Resource selectors submit chosen counts. The engine also exposes free roads and chosen Monopoly/Plenty resources explicitly, but this rule demo does not yet display their controls. Separating rules from UI avoids speculative frontend machinery. Future full-game UI should expose actual decision phases and restricted visible information through the same engine APIs.

## 11. Practical contracts and usage

Resource order is wood, sheep, wheat, brick, ore; desert is static tile data only. Development order is knight, VP, Road Building, Year of Plenty, Monopoly. Piece order is settlement, city, road. New fields must join initialization, copying/deepcopy, equality, invariant gates, fingerprints and observations when visibility requires them.

Finish consuming legal_actions before mutating its state. Do not retain a reused observation as multiple saved rollout steps. Do not share an in-place state with other workers/readers/branches. A mask aids selection; the engine still validates submitted actions. A true terminal state has no acting player or legal choice.

```python
from random import Random
from catan.state import new_game
from catan.engine import apply_action_inplace, is_terminal, winner
from catan.players.random_agent import choose_action
from rl.encoding import FlatEncoder

state = new_game(42, n_players=4)
policy = Random(42)
encoder = FlatEncoder(state.board)
for step in range(5000):
    observation = encoder.encode(state)
    if is_terminal(state):
        break
    apply_action_inplace(state, choose_action(state, policy))
print(winner(state), state.vp.tolist())
# None means a cap, not a fabricated winner.
```

Pre-roll Knight enters robber choices then returns to ROLL; pre-roll Road Building enters its phase, consumes legal free RoadAction choices then returns to ROLL. A rolled seven instead records last_roll=7, snapshots all obligations and resumes MAIN after robber decisions. These paths share rules but never confuse a card with an actual dice outcome.

The appendix tables are generated from current source and list exact state/observation/action addresses. Source docstrings define buffer/ownership behavior. README provides runnable checks and main-project commands. Historical records have current-boundary notices so prior rates and scope remain useful evidence without being presented as current behavior.

## 12. Roadmap and deliberate deferrals

### D68 - Expand rules before learning at user request

The guide suggests rule expansion after a learned baseline. The user requested cards/awards now and held training back, so state, rules, observations and masks advance together. This avoids immediately retraining on changed rules. Sections 1-7 are implemented; part of Section 10's interface and Section 17's expansions are now implemented. GNN, spatial augmentation and learning stages remain incomplete.

### D69 - Define environment/rewards before PPO

The next integration should specify reset/observe/step, actor identity, per-seat rewards and terminated/truncated separately. Conditional discards need a policy interface and correct decision-seat attribution. Save input/action schema versions in metadata. The current engine remains usable without a wrapper. This is planning, not an implemented reward or trainer hidden inside utilities.

### D70 - Make network/scaling choices conditional

An MLP can consume the current flat features/catalog; a graph model can reuse validated incidences and spatial groups later. Compare learning quality, forward latency and complete collection time before committing to heavier message passing, compilation or GPU transfer. Current batch/process numbers are measured examples, not established optimal training settings.

### D71 - Delay symmetry, search and negotiation contracts

Spatial transforms must permute ports, features, actions, masks, robber and setup anchor together. Fixed sampler port slots do not automatically make every transformed layout generatable by that sampler. Search also requires chance outcomes, hidden-hand/card beliefs and multiplayer values. Negotiation requires a bounded protocol and remains the largest base-game action expansion. None is needed for this requested engine increment.

## 13. Evidence and limits

Authoritative order is current source/tests, current expanded-engine contract, dated JSON reports, continuity/milestone records, then original roadmap. The guide is paraphrased where discussed. Official base rules and FAQ were consulted for card purchases/timing, robber effects, award thresholds/ties and own-turn victory. URLs are listed in the appendix. No formal completeness or universal-termination proof is asserted.

Timing medians use warmed alternating order without competing test/stress jobs. Minimal and expanded engines have different actions, trajectories and observation dimensions; their raw rates give context, not a causal optimization comparison. The paired optimization instead holds expanded rules/actions/state/RNG fixed. Stress elapsed time includes validation and concurrent work, so it is not a performance benchmark.

Randomized checks are finite and bounded. Fingerprints reproduce ordered summaries under the same implementation/runtime, not portable saved games. Board/RNG fingerprints remain unchanged because deck initialization draws nothing. Tests independently cover physical conservation, timing, connectivity, score equations, hidden information, copying, rejected choices and interface agreement. The following appendices retain detailed milestone records and current verification reports.
