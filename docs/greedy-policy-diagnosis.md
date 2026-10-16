# Why the trained bot loses to greedy opponents

## Finding

The main weakness is resource planning: the frozen update-350 model tends to
choose familiar locations, starts with weaker production, and spends or discards
cards without consistently preserving a route to points. It occasionally ignores
a guaranteed winning build. The results support several interacting weaknesses;
there is no single identified engine or sampling bug that explains the gap.

This investigation made no model updates. The greedy opponent is an observation-only
handwritten heuristic, not a trained neural model. All diagnostic hybrid controllers
are experiments that temporarily replace selected decisions. They have not been
installed as the normal bot or used as PPO training data.

## Protocol and evidence

The checkpoint is the tuning-selected u00000350.pt from the completed two-hour run.
Its SHA256 remains 67a0ad501d00b5031e3e53f3f87383f96a21b1224ce812907bf6f29d8c1de8a2.
The existing model, engine, observations, action catalog, training configuration,
manual-discard rules and board generation were preserved.

An exploratory probe used eight fresh board seeds 3000002000-3000002007, both
player counts and every candidate seat: 56 games against greedy and 56 against
random. The unchanged model won 1/56 and 34/56 respectively. Those eight-board
results are descriptive and do not replace the original 32-board held-out test.

The first controlled experiment used sixteen other seeds, 3000003000-3000003015.
For each controller it played 112 games against greedy: three- and four-player
matches, rotating the candidate through every seat. A second, explicitly declared
follow-up used eight further seeds, 3000004000-3000004007, for 56 games per controller.
Each group preserves a separate baseline. A targeted seven-game replay saved the
full trace of an identified missed winning opportunity. Total annotated work was
847 games / 292,899 actions, with no cutoffs and full invariant checks. The earlier
112 exploratory games are additional and are not included in that annotated total.

Same seeds pair initial boards and roles. Changes in choices can change later dice,
steals and policy trajectories. Pointwise intervals bootstrap whole seeds, not
individual seat rotations: 10,000 draws with RNG seed 91. These are small exploratory
controller comparisons, not multiplicity-adjusted claims or trained-policy improvements.
No checkpoint was selected using these diagnostic results.

## What the bot does differently

### Placement: a fixed-location preference outweighs board quality

When the learner had the first seat, it placed its first settlement on vertex 19
in 28 of 32 cases across sixteen boards and two player counts. Vertex 33 and vertex
26 each appeared twice. The mean chosen production score was 8.94 versus 12.50 for
the best legal location. The two player-count cases on a board are correlated;
these are not 32 independent board samples.

Across all 224 learner setup placements, the mean ratio of chosen production to
the best legal production was 75.75%. The two starting settlements provided 16.21
production units per game, compared with 21.92 for a greedy opponent participant.
This is a strong observed position bias. It does not prove that the representation
cannot learn board-aware placement, or that maximum production is always optimal.
Resource balance, ports, expansion and opponents also matter.

A production unit is one of the 36 equally likely two-dice outcomes. A tile numbered
6 or 8 has five units, 5 or 9 four, and so on. The scores sum adjacent producing
tiles and ignore robber blocking and bank scarcity. They describe income potential,
not realized cards or a guaranteed best strategy.

In one three-player game on seed 3000003000, seat 0:

- First settlement: vertex 19, touching brick 5, ore 11 and ore 5. It had 10 units;
  a legal alternative had 13. The model assigned its chosen vertex about 85.3%
  probability and the greedy alternative about 0.19%.
- Second settlement: vertex 33, touching wheat 3, wood 10 and sheep 11. It had
  7 units; a legal alternative had 10. The alternative received about 0.00052%
  probability from the model.

The raw examples are in [baseline.json](../output/greedy-diagnosis/phase-controls/baseline.json).

### Economy: too little ore and wheat, followed by expensive recovery

In the 112-game baseline, 41 starts had no ore production (36.6%) and 26 had no
wheat production (23.2%). This can make growth expensive: a city requires three ore
and two wheat; a development card requires one sheep, one wheat and one ore.
A port-based economy can sometimes justify missing a resource, so this count alone
is not a proof of bad setup. The spending traces show why it is problematic here.

The learner received an average 49.67 cards from dice production by the end of a
game; a greedy opponent participant received 77.02. These are actual produced
cards, including growth and blocking effects, not pure initial-placement measures.
They also reflect differing trajectories and the common game-ending race.

The learner bought 171 development cards across 112 games: 1.53 per participant.
Greedy opponents bought 1,829 across 288 participant appearances: 6.35 each.
The learner played 0.80 knights per game versus 3.35 per greedy participant. Cards
and Largest Army are therefore another visible route to points used much more
heavily by the reference. This comparison does not isolate card preference from
its ability to afford cards.

### Trading: it frequently spends the material needed for growth

Of 898 learner bank trades in the main baseline:

| Material shortfall after trade | Count | Share |
| --- | ---: | ---: |
| Reduced | 149 | 16.6% |
| Increased | 304 | 33.9% |
| Unchanged | 445 | 49.6% |

The diagnostic shortfall is the minimum missing-card count for a settlement, city,
or development card with pieces/deck stock remaining. It ignores placement
reachability, road and award strategy, robber risk and long-term plans. An unchanged
or increased value is a warning signal, not a universal declaration that a trade is wrong.

A concrete four-player example on seed 3000003000, seat 0, action 254 / own turn 27:

| | Wood | Sheep | Wheat | Brick | Ore |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before | 0 | 0 | 1 | 1 | 4 |
| After trading four ore for one wheat | 0 | 0 | 2 | 1 | 0 |

Before the trade, the hand was one wheat short of a city and one sheep short of a
card purchase. The trade removed all four ore; neither purchase was now close.
The best material shortfall rose from one card to two. The model assigned this
trade about 72.7% probability. The greedy reference would have kept the cards and
ended the turn. This hand contained six cards, so it was not already over the
seven-card discard threshold.

In a three-player game on seed 3000002000, seat 1, action 152, the bot also paid for
a road while holding [2 wood, 2 sheep, 1 wheat, 1 brick, 0 ore]. A settlement was
legal, but the road consumed the last brick required to buy it. Road-first decisions
can have strategic value; this example demonstrates the immediate opportunity cost.
The original probe is [probe.json](../output/greedy-diagnosis/probe.json).

### Discards: it can break a ready purchase

In the main baseline, 74 of 124 learner discards left a worse material shortfall
than the greedy choice from the same hand; none left a better value under that
heuristic. The remaining choices tied. This is not a universal optimal-discard test.

In the four-player seed-3000003000 game, action 43, the bot held
[3 wood, 2 sheep, 2 wheat, 0 brick, 1 ore] and had to discard four cards.

| Choice | Wood left | Sheep left | Wheat left | Brick left | Ore left |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neural discard | 2 | 1 | 1 | 0 | 0 |
| Greedy discard from the same hand | 1 | 1 | 1 | 0 | 1 |

The neural choice removed its only ore. The greedy choice preserved the materials
for a development card. This concerns the bot's submitted count vector; the manual
engine/browser rule still accepts explicitly selected cards and was not changed.

### A specific discard-decoder limitation

Code inspection also found a representational constraint: model1's discard decoder
is one affine layer applied to shared hidden state, the earlier counts and a resource
one-hot code. At an equal prefix, changing the resource changes a fixed logit offset;
that raw resource-to-resource offset does not adapt to the current hidden state.
There is no explicit hidden-state/resource interaction in that head. Conditional
masks and previous sampled counts still affect actual choices, so this does not
mean the bot is unable to produce different legal discards or adapt in any way.

A frozen-weight probe compared wood/sheep count logits under two different encoded
board contexts. The resource-difference change was only 4.77e-7, consistent with
float32 rounding, and the count-1 versus count-0 relative log-odds difference was
-0.13024247 in both contexts. The result matches the layer's algebra, not just a
particular sampled discard. Evidence is
[discard-head-structure.json](../output/greedy-diagnosis/discard-head-structure.json).

This limits how freely the head can learn context-dependent resource preferences.
It is a candidate for a small resource-conditioned decoder comparison, with an
intentional model/checkpoint version change if implemented. It has not been shown
to cause the overall win gap: replacing discards alone produced no wins in the
follow-up. The decoder and its saved weights were not changed here.

### Finishing: a legal immediate win was declined

In a four-player game on seed 3000003013, candidate seat 2, action 531 / own turn 37:

- The bot had nine points and [2 wood, 1 sheep, 2 wheat, 1 brick, 0 ore].
- Settlements at vertices 43 and 48 were legal. Either would immediately reach ten.
- It chose road 26 instead, spending one wood and the last brick, remaining at nine.
- The selected road had 80.7% model probability; the greedy settlement suggestion
  had 5.55%. Total settlement probability was only 5.58%.
- The real game ended with seat 1 winning; the learner's final score was eight.

An independent engine fork applied each settlement to the exact pre-choice state:
both returned winner seat 2 with score ten, and the original state/RNG remained
unchanged. This is an actual missed guaranteed win, not an inference from a heuristic.
See [trace-baseline.jsonl](../output/greedy-diagnosis/winning-opportunity-replay/trace-baseline.jsonl)
and the [verified fork results](../output/greedy-diagnosis/verification.json).

It is rare, not the main explanation of the broad gap. The main sample had only
three immediate winning-build opportunities, and the metric counted two choices
of another action. One of those games was eventually won. Choosing a non-building
action can also win through an award, so the aggregate metric is not automatically
a count of lost games. The specific losing game above was checked separately.

## Which replacements actually helped?

All rows below use the same frozen weights. These are diagnostic interventions;
the model itself has not improved or been replaced in normal play.

First sample: sixteen boards, 112 games per controller:

| Controller | What changes | Wins | Mean final learner score |
| --- | --- | ---: | ---: |
| Baseline | Original stochastic policy | 1 / 112 | 3.87 |
| Deterministic | Select highest-logit legal action | 3 / 112 | 4.06 |
| Greedy setup | Replace initial settlements and roads | 3 / 112 | 4.29 |
| Build first | Use greedy choice when a settlement/city is immediately legal | 2 / 112 | 4.08 |
| Greedy main | Replace decisions during MAIN only | 9 / 112 | 5.38 |

The main-phase replacement improved the paired win rate by 7.14 percentage points,
with a pointwise seed-cluster interval +2.68 to +12.50. Deterministic inference had
an interval -1.79 to +6.25; removing randomness alone did not establish a reliable
solution. Better setup raised actual production from 49.67 to 61.98 cards/game,
but the later neural decisions still produced very few wins. Building immediately
was also insufficient by itself.

Follow-up sample: eight new boards, 56 games per controller:

| Controller | What changes | Wins | Mean final learner score |
| --- | --- | ---: | ---: |
| Baseline | Original policy on the new boards | 0 / 56 | 3.43 |
| Greedy trades | Replace MAIN choice whenever either proposal requests a bank trade | 1 / 56 | 4.00 |
| Greedy discards | Replace only the complete discard vector | 0 / 56 | 3.64 |
| Greedy cards | Replace MAIN/ROLL choice whenever either proposal requests a development-card action | 1 / 56 | 3.91 |
| Greedy setup and main | Replace setup and MAIN; retain neural roll/robber/discard/free-road behavior | 9 / 56 | 5.95 |

The combined replacement improved the paired rate by 16.07 points, interval
+10.71 to +21.43. Isolated trade and card controls had intervals starting at zero;
discard replacement produced no wins in this sample. It would be incorrect to
claim that those individual changes have proved a strength improvement. They do
expose resource mistakes and sometimes raise average score. Score is descriptive;
the actual objective remains winning.

The trade/card controls change a whole decision where either proposal requests
that category. They can therefore select a purchase instead of a proposed trade,
or a trade instead of a proposed card. They are not a perfect isolation of every
mechanism within MAIN. Each controller consumes the neural proposal before an
override; only the deterministic controller stops stochastic neural draws.
Conditional discard masks, legality and engine chance still follow the existing rules.

The results favor the diagnosis that placement and later economic decisions
reinforce one another. They do not prove why the optimizer learned these habits,
or that greedy rules are optimal. Remaining robber, discard, card timing, free-road
and expansion decisions deserve further targeted evaluation.

## Why self-play can produce this gap

The inspected training runner used the current neural policy in every seat,
with one frozen version per cohort and only final-win rewards. Greedy was used
for evaluation, not for collecting training experience. The bot was rewarded for
beating copies of itself. That is a different challenge from racing an opponent
that explicitly scores locations, preserves useful cards and buys point-producing
assets efficiently.

The position bias and confident resource mistakes are observed facts. The likely
training explanations are limited opponent diversity and difficult long-term
credit assignment with sparse rewards. These are hypotheses; this investigation
did not run matched training experiments or establish that a larger network,
a different lambda, more time or a GPU would solve them. The present MLP already
receives tile/resource/number, hand and point information. Missing information is
not needed to explain the checked examples.

## Tooling, reproducibility and checks

Added [scripts/diagnose_policy.py](../scripts/diagnose_policy.py) and
[tests/test_diagnose_policy.py](../tests/test_diagnose_policy.py). The CLI owns
single-thread CPU inference and bounded seed/action counts. It reuses CatanEnv,
NeuralPolicy and the existing greedy policy. Debug state snapshots, invariant
checks, probability inspection and streaming traces exist only in this explicitly
run diagnostic tool; nothing was added to the normal collector/trainer hot path.

The CLI refuses an existing output directory. Reports preserve checkpoint hash,
controller names, seeds, outcomes, scores, setup resources, action counts, actual
production, opportunity counters, selected examples and paired intervals. A
requested trace writes one complete game per controller without retaining a
second full rollout or full observation history. Trace fields expose the candidate's
hand and own score; other players' private hands/card composition are not added
to the policy inputs or trace. Final scores are diagnostic reporting data.

Reproduce the first phase experiment in a new output directory:

```powershell
docker build --provenance=false --target learning-checks -t catan-learning-checks .
docker run --rm --cpus 4 --memory 4g --network none --mount 'type=bind,source=C:\Users\david\Documents\Coding\Catan\output\training\run-20261014\checkpoints,target=/checkpoints,readonly' --mount 'type=bind,source=C:\Users\david\Documents\Coding\Catan\output\greedy-diagnosis,target=/diagnosis' catan-learning-checks python -m scripts.diagnose_policy --checkpoint /checkpoints/u00000350.pt --output /diagnosis/review-new --seeds 16 --seed-start 3000003000 --trace-seed 3000003000 --validate
```

For the targeted follow-up, use --seeds 8 --seed-start 3000004000 and
--variants baseline greedy_trades greedy_discards greedy_cards greedy_setup_main.
Use a new output directory. The complete first report is
[phase-controls/summary.json](../output/greedy-diagnosis/phase-controls/summary.json);
the second is [resource-controls/summary.json](../output/greedy-diagnosis/resource-controls/summary.json).

The optional image built successfully; 27 focused tests passed, including exact
uninstrumented-game replay, model/global-RNG preservation, candidate-only trace
privacy, controller phase boundaries, complete discards, caps, bootstrap pairing
and CLI output/overwrite behavior. Repository-wide Ruff lint and all 70 Python
files' formatting passed. The final audit verified every experimental role,
summary, action/trade/discard total and both families of bootstrap intervals.

All eleven saved traces then replayed through the direct engine: 4,621 actions,
exact neural proposals/values, full engine and policy RNG fingerprints, and
invariants at every step. The winning settlements were checked on isolated
successor states. Checkpoint hash and original 64-source manifest remained intact;
training.py also retained its previously verified hash. No training, checkpoint
save, host package installation, engine/schema change or PDF regeneration occurred.
Temporary probe/audit scripts were removed after retaining their results. The
reusable diagnostic CLI and its tests remain.

## Next priorities

1. Address the verified immediate-win mistake with correctly recorded policy
   behavior or targeted learning examples. Do not silently override actions while
   continuing to report the original neural probabilities to PPO.
2. Teach board-dependent settlement evaluation and resource coverage. The observed
   vertex-19 preference is a useful regression target. Simple derived production
   and purchase-shortfall features or placement examples are candidate experiments,
   not reasons to assume a GNN is required.
3. Train against a controlled mixture of stronger references and older policies,
   preserving learner-seat filtering, actual behavior probabilities and private
   observation boundaries. Compare multiple initializations and fresh final-test
   boards under matched budgets.
4. Evaluate economic improvements together, while retaining individual diagnostic
   controls. Development-card opportunities, resource-preserving trades/discards
   and road expansion interact; fixing one in isolation may remain insufficient.

These are next-step proposals. The currently deployed learned policy and completed
training run were not changed by this investigation.

## Simple explanation

The bot is usually willing to buy points when it has the right cards. The bigger
problem is reaching that situation. It often starts on the same familiar spot,
earns fewer useful cards, then trades away or discards material needed for growth.
The greedy opponent has straightforward rules that preserve a route to purchases.

For example, the bot exchanged all four ore for one wheat even though it needed
ore for both a city and a development card. In another game it had nine points and
could buy the tenth immediately, but spent its last brick on a road and later lost.
Those are concrete decision problems we can reproduce and test.

Giving the bot help only at setup or only with discards did not solve the match.
Giving it help with both starting locations and later economic choices helped
much more. We have useful targets for the next training changes, while the fast
engine and the existing trained checkpoint remain intact.
