# Plan: placements, victory-point rewards and opponent resource knowledge

Status: implemented, trained and independently verified on2026-10-15. No recipe
met the improvement gate; no checkpoint was promoted. Actual results, checks and
next proposals are described in [the completed report](training-hypotheses.md).
The saved run protocol is authoritative for launched settings; this document
records the design rationale. Work belongs in C:\Users\david\Documents\Coding\Catan.

## Goal and starting evidence

Determine whether three specific changes improve held-out play against greedy
opponents, while measuring regressions against random opponents. Keep genuine
wins, cutoffs and point totals distinct.

The original selected update-350 model is the common weight source. The last
comparison found 1/252 greedy wins and 202/252 random wins for its original/control
policies, versus 2/252 and 162/252 for broad imitation plus mixed opponents. The
greedy difference was inconclusive; lower teacher loss was not stronger play.

The earlier diagnostic interventions showed setup contributes to the problem,
but later resource decisions also matter. Greedy setup won 3/112 versus 1/112 for
the baseline; greedy main-turn decisions won 9/112 on the same diagnostic sample.
These controls are evidence for designing experiments, not trained model results.

Sources: [decision diagnosis](greedy-policy-diagnosis.md),
[completed curriculum comparison](imitation-curriculum.md),
[observation contract](representation-encoding.md),
[original training contract](training-experiment.md).

## First comparison: four variants

| Variant | Single experimental change | Reward | Resource history |
| --- | --- | --- | --- |
| B: control | Continue from original weights | Actual win only | Existing public snapshot |
| P: placements | Setup-only supervised warm start, then the same PPO practice | Actual win only | Existing public snapshot |
| R: point progress | Same PPO practice with a small net VP-progress bonus | Win plus progress | Existing public snapshot |
| K: resource knowledge | Same PPO practice with observer-specific inferred resource bounds | Actual win only | Public-event tracker |

Do not initially combine changes. P tests a particular setup-teaching recipe;
it does not establish whether every possible placement method works. New placement
features are a separate follow-up if this recipe fails. Broad all-phase imitation
is excluded because the previous experiment already showed a random-play regression.

All four variants use the same experimental model architecture. Only K receives
active knowledge features. Control features are zero. All begin with the same
original policy behavior before P's explicit teaching step. PPO Adam/counters are
reset consistently; the original source checkpoint is never overwritten.

A frozen original-policy evaluation is an additional reference, not a fifth
trained variant. Evaluate it once per final board set and reuse its results.

## P: teach initial placement without mixing interventions

Use the existing greedy policy as a transparent heuristic teacher for initial
settlements and roads. Its choices are not assumed globally optimal.

Proposed fixed recipe:

- 512 fresh setup segments, half three-player and half four-player.
- 64 separate setup-validation segments, half each player count.
- Stop each segment after its 4 * player_count setup actions: 12 or 16 actions.
  These are intentionally truncated teaching segments, not completed games or wins.
- Retain non-forced SETUP_SETTLEMENT and SETUP_ROAD labels only. No critic,
  terminal outcomes or future observations enter supervised inputs.
- Use bounded shards of at most 64 segments with at most eight active games.
  Setup-only collection avoids simulating hundreds of unrelated main-turn moves.
- Reuse ImitationTrainer with two passes, learning rate 0.0001, minibatch 256,
  max gradient norm 0.5 and the existing phase weighting. This is a predeclared
  gentler recipe than the prior all-phase 0.001/three-pass warm start.
- Charge generation, loading and fitting costs to P's training allocation.
  If data are reused across replicates, charge the full generation cost to each P
  replicate rather than granting it free setup compute.
- Start PPO with fresh Adam after imitation, as for the other variants.

The model shares its trunk and action heads across phases. Setup-only teaching can
therefore change later policy behavior. Measure that collateral change; do not
claim that later decisions remain identical. Use a separate, bounded development
sample of full teacher games to inspect non-setup likelihood and decisions.

Report setup teacher agreement/loss, chosen production relative to legal choices,
first-location concentration, ore/wheat coverage, ports, expansion opportunities,
realized production and later purchases. Production heuristics are diagnostics,
not optimality labels. All evaluation moves must come from the learned model:
no hidden greedy replacement while recording neural PPO probabilities.

## R: give credit for net point progress

Keep the terminal winner bonus at +1 and other terminal win bonuses at zero.

For player p define:

    progress_p = clip((own_total_VP_p - 2) / 8, 0, 1)
    incremental_reward_p = win_reward_p
                           + 0.1 * (progress_after_p - progress_before_p)

This gives +0.0125 for each point gained between two and ten, and subtracts the
corresponding amount when points are lost. The two automatic starting points give
no progress bonus. Gaining then losing an award yields no net bonus for the cycle.
Completed-episode progress reward is bounded between zero and 0.1 from the start.

Use the player's own total score, including their own hidden VP cards. Do not
supply opponents' private scores as policy features. A ten-point loser who has
not yet had the turn required to claim victory still receives only the progress
bonus; the actual winner remains the winner reported by the engine.

This deliberately adds a secondary point objective. It does not guarantee an
unchanged win-optimal policy. Potential-based shaping can preserve an objective
under the appropriate discount and terminal assumptions, but retaining terminal
point potential here rewards final points. See
[Ng, Harada and Russell (1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).

Hold gamma=1 and lambda=1 across all variants. Under these complete-game settings,
net progress rewards telescope to a final progress bonus plus a current-state
baseline shift. The experiment tests whether credit for points helps; it should
not claim that delivering the same undiscounted total earlier necessarily gives
additional learning information. A lambda comparison would be a separate study.

Implementation must accumulate each recorded player's rewards across all actions
until their next decision or closure. Points can change on another player's turn,
particularly through award transfers. The present collector only writes terminal
rewards, so changing the environment's reward alone would silently drop progress.

At a real terminal, include actual final progress and the real win bonus. At a
cutoff, do not fabricate victory, final-game bonuses or termination; preserve
observed progress and bootstrap with the corresponding shaped critic. Keep raw
true-win flags and evaluation separate from shaped reward totals.

Give the experimental reward its own version and persist the coefficient in
rollouts, checkpoints and run configuration. Do not relabel shaped data as
terminal_win_v1 or silently accept incompatible reward metadata.

## K: infer resources from observable events

Maintain lower and upper bounds for five resources for each opponent, separately
for every observing player. The policy receives 3 * 5 * 2 = 30 normalized values
in relative-seat order, with absent seats padded. Existing hand totals constrain
the bounds. Equal lower/upper bounds mean a known count.

Start from known setup receipts and stream updates from publicly observable
production, paid construction/development costs, bank trades, publicly observed
card effects, hand totals and supply changes. Do not retain an unbounded log or
enumerate all possible hidden hands.

For a steal, the thief and victim can observe which card moved through their own
hands. Other observers receive only the participants and total-card changes;
their resource bounds widen conservatively. Subsequent public spending or receipts
can narrow uncertainty again. Bounds are conservative estimates, not calibrated
probabilities or necessarily the tightest feasible joint distribution.

Important information-profile decision: this project currently exposes exact
bank stocks. Observing the bank before/after a player's discard can reveal the
discard composition. Preserve that profile for the first comparison and use those
public deltas consistently. Do not call discards secret while publishing equivalent
exact supply changes. A profile hiding supply information would require a separate,
versioned experiment.

Construct an observer-specific event view before updating their tracker. The
tracker must not read actual opponent hands, hidden steal types for third parties,
future events or game RNG state. Privileged true hands may be inspected only by
independent tests/diagnostics to verify that all true counts remain within bounds.

Expose the compact bounds through a versioned learning observation extension.
A small, bias-free 30-to-128 projection can enter the existing first hidden layer.
Zero-initialize this additional branch and preserve the original base matrix shape,
rather than widening the original matrix and assuming floating-point equivalence.
It adds 3,840 parameters, about 1.6% of the current 244,955. All four variants share
the branch; B/P/R receive zeros. Verify initial logits, values, discard probabilities
and complete game/RNG behavior against the original model before training.

Keep the engine state and its copying contract unchanged. Beliefs belong to
owned episode/observer contexts, reset on new games and reproduce through replay.
Do not give one observer another observer's private knowledge. Old observation
and checkpoint schemas need an explicit compatible load/migration path; new
knowledge schema versions must not be confused with the original files.

Measure interval coverage, width, exactness when fully deducible, and downstream
robber/Monopoly decisions. Better tracker accuracy alone does not count as a
game-strength improvement.

## Common training protocol

Use a fixed common opponent mix so each experiment faces the same reference
policies instead of different evolving self-play leagues:

    greedy, frozen_original, greedy, random

This is 50% greedy, 25% the original neural policy and 25% random by cohort count.
Reuse the existing seat router and frozen-checkpoint loading. Keep the pool bounded.
Rotate learner seats and alternate three/four players; record only the learner's
actual probabilities, values and version. All opponent actions execute normally.

Other proposed settings remain the verified eager-CPU recipe:

| Setting | Proposal |
| --- | --- |
| Training replicates | 3 practice/shuffle seeds per variant; 12 runs total |
| Initial weights | Original selected update-350 weights, preserved SHA256 |
| Cohort | 8 complete games, up to 8 active |
| Action cap | 5,000; true terminals and cutoffs distinguished |
| PPO | 4 epochs, minibatch 256, learning rate 0.0003 |
| Other PPO settings | Clip 0.2, value clip 0.2, grad norm 0.5, KL gate 0.02, entropy 0 |
| Targets | Gamma 1, lambda 1, discount per personal decision |
| Threads | 1 collection/inference; 4 updates |
| Runtime | Existing CPU container, network disabled, 4 CPUs, 4 GiB |
| Checkpoints | Atomic commits, bounded generations, explicit schemas/source hashes |

These are three training-seed replicates, not independently initialized network
architectures. Same practice seeds fix initial boards, not later chance trajectories.

Use the final checkpoint at the fixed budget as the primary result. Do not select
from rare greedy wins on three tuning boards. Hyperparameters are frozen before
the final experiment, and test results never choose a checkpoint. Intermediate
development evaluations may be used before the protocol is frozen, with their
costs and decisions recorded. They are not final-test evidence.

## Budget proposal

The user confirmed approximately three hours total, including training and
evaluation, for this first comparison. That answer defines the plan's target;
it does not launch or authorize an unattended run in this planning turn.

For that first pass:

- 8 minutes of charged active training per variant/replicate: 12 * 8 = 96 minutes.
  Include teacher preparation for P, rollout collection, PPO and training persistence.
  Knowledge tracking overhead counts against K's allocation.
- Approximately 40-60 minutes for final evaluations, based on earlier normal
  evaluation timings; this is an estimate, not a new benchmark.
- Approximately 15-20 minutes for runtime preflight, independent replay/audit and
  report completion. Implementation and debugging time before the experiment
  are separate from the training-session estimate.

Expected session total is roughly 2.5-3 hours after implementation checks pass.
An in-progress complete cohort and final verification can cause a modest overrun.
No automatic budget extensions, extra training seeds or new arms after seeing
results. Recalibrate the runtime estimate on development fixtures before freezing
the protocol; if it will not fit, resolve the allocation before launch.

Final evaluation is outside each arm's training allowance. Reserve it in the total
session estimate. The prior 90-second pilot counted baseline/tuning evaluation
inside training time and produced only 5-10 imitation-mixed updates; do not repeat
that design and then describe the tiny practice amount as convergence evidence.

Before any long run, implement cooperative pause between cohorts: commit the
learner, RNG/shuffle/config and remaining budget before stopping. Explicit pause
time must not consume or reset training allowance. Test resume equality and
single-writer rejection. The earlier Docker pause happened during evaluation;
general mid-training pause accounting is still absent.

## Board sets and final evaluation

Proposed final sample: 64 fresh boards per replicate, both player counts, every
candidate seat, against greedy and random.

- 64 * (3 + 4) = 448 greedy roles and 448 random roles per variant/replicate.
- Across three replicates: 1,344 greedy and 1,344 random roles per variant.
- Across four variants: 10,752 final candidate games.
- Evaluate the frozen original once on the same three sets: 2,688 additional roles.
- All seats on a board belong to the same statistical cluster.

Use disjoint final-board sets across replicates and matched sets across variants.
Existing original, curriculum and diagnostic final seeds must not be reused.
Proposed ranges, to validate against all saved protocols before launch:

| Purpose | Proposed range |
| --- | --- |
| Setup teacher training | 260000000 and 260010000, 256 segments each |
| Setup teacher validation | 260100000 and 260110000, 32 segments each |
| PPO practice | 400000000 + replicate * 1000000 + update * 8 |
| Development fixtures | Reserved ranges around 420000000 |
| Optional development evaluation | 1000200000 + replicate * 1000 |
| Final evaluation | 2000200000 + replicate * 1000, 64 boards each |

Persist the complete selected seed lists, policy seeds and source hashes before
launch. Enforce finite update/seed bounds so practice cannot reach another family.
True wins on the final action take precedence over caps. Report cutoffs separately
and include every scheduled role in the all-game rate.

## Evidence and decision criteria

Primary comparisons are P versus B, R versus B and K versus B. The original frozen
reference shows whether the common continuation itself helped or regressed.

Report each training replicate separately, paired board-seed differences and
whole-board bootstrap intervals. Pooled board intervals are conditional on the
trained models; three replicates do not establish universal training-seed uncertainty.
Do not treat rotated seats as independent games in uncertainty calculations.

Proposed practical threshold for a clear candidate:

1. At least +2 percentage points in pooled greedy win rate versus B.
2. A positive paired difference in at least two of the three training replicates.
3. A simultaneous interval excluding zero for the greedy difference, correcting
   the three primary comparisons (e.g. 98.33% individual two-sided intervals).
4. The random-opponent difference does not show a material regression; predeclare
   a five-percentage-point noninferiority margin and report its uncertainty.
5. All correctness/privacy/performance gates pass.

These thresholds are proposed decisions to freeze before launch, not claims about
the sample's proven statistical power. Smaller or uncertain improvements are
promising/inconclusive, not automatically worthless. If random performance is
uncertain, do not claim the noninferiority gate passed.

Mean final VP, income, placement scores, economic actions and tracker accuracy
explain outcomes. Higher shaped rewards or teacher agreement alone do not pass.

Test combinations later only when individual results or predeclared mechanism
evidence justify them. Good placement may interact with progress learning or
opponent information; an inconclusive individual result does not prove no
interaction. A combination needs its own declared budget, fresh final tests and
clear baseline. Do not choose combinations by repeatedly querying this final set.

## Implementation order and required gates

1. Add explicit experiment configuration, shared model compatibility and bounded
   output management. Preserve the original engine, rules, IDs, ports and RNG.
2. Build the public-event projector and compact observer trackers. Add new
   observation/model versions, zero-feature compatibility and privacy tests.
3. Add versioned reward transformation and per-seat interval accumulation.
   Verify exact net rewards through award gains/losses, off-turn changes,
   terminal wins, hidden own VP and genuine cutoffs.
4. Add setup-segment teacher generation and the fixed imitation recipe. Verify
   every label against direct engine replay and declared observation-only inputs.
5. Extend the runner with the fixed common pool, active training budget,
   cooperative pause/resume and predeclared final-checkpoint evaluation.
6. Run container build, lint/format, relevant unit tests and full checks. Independently
   replay teacher labels, same-seat trajectories and restored updates.
7. Run a bounded mechanics smoke test and isolated timing/array-payload comparisons.
   It is not the strength experiment and must not touch final boards.
8. Freeze the complete manifest and budget; only then launch the 12 declared runs.
9. Finish final evaluations and independently reconcile checkpoints, configs,
   totals, role coverage, seed separation and comparisons.
10. Save detailed and plain-language results, preserve checkpoints/raw evidence,
    and remove only verified task-created prototypes.

Performance targets: fixed small tracker arrays, no unbounded history/full hidden-hand
enumeration, no whole-state copy per action, no extra random draws, and no debug
invariants in normal training. Aim for no more than approximately 5% added
collection/inference cost for tracking on unchanged-policy trajectories; measure
before accepting that as a result. Report raw array memory separately from Python,
Torch and optimizer overhead. Keep forced PPO learner rows and complete discard
actions; do not invent teacher critic values or replace learner moves invisibly.

Privacy gates: true test-oracle hands always fall inside inferred bounds; public-only
observers cannot distinguish hidden outcomes with identical observable histories;
steal participants can use their own private observations; reset/seat rotation,
batch width and replay preserve ownership and RNG. Discard visibility and public
supply changes must follow the declared profile.

## What this means in plain language

We will compare four students starting from the same bot. One practices normally.
One first gets lessons only on starting positions. One gets a small amount of
credit for progress toward ten points. One remembers what opponents collected
and spent, while keeping track of what it cannot know.

They receive comparable practice time and face the same kinds of opponents.
Then all play on new boards. We measure wins first and use points and decision
traces to explain the result. We will not call a bot better simply because it
earns more artificial reward or copies the teacher more closely.

