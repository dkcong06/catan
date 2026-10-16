# Catan working agreements

- Use the container for dependency installation and checks; do not install host packages.
- Build the checks image: `docker build -t catan-checks .`
- Run tests: `docker run --rm catan-checks`
- Run lint: `docker run --rm catan-checks python -m ruff check catan rl learning scripts tests`
- Check formatting: `docker run --rm catan-checks python -m ruff format --check catan rl learning scripts tests`
- Test viewer JavaScript: `docker build --target viewer-checks -t catan-viewer-checks .`, then `docker run --rm catan-viewer-checks`. This separate Node target executes the inline script with controlled DOM/canvas/fetch adapters; verify appearance separately in a browser.
- Check the standalone guide generator with a read-only mount: `docker run --rm --mount "type=bind,source=${PWD}/generate_guide.py,target=/guide.py,readonly" catan-checks python -m ruff check --select F --no-cache /guide.py`. Its existing document layout code is outside the application formatter scope.
- Keep static geometry immutable and generated once. Preserve tile, vertex, and edge ID ordering when optimizing.
- Enforce no edge-adjacent red tokens (6/6, 6/8, or 8/8). Keep nine fixed separated coastal port slots with four generic ports and one per resource; use the caller's RNG for all setup randomness.
- Reproduce setup measurements with `docker run --rm catan-checks python -m scripts.benchmark_board`; compare samplers in the same environment, and do not infer gameplay throughput from setup timings.
- Run the container viewer with `docker run --rm -p 127.0.0.1:8000:8000 catan-checks python -m scripts.board_viewer --host 0.0.0.0 --no-browser`.
- Keep the engine and tile viewer free of ML dependencies until implemented features need them.
- NumPy is required by state/engine. Keep board/viewer module imports and board GET routes stdlib-only; import the NumPy discard demo lazily on its POST routes. Keep PyTorch out of the engine.
- Mutable state copies must isolate every array and the complete RNG while sharing immutable Board/BoardGeometry. Include new fields in copying, equality, invariants, and isolation tests.
- Resource arrays use int16; owners/buildings/pieces use int8. Keep ownership sentinel -1 and building codes 0/1/2. Add no duplicate mutable port-access cache.
- Reproduce state measurements with `docker run --rm catan-checks python -m scripts.benchmark_state`; array payload sizes exclude object/RNG overhead, and construction/copy timings are not gameplay throughput.
- Run the expanded board/state stress gate with `docker run --rm catan-checks python -m scripts.stress_foundations`; benchmark cache/equality/ports/previews with `python -m scripts.benchmark_foundations` in the same container. Run performance batches without competing stress/test jobs.
- Preserve the geometry and 1,000-board/RNG compatibility fingerprints when optimizing. Changing their expected values requires an intentional, documented change in ID ordering or seeded setup behavior.
- Preview geometry is encoded once; retain the flat JSON schema and bounded per-request state. Normalize numeric UI seeds and keep obsolete responses from overwriting newer requests.
- Manual discards snapshot half the hand (rounded down) only when its total exceeds seven. Keep the roller in `current_player`; derive `discard_player` from pending counts. Do not choose cards automatically or recompute a completed player's discard. Finish all choices before `ROBBER_MOVE`.
- Functional transitions validate DiscardAction before copying and preserve source arrays/RNG on accepted and rejected choices. Optional *_inplace transitions require valid exclusively owned state, return the same object, retain array/RNG identities, and reject illegal choices before mutation. Both paths share validation/mutation logic; browser sessions retain functional calls. Add discard state fields to copying/equality/invariant gates. Browser demo sessions are bounded and revision checked; sample hands are not full gameplay.
- Compare transition paths with `docker run --rm catan-checks python -m scripts.benchmark_engine` (optional --players 3, --iterations, --repeats). Run without competing check/stress jobs. Both paths use identical prebuilt state pools; fixture allocation/reset and validation are excluded. Fixed seven/discard timings are not complete-game or RL throughput.
- Keep the guide PDF, its generator, and diagram assets as documentation sources. Regenerate the PDF only when requested.
- Maintain `.agent/CONTINUITY.md` for material changes and verification results.
- Stage 4 supports the minimal game plus existing port discounts; no development cards, road/army awards, or negotiated trades. Share validators/predicates/mutations between legal queries and both transition APIs; keep RNG draws after rule validation and on the state RNG only.
- legal_actions is a lazy iterator. Consume before mutating state; avoid materializing discard combinations. The explicit random agent samples physical discard cards with its separate policy RNG; engine/manual-browser choices remain explicit.
- Setup grants only second-round resources. Roads cannot extend through opponent buildings. Aggregate production before handling shortages: multiple recipients get none of the scarce resource; a sole recipient gets the remainder. Steal one uniformly random physical card from the chosen nonempty adjacent opponent.
- BankTradeAction pays the best current bank/port ratio for one different received resource, without mutable port caches. GAME_OVER requires ten building VP on the current player's turn; derive winner from current_player rather than storing duplicate state.
- Run headless gameplay measurements with `docker run --rm catan-checks python -m scripts.benchmark_games` (optional --players 3, --games, --repeats, --seed, --max-actions). Run without competing checks. Timings include reset/legal queries/random policy; capped runs are reported separately and are not wins. Minimal-rule stalemates are possible; future RL must distinguish truncations from terminal wins.

- Keep optimized settlement candidates transient and sorted by vertex ID; trade choices remain ordered by give then receive. Derive internal port ratios without a mutable cache, retaining 2:1 precedence over generic ports. Aggregate all production demands before scarcity decisions; use small per-call Python counts rather than persistent scratch state.
- Stage 5 full checks are opt-in through catan.validation.assert_invariants / validate_game; keep them out of normal engine steps/legal queries/copies. Storage-check optimization must retain shape/dtype/contiguous/writable checks and diagnostic behavior. Road components need an owned building root; later opponent buildings do not invalidate existing roads, while construction still blocks extension through them.
- Run the continuous randomized gate with `docker run --rm catan-checks python -m scripts.validate_games --games 10000 --players both --workers 8`. Each worker exclusively owns states/RNGs; submission is bounded. Report actual wins/caps separately and preserve replay coordinates/fingerprints; do not claim that a zero-error bounded run proves all games terminate. --games is total episodes, alternating 3/4 seats for both.

- Guide Sections 6-7 observations live in rl.encoding with no torch dependency or engine import of RL. Schema v1 is contiguous float32 (1461,); update version/docs/tests intentionally if layout changes. Keep relative clockwise seats/padding, own exact hand vs opponent totals, public bank/ports, token identity, setup anchor and turn/actor distinction. Default actor is discard_player during DISCARD, current_player otherwise, None at GAME_OVER (default observer is winner).
- Prepare FlatEncoder once per immutable board; reuse it across copies/successors, replace it on reset. Static arrays/lookups are immutable and bounded; no global board cache or new GameState fields. Encoding assumes valid states and does not copy/validate/mutate state/RNG. Caller out buffers must be contiguous writable float32, shape (OBS_DIM,), disjoint from all state arrays, and consumed/copied before reuse; fully overwrite every feature/padding/flag.
- Measure only encoding with `docker run --rm catan-checks python -m scripts.benchmark_encoding` (optional --iterations, --repeats, --report). Run without competing stress/check jobs. Prepared allocation and checked buffer reuse have different ownership/costs; do not infer training throughput. Keep independent scalar-oracle, hidden-hand, all-phase/discard, cyclic-seat, alias-rejection and RNG isolation tests.

- Development cards use five remaining type counts and weighted RNG draws; no eager deck shuffle. Preserve new-game board/RNG fingerprints. Track held/new/spent cards, played knights, exact cached road lengths and award holders in copy/equality/invariants/fingerprints.
- Longest Road is an edge trail (no repeated edges), including loops; opponent buildings block traversal, own buildings do not. Recompute on road placement and opponent settlement blocking only. Holder retains a top tie; ties without incumbent remain unheld. Thresholds are five roads/three knights, each worth two VP.
- One older non-VP card per turn, permitted before rolling. Knights never cause discards; resume ROLL or MAIN. Road Building chooses sequential free legal roads and finishes when count/pieces/spaces are exhausted. Fresh VP can win; off-turn award scoring waits until that player's turn.
- Observation schema v2 has 1505 float32 values. Opponent card composition/private VP/deck composition are hidden. Flat action catalog has 254 slots; discards use five conditional twenty-count masks. Keep masks in rl, sharing engine predicates; no engine dependency on RL or PyTorch.
- Batch encoding uses explicit per-row prepared contexts, validates all contexts/aliases before writes, and bounds spatial scratch to 128 rows. Single observations keep the scalar API. Run benchmark_system/benchmark_lookup without competing jobs; compare identical expanded-rule trajectories when claiming optimization gains.
- The decision report is separate from Catan_RL_Guide.pdf. Its authoring tools are documentation-only. Empty future RL modules were removed; create them when implemented.

- Agent environment/evaluation live in rl with NumPy only. CatanEnv exclusively owns state; lazy Decision requests have independent observations/masks and explicit actor. step requires player, counts only accepted atomic engine actions and rejects wrong seats before mutation. Complete discard vectors remain one engine action; snapshot is privileged copy/debug work, not policy input.
- Reward terminal_win_v1 is one for the actual winning seat, zero otherwise. Distinguish true termination from external action caps; a real final-action win takes precedence. Preserve engine phase/private information and explicit per-seat final observations at cutoff. Do not fabricate losses, GAME_OVER or intermediate shaping.
- Reference policies receive only Decision and per-seat policy RNGs. Preserve policy schema/version and exact choices when optimizing. Same-policy controls can reuse one immutable finished summary per seed; report matches_played/executed_actions separately from role records/logical actions, and use executed_actions for throughput. Different-policy rotations must execute every matchup; seed equality fixes initial boards, not identical future chance streams.
- Run evaluation with python -m scripts.evaluate_policies (--players 3/4, --seeds, --candidate/--opponent random/greedy, --validate, --report). Run python -m scripts.benchmark_environment without competing jobs; full observer/policy/query rates are not engine-only or neural-training throughput. Keep raw paired reports and discard temporary prototypes after exactness checks.
- Collector schema v1 lives in rl.collector; NumPy only, finite complete episodes, at most batch_size owned active games. Keep policy inputs observation-only/read-only, independent output columns, same-episode/same-seat successor links and positive engine-action gaps. Preserve exact existing choices/full game+policy RNGs across scheduling widths.
- Attach terminal_win_v1 to every acted seat's final record, using actual winner rather than last actor. Caps have zero reward/no winner, truncated flags and permitted per-seat final views; do not fabricate records for unacted seats, bootstrap true terminals or invent missing critic estimates (NaN).
- Complete discards stay one engine action; sparse hand/count/conditional-log-probability records reconstruct feasible prefix masks. Random physical-card sampling needs hypergeometric probabilities, not uniform count-vector probabilities. Validate every callback output before stepping any game in its batch.
- Reproduce collector performance with python -m scripts.benchmark_collector without competing tests/stress; run python -m scripts.validate_collector --seeds 50 --workers 8 for independent replay/invariants. Keep data memory bounded by caller seed/cap choices; synchronous batching is not process parallelism. Optional numeric NPZ saving is outside collection and carries schema/policy metadata; no pickle, training or automatic unbounded archive retention.
- Optional PyTorch modules live in learning; catan and every rl module stay NumPy/stdlib-only. Build docker build --target learning-checks -t catan-learning-checks . for model checks; the default checks image deliberately skips the optional learning test modules. Lint/format/compile scopes include learning. Dependency installation remains container-only; source changes must not invalidate installed dependency layers.
- Model1 is a float32 two-layer MLP with shared fused flat-policy/value output and prefix-conditioned resource-count head. Preserve private observation boundaries, exact conditional masks, one complete discard action, scoped CPU initialization RNG and caller-owned thread settings. Stochastic sampling draws only supplied per-seat policy Random streams; forced choices draw none. Deterministic point-mass logs are evaluation-only for future PPO actor updates.
- Target1 uses same-seat forward links, explicit real cutoff bootstrap values and zero true-terminal futures. Defaults gamma1/lambda0.95 discount per personal decision; engine mode exponentiates both by elapsed_actions. Validate absent/cross-seat/cross-game/forked links and closures, reject missing critic estimates, use finite float64 intermediates and owned float32 outputs. Model parameters must stay fixed throughout each collection call; increment policy versions after future updates.
- Measure actual neural collection with python -m scripts.benchmark_learning, isolated from tests/stress. Run python -m scripts.validate_learning --seeds 25 --workers 4 for direct replay, probability/value/target/gradient checks and unchanged weights. Keep eager CPU as the verified runtime; GPU/compilation/learning quality remain unverified. The earlier model increment made no updates; the subsequent PPO increment permits bounded mechanics tests. Long training experiments remain held.

- PPO schema1 lives only in learning.ppo. Accept the current updater's stochastic policy tag and numerically matching recorded probabilities/values; reject stale/reference/deterministic data and malformed choices before model/Adam/shuffle mutation. Compute same-seat targets/cutoff guesses once from the fixed behavior model. Normalize advantages globally with population variance, detach old data/targets and include every tail row.
- A complete discard uses one joint PPO ratio (sum of five conditional log differences). Recorded prefixes allow fused head reevaluation; sampling remains sequential with existing per-seat RNGs/masks. Preserve model1 parameter shapes/state_dict keys and schemas; compare joint logs/entropy/gradients with an independent sequential oracle.
- PPO clips actor incentive, optionally clips critic references, clips global gradients and checks finite outputs. KL is sampled expm1(logratio)-logratio: stop a proposed minibatch above threshold, and check all rollout rows after each visited epoch. Prior steps remain applied; this is not a hard KL bound. Default entropy coefficient0; discard entropy is explicitly a behavior-prefix conditional regularizer.
- Save only requested local checkpoints atomically; restore with weights_only=True into a new updater. Preserve model/config/schema/reward identity, Adam settings/moments, versions/counters and local shuffle RNG. Exact CPU update resumption is verified; in-flight game/policy RNGs and external experiment schedules are not included. Runtime update failures quarantine the updater; recover from a valid checkpoint rather than copying rollback state every minibatch.
- Run python -m scripts.benchmark_ppo without competing tests/stress and python -m scripts.validate_ppo --seeds 12 --workers 4 for bounded update/replay/resume checks. Minibatch-size timing comparisons do not prove learning quality. Keep eager CPU verified; GPU/CUDA, long training, opponent pools and fragment collection remain future work. Delete only task-created comparison prototypes after preserving results.

- Local Docker cached-snapshot exports were repaired with docker build --provenance=false (plus --target learning-checks/viewer-checks as needed), without pruning cache/data or changing dependencies. Keep this environment workaround separate from application performance claims.

- The user authorized real time-budgeted training in the subsequent experiment increment, superseding the earlier held-long-training boundaries. Keep orchestration/evaluation in learning, with no core/rl Torch imports or schema changes. Shared-policy games hold weights fixed during collection; reference evaluation batches with missing critic values never enter PPO.
- Training seed schedules derive from committed update counters; reserve distinct tuning/test ranges, rotate both player counts/all seats against random/greedy, select only on tuning wins and retain earlier ties. Test selected/latest versus the initial policy with separate wins/caps and paired whole-seed bootstrap intervals; self-play completion rates are not strength.
- TrainingRun owns explicit phase-dependent CPU thread settings. Use a bounded named container and a fixed built source image for long runs. Preserve the original wall deadline on resume, commit learner+state atomically, reject concurrent writers/config changes and trim only uncommitted owned metric tails. Retain initial/current/previous/best checkpoint generations; remove older owned generations only after manifest commit.
- Run new experiment tests in the optional image and python -m scripts.train_ppo with explicit --output/--minutes; --resume uses stored config/deadline. Keep normal training free of debug invariant sweeps and unbounded rollout archives. Monitor only the authorized job; finish held-out evaluation and honest detailed/plain reports before declaring training complete.

- Frozen-policy diagnostics live only in scripts.diagnose_policy. Keep debug snapshots/probability inspection outside collector/trainer hot paths; policies receive only their existing Decision. Phase overrides are evaluation-only controllers and never fresh neural PPO data. Use bounded fresh diagnostic seeds, all-seat roles, real win/cap accounting and whole-seed paired intervals. Preserve checkpoints and original final-test reports; record controller definitions and validate exact model/RNG/direct-engine replay. Trace only the candidate's private hand; keep privileged aggregate/final reporting separate from policy inputs.

- Imitation lives in optional learning.imitation and uses bounded numeric Demonstrations, separate from PPO RolloutBatch. Drop forced flat labels only for actor imitation; keep forced PPO learner records. Preserve complete discard likelihood/prefix masks, actual teacher labels, private observation boundaries and local shuffle RNG. Keep shard loading/generation out of engine/inference; no foreign pickle or human-derived approximate labels.
- record_players filters storage only: all seats still act with unchanged per-seat RNGs; excluded seats have last_rows=-1 and zero final views. Mixed training records only the current learner and its real probabilities/values/version. Reject caps below eight for mixed cohorts. Distinguish physical engine actions, retained learner rows and true win/cutoff totals.
- Training run schema2 reads schema1. New-run warm starts explicitly reset Adam/counters and preserve source hashes; resume restores committed config/deadline instead. Retain only bounded immutable initial/previous opponent models. Compare curricula with fresh practice/tuning/test ranges, all-seat roles and multiple training seeds; select on predeclared tuning objective, never final test wins. Charge teacher setup to imitation budgets and report actual elapsed times/cohort overruns.
- Model compute_entropy=False is an explicit likelihood-only optimization for imitation, returning zero entropy placeholders with unchanged logs/values/gradients. Preserve ordinary PPO/inference defaults, model parameter shapes and existing observation/action/model/PPO checkpoint schemas. Reproduce the bounded comparison with python -m scripts.compare_training; do not infer strength from teacher loss or array-payload savings.

- The authorized hypotheses experiment intentionally adds optional observation3 (1535 floats), model2 (zero-initialized 30-to-hidden branch), collector2 and PPO2 profiles. Legacy model1/observation2/terminal_win_v1 checkpoints remain readable and default paths retain their contracts. Record reward identity terminal_win_net_vp_0.1_v1 for shaped data and reject profile mismatch before optimizer mutation. No Torch imports enter catan or rl.
- Resource memory in rl.hypotheses uses public bank/board/hand totals, announced payments/effects and the observer's own hand. Third-party steal types stay unknown; exact-bank visibility makes discard compositions inferable. Keep conservative bounded intervals and private observer seat rotation; privileged validate_knowledge runs only in tests/debug. Skip resource-neutral actions but retain Monopoly transfers even when bank stocks do not change.
- scripts.train_hypotheses uses a frozen common original-policy opponent, learner-only storage, final-checkpoint primary evaluation and paired fresh boards. Cooperative pause.request completes the active cohort/evaluation chunk, commits and exits. Its versioned charged_active_v1 clock supersedes absolute deadlines for this runner only; never Docker-freeze a live clock. Resume from stored config/budget after checking exclusive ownership. Preserve raw outcomes/checkpoints and independently audit all twelve cases before declaring completion; see docs/training-hypotheses.md.

- Critic-gradient controls remain opt-in: PPOConfig.critic_updates_trunk=False blocks only direct critic gradients into shared features, preserves exact fused values and trains the value head; default true and legacy schema loading remain unchanged. Global norm clipping still couples scales. Keep model/inference parameter shapes unchanged.
- Balanced RunConfig scheduling covers every player-count/learner-seat/opponent-slot combination per7*slot_count cohorts. Explicit milestones are bounded to32 and retain checkpoint-hashed development curves. Do not reinterpret development curves as final-test selection.
- scripts.compare_critic uses fixed_cohorts_v1, not an inherited wall-time allowance: exact stored cohort counts/config/source, cooperative pause, exclusive locks, final checkpoint primary and no automatic promotion. Preserve original models/raw results and independently audit final paired board-role outcomes; see docs/critic-gradient-comparison.md.

## Optional economic profile

- Public economic features live in rl/economics.py (observation4, 2000 floats); model3 uses a zero-initialized495-to-hidden branch. Preserve the original1505-float prefix and model1/model2 checkpoint keys/contracts. No Torch imports in catan or rl.
- Production is gross entitlement before bank rationing, respects the current robber and relative seats; site yields describe one settlement, not legality. Shortfalls/port ratios use public board data and the observer's own hand. Add no mutable GameState/port/board cache.
- EconomicEnvFactory(False) supplies a matched zero suffix. Keep reward/PPO/action-decoder changes separate. Reject mismatched profiles before optimizer mutation; migration resets Adam/counters, resume restores them.
- scripts.benchmark_decisions creates bounded hashed development locations; never use them as PPO/imitation/final-test data. Only engine-proved deterministic winning road/build/knight actions have correctness labels; material/production scores are diagnostics. Keep historical fixtures separate from fresh roles.
- Run python -m scripts.benchmark_economics with --checkpoint/--output in the optional container without competing tests or training. Report payload/parameters and measured end-to-end costs separately from kernel timing and playing strength. See docs/economic-features.md.

## Optional graph comparison

- Model4/observation4 in learning.graph uses two simultaneous message rounds on immutable tile/vertex/edge relations, with the existing flat254 output, conditional discard decoder and terminal win reward. No graph library or engine changes. Ordered board readout is intentional; do not claim the full policy is permutation invariant.
- scripts.compare_architecture runs the predeclared feature comparison then fresh-MLP/graph comparison. Use the same source migration, balanced opponents/seats, common hashed teacher shards, fixed per-stage cohort counts, last-checkpoint primary and disjoint development/final boards. Persist economic_enabled and actual learner feature scope. Original model1 opponents use only the unchanged1505-float prefix.
- learning.cohorts shares fixed_cohorts_v1 controls with the earlier critic comparison. Cooperative pause.request commits the current cohort/teacher shard/evaluation chunk. Resume requires unchanged source/config/profiles, exclusive ownership, original Adam/shuffle for PPO and shard-boundary Adam/shuffle for warm-up. Never extend allowances to obtain favorable results or automatically promote a model.
- Reproduce isolated costs with python -m scripts.benchmark_architecture --economic-checkpoint PATH --output PATH in the learning container. Run without competing tests/training. See docs/architecture-comparison.md for the frozen protocol, actual costs and verification boundary.

<!-- BEGIN AWS Agent Toolkit rules -->
## AWS Agent Toolkit project settings

- AWS CLI profile: `catan`.
- Selected project Region: `us-east-2`.
- Toolkit catalog and MCP endpoint use `us-east-1`; keep Regional project resources in `us-east-2`.

# AWS Guidance for the new AWS experience

This user has signed up for the new AWS experience. This experience lets you sign into AWS using a social provider and requires the following additional context.

Where this guidance conflicts with the project's own instructions, the project's instructions take precedence.

## Context

### Terminology:

- Say "project" instead of "account" — a project contains an AWS account and settings for sharing with other collaborators
- Say "team member" instead of "IAM user" — users are invited by email, not created or federated in IAM
- Say "AWS Settings" when referring to management tasks at [settings.aws.com](https://settings.aws.com/) (project management, billing, team members, spend limits). Users view their actual AWS resources in the AWS Management Console.
- Say "selected Region" when referring to the user's Region — not "home Region"
- The user has a managed IAM experience. This includes a managed service control policies (SCP) and resource control policies (RCP) that govern the use of AWS. They will still need to use IAM to create policies to let services work with each other. If there are questions about the SCPs or RCPs, go to the documentation at https://docs.aws.amazon.com/accounts/latest/reference/scps-and-rcps-for-projects.html

### Constraints:

- All projects share a single AWS Region determined by the user's contact address. Resources cannot be created in other Regions
- When developing:
  - MUST create all Regional resources in the project's assigned Region
  - You CAN create AWS WAF and Cloudwatch Logs resources in us-east-1 when there are global resources (like a global WAF instance) that require a connection to dependencies in us-east-1. You should not use these for any other reason, because resources in the selected Region will provide lower cost (due to no cross-Region traffic), increased availability (due to no cross-Region traffic), and easier manageability (due to not needing to look in another Region). When you need to do an inventory of resources, you need to look in both the selected Region and us-east-1 for Cloudwatch Logs or WAF resources.
  - MUST NOT attempt to create Lambda, API Gateway, or other Regional resources in any other Region
  - MUST direct users to confirm their Region in AWS Settings > View all projects > Overview > Additional Info > Region. If the user cannot confirm their Region, check in ~/.aws/config
  - MUST NOT use Lambda@Edge — excluded from both Lambda and CloudFront
  - MUST NOT use CloudFormation StackSets — no multi-account or multi-Region deployments
  - MUST NOT attempt cross-Region actions — no cross-Region replication for DynamoDB/S3/RDS, no multi-Region KMS keys
  - MUST NOT use Route 53 cross-Region routing — geolocation, latency-based, and failover routing policies are not available
  - CloudFront is a global service and its actions ARE allowed in `us-east-1`. A user can create a CloudFront distribution pointing to their project-region Lambda function URL or API Gateway. However, Lambda and API Gateway themselves MUST NOT be created in `us-east-1` — they must be in the project Region.
  - Reduced availability in `eu-north-1` specifically: Amazon Rekognition, Amazon Textract, Amazon Personalize, AWS App Runner are not available in that Region.
- IAM permissions for human access are managed by AWS. Don't assign roles to team members unless absolutely necessary
- The user may have a spend limit if they are on the paid plan. The limit that pauses their project if it's exceeded. If resources suddenly become inaccessible, ask if they have a spend limit configured. Only project owners can modify a spend limit.
- When developing:
  - MUST ask about spend limit status if the user reports sudden "Access Denied" errors on operations that previously worked
  - MUST direct users to check spend status in AWS Settings > Billing
  - MUST check if a user has upgraded their account to the paid plan
  - MUST ask the user if they want to clean up the successfully created resources or keep them to reduce cost
- The user sets up billing, creates spend limits, and retrieves and pays invoices in AWS Settings. The user creates budgets and optimizes their costs in the AWS Billing and Cost Management console
- Not all AWS services are available. If a service isn't working, do the following:
  1. Run the command `aws freetier get-account-plan-state`
  2. If accountPlanType": "FREE", check the [Free Tier supported services list](https://docs.aws.amazon.com/accounts/latest/reference/supported-services-sign-up-new.html#supported-services-free-tier) next,
  3. If accountPlanType": "PAID", check the [Paid Tier supported services list](https://docs.aws.amazon.com/accounts/latest/reference/supported-services-sign-up-new.html#supported-services-paid-plan).
  4. If neither list shows the service, check the [Not supported for this experience list](https://docs.aws.amazon.com/accounts/latest/reference/supported-services-sign-up-new.html#unsupported-services). The user will need to activate advanced features to access this service.
- Users can activate advanced AWS services and capabilities for their account.
- Before starting a task, check whether a relevant AWS skill is available. Load the skill with retrieve_skill and prefer its guidance over general knowledge.

### Help level

- help_level (required): LOW, MEDIUM, or HIGH. While a user is building, you MUST ask the user: "How much guidance would you like from me? Low (I only flag security risks), medium (I ask a couple of clarifying questions if something seems off), or high (I explain what I'm doing, suggest alternatives, and flag best practices)."

You CAN update this rule file to save a user's help_level.

Constraints for each level:

**LOW:**

- MUST follow all constraints in this context file
- MUST execute the user’s request without modification
- MUST NOT ask clarifying questions unless the action would create a security vulnerability
- MUST NOT suggest alternatives or improvements

**MEDIUM:**

- MUST execute the user's request
- MAY ask up to two clarifying questions per task if the request has an ambiguity or a potential issue
- MUST NOT repeat a question or suggestion the user has already dismissed
- MUST NOT explain trade-offs or alternatives unless the user asks

**HIGH:**

- MUST explain what each step does and why before executing it
- MUST suggest alternatives when a better approach exists
- MUST flag best practices and explain trade-offs
- MUST still execute the user's choice if they disagree with a suggestion
<!-- END AWS Agent Toolkit rules -->

## NextPly website

- The independent static website lives in sites/nextply. Reuse its exact .openai/hosting.json Sites project identity and separate source history; never include parent Catan files, checkpoints, AWS configuration or credentials in a website upload.
- Build the preview with docker build -t nextply-site sites/nextply. Serve only on loopback using docker run --rm -p 127.0.0.1:4179:8080 nextply-site; install no host packages. See sites/nextply/README.md for the Node syntax check and browser QA scope.
- Publish through the Sites skill. On Windows, put installed Git Bash and the bundled Node binary before WSL in process PATH; process TAR_OPTIONS=--force-local prevents GNU tar from interpreting drive-letter archive paths as remote hosts. Keep temporary source credentials in hidden stdin only.
