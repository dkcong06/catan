# Expert-video opening pilot

Completed 2026-10-16 UTC. Three DandyDrew videos yielded **six settlement choices and six road choices**, covering both expert opening placements per game. All twelve labels pass visual review and the engine audit. No training, application-source changes or model changes were made.

This establishes that manually assisted opening extraction is feasible. It does not prove optimal expert play, improved bot strength, or economical complete-game extraction.

## Sources

| Video | Expert alias / setup seat | Observed evidence times (seconds) |
|---|---|---|
| [The most frustrating game of Catan I've ever played](https://www.youtube.com/watch?v=21zUJ2gil8E&t=193s) | GuitarGuy / fourth | 0, approximately 182.576, 193.290 |
| [I Played A WTF Wacky Wheat Port In This Catan Game](https://www.youtube.com/watch?v=jVn9_SUmOG4&t=137s) | Vibes / fourth | 12.391, 72.391, 97.391, 137.391 |
| [Catan Pro Plays DANGEROUS Ore Wheat Roll In Ranked](https://www.youtube.com/watch?v=LAv_a-iM27A&t=300s) | EvilBarney / second | 6.982, 36.982, 146.967, 201.967, 246.967, 300.811 |

Titles were observed during the pilot; video IDs are the stable provenance keys. Aliases identify the channel player's visible seat. Opponent skill/rating was not independently established. These are three conveniently inspectable games, not a representative sample of top-level play.

The source videos total about 92 minutes 35 seconds. Only opening frames were sampled through normal browser controls. No whole videos were downloaded or watched end to end, and no paid access was purchased.

## Reconstruction and verification

For each game I transcribed all nineteen terrain/number tiles, all nine port types and locations, the actual starting player, and every preceding settlement/road through the expert's second road. Fourth-seat games require ten setup actions each; the second-seat game requires fourteen: **34 engine actions replayed**.

The second-seat example required extra checks because opponents act between the expert's turns. UI player-list position alone does not establish setup order. Public placement history and visible starting-resource grants resolved the order. The expert's grants match one brick/two ore in game one, two wheat/one ore in game two, and two wheat/one brick in game three. Game three also matches public hand totals 0, 3, 3, 3.

The entire source layout maps to our fixed coastal port slots by a 180-degree rotation, including ports and choices. Immutable geometry IDs remain unchanged. All three boards have correct terrain/number/port bags and no edge-adjacent 6/6, 6/8 or 8/8.

The [mapped boards](../output/video-pilot-20261016/mapped-boards.svg) show source colors, orientation, road directions and expert vertex IDs. These are derived diagrams, not archived screenshots.

The [primary receipt](../output/video-pilot-20261016/verification.json) and [independent audit](../output/video-pilot-20261016/independent-audit.json) verify:

- Functional-copy replay agrees with the primary in-place replay, with invariants after each action and complete source-state/RNG isolation.
- All twelve labels are non-forced legal actions; separately enumerated legal masks and exact base/economic observations match the export.
- An independently specified tile-ID map and adjacent terrain/number lists match all six expert settlement sites.
- Starting resources and expert port access agree with the video checks; the second game's first coastal settlement accesses the wheat port.
- Corrupted terrain, number and port bags, adjacent red tokens, and masked-out labels are rejected.

## Artifacts and compatibility

[cases.json](../output/video-pilot-20261016/cases.json) retains board transcriptions, setup prefixes, timestamps and hand checks. [examples.json](../output/video-pilot-20261016/examples.json) maps every numeric row to source game, player, setup index, phase and evidence.

[opening-examples.npz](../output/video-pilot-20261016/opening-examples.npz) is 101,678 bytes and uses the existing supervised Demonstrations format: twelve read-only rows, observation schema 4 (2,000 float32 features), action schema 1 (254 legal-action flags), and a human-source tag. There are no discard rows, critic targets, outcomes or PPO behavior probabilities. The data have not entered training.

Observations are encoded before the labeled choice. Opponent private cards and future video information are not policy inputs. Later visible grants are transcription checks only. Setup uses no random draws; the unchanged placeholder RNG does not claim to recover Colonist's original seed.

The retained [offline recipe](../output/video-pilot-20261016/validate_pilot.py) regenerates and validates the owned dataset and diagram. It adds no code to the engine, collector or learner. From the main project, reproduce with:

    docker run --rm -e PYTHONPATH=/app --mount "type=bind,source=C:/Users/david/Documents/Coding/Catan/output/video-pilot-20261016,target=/results" sha256:b2bb7931088f3554eafafaad66a3a53ac705de6a37550ae6a1b180b2b7b892ab python /results/validate_pilot.py

The recipe passes container Ruff lint, formatting and Python compilation. All 93 application Python files match the previous frozen source inventory. Original u350 SHA-256 remains 67a0ad501d00b5031e3e53f3f87383f96a21b1224ce812907bf6f29d8c1de8a2. Full application regression tests were not rerun for data/document-only changes.

## Effort and methods

First complete three-game engine validation was available after approximately **14 minutes**. [effort.json](../output/video-pilot-20261016/effort.json) records the final total including shared setup, review, export compatibility, diagrams, documentation and cleanup; annotation alone understates usable-data cost. Final measured effort was **26.84 minutes**, or approximately 134 seconds per expert decision including shared overhead.

| Approximate boundary interval | Elapsed | Includes |
|---|---:|---|
| Start to first game | 7 min 21 sec | Shared setup, first transcription and initial validator |
| First to second | 2 min 30 sec | Second annotation and browser/tool overhead |
| Second to third | 4 min 8 sec | Third annotation and extra setup-order checks |

These are coarse task boundaries, not instrumented frame-processing benchmarks. Three games cannot establish general throughput or accuracy. API-dollar, token and bandwidth costs were not instrumented. No GPU or training compute was used.

I inspected selected paused frames and crops. Seeking/full-screen changes sometimes resumed playback or showed advertisements, so actual media time, pause state and readiness were checked. An early adjacency reading in game one was corrected during review. One synthetic negative-test fixture initially failed to create adjacent red tokens; correcting the fixture made the gate pass. Neither issue was an engine defect.

I did not build OCR or a general video parser: three boards do not justify maintaining that machinery. Transcripts omit exact coordinates and road directions. A final setup picture loses action order and legal alternatives. Guessing ambiguous actions would corrupt labels. Full-game extraction introduces hidden cards, trades, edits and event-order ambiguity that this pilot does not measure.

## Recommendation and plain-language explanation

All three selected openings were accepted after manual review; this is not an automatic-extraction success rate. Legality does not prove visual correctness or strategic optimality. Source frames were inspected but not archived; retained URLs/timestamps depend on continued video availability.

Keep this as a small development demonstration set. Before scaling collection, test whether opening imitation generalizes to unseen boards and whether later PPO retains it. Split by whole game, never adjacent rows, and reserve independent games before reporting strength. Twelve correlated decisions from three games are insufficient for a credible training comparison.

A next collection should vary seat order, resource/port strategy and board quality. Mass full-game transcription or more architectural complexity is not justified by this pilot. Teaching quality and winning-decision retention remain separate problems. No additional training is authorized by this report.

In plain language: I turned three real openings into twelve examples the bot can read, reconstructing the earlier opponent moves so each example has the right context. They follow the game rules, but the bot has not learned from them yet. This looks useful for carefully chosen lessons; it is not yet a cheap substitute for a large game database.
