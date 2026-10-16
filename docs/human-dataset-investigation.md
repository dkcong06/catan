# Investigation of the 43,947-game Colonist dataset

Checked 2026-10-14; final verification at 2026-10-14T22:23:13Z
(2026-10-14T18:23:13-04:00 in America/New_York).

## Finding

The advertised raw-game download is unavailable at its public location. I found
recoverable derived CSVs and a credible contact lead, but not a downloadable,
verified corpus of expert decisions. The recovered files should not be used as
policy demonstrations in their present form. This is an availability and quality
assessment, not proof that the original games are fabricated or invalid.

The investigation did not alter the engine, observations, policies, weights,
training configuration or rules, and did not start training.

## What was checked

The [main mirror](https://github.com/OrgadYron/catan_43k_games_dataset) advertises
43,947 anonymized four-player games, JSON event histories and a games.tar.gz
archive. These remain publisher claims; the complete raw files were not available
for independent counting or replay.

Public HTTP requests to both github.com/Catan-data/dataset and its advertised
/releases/latest endpoint returned 404. That does not establish whether the
original repository was deleted, made private or moved.

I inspected the mirror and all six forks returned by GitHub:

- glatzl/catan_43k_games_dataset
- yashindc/catan_43k_games_dataset
- Minyall/catan_43k_games_dataset
- engerlina/catan_43k_games_dataset
- theocaplan-del/Catan-dataset
- Zazarti/dataset-colonist

None had release assets or raw game files in the inspected default-branch trees.
The main mirror has one branch, no tags and six accessible historical commits;
all six historical trees contain only documentation. The added files in the
larger fork are analysis scripts, not the missing event-history corpus.
The [open download request](https://github.com/OrgadYron/catan_43k_games_dataset/pull/1)
and glatzl's modified README independently describe the broken download.

Three downstream projects were checked: lblommesteyn/settler_of_catan,
adamliewehr/InfoVisFinalProj and sousa16/catan-analyzer. They contain selected
processed data or models but no raw game archive in the inspected trees/releases.
Additional targeted web searches did not locate a working public archive. This
was a bounded investigation, not an exhaustive claim about every private copy.

## Data that could actually be inspected

I downloaded two CSVs plus documentation/parser sources, without downloading
foreign model checkpoints or executing foreign Python code.

| Recovered file | Inspected contents | What is missing for imitation learning |
| --- | --- | --- |
| opening_regret/regret_data.csv | 38,670 rows, 9,927 game IDs, 21 columns: opening scores/pips/resources, seat and outcomes | Complete boards, exact selected vertex/road IDs, legal alternatives and decision sequences |
| cleaned_games_test.csv | 934 game summaries, 99 columns: final scores, resource totals, trades and dice summaries | Observations and actions at each decision |

The [opening file](https://github.com/lblommesteyn/settler_of_catan/blob/main/research/opening_regret/regret_data.csv)
and [summary file](https://github.com/adamliewehr/InfoVisFinalProj/blob/main/data/cleaned_games_test.csv)
are derived subsets, not independent reproductions of the advertised full count.
Neither inspected CSV contains player ratings or evidence of expert selection.

Opening-file checks found no duplicate game/seat pairs or blank cells. However,
only 9,383 of 9,927 games have four retained player rows, and 9,877 have a retained
winner row. Missing rows may reflect upstream filtering or failed reconstruction;
the original records are needed to establish the cause.

The opening file labels 9,575 of its 9,877 winner rows with final_vp below ten.
More decisively, all 934 game IDs with winners shared between the two CSVs have
different winner point totals. For game 191918444, the opening file reports seven
and the summary file reports ten. These are confirmed inconsistencies in the
processed representations. They do not establish which interpretation is correct,
and do not imply that the source game was illegitimate.

## Parser and documentation problems

The published README assigns victory-point category 2 to Largest Army, 3 to
Longest Road and 4 to VP cards. The downstream summary parser assigns 2 to VP
cards, 3 to Largest Army and 4 to Longest Road. The opening parser simply sums the
category values; the summary parser applies explicit city/award multipliers.
Those conflicting interpretations can change labels and strategic statistics.
Actual raw examples and an independently checked schema are necessary before
choosing a mapping or repairing results.

The [opening loader](https://github.com/lblommesteyn/settler_of_catan/blob/main/src/catan/data/loader.py)
also explicitly approximates corner IDs by sorting vertices and assigns ports by
angular ordering. It can substitute a random board when tiles are missing. These
behaviors are unsuitable for faithful replay or exact action labels in our engine.
We need a verified coordinate-based mapping and explicit rejection of missing or
ambiguous data, preserving our own immutable geometry IDs.

The larger fork's train_model.py is a LightGBM win-prediction script, not a trained
action-selection policy or a checkpoint compatible with our neural bot. It uses
game-grouped cross-validation, which is a useful boundary. I also checked the
suspected turn_frac feature: the inspected full extractor divides by a fixed
AVG_GAME_TURNS, not the individual game's future final duration. That particular
feature is therefore not a demonstrated future-information leak. Outcome fields
such as final rank and winner remain labels/metadata and must not become inputs.

Published strategy tables describe observational associations. For example,
players finishing with more knights may also have stronger economies or more
opportunity to play them. Such tables cannot by themselves establish that an
instruction to avoid one or two knights improves decisions. I did not reproduce
the tables or treat them as an optimal teacher.

## Provenance, skill and permitted use

The documentation attributes the games to Colonist.io, but I did not verify an
official platform export, collection protocol, underlying records or expert-player
selection. A rank of one in a finished game means the match winner, not a top-rated
player. General human gameplay could still be useful after validation; it should
not be described as elite demonstrations without supporting skill information.

The README limits use to research and education. The inspected main mirror has
no separate standard LICENSE file. Keep that stated scope and obtain clear source
and permitted-use information with any recovered archive; availability alone does
not establish broader reuse terms.

## A practical recovery lead

The current [InfoVis project README](https://github.com/adamliewehr/InfoVisFinalProj)
says its author can supply the removed raw dataset by email on request. It also
warns that the old End Game State documentation may contain errors. This is a
publisher's offer, not a verified transfer or guarantee that a full copy survives.
No messages, emails or repository comments were sent during this investigation.
Its linked public Colab notebook was not readable through the web tool; it was not
an independently verified raw-data source.

The next data-specific action is to obtain a small set of complete anonymized raw
games from that lead, together with game settings, schema/provenance information,
ratings if available and permitted use. Validate roughly 100 diverse complete
games before processing gigabytes or writing a training importer. Check board
coordinates, ports, red-token adjacency, action order, resource conservation,
development-card timing and scoring against actual outcomes. Record every reject.
Do not silently substitute boards, guess locations or discard trade events.

Starting placements remain the simplest potential first use if exact boards and
choices can be recovered. Full-game training additionally requires negotiated
player trades, which our engine does not currently support, and reconstruction of
only the information available to the acting player. Neither raw legality,
private-information completeness nor these rule compatibilities was verified here.

## Evidence and verification

Evidence is saved under output/dataset-audit-20261014:

- github-inventory.json: ten default-branch trees and their release inventories.
- repository-history.json / historical-trees.json: branch/tag/history checks.
- original-availability.json / pull-requests.json: original endpoints and request.
- downloads.json / blob-verification.json: source URLs, SHA256 and Git blob checks.
- derived-data-checks.json / cross-file-checks.json: CSV counts and contradictions.
- verification.json: final integrity result and unchanged selected checkpoint.

The ten manifest downloads total 5,192,264 bytes; all ten match the Git blob hashes
in the inventoried trees. Downloaded Python sources were retained as .py.txt audit
evidence and were never imported or run. No pickle or foreign checkpoint was loaded.
The selected u00000350.pt SHA256 remains
67a0ad501d00b5031e3e53f3f87383f96a21b1224ce812907bf6f29d8c1de8a2.

Docker's Linux daemon was unavailable during this task, including the escalated
check. The bounded CSV inspection used existing PowerShell/.NET facilities, with
no host package installation. This documentation/data-only task required no
application build or training tests; integrity and record checks are not a raw-game
legality validation. Previously existing project changes were preserved.
