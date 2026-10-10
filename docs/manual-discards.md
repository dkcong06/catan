# Manual discards: first Phase 4 increment

Implemented 2026-10-10 at the user's request. The engine and browser allow each
affected player to choose which resource cards to return after a seven.
This began as a focused increment. The [Stage 4 engine](stage-four-engine.md) now
integrates setup, dice/production, robber movement/stealing, building/trading,
and full turns. The browser still demonstrates manual discards with sample hands.

## Rule and state contract

This paraphrases the seven rule in the
[official CATAN rules and almanac, 2020](https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf),
Almanac section “Rolling a 7 and Activating the Robber”:

- A seven produces no resources.
- Every player with more than seven resource cards chooses half to discard,
  rounded down. Seven cards are kept; eight or nine require four discards.
- Discarded cards return to the bank. All discards finish before the roller
  moves the robber and steals.

`resolve_seven(state)` accepts a valid state in `Phase.ROLL`, resolving a seven
already supplied by the caller. It does not draw dice or consume RNG. It records
`last_roll = 7` and snapshots `discard_pending: int16[n_players]` from hand totals.
If no player owes cards, it enters `ROBBER_MOVE` immediately; otherwise it enters
the appended `Phase.DISCARD = 8`. Existing phase values 1-7 are preserved.

`state.discard_player` derives the next pending seat clockwise from the roller.
Serial order is an engine scheduling choice for independent discards; the rules
do not require that order. `current_player` always remains the roller. Pending
counts are cleared once, never recomputed: a 95-card hand discards 47 and keeps
48 without being asked to discard again.

`DiscardAction(player, resources)` is frozen/slotted and normalizes counts to an
immutable five-element tuple ordered **wood, sheep, wheat, brick, ore**. Counts
must be nonnegative integers up to the physical supply of 19; booleans, floats,
strings, and malformed lengths are rejected. NumPy integers are accepted.
`apply_action(state, action)` checks the phase, expected seat, exact required
total, and available resources before allocating an independent successor.
It subtracts only selected cards, adds the same counts to the bank, and clears
that seat's pending count. The final choice enters `ROBBER_MOVE`.

Both functional transitions preserve source state, shared immutable board identity, live
robber position, and complete RNG continuation. `GameState.copy`, equality,
deepcopy aliases, `array_bytes`, and debug `validate()` include pending counts.
The extra payload is six/eight bytes for three/four players (241/258 bytes total,
excluding array headers, RNG, and shared board). No discard combinations are
eagerly enumerated or cached: validation examines only the selected five counts
and at most four pending seats. `legal_actions` yields count combinations lazily
only when explicitly iterated; direct manual choices do not enumerate them. `validate()` remains an explicit debug check; engine
callers must provide a valid state.

## Engine example

Until setup and dice actions are implemented, construct a valid test fixture:

```python
from catan.actions import DiscardAction
from catan.engine import apply_action, resolve_seven
from catan.state import Phase, new_game

source = new_game(seed=42, n_players=4)
source.phase = Phase.ROLL
source.setup_index = len(source.setup_order)
source.hands[2] = (4, 3, 2, 0, 0)  # Nine cards; player 3 owes four.
source.bank -= source.hands.sum(axis=0)
source.validate()

pending = resolve_seven(source)
assert pending.discard_player == 2
assert pending.discard_pending[2] == 4
following = apply_action(pending, DiscardAction(2, (2, 1, 1, 0, 0)))
following.validate()
assert following.phase == Phase.ROBBER_MOVE
assert following.current_player == 0  # Original roller moves the robber.
assert source.hands[2].tolist() == [4, 3, 2, 0, 0]  # Functional source retained.
```

Run through the project Docker workflow in README. `RollAction()` now draws two
dice from the state RNG and enters this discard flow automatically on a seven;
other totals distribute resources. Functional calls draw on an independent branch;
in-place calls advance the owned RNG. Both action APIs now support all minimal
Stage 4 actions, with lazy `legal_actions` and terminal/winner queries. The known-
seven helpers remain useful for this demo and explicitly supplied test fixtures.

## Exclusive-owner transitions

`resolve_seven_inplace(state)` and `apply_action_inplace(state, action)` implement
the same rules on a valid state exclusively owned by the caller. They return that
same state and preserve array/RNG identities, avoiding a full copy. These paths
share validation and mutations with the functional APIs; invalid choices are
rejected before mutation. Retained aliases see accepted updates, so use functional
calls for snapshots, branches, or states shared with readers/workers. The browser
continues to use functional transitions.

Hand totals reduce directly into the existing pending array before thresholding;
valid per-seat totals cannot exceed 95, fitting its int16 storage. Discard counts
are converted once and reused for both hand and bank updates. No state field,
RNG algorithm, seeded stream, rule, or action format changed. See the
[optimization contract and benchmark](engine-optimization.md).

The 371-test verification below is the original manual-discard milestone. The
in-place increment adds equivalence/identity/rejection tests; its current gate and
measurements are recorded in the optimization document.

## Browser selector and API

Run the existing Docker board viewer, then open `/discards` or follow the board
page's manual-discard link. The demo loads sample four-player hands with totals
10, 7, 9, and 8. **Resolve a 7** starts the real engine discard flow: players
1, 3, and 4 owe 5, 4, and 4 cards. Player 2 keeps seven cards.

Choose each resource count with the labeled numeric controls. The page shows
held cards, the required/selected count, pending players, and bank supply.
Confirmation is enabled only for whole nonnegative counts within the hand and
the exact required total. Each accepted choice clears controls for the next seat.
The final screen reports that the original roller may now move the robber.
**Reset sample hands** starts a new independent demo. The page explicitly labels
this as a rule demo; it does not simulate unimplemented gameplay or hide hands.

The same local server serves three JSON POST routes:

| Route | Body | Result |
| --- | --- | --- |
| `/api/discard/new` | Optional `seed` (unsigned 32-bit), `n_players` (3/4), `current_player` (seat) | Fresh sample session at `ROLL`, revision 0 |
| `/api/discard/seven` | `session`, `revision` | Engine resolves seven |
| `/api/discard/choose` | `session`, `revision`, `player`, `resources` | Engine applies exact chosen counts |

Responses carry session/revision, phase, roller, decision player, pending counts,
hands, bank, and last roll. Invalid requests return JSON errors (400); missing or
evicted sessions return 404. A stale revision returns 409 plus the current
snapshot, which the page uses to recover. A lock makes revision validation and
replacement atomic so repeated/concurrent submissions cannot discard twice.
Storage is bounded to 32 least-recently-used sessions, held in memory only.
Bodies must be JSON objects, limited to 4 KiB; no third-party frontend packages.

Board generation, board GET routes, and viewer module imports remain stdlib-only.
NumPy is imported lazily on discard POST requests. Run this feature in Docker;
a host without NumPy gets an explicit 503 instruction to use the Docker viewer.

## Verification

Tests cover threshold/floor behavior, all seats and three/four players, exact
manual choices, invalid/out-of-turn/duplicate actions, no double discard of a
large remaining hand, copy/deepcopy isolation, pending-field invariants, and
500 varied complete discard rounds with resource conservation and unchanged RNG.
HTTP tests exercise the engine end to end, isolated/bounded sessions, malformed
bodies, and six concurrent identical choices (one accepted, five stale).
Node tests run the actual selector script, including counts/limits, repeated
submissions, choice reset, error/retry, stale-state recovery, and completion.
Live browser appearance and interaction are checked separately.

Final gate (2026-10-10): both Docker targets build; **371 Python tests** and
**21 JavaScript tests** pass. Ruff lint and formatting (27 Python files), Python
compilation, dependency consistency, the documented engine example, and Git
whitespace checks pass. The live browser completed a full round, rejected an
overdrawn selection, kept the seven-card player exempt, and showed the correct
bank/roller at completion with no console errors or warnings. Phone (390x844)
and desktop (1200x900) layouts have no horizontal overflow; viewport restored.
The updated preview runs at `http://127.0.0.1:8124/discards` in
`catan-main-viewer`; `output/manual-discards.jpg` holds the verified view
(ignored by Git). The old Catan preview container was replaced; other services
were unchanged.

The guide PDF and generator remain unchanged. Setup, dice/production, robber
movement/stealing, and normal turns are now integrated; see the current
[Stage 4 verification and remaining boundary](stage-four-engine.md).
