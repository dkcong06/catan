"""Interactive games use the same decisions/streams as direct policy execution."""

from random import Random

from catan.validation import assert_invariants
from rl.env import CatanEnv
from rl.policies import POLICIES
from scripts.game_viewer import GameSessions, PolicyRegistry


def test_live_steps_exactly_reproduce_direct_engine_game_and_stale_rejects():
    games = GameSessions()
    bots = ["random", "greedy", "tactical"]
    status, created = games.handle(
        "/api/game/new", dict(seed=91, players=3, bots=bots, max_actions=80)
    )
    assert status == 200 and created["revision"] == 0
    env = CatanEnv(max_actions=80)
    request = env.reset(91, n_players=3)
    streams = [Random(f"catan-evaluation-v1:17:91:{p}") for p in range(3)]
    while request is not None:
        choice = POLICIES[bots[request.player]](request, streams[request.player])
        env.step(choice, player=request.player)
        status, response = games.handle(
            "/api/game/step", dict(session=created["session"], revision=env.actions - 1)
        )
        assert status == 200 and response["revision"] == env.actions
        assert games.sessions[created["session"]]["env"].snapshot() == env.snapshot()
        if env.actions == 1:
            before = env.snapshot()
            status, stale = games.handle(
                "/api/game/step", dict(session=created["session"], revision=0)
            )
            assert status == 409 and stale["revision"] == 1
            assert games.sessions[created["session"]]["env"].snapshot() == before
        request = env.decision()
    assert response["frame"]["truncated"] and not response["frame"]["terminated"]
    assert response["frame"]["winner"] is None
    assert_invariants(env.snapshot())


def test_session_bounds_and_request_validation():
    games = GameSessions(capacity=2)
    tokens = []
    for seed in range(3):
        status, result = games.handle("/api/game/new", dict(seed=seed))
        assert status == 200
        tokens.append(result["session"])
    assert len(games.sessions) == 2 and tokens[0] not in games.sessions
    assert games.handle("/api/game/step", dict(session=tokens[0], revision=0))[0] == 404
    for payload in (
        None,
        [],
        dict(players=1000000000),
        dict(seed=True),
        dict(bots=["unknown"] * 4),
        dict(deterministic="yes"),
        dict(max_actions=5001),
        dict(players=2),
    ):
        assert games.handle("/api/game/new", payload)[0] == 400
    assert games.handle("/api/game/arbitrary", {})[0] == 404
    assert (
        games.handle("/api/game/step", dict(session=tokens[-1], revision=True))[0]
        == 400
    )


def test_registry_never_accepts_paths_from_browser():
    games = GameSessions()
    assert games.handle("/api/game/new", dict(bots=["/tmp/model.pt"] * 4))[0] == 400
    for paths in ({"greedy": "/tmp/a"}, {"../a": "/tmp/a"}, {"": "/tmp/a"}):
        try:
            PolicyRegistry(paths)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid checkpoint name accepted")
