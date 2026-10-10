import json
import builtins
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from scripts import discard_demo
from scripts.board_viewer import BoardViewerHandler, DISCARD_PAGE_PATH


@pytest.fixture(scope="module")
def viewer_url():
    with ThreadingHTTPServer(("127.0.0.1", 0), BoardViewerHandler) as server:
        worker = Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            worker.join(timeout=5)


@pytest.fixture(autouse=True)
def fresh_sessions(monkeypatch):
    monkeypatch.setattr(discard_demo, "DEMO", discard_demo.DiscardDemo())


def post(url, route, payload, *, raw=None, content_type="application/json"):
    request = Request(
        url + route,
        data=json.dumps(payload).encode() if raw is None else raw,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as error:
        response = error
    with response:
        return response.code, json.load(response)


def transition(viewer_url, route, state, **choice):
    return post(
        viewer_url,
        "/api/discard/" + route,
        {"session": state["session"], "revision": state["revision"], **choice},
    )


def test_discard_page_and_board_navigation_are_available(viewer_url):
    with urlopen(viewer_url + "/discards", timeout=5) as response:
        assert response.read() == DISCARD_PAGE_PATH.read_bytes()
    with urlopen(viewer_url, timeout=5) as response:
        assert b'href="/discards"' in response.read()


def test_numpy_unavailable_returns_a_docker_instruction_and_board_still_works(
    viewer_url, monkeypatch
):
    original_import = builtins.__import__

    def without_demo(name, *args, **kwargs):
        if name == "scripts.discard_demo":
            raise ImportError("NumPy unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_demo)
    status, result = post(viewer_url, "/api/discard/new", {})
    assert status == 503 and "Docker" in result["error"]
    with urlopen(viewer_url + "/api/board?seed=42", timeout=5) as response:
        assert json.load(response)["seed"] == 42


@pytest.mark.parametrize("players, roller, order", [(4, 0, [0, 2, 3]), (3, 2, [2, 0])])
def test_browser_api_resolves_seven_and_accepts_manual_resource_choices(
    viewer_url, players, roller, order
):
    status, state = post(
        viewer_url, "/api/discard/new", {"n_players": players, "current_player": roller}
    )
    assert status == 200 and state["phase"] == "ROLL"
    status, state = transition(viewer_url, "seven", state)
    assert status == 200 and state["phase"] == "DISCARD"
    choices = {0: [4, 0, 0, 1, 0], 2: [0, 0, 0, 0, 4], 3: [0, 0, 2, 2, 0]}
    for player in order:
        assert state["discard_player"] == player
        previous = state
        status, state = transition(
            viewer_url, "choose", state, player=player, resources=choices[player]
        )
        assert status == 200 and state["revision"] == previous["revision"] + 1
        assert state["current_player"] == roller
        assert state["hands"][player] == [
            n - c for n, c in zip(previous["hands"][player], choices[player])
        ]
        assert all(
            sum(hand[r] for hand in state["hands"]) + state["bank"][r] == 19
            for r in range(5)
        )
    assert state["phase"] == "ROBBER_MOVE" and state["discard_player"] is None
    assert state["hands"][1] == list(discard_demo.SAMPLE_HANDS[1])
    assert state["discard_pending"] == [0] * players


@pytest.mark.parametrize(
    "choice",
    [
        {"player": 0, "resources": [4, 0, 0, 0, 0]},
        {"player": 0, "resources": [0, 0, 0, 0, 5]},
        {"player": 2, "resources": [0, 0, 0, 0, 4]},
        {"player": 0, "resources": [5.0, 0, 0, 0, 0]},
        {"player": 0, "resources": [True, 3, 1, 0, 0]},
        {"player": 0, "resources": [-1, 3, 3, 0, 0]},
        {"player": 0, "resources": [5, 0]},
        {},
    ],
)
def test_api_rejects_bad_choices_without_changes(viewer_url, choice):
    _, state = post(viewer_url, "/api/discard/new", {})
    _, state = transition(viewer_url, "seven", state)
    status, error = transition(viewer_url, "choose", state, **choice)
    assert status == 400 and error["error"]
    stored, revision = discard_demo.DEMO.sessions[state["session"]]
    assert discard_demo.snapshot(state["session"], stored, revision) == state


def test_duplicate_concurrent_requests_apply_only_once_and_return_current_state(
    viewer_url,
):
    _, state = post(viewer_url, "/api/discard/new", {})
    _, state = transition(viewer_url, "seven", state)

    def submit(_):
        return transition(
            viewer_url, "choose", state, player=0, resources=[4, 0, 0, 1, 0]
        )

    with ThreadPoolExecutor(max_workers=6) as workers:
        results = list(workers.map(submit, range(6)))
    accepted = [result for status, result in results if status == 200]
    rejected = [result for status, result in results if status == 409]
    assert len(accepted) == 1 and len(rejected) == 5
    assert all(result["current"] == accepted[0] for result in rejected)
    assert accepted[0]["hands"][0] == [0, 3, 2, 0, 0]


def test_separate_browser_sessions_and_bounded_storage():
    demo = discard_demo.DiscardDemo(capacity=2)
    _, first = demo.handle("/api/discard/new", {})
    _, second = demo.handle("/api/discard/new", {})
    assert first["session"] != second["session"]
    _, advanced = demo.handle(
        "/api/discard/seven", {"session": first["session"], "revision": 0}
    )
    assert advanced["phase"] == "DISCARD"
    assert demo.sessions[second["session"]][0].phase.name == "ROLL"
    demo.handle("/api/discard/new", {})  # Evict least recently used second session.
    assert len(demo.sessions) == 2
    status, result = demo.handle(
        "/api/discard/seven", {"session": second["session"], "revision": 0}
    )
    assert status == 404 and "expired" in result["error"]


@pytest.mark.parametrize(
    "payload", [[], None, {}, {"revision": False}, {"session": []}]
)
def test_missing_or_invalid_session_returns_structured_error(viewer_url, payload):
    status, result = post(viewer_url, "/api/discard/seven", payload)
    assert status in (400, 404) and result["error"]


@pytest.mark.parametrize(
    "payload",
    [
        {"seed": -1},
        {"seed": 2**32},
        {"seed": True},
        {"n_players": 2},
        {"current_player": 4},
    ],
)
def test_invalid_sample_setup_creates_no_session(viewer_url, payload):
    status, result = post(viewer_url, "/api/discard/new", payload)
    assert status == 400 and result["error"]
    assert not discard_demo.DEMO.sessions


@pytest.mark.parametrize(
    "raw, content_type",
    [
        (b"{", "application/json"),
        (b"\xff", "application/json"),
        (b"{}", "text/plain"),
        (b"", "application/json"),
        (b" " * 4097, "application/json"),
    ],
)
def test_invalid_http_bodies_return_400(viewer_url, raw, content_type):
    status, result = post(
        viewer_url, "/api/discard/new", {}, raw=raw, content_type=content_type
    )
    assert status == 400 and result["error"]
