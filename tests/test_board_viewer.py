import json
from http.server import ThreadingHTTPServer
from random import Random
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from catan.board import GEOMETRY, make_board
from scripts.board_viewer import BoardViewerHandler, PAGE_PATH


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


@pytest.mark.parametrize("route", ("/", "/index.html"))
def test_viewer_serves_html(viewer_url: str, route: str) -> None:
    with urlopen(viewer_url + route, timeout=5) as response:
        assert response.status == 200
        assert response.read() == PAGE_PATH.read_bytes()


def test_tile_coordinate_endpoint_remains_compatible(viewer_url: str) -> None:
    with urlopen(viewer_url + "/api/tiles", timeout=5) as response:
        assert json.load(response)["tiles"] == [
            list(t) for t in GEOMETRY.tile_coordinates
        ]


def test_board_preview_matches_seeded_board(viewer_url: str) -> None:
    with urlopen(viewer_url + "/api/board?refresh=1", timeout=5) as response:
        assert response.headers.get_content_type() == "application/json"
        preview = json.load(response)
    board = make_board(Random(preview["seed"]))
    assert preview["resources"] == [
        resource.name.lower() for resource in board.resources
    ]
    assert preview["numbers"] == list(board.numbers)
    assert preview["robber_tile"] == board.robber_tile
    for name in (
        "tile_positions",
        "vertex_positions",
        "edge_positions",
        "tile_vertices",
        "edge_vertices",
        "port_vertices",
        "port_positions",
    ):
        assert preview[name] == [list(value) for value in getattr(GEOMETRY, name)]
    assert preview["ports"] == [
        {"type": port.name.lower(), "ratio": port.ratio} for port in board.ports
    ]


def test_seed_query_reproduces_requested_board(viewer_url: str) -> None:
    with urlopen(viewer_url + "/api/board?seed=4294967295", timeout=5) as response:
        first = response.read()
    with urlopen(viewer_url + "/api/board?seed=4294967295", timeout=5) as response:
        assert response.read() == first
    preview = json.loads(first)
    assert preview["seed"] == 4294967295
    board = make_board(Random(preview["seed"]))
    assert preview["numbers"] == list(board.numbers)
    assert preview["ports"] == [
        {"type": port.name.lower(), "ratio": port.ratio} for port in board.ports
    ]


def test_parallel_previews_keep_seed_and_geometry_isolated(viewer_url: str) -> None:
    def fetch(seed: int) -> dict:
        with urlopen(f"{viewer_url}/api/board?seed={seed}", timeout=5) as response:
            body = response.read()
            assert int(response.headers["Content-Length"]) == len(body)
            return json.loads(body)

    seeds = [0, 42, 4294967295, 1, 0, 42, 7, 9] * 4
    with ThreadPoolExecutor(max_workers=8) as workers:
        previews = list(workers.map(fetch, seeds))
    for seed, preview in zip(seeds, previews):
        board = make_board(Random(seed))
        assert preview["seed"] == seed
        assert preview["numbers"] == list(board.numbers)
        assert preview["resources"] == [r.name.lower() for r in board.resources]
        assert preview["port_vertices"] == [list(v) for v in GEOMETRY.port_vertices]
    assert previews[0] == previews[4]


@pytest.mark.parametrize(
    "query",
    (
        "seed=-1",
        "seed=",
        "seed=1.5",
        "seed=x",
        "seed=4294967296",
        "seed=1&seed=2",
        "seed=%D9%A1",
    ),
)
def test_invalid_seed_returns_400(viewer_url: str, query: str) -> None:
    with pytest.raises(HTTPError) as error:
        urlopen(viewer_url + "/api/board?" + query, timeout=5)
    assert error.value.code == 400
    error.value.close()


def test_unknown_viewer_route_returns_404(viewer_url: str) -> None:
    with pytest.raises(HTTPError) as error:
        urlopen(viewer_url + "/missing", timeout=5)
    assert error.value.code == 404
    error.value.close()
