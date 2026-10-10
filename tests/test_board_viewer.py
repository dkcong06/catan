import json
from http.server import ThreadingHTTPServer
from random import Random
from threading import Thread
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
    ):
        assert preview[name] == [list(value) for value in getattr(GEOMETRY, name)]


def test_unknown_viewer_route_returns_404(viewer_url: str) -> None:
    with pytest.raises(HTTPError) as error:
        urlopen(viewer_url + "/missing", timeout=5)
    assert error.value.code == 404
    error.value.close()
