"""Serve the current Catan board geometry in a local web browser."""

from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from random import Random
from urllib.parse import urlparse

from catan.board import GEOMETRY, make_board


HOST = "127.0.0.1"
PORT = 8000
PAGE_PATH = Path(__file__).with_name("board_viewer.html")
TILE_PAYLOAD = json.dumps({"tiles": GEOMETRY.tile_coordinates}).encode("utf-8")
_PREVIEW_BOARD = make_board(Random(0))
BOARD_PAYLOAD = json.dumps(
    {
        "seed": 0,
        "tile_positions": GEOMETRY.tile_positions,
        "vertex_positions": GEOMETRY.vertex_positions,
        "edge_positions": GEOMETRY.edge_positions,
        "tile_vertices": GEOMETRY.tile_vertices,
        "edge_vertices": GEOMETRY.edge_vertices,
        "resources": [resource.name.lower() for resource in _PREVIEW_BOARD.resources],
        "numbers": _PREVIEW_BOARD.numbers,
        "robber_tile": _PREVIEW_BOARD.robber_tile,
    }
).encode("utf-8")


class BoardViewerHandler(BaseHTTPRequestHandler):
    """Serve the viewer page and its current tile coordinates."""

    def do_GET(self) -> None:
        route = urlparse(self.path).path

        if route in ("/", "/index.html"):
            self._send(PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        elif route == "/api/tiles":
            self._send(TILE_PAYLOAD, "application/json; charset=utf-8")
        elif route == "/api/board":
            self._send(BOARD_PAYLOAD, "application/json; charset=utf-8")
        else:
            self.send_error(404, "Not found")

    def log_message(self, format: str, *args: object) -> None:
        """Keep routine browser requests out of the terminal."""

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    """Start the local viewer and open it in the default browser."""
    with ThreadingHTTPServer((HOST, PORT), BoardViewerHandler) as server:
        url = f"http://{HOST}:{PORT}"
        print(f"Board viewer running at {url}")
        print("Press Ctrl+C to stop the viewer.")
        webbrowser.open(url)
        server.serve_forever()


if __name__ == "__main__":
    main()
