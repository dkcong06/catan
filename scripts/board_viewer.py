"""Serve the current Catan board geometry in a local web browser."""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from random import Random
from urllib.parse import parse_qs, urlparse

from catan.board import GEOMETRY, make_board


HOST = "127.0.0.1"
PORT = 8000
PAGE_PATH = Path(__file__).with_name("board_viewer.html")
DISCARD_PAGE_PATH = Path(__file__).with_name("discard_viewer.html")
TILE_PAYLOAD = json.dumps({"tiles": GEOMETRY.tile_coordinates}).encode("utf-8")
_GEOMETRY_PAYLOAD = {
    "tile_positions": GEOMETRY.tile_positions,
    "vertex_positions": GEOMETRY.vertex_positions,
    "edge_positions": GEOMETRY.edge_positions,
    "tile_vertices": GEOMETRY.tile_vertices,
    "edge_vertices": GEOMETRY.edge_vertices,
    "port_vertices": GEOMETRY.port_vertices,
    "port_positions": GEOMETRY.port_positions,
}
# Geometry never changes: encode it once, retaining the existing flat JSON schema.
_GEOMETRY_JSON_PREFIX = (
    json.dumps(_GEOMETRY_PAYLOAD, separators=(",", ":")).encode("utf-8")[:-1] + b","
)


def board_payload(seed: int) -> bytes:
    """Encode a preview without retaining per-seed boards in an unbounded cache."""
    board = make_board(Random(seed))
    layout = json.dumps(
        {
            "seed": seed,
            "resources": [resource.name.lower() for resource in board.resources],
            "numbers": board.numbers,
            "robber_tile": board.robber_tile,
            "ports": [
                {"type": port.name.lower(), "ratio": port.ratio} for port in board.ports
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return _GEOMETRY_JSON_PREFIX + layout[1:]


BOARD_PAYLOAD = board_payload(0)


class BoardViewerHandler(BaseHTTPRequestHandler):
    """Serve the viewer page, geometry, and reproducible board previews."""

    def do_GET(self) -> None:
        request = urlparse(self.path)
        route = request.path

        if route in ("/", "/index.html"):
            self._send(PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        elif route == "/discards":
            self._send(DISCARD_PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        elif route == "/api/tiles":
            self._send(TILE_PAYLOAD, "application/json; charset=utf-8")
        elif route == "/api/board":
            seeds = parse_qs(request.query, keep_blank_values=True).get("seed", ["0"])
            if (
                len(seeds) != 1
                or not seeds[0].isascii()
                or not seeds[0].isdigit()
                or len(seeds[0]) > 10
                or int(seeds[0]) > 2**32 - 1
            ):
                self.send_error(400, "Seed must be an integer from 0 to 4294967295")
                return
            seed = int(seeds[0])
            self._send(
                BOARD_PAYLOAD if seed == 0 else board_payload(seed),
                "application/json; charset=utf-8",
            )
        else:
            self.send_error(404, "Not found")

    def do_POST(self) -> None:
        """Engine-backed demo choices; board GET routes need no NumPy import."""
        route = urlparse(self.path).path
        if route not in (
            "/api/discard/new",
            "/api/discard/seven",
            "/api/discard/choose",
        ):
            self.send_error(404, "Not found")
            return
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Send application/json.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("JSON body must be between 1 and 4096 bytes.")
            payload = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError) as error:
            self._send(
                json.dumps({"error": str(error)}).encode(), "application/json", 400
            )
            return
        try:
            from scripts.discard_demo import DEMO
        except ImportError:
            self._send(
                b'{"error":"Manual discard demo requires NumPy. Run the Docker viewer from README."}',
                "application/json",
                503,
            )
            return
        status, result = DEMO.handle(route, payload)
        self._send(
            json.dumps(result).encode(), "application/json; charset=utf-8", status
        )

    def log_message(self, format: str, *args: object) -> None:
        """Keep routine browser requests out of the terminal."""

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    """Start the local viewer and open it in the default browser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    with ThreadingHTTPServer((args.host, args.port), BoardViewerHandler) as server:
        url = f"http://{args.host}:{server.server_port}"
        print(f"Board viewer running at {url}")
        print("Press Ctrl+C to stop the viewer.")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
