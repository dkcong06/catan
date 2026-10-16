"""Local interactive bot matches; diagnostics never enter the training collector."""

import argparse
from collections import OrderedDict
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from random import Random
from secrets import token_urlsafe
from threading import Lock
from urllib.parse import urlparse
import webbrowser

import numpy as np

from catan.board import GEOMETRY
from catan.state import Phase, _integer
from rl.actions import indexed_action
from rl.env import CatanEnv
from rl.policies import POLICIES, POLICY_VERSIONS
from rl.tactics import winning_actions


PAGE = Path(__file__).with_name("game_viewer.html")
RESOURCE_NAMES = ("wood", "sheep", "wheat", "brick", "ore")
CARD_NAMES = ("knight", "victory point", "road building", "year of plenty", "monopoly")
GEOMETRY_DATA = {
    name: getattr(GEOMETRY, name)
    for name in (
        "tile_positions",
        "vertex_positions",
        "tile_vertices",
        "edge_vertices",
        "port_vertices",
        "port_positions",
    )
}


def action_label(state, choice):
    if state.phase == Phase.DISCARD:
        return "Discard " + ", ".join(
            f"{int(n)} {name}" for n, name in zip(choice, RESOURCE_NAMES) if n
        )
    action = indexed_action(state, int(choice))
    name = type(action).__name__
    if name == "RoadAction":
        first, second = GEOMETRY.edge_vertices[action.edge]
        return f"Build road e{action.edge} (v{first}–v{second})"
    if name in ("SettlementAction", "CityAction"):
        return (
            f"Build {'city' if name == 'CityAction' else 'settlement'} v{action.vertex}"
        )
    if name == "RobberMoveAction":
        return f"Move robber to tile {action.tile}"
    if name == "StealAction":
        return f"Steal from P{action.victim + 1}"
    if name == "BankTradeAction":
        return (
            f"Trade {RESOURCE_NAMES[action.give]} for {RESOURCE_NAMES[action.receive]}"
        )
    if name == "YearOfPlentyAction":
        return "Year of Plenty: " + ", ".join(
            f"{n} {resource}"
            for n, resource in zip(action.resources, RESOURCE_NAMES)
            if n
        )
    if name == "MonopolyAction":
        return f"Monopoly: {RESOURCE_NAMES[action.resource]}"
    return {
        "RollAction": "Roll dice",
        "EndTurnAction": "End turn",
        "BuyDevelopmentAction": "Buy development card",
        "KnightAction": "Play knight",
        "RoadBuildingAction": "Play Road Building",
    }[name]


class PolicyRegistry:
    """Explicit CLI allowlist, lazy bounded model loading, no browser file paths."""

    def __init__(self, checkpoints=None):
        self.checkpoints = dict(checkpoints or {})
        if len(self.checkpoints) > 12 or any(
            not name
            or not name.isascii()
            or not name.replace("-", "").replace("_", "").isalnum()
            or name in POLICIES
            for name in self.checkpoints
        ):
            raise ValueError("use at most twelve unique named checkpoints")
        self.loaded = {}

    @property
    def names(self):
        return tuple(POLICIES) + tuple(self.checkpoints)

    def _model(self, name):
        if name not in self.loaded:
            from learning.ppo import PPOUpdater
            import torch

            torch.set_num_threads(1)
            path = Path(self.checkpoints[name])
            updater = PPOUpdater.load(path)
            if updater.model.schema[1] != 2:
                raise ValueError(
                    "This viewer accepts observation-schema-2 checkpoints; explicit other-profile support is not implemented."
                )
            self.loaded[name] = updater.model, sha256(path.read_bytes()).hexdigest()
        return self.loaded[name]

    def sample(self, name, decision, rng, *, deterministic=False):
        if name in POLICIES:
            return (
                POLICIES[name](decision, rng),
                [],
                None,
                f"{name}-v{POLICY_VERSIONS[name]}",
            )
        if name not in self.checkpoints:
            raise ValueError("Choose an available bot")
        from learning.policy import NeuralPolicy
        import torch

        model, digest = self._model(name)
        policy = NeuralPolicy(model, deterministic=deterministic, policy_version=name)
        top = []
        if decision.mask is not None:
            with torch.inference_mode():
                output = model(torch.from_numpy(decision.observation.copy()[None]))
                legal = torch.from_numpy(decision.mask.copy())
                probabilities = (
                    output.logits[0].masked_fill(~legal, -torch.inf).softmax(0).numpy()
                )
            indices = np.flatnonzero(decision.mask)
            selected = sorted(
                indices, key=lambda i: (-float(probabilities[i]), int(i))
            )[:5]
            top = [
                dict(action=int(i), probability=float(probabilities[i]))
                for i in selected
            ]
        sample = policy((decision,), (rng,))[0]
        return sample.action, top, float(sample.value), digest


def frame(env, event=None):
    state = env.snapshot()
    return dict(
        step=env.actions,
        phase=state.phase.name,
        turn=state.current_player,
        actor=state.discard_player
        if state.phase == Phase.DISCARD
        else state.current_player,
        roll=state.last_roll,
        robber=state.robber_tile,
        vertices=state.vertex_owner.tolist(),
        buildings=state.vertex_building.tolist(),
        edges=state.edge_owner.tolist(),
        hands=state.hands.tolist(),
        cards=state.development.tolist(),
        new_cards=state.development_new.tolist(),
        bank=state.bank.tolist(),
        players=[
            dict(
                public_vp=int(state.vp[p] - state.development[p, 1]),
                own_vp=int(state.vp[p]),
                resources=int(state.hands[p].sum()),
                cards=int(state.development[p].sum()),
                road=int(state.road_lengths[p]),
                knights=int(state.knights_played[p]),
                pieces=state.pieces_left[p].tolist(),
            )
            for p in range(state.n_players)
        ],
        longest=int(state.longest_road_owner),
        army=int(state.largest_army_owner),
        terminated=env.terminated,
        truncated=env.truncated,
        winner=state.current_player if env.terminated else None,
        event=event,
    )


class GameSessions:
    """Small bounded live states; browser retains one finite replay timeline."""

    def __init__(self, registry=None, capacity=8):
        self.registry = registry or PolicyRegistry()
        self.capacity = capacity
        self.sessions = OrderedDict()
        self.lock = Lock()

    def handle(self, route, payload):
        if not isinstance(payload, dict):
            return 400, {"error": "Send a JSON object"}
        with self.lock:
            try:
                if route == "/api/game/new":
                    seed = _integer(payload.get("seed", 0), "seed")
                    players = _integer(payload.get("players", 4), "players")
                    if players not in (3, 4):
                        raise ValueError("Choose three or four players")
                    cap = _integer(payload.get("max_actions", 5000), "max_actions")
                    deterministic = payload.get("deterministic", False)
                    names = payload.get("bots", ["tactical"] * players)
                    if not 0 <= seed <= 2**32 - 1 or not 16 <= cap <= 5000:
                        raise ValueError("Seed or action limit outside bounds")
                    if type(deterministic) is not bool or not isinstance(names, list):
                        raise ValueError("Invalid bot settings")
                    if len(names) != players or any(
                        name not in self.registry.names for name in names
                    ):
                        raise ValueError("Choose one available bot per player")
                    # Load/reject profiles before replacing any active game.
                    for name in set(names) - set(POLICIES):
                        self.registry._model(name)
                    env = CatanEnv(max_actions=cap)
                    env.reset(seed, n_players=players)
                    token = token_urlsafe(18)
                    self.sessions[token] = dict(
                        env=env,
                        names=names,
                        rngs=[
                            Random(f"catan-evaluation-v1:17:{seed}:{p}")
                            for p in range(players)
                        ],
                        deterministic=deterministic,
                        last=None,
                    )
                    if len(self.sessions) > self.capacity:
                        self.sessions.popitem(last=False)
                    board = env.snapshot().board
                    return 200, dict(
                        session=token,
                        seed=seed,
                        bots=names,
                        revision=0,
                        deterministic=deterministic,
                        max_actions=cap,
                        policy_seed=17,
                        geometry=GEOMETRY_DATA,
                        board=dict(
                            resources=[r.name.lower() for r in board.resources],
                            numbers=board.numbers,
                            ports=[p.name.lower() for p in board.ports],
                        ),
                        frame=frame(env),
                    )
                if route != "/api/game/step":
                    return 404, {"error": "Unknown game route"}
                token = payload.get("session")
                if not isinstance(token, str) or token not in self.sessions:
                    return 404, {"error": "Game expired. Start a new game."}
                game = self.sessions[token]
                env = game["env"]
                revision = _integer(payload.get("revision"), "revision")
                if revision != env.actions:
                    return 409, dict(
                        error="Outdated step. Review the current game.",
                        revision=env.actions,
                        frame=frame(env, game["last"]),
                    )
                decision = env.decision()
                if decision is None:
                    return 200, dict(
                        revision=env.actions, frame=frame(env, game["last"])
                    )
                before = env.snapshot()
                wins = winning_actions(decision)
                choice, top, value, identity = self.registry.sample(
                    game["names"][decision.player],
                    decision,
                    game["rngs"][decision.player],
                    deterministic=game["deterministic"],
                )
                label = action_label(before, choice)
                alternatives = [
                    dict(action=int(i), label=action_label(before, int(i)))
                    for i in np.flatnonzero(wins)
                ]
                for item in top:
                    item["label"] = action_label(before, item["action"])
                    item["winning"] = bool(wins[item["action"]])
                event = dict(
                    player=decision.player,
                    phase=decision.phase.name,
                    label=label,
                    action=None if decision.mask is None else int(choice),
                    discard=list(map(int, choice)) if decision.mask is None else None,
                    top=top,
                    value=value,
                    policy=identity,
                    winning=alternatives,
                    missed_win=bool(wins.any() and not wins[int(choice)])
                    if decision.mask is not None
                    else False,
                )
                env.step(choice, player=decision.player)
                result = frame(env, event)
                if label == "Roll dice":
                    event["label"] += f": {result['roll']}"
                game["last"] = event
                self.sessions.move_to_end(token)
                return 200, dict(revision=env.actions, frame=result)
            except (ValueError, TypeError, KeyError) as error:
                return 400, {"error": str(error)}


class GameViewerHandler(BaseHTTPRequestHandler):
    def _send(self, payload, content_type="application/json", status=200):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/":
            self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif route == "/game.js":
            self._send(
                PAGE.with_suffix(".js").read_bytes(), "text/javascript; charset=utf-8"
            )
        elif route == "/api/game/bots":
            self._send(dict(bots=self.server.games.registry.names))
        elif route == "/api/experiment":
            root = getattr(self.server, "experiment", None)
            result = {"status": "No experiment attached"}
            if root is not None:
                progress = root / "progress.json"
                if progress.is_file() and progress.stat().st_size < 65536:
                    result = json.loads(progress.read_text())
                rep, arm = result.get("replicate", 0), result.get("arm")
                if type(rep) is int and 0 <= rep <= 2:
                    curve = root / f"r{rep}" / "learning-curve.json"
                    if curve.is_file() and curve.stat().st_size < 262144:
                        result["warmup"] = json.loads(curve.read_text())[-1]
                    if arm in ("ppo", "anchored"):
                        state = root / f"r{rep}" / arm / "state.json"
                        if state.is_file() and state.stat().st_size < 2_000_000:
                            metrics = json.loads(state.read_text())["metrics"]
                            result["learner_wins"] = sum(
                                e["winner"] == m["seat"]
                                for m in metrics
                                for e in m["episodes"]
                            )
                            result["practice_games"] = sum(
                                len(m["episodes"]) for m in metrics
                            )
                            scored = [m for m in metrics if "validation" in m]
                            if scored:
                                result["validation"] = scored[-1]["validation"]
            self._send(result)
        else:
            self._send(dict(error="Not found"), status=404)

    def do_POST(self):
        route = urlparse(self.path).path
        if route not in ("/api/game/new", "/api/game/step"):
            self._send(dict(error="Not found"), status=404)
            return
        origin = self.headers.get("Origin")
        if origin is not None and origin != f"http://{self.headers.get('Host')}":
            self._send(dict(error="Use the local dashboard"), status=403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if (
                self.headers.get_content_type() != "application/json"
                or not 0 < length <= 4096
            ):
                raise ValueError("Send a JSON body of at most 4096 bytes")
            payload = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError) as error:
            self._send(dict(error=str(error)), status=400)
            return
        status, response = self.server.games.handle(route, payload)
        self._send(response, status=status)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8124, type=int)
    parser.add_argument(
        "--checkpoint", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--experiment", type=Path, help="Read-only progress directory")
    args = parser.parse_args()
    paths = {}
    for item in args.checkpoint:
        name, separator, path = item.partition("=")
        if not separator or name in paths or not Path(path).is_file():
            parser.error("Checkpoints must be unique NAME=existing-file entries")
        paths[name] = path
    with ThreadingHTTPServer((args.host, args.port), GameViewerHandler) as server:
        server.games = GameSessions(PolicyRegistry(paths))
        server.experiment = args.experiment
        url = f"http://{args.host}:{server.server_port}"
        print(f"Bot game dashboard: {url}")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
