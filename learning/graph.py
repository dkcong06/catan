"""Compact two-round board encoder, using the unchanged economic observation.

Only the board encoder changes: flat catalog output, value head and complete
conditional discard evaluation retain ActorCritic's contracts. No graph library,
dynamic edge construction, hidden state or engine access is needed at inference.
"""

from math import sqrt

import torch
from torch import nn

from catan.board import GEOMETRY
from catan.state import _integer
from learning.model import ActorCritic, NetworkOutput
from rl.actions import ACTION_DIM, ACTION_SCHEMA_VERSION
from rl.economics import ECONOMIC_OBS_DIM, ECONOMIC_SCHEMA_VERSION

NODE_WIDTH = 12
BOARD_DIM = 19 * 9 + 54 * 16 + 72 * 5


def _linear(inputs, outputs):
    return nn.Linear(inputs, outputs, dtype=torch.float32, device="cpu")


def _incidence(rows, columns):
    result = torch.zeros((len(rows), columns), dtype=torch.float32, device="cpu")
    for row, neighbors in enumerate(rows):
        for column in neighbors:
            result[row, column] = 1 / len(neighbors)
    return result


class _Round(nn.Module):
    def __init__(self):
        super().__init__()
        self.tiles = _linear(2 * NODE_WIDTH, NODE_WIDTH)
        self.vertices = _linear(3 * NODE_WIDTH, NODE_WIDTH)
        self.edges = _linear(2 * NODE_WIDTH, NODE_WIDTH)

    def forward(self, tiles, vertices, edges, tv, vt, ve, ev):
        # All relations read the previous round, not partially updated nodes.
        return (
            self.tiles(torch.cat((tiles, tv @ vertices), dim=2)).relu(),
            self.vertices(torch.cat((vertices, vt @ tiles, ve @ edges), dim=2)).relu(),
            self.edges(torch.cat((edges, ev @ vertices), dim=2)).relu(),
        )


class GraphActorCritic(ActorCritic):
    """Model4/observation4; width112 gives capacity close to the economic MLP."""

    observation_dim = ECONOMIC_OBS_DIM
    fixed_buffers = ("resource_codes", "count_choices", "tv", "vt", "ve", "ev")

    def __init__(self, hidden_size=112, *, seed=0):
        nn.Module.__init__(self)
        hidden_size = _integer(hidden_size, "hidden_size")
        seed = _integer(seed, "seed")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.hidden_size, self.seed = hidden_size, seed
        self.schema = (4, ECONOMIC_SCHEMA_VERSION, ACTION_SCHEMA_VERSION)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.tile_input = _linear(9, NODE_WIDTH)
            self.vertex_input = _linear(16, NODE_WIDTH)
            self.edge_input = _linear(5, NODE_WIDTH)
            self.rounds = nn.ModuleList((_Round(), _Round()))
            width = 145 * NODE_WIDTH + ECONOMIC_OBS_DIM - BOARD_DIM
            self.trunk = nn.Sequential(
                _linear(width, hidden_size),
                nn.ReLU(),
                _linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            self.output = _linear(hidden_size, ACTION_DIM + 1)
            self.discard = _linear(hidden_size + 10, 20)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, sqrt(2))
                    nn.init.zeros_(module.bias)
            nn.init.orthogonal_(self.output.weight, 0.01)
            nn.init.orthogonal_(self.output.weight[-1:], 1)
            nn.init.orthogonal_(self.discard.weight, 0.01)
        self.register_buffer(
            "resource_codes", torch.eye(5, dtype=torch.float32, device="cpu")
        )
        self.register_buffer("count_choices", torch.arange(20, device="cpu"))
        self.register_buffer("tv", _incidence(GEOMETRY.tile_vertices, 54))
        self.register_buffer("vt", _incidence(GEOMETRY.vertex_tiles, 19))
        self.register_buffer("ve", _incidence(GEOMETRY.vertex_edges, 72))
        self.register_buffer("ev", _incidence(GEOMETRY.edge_vertices, 54))

    def board_features(self, observations):
        n = len(observations)
        tiles = self.tile_input(observations[:, :171].reshape(n, 19, 9)).relu()
        vertices = self.vertex_input(
            observations[:, 171:1035].reshape(n, 54, 16)
        ).relu()
        edges = self.edge_input(observations[:, 1035:1395].reshape(n, 72, 5)).relu()
        for layer in self.rounds:
            tiles, vertices, edges = layer(
                tiles, vertices, edges, self.tv, self.vt, self.ve, self.ev
            )
        return tiles, vertices, edges

    def forward(self, observations):
        if (
            observations.ndim != 2
            or observations.shape[1] != self.observation_dim
            or observations.dtype != torch.float32
        ):
            raise ValueError("graph observations must be float32 (N,2000)")
        nodes = self.board_features(observations)
        joined = torch.cat(
            (*(x.flatten(1) for x in nodes), observations[:, BOARD_DIM:]), dim=1
        )
        hidden = self.trunk(joined)
        output = self.output(hidden)
        return NetworkOutput(output[:, :ACTION_DIM], output[:, ACTION_DIM], hidden)
