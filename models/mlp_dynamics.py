import torch
from torch import nn


class MLPDynamics(nn.Module):
    def __init__(self, input_dim, output_dim=12, hidden_dim=512, num_layers=5):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(config):
    return MLPDynamics(
        input_dim=int(config["input_dim"]),
        output_dim=int(config.get("output_dim", 12)),
        hidden_dim=int(config.get("hidden_dim", 512)),
        num_layers=int(config.get("num_layers", 5)),
    )
