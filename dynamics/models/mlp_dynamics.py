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


class GRUDynamics(nn.Module):
    def __init__(
        self,
        input_dim,
        state_dim,
        action_dim,
        k,
        output_dim=12,
        hidden_dim=512,
        num_layers=3,
        dropout=0.0,
    ):
        super().__init__()
        self.k = int(k)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.step_dim = self.state_dim + self.action_dim
        expected_input_dim = (self.k + 1) * self.step_dim
        if int(input_dim) != expected_input_dim:
            raise ValueError(
                f"input_dim={input_dim} does not match "
                f"(k+1)*(state_dim+action_dim)={expected_input_dim}"
            )
        self.gru = nn.GRU(
            input_size=self.step_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        seq = x.reshape(x.shape[0], self.k + 1, self.step_dim)
        out, _ = self.gru(seq)
        return self.head(out[:, -1, :])


def build_model(config):
    model_type = str(config.get("model_type", "mlp")).lower()
    if model_type == "gru":
        return GRUDynamics(
            input_dim=int(config["input_dim"]),
            output_dim=int(config.get("output_dim", 12)),
            hidden_dim=int(config.get("hidden_dim", 512)),
            num_layers=int(config.get("num_layers", 3)),
            dropout=float(config.get("dropout", 0.0)),
            k=int(config["k"]),
            state_dim=int(config["state_dim"]),
            action_dim=int(config["action_dim"]),
        )
    if model_type != "mlp":
        raise ValueError(f"Unknown model_type: {model_type}")
    return MLPDynamics(
        input_dim=int(config["input_dim"]),
        output_dim=int(config.get("output_dim", 12)),
        hidden_dim=int(config.get("hidden_dim", 512)),
        num_layers=int(config.get("num_layers", 5)),
    )
