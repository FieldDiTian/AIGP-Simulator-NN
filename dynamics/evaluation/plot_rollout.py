import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Plot an approximate model rollout against processed dataset deltas.")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best_val_model.pt"))
    parser.add_argument("--data", default=str(ROOT / "logs" / "processed" / "test.npz"))
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--output", default=str(ROOT / "logs" / "processed" / "rollout_plot.png"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    from dynamics.models.dynamics_math import rotvec_to_rotmat
    from dynamics.models.mlp_dynamics import build_model
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    stats = checkpoint["normalization_stats"]
    model = build_model(checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data = np.load(args.data, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    if len(x) == 0:
        raise RuntimeError("No samples to plot.")

    steps = min(args.steps, len(x))
    pred, true = rollout_positions(model, stats, x[:steps], y[:steps], args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(true[:, 0], true[:, 1], label="true cumulative delta p")
    plt.plot(pred[:, 0], pred[:, 1], label="model cumulative delta p")
    plt.xlabel("body-delta x accumulation [m]")
    plt.ylabel("body-delta y accumulation [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved rollout plot to {output}", flush=True)


def rollout_positions(model, stats, x, y, device):
    x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
    x_std = np.asarray(stats["x_std"], dtype=np.float32)
    y_mean = np.asarray(stats["y_mean"], dtype=np.float32)
    y_std = np.asarray(stats["y_std"], dtype=np.float32)
    state_dim = int(stats["state_dim"])
    action_dim = int(stats["action_dim"])
    k = int(stats["k"])
    step_dim = state_dim + action_dim

    history = x[0].copy().reshape(k + 1, step_dim)
    pred_positions = [np.zeros(3)]
    true_positions = [np.zeros(3)]
    pred_p = np.zeros(3, dtype=np.float64)
    true_p = np.zeros(3, dtype=np.float64)

    for idx in range(len(x)):
        model_input = history.reshape(-1)
        model_input_n = (model_input - x_mean) / x_std
        with torch.no_grad():
            pred_n = model(torch.from_numpy(model_input_n[None, :]).to(device)).cpu().numpy()[0]
        pred = pred_n * y_std + y_mean
        pred_p += pred[0:3]
        true_p += y[idx, 0:3]
        pred_positions.append(pred_p.copy())
        true_positions.append(true_p.copy())

        next_state = history[-1, :state_dim].copy()
        next_state[0:3] = pred[6:9]
        next_state[3:6] = pred[9:12]
        if state_dim >= 9:
            next_state[6:9] = rotvec_to_rotmat(pred[3:6]).T @ history[-1, 6:9]
        next_action = x[idx].reshape(k + 1, step_dim)[-1, state_dim:state_dim + action_dim]
        history = np.concatenate([
            history[1:],
            np.concatenate([next_state, next_action]).astype(np.float32)[None, :],
        ], axis=0)

    return np.asarray(pred_positions), np.asarray(true_positions)


if __name__ == "__main__":
    main()
