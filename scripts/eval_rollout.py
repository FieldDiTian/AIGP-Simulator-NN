import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Approximate rollout evaluation on processed dynamics dataset.")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best_val_model.pt"))
    parser.add_argument("--data", default=str(ROOT / "logs" / "processed" / "test.npz"))
    parser.add_argument("--horizons-s", nargs="+", type=float, default=[1.0, 3.0, 5.0, 10.0])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    from models.mlp_dynamics import build_model
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
    run_id = data["run_id"]
    time_s = data["time_s"]
    if len(x) == 0:
        raise RuntimeError("No rollout samples found.")

    grouped = group_indices(run_id, time_s)
    results = {}
    for horizon_s in args.horizons_s:
        steps = max(1, int(round(horizon_s * float(stats["hz"]))))
        errors = []
        diverged = 0
        for indices in grouped:
            if len(indices) <= steps:
                continue
            for start in range(0, len(indices) - steps, max(1, steps // 4)):
                idxs = indices[start:start + steps]
                err, is_diverged = rollout_error(model, stats, x, y, idxs, args.device)
                errors.append(err)
                diverged += int(is_diverged)
        if errors:
            errors = np.asarray(errors, dtype=np.float64)
            results[f"{horizon_s:.1f}s"] = {
                "rollouts": int(len(errors)),
                "position_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
                "position_median_m": float(np.median(errors)),
                "diverged": int(diverged),
            }
        else:
            results[f"{horizon_s:.1f}s"] = {
                "rollouts": 0,
                "position_rmse_m": None,
                "position_median_m": None,
                "diverged": 0,
            }
    print(json.dumps(results, indent=2), flush=True)


def group_indices(run_ids, times):
    buckets = defaultdict(list)
    for idx, (run, t) in enumerate(zip(run_ids, times)):
        buckets[str(run)].append((float(t), idx))
    groups = []
    for items in buckets.values():
        items = sorted(items)
        groups.append([idx for _, idx in items])
    return groups


def rollout_error(model, stats, x, y, indices, device):
    x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
    x_std = np.asarray(stats["x_std"], dtype=np.float32)
    y_mean = np.asarray(stats["y_mean"], dtype=np.float32)
    y_std = np.asarray(stats["y_std"], dtype=np.float32)
    state_dim = int(stats["state_dim"])
    action_dim = int(stats["action_dim"])
    k = int(stats["k"])
    step_dim = state_dim + action_dim

    history = x[indices[0]].copy().reshape(k + 1, step_dim)
    pred_p = np.zeros(3, dtype=np.float64)
    true_p = np.zeros(3, dtype=np.float64)

    for idx in indices:
        model_input = history.reshape(-1)
        model_input_n = (model_input - x_mean) / x_std
        with torch.no_grad():
            pred_n = model(torch.from_numpy(model_input_n[None, :]).to(device)).cpu().numpy()[0]
        pred = pred_n * y_std + y_mean
        true = y[idx]

        pred_p += pred[0:3]
        true_p += true[0:3]

        next_state = history[-1, :state_dim].copy()
        next_state[0:3] = pred[6:9]
        next_state[3:6] = pred[9:12]
        next_action = x[idx].reshape(k + 1, step_dim)[-1, state_dim:state_dim + action_dim]
        next_step = np.concatenate([next_state, next_action]).astype(np.float32)
        history = np.concatenate([history[1:], next_step[None, :]], axis=0)

    error = float(np.linalg.norm(pred_p - true_p))
    return error, error > 25.0 or not np.isfinite(error)


if __name__ == "__main__":
    main()
