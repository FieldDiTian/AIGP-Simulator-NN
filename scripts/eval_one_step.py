import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Evaluate one-step drone dynamics prediction.")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best_val_model.pt"))
    parser.add_argument("--data", default=str(ROOT / "logs" / "processed" / "test.npz"))
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
    if len(x) == 0:
        raise RuntimeError("No evaluation samples found.")

    x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
    x_std = np.asarray(stats["x_std"], dtype=np.float32)
    y_mean = np.asarray(stats["y_mean"], dtype=np.float32)
    y_std = np.asarray(stats["y_std"], dtype=np.float32)
    x_n = (x - x_mean) / x_std

    preds = []
    batch = 4096
    with torch.no_grad():
        for start in range(0, len(x_n), batch):
            xb = torch.from_numpy(x_n[start:start + batch]).to(args.device)
            pred_n = model(xb).cpu().numpy()
            preds.append(pred_n)
    pred = np.concatenate(preds, axis=0) * y_std + y_mean

    metrics = {
        "samples": int(len(x)),
        "delta_p_body_rmse_m": rmse(pred[:, 0:3], y[:, 0:3]),
        "delta_rotvec_body_rmse_rad": rmse(pred[:, 3:6], y[:, 3:6]),
        "v_body_next_rmse_mps": rmse(pred[:, 6:9], y[:, 6:9]),
        "omega_body_next_rmse_radps": rmse(pred[:, 9:12], y[:, 9:12]),
    }
    print(json.dumps(metrics, indent=2), flush=True)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


if __name__ == "__main__":
    main()
