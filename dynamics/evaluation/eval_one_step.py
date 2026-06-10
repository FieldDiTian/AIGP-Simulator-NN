import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Evaluate one-step drone dynamics prediction.")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best_val_model.pt"))
    parser.add_argument("--data", default=str(ROOT / "logs" / "processed" / "test.npz"))
    parser.add_argument("--raw-input", nargs="+", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--hz", type=float, default=None)
    parser.add_argument("--max-rate", type=float, default=None)
    parser.add_argument("--no-actuator", action="store_true")
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    from dynamics.models.flightsim_wrapper import FlightSimBodyCentricWrapper
    from dynamics.models.mlp_dynamics import build_model
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    stats = checkpoint["normalization_stats"]
    model = build_model(checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.raw_input:
        k = int(args.k if args.k is not None else stats.get("k", 10))
        hz = float(args.hz if args.hz is not None else stats.get("hz", 60.0))
        max_rate = float(args.max_rate if args.max_rate is not None else stats.get("max_rate", 1.0))
        use_actuator = False if args.no_actuator else bool(stats.get("use_actuator", True))
        use_imu = False if args.no_imu else bool(stats.get("use_imu", True))
        x, y = load_raw_eval_arrays(
            args.raw_input,
            k=k,
            hz=hz,
            max_rate=max_rate,
            use_actuator=use_actuator,
            use_imu=use_imu,
        )
    else:
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

    metrics = FlightSimBodyCentricWrapper.prediction_metrics(pred, y)
    metrics["source"] = "raw_input" if args.raw_input else "processed_npz"
    print(json.dumps(metrics, indent=2), flush=True)


def load_raw_eval_arrays(inputs, k, hz, max_rate, use_actuator, use_imu):
    from dynamics.dataset.build_dataset import resample_segment, split_segments
    from dynamics.models.flightsim_wrapper import FlightSimBodyCentricWrapper, required_fields_present

    paths = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.jsonl")))
        else:
            paths.append(path)

    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("sample_type") == "dynamics" and required_fields_present(row):
                    rows.append(row)

    wrapper = FlightSimBodyCentricWrapper(
        max_rate=max_rate,
        use_actuator=use_actuator,
        use_imu=use_imu,
    )
    x_values = []
    y_values = []
    dt = 1.0 / hz
    for segment in split_segments(rows):
        resampled = resample_segment(segment["rows"], dt=dt, max_gap_s=dt * 1.5)
        if len(resampled) < k + 2:
            continue
        frames = [wrapper.frame_from_log_row(row) for row in resampled]
        for i in range(k, len(frames) - 1):
            history = frames[i - k:i + 1]
            x_values.append(wrapper.model_input_from_history(history))
            y_values.append(wrapper.target_from_frames(frames[i], frames[i + 1]).y)

    if not x_values:
        state_dim = 9 + (3 if use_imu else 0) + (4 if use_actuator else 0)
        input_dim = (k + 1) * (state_dim + 4)
        return (
            np.zeros((0, input_dim), dtype=np.float32),
            np.zeros((0, 12), dtype=np.float32),
        )
    return (
        np.stack(x_values).astype(np.float32),
        np.stack(y_values).astype(np.float32),
    )


if __name__ == "__main__":
    main()
