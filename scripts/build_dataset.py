import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.dynamics_math import action_from_row, build_body_state, target_from_rows


def load_jsonl(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("sample_type") != "dynamics":
                    continue
                if required_fields_present(row):
                    rows.append(row)
    return rows


def required_fields_present(row):
    return (
        row.get("action") is not None
        and row.get("odometry") is not None
        and row.get("imu") is not None
        and row.get("attitude") is not None
        and row["odometry"].get("p_ned") is not None
        and row["odometry"].get("q_wxyz") is not None
        and row["odometry"].get("omega") is not None
        and (
            row.get("local_position_ned") is not None
            or row["odometry"].get("v") is not None
        )
    )


def split_segments(rows):
    grouped = defaultdict(list)
    for row in rows:
        run_id = row.get("run_id", "unknown_run")
        reset_counter = row.get("reset_counter")
        if reset_counter is None and row.get("odometry"):
            reset_counter = row["odometry"].get("reset_counter")
        grouped[(run_id, int(reset_counter or 0))].append(row)

    segments = []
    for (run_id, reset_counter), segment_rows in grouped.items():
        segment_rows = sorted(segment_rows, key=lambda r: int(r["t_wall_ns"]))
        segment_rows = drop_collision_suffix(segment_rows)
        if len(segment_rows) >= 3:
            segments.append({
                "run_id": run_id,
                "reset_counter": reset_counter,
                "rows": segment_rows,
            })
    return segments


def drop_collision_suffix(rows):
    if not rows:
        return rows
    baseline = collision_count(rows[0])
    kept = []
    for row in rows:
        if collision_count(row) > baseline:
            break
        kept.append(row)
    return kept


def collision_count(row):
    collision = row.get("collision") or {}
    return int(collision.get("count") or 0)


def resample_segment(rows, dt, max_gap_s):
    rows = sorted(rows, key=lambda r: int(r["t_wall_ns"]))
    times = np.asarray([int(r["t_wall_ns"]) / 1e9 for r in rows], dtype=np.float64)
    start = times[0]
    end = times[-1]
    if end <= start:
        return []

    grid = np.arange(start, end + 0.5 * dt, dt)
    resampled = []
    for t in grid:
        idx = int(np.searchsorted(times, t))
        candidates = []
        if idx < len(rows):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda i: abs(times[i] - t))
        if abs(times[best] - t) <= max_gap_s:
            row = dict(rows[best])
            row["_resampled_time_s"] = float(t)
            resampled.append(row)
    return resampled


def build_samples(segments, k, dt, max_rate, use_actuator):
    samples = []
    dropped_windows = 0
    for segment_idx, segment in enumerate(segments):
        rows = resample_segment(segment["rows"], dt=dt, max_gap_s=dt * 1.5)
        if len(rows) < k + 2:
            continue
        states = [build_body_state(row, use_actuator=use_actuator) for row in rows]
        actions = [action_from_row(row, max_rate=max_rate) for row in rows]
        targets = [target_from_rows(rows[i], rows[i + 1]) for i in range(len(rows) - 1)]
        for i in range(k, len(rows) - 1):
            history_states = states[i - k:i + 1]
            history_actions = actions[i - k:i + 1]
            if len(history_states) != k + 1:
                dropped_windows += 1
                continue
            x = np.concatenate([
                np.concatenate([s, a]).astype(np.float32)
                for s, a in zip(history_states, history_actions)
            ])
            y = targets[i]
            samples.append({
                "x": x,
                "y": y,
                "run_id": segment["run_id"],
                "segment_idx": segment_idx,
                "time_s": rows[i].get("_resampled_time_s", int(rows[i]["t_wall_ns"]) / 1e9),
            })
    return samples, dropped_windows


def split_by_run(samples):
    by_run = defaultdict(list)
    for sample in samples:
        by_run[sample["run_id"]].append(sample)
    run_ids = sorted(by_run)
    if not run_ids:
        return [], [], []
    n = len(run_ids)
    train_n = max(1, int(round(0.70 * n)))
    val_n = max(1, int(round(0.15 * n))) if n >= 3 else 0
    if train_n + val_n >= n and n >= 2:
        train_n = n - 1
        val_n = 0
    train_runs = set(run_ids[:train_n])
    val_runs = set(run_ids[train_n:train_n + val_n])
    test_runs = set(run_ids[train_n + val_n:])
    if not test_runs and n >= 2:
        test_runs = {run_ids[-1]}
        train_runs.discard(run_ids[-1])

    return (
        [s for s in samples if s["run_id"] in train_runs],
        [s for s in samples if s["run_id"] in val_runs],
        [s for s in samples if s["run_id"] in test_runs],
    )


def save_split(path, samples, input_dim, output_dim):
    if samples:
        x = np.stack([s["x"] for s in samples]).astype(np.float32)
        y = np.stack([s["y"] for s in samples]).astype(np.float32)
        run_id = np.asarray([s["run_id"] for s in samples])
        time_s = np.asarray([s["time_s"] for s in samples], dtype=np.float64)
        segment_idx = np.asarray([s["segment_idx"] for s in samples], dtype=np.int32)
    else:
        x = np.zeros((0, input_dim), dtype=np.float32)
        y = np.zeros((0, output_dim), dtype=np.float32)
        run_id = np.asarray([], dtype=str)
        time_s = np.asarray([], dtype=np.float64)
        segment_idx = np.asarray([], dtype=np.int32)
    np.savez_compressed(path, x=x, y=y, run_id=run_id, time_s=time_s, segment_idx=segment_idx)


def compute_stats(samples, input_dim, output_dim):
    if samples:
        x = np.stack([s["x"] for s in samples]).astype(np.float32)
        y = np.stack([s["y"] for s in samples]).astype(np.float32)
    else:
        x = np.zeros((1, input_dim), dtype=np.float32)
        y = np.zeros((1, output_dim), dtype=np.float32)
    x_mean = x.mean(axis=0)
    x_std = np.maximum(x.std(axis=0), 1e-6)
    y_mean = y.mean(axis=0)
    y_std = np.maximum(y.std(axis=0), 1e-6)
    return {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Build FlightSim dynamics dataset from raw jsonl logs.")
    parser.add_argument("--input", nargs="+", default=[str(ROOT / "logs" / "raw")])
    parser.add_argument("--output-dir", default=str(ROOT / "logs" / "processed"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--max-rate", type=float, default=1.0)
    parser.add_argument("--no-actuator", action="store_true")
    args = parser.parse_args()

    input_paths = []
    for item in args.input:
        path = Path(item)
        if path.is_dir():
            input_paths.extend(sorted(path.glob("*.jsonl")))
        else:
            input_paths.append(path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_paths)
    segments = split_segments(rows)
    use_actuator = not args.no_actuator
    dt = 1.0 / args.hz
    samples, dropped_windows = build_samples(
        segments,
        k=args.k,
        dt=dt,
        max_rate=args.max_rate,
        use_actuator=use_actuator,
    )
    train, val, test = split_by_run(samples)

    state_dim = 16 if use_actuator else 12
    action_dim = 4
    input_dim = (args.k + 1) * (state_dim + action_dim)
    output_dim = 12
    save_split(output_dir / "train.npz", train, input_dim, output_dim)
    save_split(output_dir / "val.npz", val, input_dim, output_dim)
    save_split(output_dir / "test.npz", test, input_dim, output_dim)

    stats = compute_stats(train, input_dim, output_dim)
    stats.update({
        "k": args.k,
        "hz": args.hz,
        "dt": dt,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "use_actuator": use_actuator,
        "max_rate": args.max_rate,
    })
    with open(output_dir / "normalization_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    summary = {
        "raw_rows": len(rows),
        "segments": len(segments),
        "samples": len(samples),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "dropped_windows": dropped_windows,
    }
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
