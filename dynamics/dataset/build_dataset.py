import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamics.models.flightsim_wrapper import (
    FlightSimBodyCentricWrapper,
    collision_count,
    required_fields_present,
)


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


def split_segments(
    rows,
    *,
    stuck_position_epsilon_m=1e-4,
    stuck_min_duration_s=0.5,
    disable_stuck_position_filter=False,
    return_stats=False,
):
    grouped = defaultdict(list)
    for row in rows:
        run_id = row.get("run_id", "unknown_run")
        reset_counter = row.get("reset_counter")
        if reset_counter is None and row.get("odometry"):
            reset_counter = row["odometry"].get("reset_counter")
        grouped[(run_id, int(reset_counter or 0))].append(row)

    segments = []
    stats = {
        "collision_filtered_rows": 0,
        "stuck_position_filtered_rows": 0,
    }
    for (run_id, reset_counter), segment_rows in grouped.items():
        segment_rows = sorted(segment_rows, key=lambda r: int(r["t_wall_ns"]))
        before_collision = len(segment_rows)
        segment_rows = drop_collision_suffix(segment_rows)
        stats["collision_filtered_rows"] += before_collision - len(segment_rows)
        if not disable_stuck_position_filter:
            before_stuck = len(segment_rows)
            segment_rows = drop_stuck_position_suffix(
                segment_rows,
                epsilon_m=stuck_position_epsilon_m,
                min_duration_s=stuck_min_duration_s,
            )
            stats["stuck_position_filtered_rows"] += before_stuck - len(segment_rows)
        if len(segment_rows) >= 3:
            segments.append({
                "run_id": run_id,
                "reset_counter": reset_counter,
                "rows": segment_rows,
            })
    if return_stats:
        return segments, stats
    return segments


def drop_collision_suffix(rows):
    """Drop only the post-collision suffix inside one reset segment.

    Rows after an automatic simulator reset can still be used because
    split_segments() groups by (run_id, reset_counter) before this function runs.
    """
    if not rows:
        return rows
    baseline = collision_count(rows[0])
    kept = []
    for row in rows:
        if collision_count(row) > baseline:
            break
        kept.append(row)
    return kept


def drop_stuck_position_suffix(rows, epsilon_m=1e-4, min_duration_s=0.5):
    """Drop the suffix after sustained unchanged position in one reset segment.

    This catches boundary/contact states where the aircraft keeps receiving
    controls but its reported position is frozen. Automatic reset data is kept
    because split_segments() runs this per (run_id, reset_counter) segment.
    """
    if len(rows) < 2:
        return rows

    stationary_start_idx = None
    stationary_start_s = None
    prev_pos = position_ned_from_row(rows[0])
    if prev_pos is None:
        return rows

    for idx in range(1, len(rows)):
        pos = position_ned_from_row(rows[idx])
        if pos is None:
            stationary_start_idx = None
            stationary_start_s = None
            prev_pos = None
            continue

        if prev_pos is not None and np.linalg.norm(pos - prev_pos) <= epsilon_m:
            if stationary_start_idx is None:
                stationary_start_idx = idx - 1
                stationary_start_s = wall_time_s(rows[stationary_start_idx])
            if wall_time_s(rows[idx]) - stationary_start_s >= min_duration_s:
                return rows[:stationary_start_idx + 1]
        else:
            stationary_start_idx = None
            stationary_start_s = None

        prev_pos = pos
    return rows


def position_ned_from_row(row):
    odom = row.get("odometry") or {}
    pos = odom.get("p_ned")
    if pos is None:
        return None
    return np.asarray(pos, dtype=np.float64)


def wall_time_s(row):
    return int(row["t_wall_ns"]) / 1e9


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


def build_samples(segments, k, dt, max_rate, use_actuator, use_imu):
    samples = []
    dropped_windows = 0
    wrapper = FlightSimBodyCentricWrapper(
        max_rate=max_rate,
        use_actuator=use_actuator,
        use_imu=use_imu,
    )
    for segment_idx, segment in enumerate(segments):
        rows = resample_segment(segment["rows"], dt=dt, max_gap_s=dt * 1.5)
        if len(rows) < k + 2:
            continue
        frames = [wrapper.frame_from_log_row(row) for row in rows]
        targets = [
            wrapper.target_from_frames(frames[i], frames[i + 1]).y
            for i in range(len(frames) - 1)
        ]
        for i in range(k, len(rows) - 1):
            history_frames = frames[i - k:i + 1]
            if len(history_frames) != k + 1:
                dropped_windows += 1
                continue
            x = wrapper.model_input_from_history(history_frames)
            y = targets[i]
            samples.append({
                "x": x,
                "y": y,
                "run_id": segment["run_id"],
                "segment_idx": segment_idx,
                "time_s": frames[i].t_s,
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
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--stuck-position-epsilon-m", type=float, default=1e-4)
    parser.add_argument("--stuck-min-duration-s", type=float, default=0.5)
    parser.add_argument("--disable-stuck-position-filter", action="store_true")
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
    segments, filter_stats = split_segments(
        rows,
        stuck_position_epsilon_m=args.stuck_position_epsilon_m,
        stuck_min_duration_s=args.stuck_min_duration_s,
        disable_stuck_position_filter=args.disable_stuck_position_filter,
        return_stats=True,
    )
    use_actuator = not args.no_actuator
    use_imu = not args.no_imu
    dt = 1.0 / args.hz
    samples, dropped_windows = build_samples(
        segments,
        k=args.k,
        dt=dt,
        max_rate=args.max_rate,
        use_actuator=use_actuator,
        use_imu=use_imu,
    )
    train, val, test = split_by_run(samples)

    state_dim = 9 + (3 if use_imu else 0) + (4 if use_actuator else 0)
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
        "use_imu": use_imu,
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
        "collision_filtered_rows": filter_stats["collision_filtered_rows"],
        "stuck_position_filtered_rows": filter_stats["stuck_position_filtered_rows"],
        "collision_filter_policy": (
            "drop collision row and suffix within each run_id/reset_counter segment; "
            "post-reset data is kept as a new segment"
        ),
        "stuck_position_filter_policy": (
            "drop suffix within each run_id/reset_counter segment after reported "
            f"p_ned changes by <= {args.stuck_position_epsilon_m} m for at least "
            f"{args.stuck_min_duration_s} s; post-reset data is kept as a new segment"
        ),
    }
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
