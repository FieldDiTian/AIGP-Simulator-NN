import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamics.dataset.build_dataset import resample_segment, split_segments
from dynamics.models.flightsim_wrapper import (
    FlightSimBodyCentricWrapper,
    collision_count,
    required_fields_present,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze body-centric coverage and optional MLP error bins.")
    parser.add_argument("--input", nargs="+", default=[str(ROOT / "logs" / "raw")])
    parser.add_argument("--output-dir", default=str(ROOT / "logs" / "coverage"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--max-rate", type=float, default=1.0)
    parser.add_argument("--no-actuator", action="store_true")
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--min-samples-per-bin", type=int, default=100)
    parser.add_argument("--max-reported-bins", type=int, default=80)
    parser.add_argument("--stuck-position-epsilon-m", type=float, default=1e-4)
    parser.add_argument("--stuck-min-duration-s", type=float, default=0.5)
    parser.add_argument("--disable-stuck-position-filter", action="store_true")
    args = parser.parse_args()

    input_paths = expand_inputs(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, missing_rows = load_rows(input_paths)
    wrapper = FlightSimBodyCentricWrapper(
        max_rate=args.max_rate,
        use_actuator=not args.no_actuator,
        use_imu=not args.no_imu,
    )
    collision_frames = build_collision_frames(rows, wrapper)
    samples, summary = build_coverage_samples(
        rows=rows,
        wrapper=wrapper,
        k=args.k,
        hz=args.hz,
        stuck_position_epsilon_m=args.stuck_position_epsilon_m,
        stuck_min_duration_s=args.stuck_min_duration_s,
        disable_stuck_position_filter=args.disable_stuck_position_filter,
    )
    if args.checkpoint:
        attach_model_errors(samples, args.checkpoint, args.device)

    views = make_views()
    reports = {}
    for view in views:
        reports[view["name"]] = analyze_view(
            view=view,
            samples=samples,
            collision_frames=collision_frames,
            bins=args.bins,
            min_samples=args.min_samples_per_bin,
            max_reported_bins=args.max_reported_bins,
        )

    report = {
        "inputs": [str(p) for p in input_paths],
        "raw_rows_with_required_fields": len(rows),
        "raw_rows_missing_required_fields": missing_rows,
        "segments_after_collision_filter": summary["segments"],
        "valid_history_samples": len(samples),
        "collision_filtered_rows_with_required_fields": len(collision_frames),
        "collision_filtered_rows": summary["collision_filtered_rows"],
        "stuck_position_filtered_rows": summary["stuck_position_filtered_rows"],
        "collision_filter_policy": (
            "drop collision row and suffix within each run_id/reset_counter segment; "
            "post-reset data is kept as a new segment"
        ),
        "stuck_position_filter_policy": (
            "drop suffix within each run_id/reset_counter segment after reported "
            f"p_ned changes by <= {args.stuck_position_epsilon_m} m for at least "
            f"{args.stuck_min_duration_s} s; post-reset data is kept as a new segment"
        ),
        "k": args.k,
        "hz": args.hz,
        "max_rate": args.max_rate,
        "use_actuator": not args.no_actuator,
        "use_imu": not args.no_imu,
        "bins": args.bins,
        "min_samples_per_bin": args.min_samples_per_bin,
        "checkpoint": args.checkpoint,
        "views": reports,
    }
    with open(output_dir / "coverage_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    undercovered = {
        name: {
            "undercovered_bins_count": view_report["undercovered_bins_count"],
            "high_error_bins_count": view_report["high_error_bins_count"],
            "reported_bins": view_report["reported_bins"],
        }
        for name, view_report in reports.items()
    }
    with open(output_dir / "undercovered_bins.json", "w", encoding="utf-8") as handle:
        json.dump(undercovered, handle, indent=2)

    printable = {
        "valid_history_samples": len(samples),
        "collision_filtered_rows_with_required_fields": len(collision_frames),
        "output_dir": str(output_dir),
        "views": {
            name: {
                "undercovered_bins_count": view_report["undercovered_bins_count"],
                "high_error_bins_count": view_report["high_error_bins_count"],
            }
            for name, view_report in reports.items()
        },
    }
    print(json.dumps(printable, indent=2), flush=True)


def expand_inputs(items):
    paths = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.jsonl")))
        else:
            paths.append(path)
    return paths


def load_rows(paths):
    rows = []
    missing = 0
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
                else:
                    missing += 1
    return rows, missing


def build_collision_frames(rows, wrapper):
    """Return filtered collision/post-collision frames per reset segment.

    Automatic reset data is not counted as collision-filtered unless another
    collision happens after that reset_counter baseline.
    """
    grouped = defaultdict(list)
    for row in rows:
        reset_counter = row.get("reset_counter")
        if reset_counter is None and row.get("odometry"):
            reset_counter = row["odometry"].get("reset_counter")
        grouped[(row.get("run_id", "unknown_run"), int(reset_counter or 0))].append(row)

    frames = []
    for segment_rows in grouped.values():
        segment_rows = sorted(segment_rows, key=lambda r: int(r["t_wall_ns"]))
        if not segment_rows:
            continue
        baseline = collision_count(segment_rows[0])
        for row in segment_rows:
            if collision_count(row) > baseline:
                frames.append(wrapper.frame_from_log_row(row))
    return frames


def build_coverage_samples(
    rows,
    wrapper,
    k,
    hz,
    stuck_position_epsilon_m=1e-4,
    stuck_min_duration_s=0.5,
    disable_stuck_position_filter=False,
):
    dt = 1.0 / hz
    segments, filter_stats = split_segments(
        rows,
        stuck_position_epsilon_m=stuck_position_epsilon_m,
        stuck_min_duration_s=stuck_min_duration_s,
        disable_stuck_position_filter=disable_stuck_position_filter,
        return_stats=True,
    )
    samples = []
    for segment_idx, segment in enumerate(segments):
        resampled = resample_segment(segment["rows"], dt=dt, max_gap_s=dt * 1.5)
        if len(resampled) < k + 2:
            continue
        frames = [wrapper.frame_from_log_row(row) for row in resampled]
        for i in range(k, len(frames) - 1):
            history = frames[i - k:i + 1]
            target = wrapper.target_from_frames(frames[i], frames[i + 1])
            samples.append({
                "segment_idx": segment_idx,
                "run_id": segment["run_id"],
                "frame": frames[i],
                "x": wrapper.model_input_from_history(history),
                "y": target.y,
                "model_error_norm": None,
            })
    return samples, {
        "segments": len(segments),
        "collision_filtered_rows": filter_stats["collision_filtered_rows"],
        "stuck_position_filtered_rows": filter_stats["stuck_position_filtered_rows"],
    }


def attach_model_errors(samples, checkpoint_path, device):
    if not samples:
        return
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from dynamics.models.mlp_dynamics import build_model

    checkpoint = torch.load(checkpoint_path, map_location=device)
    stats = checkpoint["normalization_stats"]
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = np.stack([sample["x"] for sample in samples]).astype(np.float32)
    y = np.stack([sample["y"] for sample in samples]).astype(np.float32)
    x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
    x_std = np.asarray(stats["x_std"], dtype=np.float32)
    y_mean = np.asarray(stats["y_mean"], dtype=np.float32)
    y_std = np.asarray(stats["y_std"], dtype=np.float32)
    if x.shape[1] != x_mean.shape[0]:
        raise RuntimeError(
            f"Checkpoint input dim {x_mean.shape[0]} does not match coverage sample dim {x.shape[1]}."
        )

    preds = []
    x_norm = (x - x_mean) / x_std
    with torch.no_grad():
        for start in range(0, len(x_norm), 4096):
            xb = torch.from_numpy(x_norm[start:start + 4096]).to(device)
            pred_n = model(xb).cpu().numpy()
            preds.append(pred_n)
    pred = np.concatenate(preds, axis=0) * y_std + y_mean
    errors = np.linalg.norm(pred.astype(np.float64) - y.astype(np.float64), axis=1)
    for sample, error in zip(samples, errors):
        sample["model_error_norm"] = float(error)


def make_views():
    return [
        {
            "name": "gravity_body_x_vs_gravity_body_y",
            "x_label": "gravity_body_x",
            "y_label": "gravity_body_y",
            "x_range": [-9.81, 9.81],
            "y_range": [-9.81, 9.81],
            "extract": lambda frame: (frame.x_body[6], frame.x_body[7]),
        },
        {
            "name": "v_body_x_vs_v_body_z",
            "x_label": "v_body_x",
            "y_label": "v_body_z",
            "x_range": [-40.0, 40.0],
            "y_range": [-40.0, 40.0],
            "extract": lambda frame: (frame.x_body[0], frame.x_body[2]),
        },
        {
            "name": "omega_body_x_vs_omega_body_y",
            "x_label": "omega_body_x",
            "y_label": "omega_body_y",
            "x_range": [-5.0, 5.0],
            "y_range": [-5.0, 5.0],
            "extract": lambda frame: (frame.x_body[3], frame.x_body[4]),
        },
        {
            "name": "roll_rate_cmd_vs_pitch_rate_cmd",
            "x_label": "roll_rate_cmd_norm",
            "y_label": "pitch_rate_cmd_norm",
            "x_range": [-1.0, 1.0],
            "y_range": [-1.0, 1.0],
            "extract": lambda frame: (frame.u[0], frame.u[1]),
        },
        {
            "name": "yaw_rate_cmd_vs_thrust_cmd",
            "x_label": "yaw_rate_cmd_norm",
            "y_label": "thrust_cmd",
            "x_range": [-1.0, 1.0],
            "y_range": [0.0, 1.0],
            "extract": lambda frame: (frame.u[2], frame.u[3]),
        },
        {
            "name": "v_body_x_vs_pitch_rate_cmd",
            "x_label": "v_body_x",
            "y_label": "pitch_rate_cmd_norm",
            "x_range": [-40.0, 40.0],
            "y_range": [-1.0, 1.0],
            "extract": lambda frame: (frame.x_body[0], frame.u[1]),
        },
        {
            "name": "roll_angle_vs_roll_rate_cmd",
            "x_label": "roll_rad",
            "y_label": "roll_rate_cmd_norm",
            "x_range": [-math.pi, math.pi],
            "y_range": [-1.0, 1.0],
            "extract": lambda frame: (frame.row["attitude"]["roll"], frame.u[0]),
        },
        {
            "name": "pitch_angle_vs_thrust_cmd",
            "x_label": "pitch_rad",
            "y_label": "thrust_cmd",
            "x_range": [-0.5 * math.pi, 0.5 * math.pi],
            "y_range": [0.0, 1.0],
            "extract": lambda frame: (frame.row["attitude"]["pitch"], frame.u[3]),
        },
    ]


def analyze_view(view, samples, collision_frames, bins, min_samples, max_reported_bins):
    x_edges = np.linspace(view["x_range"][0], view["x_range"][1], bins + 1)
    y_edges = np.linspace(view["y_range"][0], view["y_range"][1], bins + 1)
    valid_counts = np.zeros((bins, bins), dtype=np.int64)
    collision_counts = np.zeros((bins, bins), dtype=np.int64)
    action_sum = np.zeros((bins, bins, 4), dtype=np.float64)
    error_sum = np.zeros((bins, bins), dtype=np.float64)
    error_count = np.zeros((bins, bins), dtype=np.int64)
    target_values = [[[] for _ in range(bins)] for _ in range(bins)]

    for sample in samples:
        idx = bin_for_frame(view, sample["frame"], x_edges, y_edges)
        if idx is None:
            continue
        ix, iy = idx
        valid_counts[ix, iy] += 1
        action_sum[ix, iy] += sample["frame"].u
        target_values[ix][iy].append(sample["y"])
        if sample["model_error_norm"] is not None:
            error_sum[ix, iy] += float(sample["model_error_norm"])
            error_count[ix, iy] += 1

    for frame in collision_frames:
        idx = bin_for_frame(view, frame, x_edges, y_edges)
        if idx is None:
            continue
        collision_counts[idx] += 1

    sample_count = valid_counts + collision_counts
    mean_action = np.full((bins, bins, 4), np.nan, dtype=np.float64)
    target_std = np.full((bins, bins, 12), np.nan, dtype=np.float64)
    model_error_mean = np.full((bins, bins), np.nan, dtype=np.float64)

    for ix in range(bins):
        for iy in range(bins):
            if valid_counts[ix, iy] > 0:
                mean_action[ix, iy] = action_sum[ix, iy] / valid_counts[ix, iy]
                values = np.asarray(target_values[ix][iy], dtype=np.float64)
                target_std[ix, iy] = values.std(axis=0) if len(values) > 1 else np.zeros(12)
            if error_count[ix, iy] > 0:
                model_error_mean[ix, iy] = error_sum[ix, iy] / error_count[ix, iy]

    finite_errors = model_error_mean[np.isfinite(model_error_mean)]
    high_error_threshold = None
    high_error_mask = np.zeros((bins, bins), dtype=bool)
    if len(finite_errors):
        high_error_threshold = float(np.mean(finite_errors) + np.std(finite_errors))
        high_error_mask = (
            (valid_counts >= min_samples)
            & np.isfinite(model_error_mean)
            & (model_error_mean > high_error_threshold)
        )

    undercovered_mask = valid_counts < min_samples
    reported_bins = report_bins(
        valid_counts=valid_counts,
        collision_counts=collision_counts,
        model_error_mean=model_error_mean,
        mean_action=mean_action,
        target_std=target_std,
        undercovered_mask=undercovered_mask,
        high_error_mask=high_error_mask,
        x_edges=x_edges,
        y_edges=y_edges,
        max_reported_bins=max_reported_bins,
    )

    return {
        "x_label": view["x_label"],
        "y_label": view["y_label"],
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
        "sample_count": sample_count.tolist(),
        "valid_sample_count": valid_counts.tolist(),
        "collision_filtered_count": collision_counts.tolist(),
        "mean_action": nan_to_none(mean_action),
        "target_std": nan_to_none(target_std),
        "model_error_mean": nan_to_none(model_error_mean),
        "high_error_threshold": high_error_threshold,
        "undercovered_bins_count": int(np.sum(undercovered_mask)),
        "high_error_bins_count": int(np.sum(high_error_mask)),
        "reported_bins": reported_bins,
    }


def bin_for_frame(view, frame, x_edges, y_edges):
    x_value, y_value = view["extract"](frame)
    ix = bin_index(float(x_value), x_edges)
    iy = bin_index(float(y_value), y_edges)
    if ix is None or iy is None:
        return None
    return ix, iy


def bin_index(value, edges):
    if value < edges[0] or value > edges[-1]:
        return None
    if value == edges[-1]:
        return len(edges) - 2
    idx = int(np.searchsorted(edges, value, side="right") - 1)
    if idx < 0 or idx >= len(edges) - 1:
        return None
    return idx


def report_bins(
    valid_counts,
    collision_counts,
    model_error_mean,
    mean_action,
    target_std,
    undercovered_mask,
    high_error_mask,
    x_edges,
    y_edges,
    max_reported_bins,
):
    entries = []
    bins = valid_counts.shape[0]
    for ix in range(bins):
        for iy in range(bins):
            if not undercovered_mask[ix, iy] and not high_error_mask[ix, iy]:
                continue
            entry = {
                "ix": ix,
                "iy": iy,
                "x_range": [float(x_edges[ix]), float(x_edges[ix + 1])],
                "y_range": [float(y_edges[iy]), float(y_edges[iy + 1])],
                "valid_sample_count": int(valid_counts[ix, iy]),
                "collision_filtered_count": int(collision_counts[ix, iy]),
                "undercovered": bool(undercovered_mask[ix, iy]),
                "high_model_error": bool(high_error_mask[ix, iy]),
                "model_error_mean": safe_float(model_error_mean[ix, iy]),
                "mean_action": safe_list(mean_action[ix, iy]),
                "target_std": safe_list(target_std[ix, iy]),
            }
            entries.append(entry)

    entries.sort(key=lambda item: (
        not item["high_model_error"],
        item["valid_sample_count"],
        -(item["model_error_mean"] or 0.0),
    ))
    return entries[:max_reported_bins]


def safe_float(value):
    value = float(value)
    return None if not np.isfinite(value) else value


def safe_list(values):
    return [safe_float(value) for value in np.asarray(values).reshape(-1)]


def nan_to_none(array):
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 0:
        return safe_float(values)
    return convert_nan(values.tolist())


def convert_nan(value):
    if isinstance(value, list):
        return [convert_nan(item) for item in value]
    return safe_float(value)


if __name__ == "__main__":
    main()
