"""Rebuild the existing dynamics dataset format from standard ROS bag topics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from aigp_bag_tools.conversions import (
    find_repo_root,
    flu_to_frd,
    q_wxyz_to_rotmat,
    rel_time_ns,
    rpy_from_rotmat,
)
from aigp_bag_tools.rosbag_io import diagnostic_values, odom_msg_to_row_parts, read_bag_messages


def main():
    parser = argparse.ArgumentParser(description="Rebuild FlightSim dynamics .npz data from an AI-GP rosbag.")
    parser.add_argument("--bag", nargs="+", required=True, help="One or more bag directories.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=None, help="Repository root containing dynamics/. Defaults to cwd search.")
    parser.add_argument("--storage-id", default="sqlite3", choices=["sqlite3", "mcap"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--max-rate", type=float, default=1.0)
    parser.add_argument("--no-actuator", action="store_true")
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--stuck-position-epsilon-m", type=float, default=1e-4)
    parser.add_argument("--stuck-min-duration-s", type=float, default=0.5)
    parser.add_argument("--disable-stuck-position-filter", action="store_true")
    parser.add_argument("--compare-jsonl", nargs="*", default=None, help="Optional JSONL inputs for sample-count comparison.")
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root)
    if repo_root is None:
        raise RuntimeError("Could not find repository root with dynamics/. Run from repo root or pass --repo-root.")
    sys.path.insert(0, str(repo_root))

    from dynamics.dataset.build_dataset import build_samples, compute_stats, save_split, split_by_run, split_segments

    rows = []
    for bag in args.bag:
        rows.extend(load_rows_from_bag(Path(bag), storage_id=args.storage_id))

    if not rows:
        raise RuntimeError("No reconstructable rows found in bag. Use bags written with default --storage-time row.")

    segments, filter_stats = split_segments(
        rows,
        stuck_position_epsilon_m=args.stuck_position_epsilon_m,
        stuck_min_duration_s=args.stuck_min_duration_s,
        disable_stuck_position_filter=args.disable_stuck_position_filter,
        return_stats=True,
    )
    use_actuator = not args.no_actuator
    use_imu = not args.no_imu
    samples, dropped_windows = build_samples(
        segments,
        k=args.k,
        dt=1.0 / args.hz,
        max_rate=args.max_rate,
        use_actuator=use_actuator,
        use_imu=use_imu,
    )
    train, val, test = split_by_run(samples)

    state_dim = 9 + (3 if use_imu else 0) + (4 if use_actuator else 0)
    action_dim = 4
    input_dim = (args.k + 1) * (state_dim + action_dim)
    output_dim = 12
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_split(output_dir / "train.npz", train, input_dim, output_dim)
    save_split(output_dir / "val.npz", val, input_dim, output_dim)
    save_split(output_dir / "test.npz", test, input_dim, output_dim)

    stats = compute_stats(train, input_dim, output_dim)
    stats.update(
        {
            "k": args.k,
            "hz": args.hz,
            "dt": 1.0 / args.hz,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "use_actuator": use_actuator,
            "use_imu": use_imu,
            "max_rate": args.max_rate,
            "source": "rosbag",
        }
    )
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
        "source": "rosbag",
        "bags": [str(Path(bag)) for bag in args.bag],
    }
    if args.compare_jsonl:
        comparison = compare_jsonl_sample_count(
            args.compare_jsonl,
            k=args.k,
            hz=args.hz,
            max_rate=args.max_rate,
            use_actuator=use_actuator,
            use_imu=use_imu,
            repo_root=repo_root,
        )
        comparison["bag_samples"] = int(len(samples))
        comparison["sample_count_delta"] = int(len(samples) - comparison["jsonl_samples"])
        summary["jsonl_comparison"] = comparison
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def load_rows_from_bag(bag_dir: Path, storage_id="sqlite3") -> list[dict]:
    records = defaultdict(dict)
    for topic, msg, timestamp, _type_name in read_bag_messages(bag_dir, storage_id=storage_id):
        if topic == "/odom":
            records[timestamp]["odom"] = msg
        elif topic == "/imu/data":
            records[timestamp]["imu"] = msg
        elif topic == "/aigp/control/attitude_target":
            records[timestamp]["action"] = msg
        elif topic == "/aigp/actuator_output":
            records[timestamp]["actuator"] = msg
        elif topic == "/aigp/debug/sample_info":
            records[timestamp]["sample_info"] = msg
        elif topic == "/aigp/raw/attitude_rpy_frd":
            records[timestamp]["raw_attitude_rpy"] = msg
        elif topic == "/aigp/collision":
            records[timestamp]["collision"] = msg

    rows = []
    for timestamp in sorted(records):
        record = records[timestamp]
        if not all(key in record for key in ("odom", "imu", "action")):
            continue
        rows.append(record_to_row(record, timestamp, bag_dir.name))
    return rows


def record_to_row(record: dict, timestamp: int, default_run_id: str) -> dict:
    odom_parts = odom_msg_to_row_parts(record["odom"])
    info = diagnostic_values(record["sample_info"]) if "sample_info" in record else {}
    collision = diagnostic_values(record["collision"]) if "collision" in record else {}
    run_id = info.get("run_id") or default_run_id
    reset_counter = int(float(info.get("reset_counter") or 0))
    t_wall_ns = int(float(info.get("relative_t_ns") or timestamp))

    imu_msg = record["imu"]
    imu_acc_flu = np.asarray(
        [
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z,
        ],
        dtype=np.float64,
    )
    imu_gyro_flu = np.asarray(
        [
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z,
        ],
        dtype=np.float64,
    )
    imu_acc_frd = flu_to_frd(imu_acc_flu)
    imu_gyro_frd = flu_to_frd(imu_gyro_flu)

    action_msg = record["action"]
    action_rate = np.asarray(
        [
            action_msg.body_rate.x,
            action_msg.body_rate.y,
            action_msg.body_rate.z,
        ],
        dtype=np.float64,
    )

    actuator_values = [0.0, 0.0, 0.0, 0.0]
    if "actuator" in record:
        controls = list(record["actuator"].controls)
        actuator_values = (controls + [0.0, 0.0, 0.0, 0.0])[:4]

    if "raw_attitude_rpy" in record:
        raw_rpy = record["raw_attitude_rpy"].vector
        roll_pitch_yaw = [raw_rpy.x, raw_rpy.y, raw_rpy.z]
    else:
        roll_pitch_yaw = rpy_from_rotmat(q_wxyz_to_rotmat(odom_parts["q_wxyz"])).tolist()

    return {
        "sample_type": "dynamics",
        "run_id": run_id,
        "t_wall_ns": t_wall_ns,
        "t_boot_ms": _int_or_none(info.get("t_boot_ms")),
        "dt": _float_or_none(info.get("dt")),
        "action": {
            "control_mode": "SET_ATTITUDE_TARGET",
            "t_send_wall_ns": _int_or_default(info.get("action_t_send_wall_ns"), t_wall_ns),
            "time_boot_ms": _int_or_none(info.get("action_time_boot_ms")),
            "roll_rate_cmd": float(action_rate[0]),
            "pitch_rate_cmd": float(action_rate[1]),
            "yaw_rate_cmd": float(action_rate[2]),
            "thrust_cmd": float(action_msg.thrust),
        },
        "odometry": {
            "time_wall_ns": _int_or_default(info.get("odometry_time_wall_ns"), t_wall_ns),
            "time_usec": _int_or_none(info.get("odometry_time_usec")),
            "frame_id": _int_or_none(info.get("odometry_frame_id")),
            "child_frame_id": _int_or_none(info.get("odometry_child_frame_id")),
            "p_ned": odom_parts["p_ned"],
            "q_wxyz": odom_parts["q_wxyz"],
            "v": odom_parts["v_ned"],
            "omega": odom_parts["omega"],
            "reset_counter": reset_counter,
        },
        "local_position_ned": {
            "time_wall_ns": _int_or_default(info.get("local_position_time_wall_ns"), t_wall_ns),
            "time_boot_ms": _int_or_none(info.get("local_position_time_boot_ms")),
            "p_ned": odom_parts["p_ned"],
            "v_ned": odom_parts["v_ned"],
        },
        "imu": {
            "time_wall_ns": _int_or_default(info.get("imu_time_wall_ns"), t_wall_ns),
            "time_usec": _int_or_none(info.get("imu_time_usec")),
            "acc": [float(v) for v in imu_acc_frd],
            "gyro": [float(v) for v in imu_gyro_frd],
        },
        "attitude": {
            "time_wall_ns": _int_or_default(info.get("attitude_time_wall_ns"), t_wall_ns),
            "time_boot_ms": _int_or_none(info.get("attitude_time_boot_ms")),
            "roll": float(roll_pitch_yaw[0]),
            "pitch": float(roll_pitch_yaw[1]),
            "yaw": float(roll_pitch_yaw[2]),
            "rollspeed": float(imu_gyro_frd[0]),
            "pitchspeed": float(imu_gyro_frd[1]),
            "yawspeed": float(imu_gyro_frd[2]),
        },
        "actuator_output": {
            "time_wall_ns": _int_or_default(info.get("actuator_time_wall_ns"), t_wall_ns),
            "time_usec": _int_or_none(info.get("actuator_time_usec")),
            "actuator": [float(v) for v in actuator_values],
        },
        "collision": {
            "count": _int_or_default(collision.get("count"), 0),
            "last": _json_or_none(collision.get("last")),
        },
        "reset_counter": reset_counter,
    }


def compare_jsonl_sample_count(inputs, *, k, hz, max_rate, use_actuator, use_imu, repo_root: Path) -> dict:
    from aigp_bag_tools.conversions import expand_input_paths
    from dynamics.evaluation.eval_one_step import load_raw_eval_arrays

    paths = [str(path) for path in expand_input_paths(inputs)]
    x, y = load_raw_eval_arrays(paths, k, hz, max_rate, use_actuator, use_imu)
    return {
        "jsonl_inputs": paths,
        "jsonl_samples": int(len(x)),
        "jsonl_target_rows": int(len(y)),
    }


def _int_or_none(value):
    if value in (None, ""):
        return None
    return int(float(value))


def _int_or_default(value, default):
    if value in (None, ""):
        return int(default)
    return int(float(value))


def _float_or_none(value):
    if value in (None, ""):
        return None
    return float(value)


def _json_or_none(value):
    if value in (None, ""):
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


if __name__ == "__main__":
    main()
