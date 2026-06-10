import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamics.models.dynamics_math import quat_to_rotmat, rotmat_log, rotvec_to_rotmat
from dynamics.models.flightsim_wrapper import (
    TARGET_SLICES,
    FlightSimBodyCentricWrapper,
    required_fields_present,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate processed or raw-log FlightSim dynamics rollouts.")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best_val_model.pt"))
    parser.add_argument("--data", default=str(ROOT / "logs" / "processed" / "test.npz"))
    parser.add_argument("--raw-input", nargs="+", default=None)
    parser.add_argument("--horizons-s", nargs="+", type=float, default=[1.0, 3.0, 5.0, 10.0])
    parser.add_argument("--max-rollouts-per-horizon", type=int, default=0)
    parser.add_argument("--rollout-stride", type=int, default=0)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--hz", type=float, default=None)
    parser.add_argument("--max-rate", type=float, default=None)
    parser.add_argument("--no-actuator", action="store_true")
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--position-integration", choices=["delta", "velocity", "blend"], default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.raw_input:
        results = evaluate_raw_rollout(args)
        print(json.dumps(results, indent=2), flush=True)
        return

    import torch
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
            stride = args.rollout_stride if args.rollout_stride > 0 else max(1, steps // 4)
            for start in range(0, len(indices) - steps, stride):
                idxs = indices[start:start + steps]
                err, is_diverged = rollout_error(model, stats, x, y, idxs, args.device)
                errors.append(err)
                diverged += int(is_diverged)
                if args.max_rollouts_per_horizon > 0 and len(errors) >= args.max_rollouts_per_horizon:
                    break
            if args.max_rollouts_per_horizon > 0 and len(errors) >= args.max_rollouts_per_horizon:
                break
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


def evaluate_raw_rollout(args):
    from dynamics.dataset.build_dataset import resample_segment, split_segments
    from dynamics.models.neural_sim import NeuralDroneDynamics

    probe = NeuralDroneDynamics(
        args.checkpoint,
        device=args.device,
        position_integration=args.position_integration,
    )
    k = int(args.k if args.k is not None else probe.k)
    hz = float(args.hz if args.hz is not None else probe.stats.get("hz", 60.0))
    max_rate = float(args.max_rate if args.max_rate is not None else probe.max_rate)
    use_actuator = False if args.no_actuator else bool(probe.use_actuator)
    use_imu = False if args.no_imu else bool(probe.use_imu)
    wrapper = FlightSimBodyCentricWrapper(
        max_rate=max_rate,
        use_actuator=use_actuator,
        use_imu=use_imu,
    )

    rows = load_raw_rows(args.raw_input)
    segment_frames = []
    dt = 1.0 / hz
    for segment in split_segments(rows):
        resampled = resample_segment(segment["rows"], dt=dt, max_gap_s=dt * 1.5)
        if len(resampled) >= k + 2:
            segment_frames.append([wrapper.frame_from_log_row(row) for row in resampled])

    sim = NeuralDroneDynamics(
        args.checkpoint,
        device=args.device,
        position_integration=args.position_integration,
    )
    results = {}
    for horizon_s in args.horizons_s:
        steps = max(1, int(round(horizon_s * hz)))
        pos_errors = []
        final_pos_errors = []
        vel_errors = []
        att_errors = []
        omega_errors = []
        diverged = 0
        rollouts = 0
        for frames in segment_frames:
            if len(frames) <= k + steps + 1:
                continue
            stride = args.rollout_stride if args.rollout_stride > 0 else max(1, steps // 4)
            for start in range(k, len(frames) - steps - 1, stride):
                sim.reset_with_history(
                    initial_state_from_frame(
                        frames[start],
                        use_actuator=use_actuator,
                        use_imu=use_imu,
                    ),
                    [(frame.x_body, frame.u) for frame in frames[start - k:start]],
                )
                rollout_pos_errors = []
                is_diverged = False
                for offset in range(steps):
                    action = frames[start + offset].u_raw
                    pred_state = sim.step(action)
                    ref = frames[start + offset + 1]
                    errors = state_errors(pred_state, ref)
                    rollout_pos_errors.append(errors["position_error_m"])
                    pos_errors.append(errors["position_error_m"])
                    vel_errors.append(errors["velocity_error_mps"])
                    att_errors.append(errors["attitude_error_rad"])
                    omega_errors.append(errors["omega_error_radps"])
                    if (
                        errors["position_error_m"] > 25.0
                        or not np.isfinite(errors["position_error_m"])
                    ):
                        is_diverged = True
                if rollout_pos_errors:
                    final_pos_errors.append(rollout_pos_errors[-1])
                    diverged += int(is_diverged)
                    rollouts += 1
                if args.max_rollouts_per_horizon > 0 and rollouts >= args.max_rollouts_per_horizon:
                    break
            if args.max_rollouts_per_horizon > 0 and rollouts >= args.max_rollouts_per_horizon:
                break

        results[f"{horizon_s:.1f}s"] = {
            "source": "raw_input_wrapper_rollout",
            "rollouts": int(rollouts),
            "position_rmse_m": rmse_values(pos_errors),
            "position_final_rmse_m": rmse_values(final_pos_errors),
            "velocity_rmse_mps": rmse_values(vel_errors),
            "attitude_rmse_rad": rmse_values(att_errors),
            "omega_rmse_radps": rmse_values(omega_errors),
            "diverged": int(diverged),
        }
    return results


def load_raw_rows(inputs):
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
    return rows


def initial_state_from_frame(frame, use_actuator, use_imu):
    state = {
        "p_ned": frame.p_ned,
        "q_wxyz": frame.q_wxyz,
        "v_ned": frame.v_ned,
        "omega": frame.omega_body,
    }
    cursor = 9
    if use_imu:
        state["imu_acc"] = frame.x_body[cursor:cursor + 3]
        cursor += 3
    if use_actuator and len(frame.x_body) >= cursor + 4:
        state["actuator"] = frame.x_body[cursor:cursor + 4]
    return state


def state_errors(pred_state, ref_frame):
    pred_p = np.asarray(pred_state["p_ned"], dtype=np.float64)
    pred_q = np.asarray(pred_state["q_wxyz"], dtype=np.float64)
    pred_v_ned = np.asarray(pred_state["v_ned"], dtype=np.float64)
    pred_omega = np.asarray(pred_state["omega"], dtype=np.float64)
    R_pred = quat_to_rotmat(pred_q)
    R_ref = quat_to_rotmat(ref_frame.q_wxyz)
    pred_v_body = R_pred.T @ pred_v_ned
    ref_v_body = ref_frame.x_body[0:3]
    attitude_error = np.linalg.norm(rotmat_log(R_pred.T @ R_ref))
    return {
        "position_error_m": float(np.linalg.norm(pred_p - ref_frame.p_ned)),
        "velocity_error_mps": float(np.linalg.norm(pred_v_body - ref_v_body)),
        "attitude_error_rad": float(attitude_error),
        "omega_error_radps": float(np.linalg.norm(pred_omega - ref_frame.omega_body)),
    }


def rmse_values(values):
    if not values:
        return None
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values ** 2)))


def rollout_error(model, stats, x, y, indices, device):
    import torch

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

        pred_p += pred[TARGET_SLICES["delta_p_body"]]
        true_p += true[TARGET_SLICES["delta_p_body"]]

        next_state = history[-1, :state_dim].copy()
        next_state[0:3] = pred[TARGET_SLICES["v_body_next"]]
        next_state[3:6] = pred[TARGET_SLICES["omega_body_next"]]
        if state_dim >= 9:
            delta_rot = rotvec_to_rotmat(pred[TARGET_SLICES["delta_rotvec_body"]])
            next_state[6:9] = delta_rot.T @ history[-1, 6:9]
        next_action = x[idx].reshape(k + 1, step_dim)[-1, state_dim:state_dim + action_dim]
        next_step = np.concatenate([next_state, next_action]).astype(np.float32)
        history = np.concatenate([history[1:], next_step[None, :]], axis=0)

    error = float(np.linalg.norm(pred_p - true_p))
    return error, error > 25.0 or not np.isfinite(error)


if __name__ == "__main__":
    main()
