import argparse
import math
import random
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PY_EXAMPLE = ROOT / "PyAIPilotExample"
if str(PY_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(PY_EXAMPLE))

from flight_command import FlightCommand
from logger import JsonlLogger
from setup import setup_components
from controller import CONTROL_HZ, update_attitude_flight_control


SAFE_RATE = 0.8
THRUST_MIN = 0.35
THRUST_MAX = 0.75
THRUST_BASE = 0.55


PROFILE_METADATA = {
    "hover_perturb": {
        "category": "baseline_hover",
        "coverage_goal": "hover thrust response",
    },
    "forward_flight": {
        "category": "baseline_forward",
        "coverage_goal": "forward acceleration and cruise response",
    },
    "left_turn": {
        "category": "baseline_turn",
        "coverage_goal": "left turn coupled roll/yaw response",
    },
    "right_turn": {
        "category": "baseline_turn",
        "coverage_goal": "right turn coupled roll/yaw response",
    },
    "climb_descend": {
        "category": "baseline_vertical",
        "coverage_goal": "vertical thrust response",
    },
    "roll_step": {
        "category": "single_axis_step",
        "coverage_goal": "roll-rate step response",
    },
    "pitch_step": {
        "category": "single_axis_step",
        "coverage_goal": "pitch-rate step response",
    },
    "yaw_step": {
        "category": "single_axis_step",
        "coverage_goal": "yaw-rate step response",
    },
    "thrust_step": {
        "category": "single_axis_step",
        "coverage_goal": "thrust step response",
    },
    "roll_sine_sweep": {
        "category": "single_axis_sweep",
        "coverage_goal": "roll-rate frequency response",
    },
    "pitch_sine_sweep": {
        "category": "single_axis_sweep",
        "coverage_goal": "pitch-rate frequency response",
    },
    "yaw_sine_sweep": {
        "category": "single_axis_sweep",
        "coverage_goal": "yaw-rate frequency response",
    },
    "thrust_sine_sweep": {
        "category": "single_axis_sweep",
        "coverage_goal": "thrust frequency response",
    },
    "smooth_random": {
        "category": "random_combo",
        "coverage_goal": "smooth random combined controls",
    },
    "race_like": {
        "category": "race_like",
        "coverage_goal": "race-like mixed control distribution",
    },
    "high_roll_pitch_response": {
        "category": "targeted_attitude",
        "coverage_goal": "large roll/pitch attitude response",
    },
    "coupled_rpyt_combo": {
        "category": "targeted_coupled_control",
        "coverage_goal": "roll/pitch/yaw/thrust simultaneous combinations",
    },
    "high_speed_yaw_roll_correction": {
        "category": "targeted_high_speed",
        "coverage_goal": "high-speed yaw and roll correction",
    },
    "low_speed_hover_attitude_perturb": {
        "category": "targeted_low_speed",
        "coverage_goal": "low-speed or hover attitude perturbation",
    },
    "low_speed_feedback_perturb": {
        "category": "targeted_low_speed",
        "coverage_goal": "telemetry-braked low-speed attitude perturbation",
    },
    "boundary_approach_recover": {
        "category": "targeted_boundary",
        "coverage_goal": "near-boundary normal dynamics before collision",
    },
    "climb_descend_turn": {
        "category": "targeted_vertical_turn",
        "coverage_goal": "climb/descend while turning",
    },
    "high_omega_thrust_response": {
        "category": "targeted_high_omega",
        "coverage_goal": "high angular-rate thrust response",
    },
    "error_attitude_corner_response": {
        "category": "high_error_attitude",
        "coverage_goal": "high-error gravity_body corner bins",
    },
    "error_pitch_brake_velocity": {
        "category": "high_error_velocity_pitch",
        "coverage_goal": "high-error v_body_x vs pitch-rate bins",
    },
    "error_roll_angle_reversal": {
        "category": "high_error_roll_angle",
        "coverage_goal": "high-error roll angle vs roll-rate bins",
    },
    "error_yaw_thrust_grid": {
        "category": "high_error_yaw_thrust",
        "coverage_goal": "high-error yaw-rate vs thrust bins",
    },
    "error_vertical_velocity_mix": {
        "category": "high_error_vertical_velocity",
        "coverage_goal": "high-error vertical velocity and thrust/pitch bins",
    },
}


class IdentificationCommandSource:
    def __init__(self, profile, duration_s, seed=1, shared_data=None):
        self.profile = profile
        self.duration_s = duration_s
        self.start_s = time.monotonic()
        self.rng = random.Random(seed)
        self.smooth_u = [0.0, 0.0, 0.0, THRUST_BASE]
        self.shared_data = shared_data

    def read_command(self):
        elapsed = time.monotonic() - self.start_s
        if elapsed >= self.duration_s:
            return FlightCommand(0.0, 0.0, 0.0, 0.0, exit_requested=True)

        roll, pitch, yaw, thrust = self._profile_command(elapsed)
        return FlightCommand(
            roll_rate=_clip(roll, -SAFE_RATE, SAFE_RATE),
            pitch_rate=_clip(pitch, -SAFE_RATE, SAFE_RATE),
            yaw_rate=_clip(yaw, -SAFE_RATE, SAFE_RATE),
            thrust=_clip(thrust, THRUST_MIN, THRUST_MAX),
            exit_requested=False,
        )

    @staticmethod
    def close():
        pass

    def _profile_command(self, t):
        if self.profile == "hover_perturb":
            return 0.0, 0.0, 0.0, THRUST_BASE + 0.05 * math.sin(2.0 * math.pi * 0.5 * t)

        if self.profile == "forward_flight":
            pitch = -0.50 if t < 4.0 else -0.20
            thrust = 0.65
            return 0.0, pitch, 0.0, thrust

        if self.profile == "left_turn":
            roll = -0.35
            pitch = -0.20
            yaw = -0.55
            thrust = 0.63
            return roll, pitch, yaw, thrust

        if self.profile == "right_turn":
            roll = 0.35
            pitch = -0.20
            yaw = 0.55
            thrust = 0.63
            return roll, pitch, yaw, thrust

        if self.profile == "climb_descend":
            phase = (t % 4.0) / 4.0
            thrust = 0.72 if phase < 0.5 else 0.42
            return 0.0, 0.0, 0.0, thrust

        if self.profile.endswith("_step"):
            axis = self.profile.removesuffix("_step")
            value = step_wave(t, period_s=2.0, amplitude=0.45)
            return axis_command(axis, value, THRUST_BASE)

        if self.profile.endswith("_sine_sweep"):
            axis = self.profile.removesuffix("_sine_sweep")
            value = sine_sweep(t, self.duration_s, amplitude=0.45, f0=0.2, f1=2.0)
            return axis_command(axis, value, THRUST_BASE)

        if self.profile == "smooth_random":
            target = [
                self.rng.uniform(-0.6, 0.6),
                self.rng.uniform(-0.6, 0.6),
                self.rng.uniform(-0.6, 0.6),
                self.rng.uniform(0.40, 0.70),
            ]
            self.smooth_u = [
                0.9 * current + 0.1 * desired
                for current, desired in zip(self.smooth_u, target)
            ]
            return tuple(self.smooth_u)

        if self.profile == "race_like":
            roll = 0.25 * math.sin(2.0 * math.pi * 0.7 * t)
            pitch = -0.25 + 0.15 * math.sin(2.0 * math.pi * 0.35 * t)
            yaw = 0.35 * math.sin(2.0 * math.pi * 0.25 * t)
            thrust = THRUST_BASE + 0.08 * math.sin(2.0 * math.pi * 0.45 * t)
            return roll, pitch, yaw, thrust

        if self.profile == "high_roll_pitch_response":
            return high_roll_pitch_response(t)

        if self.profile == "coupled_rpyt_combo":
            return coupled_rpyt_combo(t, self.rng, self.smooth_u)

        if self.profile == "high_speed_yaw_roll_correction":
            return high_speed_yaw_roll_correction(t)

        if self.profile == "low_speed_hover_attitude_perturb":
            return low_speed_hover_attitude_perturb(t)

        if self.profile == "low_speed_feedback_perturb":
            return low_speed_feedback_perturb(t, self.current_speed_mps())

        if self.profile == "boundary_approach_recover":
            return boundary_approach_recover(t)

        if self.profile == "climb_descend_turn":
            return climb_descend_turn(t)

        if self.profile == "high_omega_thrust_response":
            return high_omega_thrust_response(t)

        if self.profile == "error_attitude_corner_response":
            return error_attitude_corner_response(t)

        if self.profile == "error_pitch_brake_velocity":
            return error_pitch_brake_velocity(t)

        if self.profile == "error_roll_angle_reversal":
            return error_roll_angle_reversal(t)

        if self.profile == "error_yaw_thrust_grid":
            return error_yaw_thrust_grid(t)

        if self.profile == "error_vertical_velocity_mix":
            return error_vertical_velocity_mix(t)

        raise ValueError(f"Unknown profile: {self.profile}")

    def current_speed_mps(self):
        if self.shared_data is None:
            return None
        lock = self.shared_data.get("_lock")
        if lock is None:
            telemetry = self.shared_data.get("telemetry", {})
        else:
            with lock:
                telemetry = dict(self.shared_data.get("telemetry", {}))
        local = telemetry.get("local_position_ned") or {}
        velocity = local.get("v_ned")
        if velocity is None:
            odom = telemetry.get("odometry") or {}
            velocity = odom.get("v")
        if velocity is None:
            return None
        return math.sqrt(sum(float(component) ** 2 for component in velocity))


def step_wave(t, period_s, amplitude):
    phase = (t % period_s) / period_s
    if phase < 0.25:
        return amplitude
    if phase < 0.5:
        return 0.0
    if phase < 0.75:
        return -amplitude
    return 0.0


def sine_sweep(t, duration_s, amplitude, f0, f1):
    duration_s = max(duration_s, 1e-6)
    k = (f1 - f0) / duration_s
    phase = 2.0 * math.pi * (f0 * t + 0.5 * k * t * t)
    return amplitude * math.sin(phase)


def axis_command(axis, value, base_thrust):
    if axis == "roll":
        return value, 0.0, 0.0, base_thrust
    if axis == "pitch":
        return 0.0, value, 0.0, base_thrust
    if axis == "yaw":
        return 0.0, 0.0, value, base_thrust
    if axis == "thrust":
        return 0.0, 0.0, 0.0, base_thrust + 0.12 * (value / max(abs(value), 1e-6))
    raise ValueError(f"Unknown axis: {axis}")


def high_roll_pitch_response(t):
    phase = (t % 8.0) / 8.0
    if phase < 0.18:
        return 0.70, 0.00, 0.05, 0.63
    if phase < 0.32:
        return 0.00, -0.62, 0.10, 0.65
    if phase < 0.48:
        return -0.42, 0.30, -0.18, 0.58 + 0.06 * math.sin(2.0 * math.pi * 2.0 * t)
    if phase < 0.64:
        return -0.70, 0.00, -0.05, 0.63
    if phase < 0.80:
        return 0.00, 0.62, -0.10, 0.57
    return 0.42, -0.30, 0.18, 0.58 + 0.06 * math.sin(2.0 * math.pi * 2.3 * t)


def coupled_rpyt_combo(t, rng, smooth_u):
    target = [
        0.55 * math.sin(2.0 * math.pi * 0.37 * t) + 0.20 * math.sin(2.0 * math.pi * 1.13 * t + 0.4),
        0.50 * math.sin(2.0 * math.pi * 0.29 * t + 1.1) + 0.20 * math.sin(2.0 * math.pi * 0.91 * t),
        0.50 * math.sin(2.0 * math.pi * 0.23 * t + 2.0) + 0.18 * math.sin(2.0 * math.pi * 1.37 * t),
        THRUST_BASE + 0.13 * math.sin(2.0 * math.pi * 0.19 * t + 0.8),
    ]
    if int(t * 2.0) != int(max(0.0, t - 1.0 / CONTROL_HZ) * 2.0):
        target[0] += rng.uniform(-0.18, 0.18)
        target[1] += rng.uniform(-0.18, 0.18)
        target[2] += rng.uniform(-0.18, 0.18)
        target[3] += rng.uniform(-0.04, 0.04)
    for i, desired in enumerate(target):
        smooth_u[i] = 0.85 * smooth_u[i] + 0.15 * desired
    return tuple(smooth_u)


def high_speed_yaw_roll_correction(t):
    pitch = -0.55 if (t % 10.0) < 3.0 else -0.34
    roll = 0.42 * math.sin(2.0 * math.pi * 0.75 * t)
    yaw = 0.44 * math.sin(2.0 * math.pi * 0.53 * t + 0.7)
    thrust = 0.68 + 0.04 * math.sin(2.0 * math.pi * 0.25 * t)
    return roll, pitch, yaw, thrust


def low_speed_hover_attitude_perturb(t):
    roll = 0.22 * math.sin(2.0 * math.pi * 0.80 * t)
    pitch = 0.22 * math.sin(2.0 * math.pi * 0.67 * t + 1.2)
    yaw = 0.16 * math.sin(2.0 * math.pi * 0.43 * t + 2.1)
    thrust = THRUST_BASE + 0.035 * math.sin(2.0 * math.pi * 0.55 * t)
    return roll, pitch, yaw, thrust


def low_speed_feedback_perturb(t, speed_mps):
    if speed_mps is None or speed_mps > 8.0:
        roll = 0.10 * math.sin(2.0 * math.pi * 0.40 * t)
        pitch = 0.58
        yaw = 0.12 * math.sin(2.0 * math.pi * 0.31 * t)
        thrust = 0.48
        return roll, pitch, yaw, thrust
    if speed_mps > 4.0:
        roll = 0.14 * math.sin(2.0 * math.pi * 0.55 * t)
        pitch = 0.35
        yaw = 0.14 * math.sin(2.0 * math.pi * 0.37 * t + 0.5)
        thrust = 0.52
        return roll, pitch, yaw, thrust
    return low_speed_hover_attitude_perturb(t)


def boundary_approach_recover(t):
    phase = (t % 18.0) / 18.0
    if phase < 0.28:
        return 0.10, -0.52, 0.10, 0.67
    if phase < 0.42:
        return -0.45, -0.25, -0.35, 0.64
    if phase < 0.56:
        return 0.45, -0.20, 0.35, 0.64
    if phase < 0.72:
        return 0.00, 0.45, 0.00, 0.55
    if phase < 0.86:
        return 0.00, 0.00, 0.00, 0.72
    return 0.00, 0.00, 0.00, 0.45


def climb_descend_turn(t):
    phase = (t % 8.0) / 8.0
    thrust = 0.72 if phase < 0.5 else 0.40
    roll = 0.42 * math.sin(2.0 * math.pi * 0.31 * t)
    yaw = 0.48 * math.sin(2.0 * math.pi * 0.31 * t + 0.7)
    pitch = -0.24 + 0.18 * math.sin(2.0 * math.pi * 0.22 * t)
    return roll, pitch, yaw, thrust


def high_omega_thrust_response(t):
    roll = 0.68 * math.sin(2.0 * math.pi * 1.35 * t)
    pitch = 0.50 * math.sin(2.0 * math.pi * 1.05 * t + 1.1)
    yaw = 0.62 * step_wave(t, period_s=1.2, amplitude=1.0)
    thrust = 0.56 + 0.16 * step_wave(t + 0.3, period_s=1.6, amplitude=1.0)
    return roll, pitch, yaw, thrust


def error_attitude_corner_response(t):
    phase = (t % 12.0) / 12.0
    if phase < 0.16:
        return 0.72, 0.28, 0.12, 0.60
    if phase < 0.32:
        return -0.72, 0.28, -0.12, 0.60
    if phase < 0.48:
        return 0.72, -0.28, -0.12, 0.64
    if phase < 0.64:
        return -0.72, -0.28, 0.12, 0.64
    if phase < 0.80:
        return 0.25 * math.sin(2.0 * math.pi * 1.1 * t), 0.62, 0.08, 0.50
    return 0.25 * math.sin(2.0 * math.pi * 1.1 * t), -0.62, -0.08, 0.68


def error_pitch_brake_velocity(t):
    phase = (t % 14.0) / 14.0
    roll = 0.18 * math.sin(2.0 * math.pi * 0.77 * t)
    yaw = 0.18 * math.sin(2.0 * math.pi * 0.41 * t + 0.6)
    if phase < 0.38:
        return roll, -0.62, yaw, 0.69
    if phase < 0.70:
        return -roll, 0.62, -yaw, 0.46
    if phase < 0.86:
        return 0.35 * math.sin(2.0 * math.pi * 0.55 * t), 0.42, yaw, 0.54
    return -0.35 * math.sin(2.0 * math.pi * 0.55 * t), -0.42, -yaw, 0.62


def error_roll_angle_reversal(t):
    phase = (t % 10.0) / 10.0
    if phase < 0.24:
        return 0.72, 0.02, 0.06, 0.63
    if phase < 0.48:
        return -0.72, -0.02, -0.06, 0.63
    if phase < 0.64:
        return 0.72, 0.55, 0.08, 0.50
    if phase < 0.80:
        return -0.72, -0.55, -0.08, 0.68
    return 0.72 * math.sin(2.0 * math.pi * 1.2 * t), 0.0, 0.0, 0.60


def error_yaw_thrust_grid(t):
    yaw_values = [-0.68, 0.0, 0.68, -0.52, 0.52]
    thrust_values = [0.40, 0.48, 0.56, 0.66, 0.72]
    cell = int((t % 25.0) // 1.0)
    yaw = yaw_values[cell % len(yaw_values)]
    thrust = thrust_values[(cell // len(yaw_values)) % len(thrust_values)]
    roll = 0.12 * math.sin(2.0 * math.pi * 0.33 * t)
    pitch = 0.12 * math.sin(2.0 * math.pi * 0.47 * t + 0.8)
    return roll, pitch, yaw, thrust


def error_vertical_velocity_mix(t):
    phase = (t % 12.0) / 12.0
    roll = 0.22 * math.sin(2.0 * math.pi * 0.49 * t)
    yaw = 0.20 * math.sin(2.0 * math.pi * 0.37 * t + 1.3)
    if phase < 0.25:
        return roll, -0.30, yaw, 0.74
    if phase < 0.50:
        return -roll, 0.30, -yaw, 0.38
    if phase < 0.75:
        return 0.40 * math.sin(2.0 * math.pi * 0.65 * t), 0.52, yaw, 0.48
    return -0.40 * math.sin(2.0 * math.pi * 0.65 * t), -0.52, -yaw, 0.68


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def main():
    parser = argparse.ArgumentParser(description="Collect FlightSim dynamics identification logs.")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILE_METADATA))
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=14550)
    parser.add_argument("--heartbeat-timeout-s", type=float, default=120.0)
    parser.add_argument("--telemetry-timeout-s", type=float, default=120.0)
    parser.add_argument("--skip-flight-response-probe", action="store_true")
    parser.add_argument("--probe-duration-s", type=float, default=1.5)
    parser.add_argument("--probe-min-displacement-m", type=float, default=0.05)
    parser.add_argument("--probe-min-speed-mps", type=float, default=0.08)
    parser.add_argument("--log-dir", default=str(ROOT / "logs" / "raw"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start-vision", action="store_true")
    args = parser.parse_args()

    shared_data = {"_lock": threading.RLock()}
    system_boot_ms = int(time.time() * 1000)
    command_source = IdentificationCommandSource(
        args.profile,
        args.duration_s,
        seed=args.seed,
        shared_data=shared_data,
    )
    profile_metadata = PROFILE_METADATA[args.profile]
    logger_metadata = {
        "profile": args.profile,
        "profile_category": profile_metadata["category"],
        "coverage_goal": profile_metadata["coverage_goal"],
        "duration_s": args.duration_s,
        "control_mode": "SET_ATTITUDE_TARGET",
        "rate_limit_rad_s": SAFE_RATE,
        "thrust_min": THRUST_MIN,
        "thrust_max": THRUST_MAX,
        "flight_response_probe": not args.skip_flight_response_probe,
    }

    components = setup_components(
        shared_data,
        system_boot_ms,
        args.host,
        args.port,
        command_source=command_source,
        logger=None,
        start_vision=args.start_vision,
        heartbeat_timeout_s=args.heartbeat_timeout_s,
    )

    controller = components["controller"]
    print("Waiting for simulator flight telemetry...", flush=True)
    if not wait_for_required_telemetry(shared_data, args.telemetry_timeout_s):
        controller.shutdown()
        for key in ("ts_loop", "mavlink_rx", "vision_rx"):
            component = components.get(key)
            if component is None:
                continue
            thread = component.get_thread_for_join()
            if thread is not None:
                thread.join(timeout=1.0)
        raise TimeoutError(
            "Timed out waiting for ODOMETRY + HIGHRES_IMU + ATTITUDE. "
            "FlightSim is connected but likely not inside the active flyable simulator UI."
        )

    print("Arming drone...", flush=True)
    controller.arm()
    if not args.skip_flight_response_probe:
        print("Checking that the simulator responds to rate/thrust commands...", flush=True)
        response = probe_flight_response(
            controller,
            shared_data,
            system_boot_ms,
            duration_s=args.probe_duration_s,
            min_displacement_m=args.probe_min_displacement_m,
            min_speed_mps=args.probe_min_speed_mps,
        )
        if not response["valid"]:
            controller.shutdown()
            for key in ("ts_loop", "mavlink_rx", "vision_rx"):
                component = components.get(key)
                if component is None:
                    continue
                thread = component.get_thread_for_join()
                if thread is not None:
                    thread.join(timeout=1.0)
            raise RuntimeError(
                "FlightSim telemetry did not respond to the preflight control probe. "
                "Do not collect: the UI is probably not in the flyable HUD or the vehicle is not accepting "
                "SET_ATTITUDE_TARGET commands. "
                f"probe_displacement_m={response['displacement_m']:.3f}, "
                f"probe_max_speed_mps={response['max_speed_mps']:.3f}"
            )
        print(
            "Flight response ready: "
            f"probe_displacement_m={response['displacement_m']:.3f}, "
            f"probe_max_speed_mps={response['max_speed_mps']:.3f}",
            flush=True,
        )

    logger = JsonlLogger(
        log_dir=args.log_dir,
        run_id=args.run_id,
        metadata=logger_metadata,
    )
    controller.logger = logger
    print(f"Writing raw log to {logger.path}", flush=True)

    try:
        running = True
        while running:
            running = controller.update()
    except KeyboardInterrupt:
        print("\nCollection interrupted.", flush=True)
    finally:
        controller.shutdown()
        for key in ("ts_loop", "mavlink_rx", "vision_rx"):
            component = components.get(key)
            if component is None:
                continue
            thread = component.get_thread_for_join()
            if thread is not None:
                thread.join(timeout=1.0)

    print(f"Collection complete: {logger.path}", flush=True)


def wait_for_required_telemetry(shared_data, timeout_s):
    deadline = time.time() + timeout_s
    required_all = ("odometry", "imu", "attitude")
    while time.time() < deadline:
        telemetry = snapshot_telemetry(shared_data)
        if all(telemetry.get(key) is not None for key in required_all):
            return True
        time.sleep(0.05)
    return False


def snapshot_telemetry(shared_data):
    lock = shared_data.get("_lock")
    if lock is None:
        telemetry = shared_data.get("telemetry", {})
        return dict(telemetry)
    with lock:
        return dict(shared_data.get("telemetry", {}))


def probe_flight_response(
        controller,
        shared_data,
        system_boot_ms,
        duration_s,
        min_displacement_m,
        min_speed_mps):
    start = snapshot_telemetry(shared_data)
    start_position = telemetry_position(start)
    max_speed_mps = 0.0

    deadline = time.time() + max(0.1, duration_s)
    probe_command = FlightCommand(
        roll_rate=0.0,
        pitch_rate=-0.25,
        yaw_rate=0.0,
        thrust=min(THRUST_MAX, THRUST_BASE + 0.12),
    )
    while time.time() < deadline:
        update_attitude_flight_control(controller.sim_conn, system_boot_ms, probe_command)
        telemetry = snapshot_telemetry(shared_data)
        velocity = telemetry_velocity(telemetry)
        if velocity is not None:
            max_speed_mps = max(max_speed_mps, vector_norm(velocity))
        time.sleep(1.0 / CONTROL_HZ)

    settle_command = FlightCommand(0.0, 0.0, 0.0, THRUST_BASE)
    for _ in range(max(1, int(0.3 * CONTROL_HZ))):
        update_attitude_flight_control(controller.sim_conn, system_boot_ms, settle_command)
        time.sleep(1.0 / CONTROL_HZ)

    end = snapshot_telemetry(shared_data)
    end_position = telemetry_position(end)
    displacement_m = 0.0
    if start_position is not None and end_position is not None:
        displacement_m = vector_norm([
            float(end_position[index]) - float(start_position[index])
            for index in range(3)
        ])
    end_velocity = telemetry_velocity(end)
    if end_velocity is not None:
        max_speed_mps = max(max_speed_mps, vector_norm(end_velocity))

    return {
        "valid": displacement_m >= min_displacement_m or max_speed_mps >= min_speed_mps,
        "displacement_m": displacement_m,
        "max_speed_mps": max_speed_mps,
        "min_displacement_m": min_displacement_m,
        "min_speed_mps": min_speed_mps,
    }


def telemetry_position(telemetry):
    odometry = telemetry.get("odometry") or {}
    position = odometry.get("p_ned")
    if position is not None:
        return position
    local_position = telemetry.get("local_position_ned") or {}
    return local_position.get("p_ned")


def telemetry_velocity(telemetry):
    local_position = telemetry.get("local_position_ned") or {}
    velocity = local_position.get("v_ned")
    if velocity is not None:
        return velocity
    odometry = telemetry.get("odometry") or {}
    return odometry.get("v")


def vector_norm(values):
    return math.sqrt(sum(float(value) * float(value) for value in values))


if __name__ == "__main__":
    main()
