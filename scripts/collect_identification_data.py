import argparse
import math
import random
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY_EXAMPLE = ROOT / "PyAIPilotExample"
if str(PY_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(PY_EXAMPLE))

from flight_command import FlightCommand
from logger import JsonlLogger
from setup import setup_components


SAFE_RATE = 0.8
THRUST_MIN = 0.35
THRUST_MAX = 0.75
THRUST_BASE = 0.55


class IdentificationCommandSource:
    def __init__(self, profile, duration_s, seed=1):
        self.profile = profile
        self.duration_s = duration_s
        self.start_s = time.monotonic()
        self.rng = random.Random(seed)
        self.smooth_u = [0.0, 0.0, 0.0, THRUST_BASE]

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

        raise ValueError(f"Unknown profile: {self.profile}")


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


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def main():
    parser = argparse.ArgumentParser(description="Collect FlightSim dynamics identification logs.")
    parser.add_argument("--profile", required=True, choices=[
        "hover_perturb",
        "forward_flight",
        "left_turn",
        "right_turn",
        "climb_descend",
        "roll_step",
        "pitch_step",
        "yaw_step",
        "thrust_step",
        "roll_sine_sweep",
        "pitch_sine_sweep",
        "yaw_sine_sweep",
        "thrust_sine_sweep",
        "smooth_random",
        "race_like",
    ])
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=14550)
    parser.add_argument("--heartbeat-timeout-s", type=float, default=120.0)
    parser.add_argument("--telemetry-timeout-s", type=float, default=120.0)
    parser.add_argument("--log-dir", default=str(ROOT / "logs" / "raw"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start-vision", action="store_true")
    args = parser.parse_args()

    shared_data = {"_lock": threading.RLock()}
    system_boot_ms = int(time.time() * 1000)
    command_source = IdentificationCommandSource(args.profile, args.duration_s, seed=args.seed)
    logger = JsonlLogger(
        log_dir=args.log_dir,
        run_id=args.run_id,
        metadata={
            "profile": args.profile,
            "duration_s": args.duration_s,
            "control_mode": "SET_ATTITUDE_TARGET",
            "rate_limit_rad_s": SAFE_RATE,
            "thrust_min": THRUST_MIN,
            "thrust_max": THRUST_MAX,
        },
    )

    components = setup_components(
        shared_data,
        system_boot_ms,
        args.host,
        args.port,
        command_source=command_source,
        logger=logger,
        start_vision=args.start_vision,
        heartbeat_timeout_s=args.heartbeat_timeout_s,
    )

    controller = components["controller"]
    print(f"Writing raw log to {logger.path}", flush=True)
    print("Waiting for simulator telemetry...", flush=True)
    if not wait_for_telemetry(shared_data, args.telemetry_timeout_s):
        controller.shutdown()
        for key in ("ts_loop", "mavlink_rx", "vision_rx"):
            component = components.get(key)
            if component is None:
                continue
            thread = component.get_thread_for_join()
            if thread is not None:
                thread.join(timeout=1.0)
        raise TimeoutError(
            "Timed out waiting for ATTITUDE/ODOMETRY/HIGHRES_IMU/LOCAL_POSITION_NED. "
            "FlightSim is connected but likely not inside the active simulator UI."
        )

    print("Arming drone...", flush=True)
    controller.arm()

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


def wait_for_telemetry(shared_data, timeout_s):
    deadline = time.time() + timeout_s
    required_any = ("odometry", "local_position_ned", "imu", "attitude")
    while time.time() < deadline:
        lock = shared_data.get("_lock")
        if lock is None:
            telemetry = shared_data.get("telemetry", {})
        else:
            with lock:
                telemetry = dict(shared_data.get("telemetry", {}))
        if any(telemetry.get(key) is not None for key in required_any):
            return True
        time.sleep(0.05)
    return False


if __name__ == "__main__":
    main()
