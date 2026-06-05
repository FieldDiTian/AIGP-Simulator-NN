import copy
import json
import os
import time
import uuid


class JsonlLogger:
    def __init__(self, log_dir="logs/raw", run_id=None, metadata=None):
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        self.path = os.path.join(self.log_dir, self.run_id + ".jsonl")
        self.metadata = metadata or {}
        self._file = open(self.path, "a", encoding="utf-8")
        self._last_wall_ns = None

        if self.metadata:
            self._file.write(json.dumps({
                "sample_type": "metadata",
                "run_id": self.run_id,
                "t_wall_ns": time.time_ns(),
                "metadata": self.metadata,
            }, ensure_ascii=True) + "\n")
            self._file.flush()

    def write_sample(self, shared_data, action):
        snapshot = self._snapshot(shared_data)
        now_ns = int(action.get("t_send_wall_ns", time.time_ns()))
        dt = None
        if self._last_wall_ns is not None:
            dt = (now_ns - self._last_wall_ns) / 1e9
        self._last_wall_ns = now_ns

        telemetry = snapshot.get("telemetry", {})
        events = snapshot.get("events", {})
        odometry = telemetry.get("odometry")

        row = {
            "sample_type": "dynamics",
            "run_id": self.run_id,
            "t_wall_ns": now_ns,
            "t_boot_ms": action.get("time_boot_ms"),
            "dt": dt,
            "action": action,
            "odometry": odometry,
            "local_position_ned": telemetry.get("local_position_ned"),
            "imu": telemetry.get("imu"),
            "attitude": telemetry.get("attitude"),
            "actuator_output": telemetry.get("actuator_output"),
            "collision": {
                "count": int(events.get("collision_count", 0)),
                "last": events.get("last_collision"),
            },
            "reset_counter": odometry.get("reset_counter") if odometry else None,
        }

        self._file.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    @staticmethod
    def _snapshot(shared_data):
        lock = shared_data.get("_lock")
        if lock is None:
            return _json_safe_copy(shared_data)
        with lock:
            return _json_safe_copy(shared_data)


def _json_safe_copy(value):
    if isinstance(value, dict):
        return {
            str(k): _json_safe_copy(v)
            for k, v in value.items()
            if k != "_lock"
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_copy(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return copy.deepcopy(value)
    except Exception:
        return repr(value)
