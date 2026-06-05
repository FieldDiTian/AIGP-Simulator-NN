import json
from pathlib import Path

import numpy as np
import torch

from models.dynamics_math import (
    GRAVITY_NED,
    normalize_quat,
    quat_to_rotmat,
    rotmat_to_quat,
    rotvec_to_rotmat,
)
from models.mlp_dynamics import build_model


class NeuralDroneDynamics:
    def __init__(self, model_path, norm_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint["model_config"]
        self.stats = checkpoint.get("normalization_stats")
        if norm_path is not None:
            with open(norm_path, "r", encoding="utf-8") as handle:
                self.stats = json.load(handle)
        if self.stats is None:
            raise ValueError("Normalization stats are required.")

        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.k = int(self.stats["k"])
        self.max_rate = float(self.stats.get("max_rate", 1.0))
        self.use_actuator = bool(self.stats.get("use_actuator", True))
        self.x_mean = np.asarray(self.stats["x_mean"], dtype=np.float32)
        self.x_std = np.asarray(self.stats["x_std"], dtype=np.float32)
        self.y_mean = np.asarray(self.stats["y_mean"], dtype=np.float32)
        self.y_std = np.asarray(self.stats["y_std"], dtype=np.float32)
        self.state = None
        self.history = []

    def reset(self, initial_state):
        self.state = {
            "p_ned": np.asarray(initial_state["p_ned"], dtype=np.float64),
            "q_wxyz": normalize_quat(initial_state["q_wxyz"]),
            "v_ned": np.asarray(initial_state["v_ned"], dtype=np.float64),
            "omega": np.asarray(initial_state.get("omega", [0.0, 0.0, 0.0]), dtype=np.float64),
            "imu_acc": np.asarray(initial_state.get("imu_acc", [0.0, 0.0, 0.0]), dtype=np.float64),
            "actuator": np.asarray(initial_state.get("actuator", [0.0, 0.0, 0.0, 0.0]), dtype=np.float64),
        }
        zero_action = np.zeros(4, dtype=np.float32)
        feature = self._state_feature(self.state)
        self.history = [(feature, zero_action) for _ in range(self.k + 1)]
        return self.current_state()

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Call reset(initial_state) before step(action).")
        action_n = self._normalize_action(action)
        feature = self._state_feature(self.state)
        self.history.append((feature, action_n))
        self.history = self.history[-(self.k + 1):]

        model_input = np.concatenate([
            np.concatenate([s, a]).astype(np.float32)
            for s, a in self.history
        ])
        model_input_n = (model_input - self.x_mean) / self.x_std
        with torch.no_grad():
            pred_n = self.model(torch.from_numpy(model_input_n[None, :]).to(self.device)).cpu().numpy()[0]
        pred = pred_n * self.y_std + self.y_mean
        self._integrate_prediction(pred)
        return self.current_state()

    def current_state(self):
        return {
            "p_ned": self.state["p_ned"].copy(),
            "q_wxyz": self.state["q_wxyz"].copy(),
            "v_ned": self.state["v_ned"].copy(),
            "omega": self.state["omega"].copy(),
            "imu_acc": self.state["imu_acc"].copy(),
            "actuator": self.state["actuator"].copy(),
        }

    def _state_feature(self, state):
        R = quat_to_rotmat(state["q_wxyz"])
        v_body = R.T @ state["v_ned"]
        gravity_body = R.T @ GRAVITY_NED
        parts = [
            v_body,
            state["omega"],
            gravity_body,
            state["imu_acc"],
        ]
        if self.use_actuator:
            parts.append(state["actuator"][:4])
        return np.concatenate(parts).astype(np.float32)

    def _normalize_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        return np.asarray([
            action[0] / self.max_rate,
            action[1] / self.max_rate,
            action[2] / self.max_rate,
            action[3],
        ], dtype=np.float32)

    def _integrate_prediction(self, pred):
        delta_p_body = pred[0:3].astype(np.float64)
        delta_rotvec_body = pred[3:6].astype(np.float64)
        v_body_next = pred[6:9].astype(np.float64)
        omega_next = pred[9:12].astype(np.float64)

        R = quat_to_rotmat(self.state["q_wxyz"])
        R_next = R @ rotvec_to_rotmat(delta_rotvec_body)
        self.state["p_ned"] = self.state["p_ned"] + R @ delta_p_body
        self.state["q_wxyz"] = rotmat_to_quat(R_next)
        self.state["v_ned"] = R_next @ v_body_next
        self.state["omega"] = omega_next
        # First-version MLP does not predict future IMU acceleration or actuator feedback.
        # Keep their last observed values so history dimensions remain consistent.


def load_neural_dynamics(model_path, norm_path=None, device=None):
    return NeuralDroneDynamics(Path(model_path), norm_path=norm_path, device=device)
