import time

import keyboard as key_input

from flight_command import FlightCommand


ROLL_RATE_COMMAND = 1.0
PITCH_RATE_COMMAND = 1.0
YAW_RATE_COMMAND = 1.0
THRUST_IDLE = 0.0
THRUST_ACTIVE = 0.6


class KeyboardControlConsole:
    def __init__(self):
        self.last_status_print_s = 0.0
        print(
            "Keyboard control: "
            "A/D roll CCW/CW | W/S pitch forward/back | "
            "Q/E yaw CCW/CW | Space thrust | Esc quit",
            flush=True
        )

    def read_command(self):
        # MAVLink body-rate signs: A/Q/W are negative, D/E/S are positive.
        roll_axis = self._axis("a", "d")
        pitch_axis = self._axis("w", "s")
        yaw_axis = self._axis("q", "e")
        thrust = THRUST_ACTIVE if key_input.is_pressed("space") else THRUST_IDLE

        command = FlightCommand(
            roll_rate=roll_axis * ROLL_RATE_COMMAND,
            pitch_rate=pitch_axis * PITCH_RATE_COMMAND,
            yaw_rate=yaw_axis * YAW_RATE_COMMAND,
            thrust=thrust,
            exit_requested=key_input.is_pressed("esc")
        )
        self.print_status(command)
        return command

    @staticmethod
    def _axis(negative_key, positive_key):
        negative = int(key_input.is_pressed(negative_key))
        positive = int(key_input.is_pressed(positive_key))
        return positive - negative

    def print_status(self, command):
        now_s = time.time()
        if now_s - self.last_status_print_s < 0.2:
            return

        self.last_status_print_s = now_s
        print(
            "\r"
            f"roll {command.roll_rate:+.2f} rad/s | "
            f"pitch {command.pitch_rate:+.2f} rad/s | "
            f"yaw {command.yaw_rate:+.2f} rad/s | "
            f"thrust {command.thrust:.2f}      ",
            end="",
            flush=True
        )

    @staticmethod
    def close():
        print("", flush=True)
