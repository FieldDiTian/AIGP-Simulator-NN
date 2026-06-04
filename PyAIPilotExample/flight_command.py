from dataclasses import dataclass


@dataclass(frozen=True)
class FlightCommand:
    roll_rate: float
    pitch_rate: float
    yaw_rate: float
    thrust: float
    exit_requested: bool = False
