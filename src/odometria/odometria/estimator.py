"""Causal longitudinal observer. Time in seconds, speed in m/s, distance in metres.

The GNSS reference is never read here. Absolute map position and heading are not
observable from the three scalar inputs; `distance` is relative along track.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Config:
    wheel_speed_scale: float = 1.0 / 3.6
    traction: float = 1.2
    braking: float = 1.6
    speed_scale: float = 18.0
    rolling: float = 0.025
    drag: float = 0.0005
    actuator_tau: float = 0.45
    innovation_floor: float = 0.8
    bogie_agreement: float = 0.65
    velocity_timeout: float = 0.25
    command_timeout: float = 0.5
    max_gap: float = 2.0
    max_speed: float = 30.0
    learning_rate: float = 0.04
    max_bias: float = 0.35

    def __post_init__(self):
        for field in self.__dataclass_fields__:
            if not math.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
                raise ValueError(f'{field} must be finite and positive')


@dataclass(frozen=True)
class Estimate:
    t: float
    distance: float
    velocity: float
    acceleration: float
    sigma_velocity: float
    sigma_distance: float
    status: str
    trusted_bogies: int


class Observer:
    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.time = None
        self.distance = 0.0
        self.velocity = 0.0
        self.actuator = 0.0
        self.bias = 0.0
        self.sigma_v = 0.35
        self.sigma_s = 0.5
        self.last_velocity = None
        self.last_good = None

    def _acceleration(self, speed: float) -> float:
        c = self.cfg
        u = self.actuator
        drive = c.traction * max(u, 0.0) / (1.0 + speed / c.speed_scale)
        brake = c.braking * min(u, 0.0)
        return drive + brake - c.rolling * math.tanh(speed / 0.3) - c.drag * speed * speed + self.bias

    def step(self, t: float, command: int | None, front: float | None,
             rear: float | None, *, command_age: float = 0.0,
             front_age: float = 0.0, rear_age: float = 0.0) -> Estimate:
        c = self.cfg
        if not math.isfinite(t) or t < 0 or (self.time is not None and t <= self.time):
            raise ValueError('timestamps must be finite, nonnegative, and strictly increasing')
        if command is not None and (not isinstance(command, int) or not -15 <= command <= 15):
            raise ValueError('controller position must be an integer between -15 and 15')
        ages = (command_age, front_age, rear_age)
        if any(not math.isfinite(a) or a < 0 for a in ages):
            raise ValueError('input ages must be finite and nonnegative')
        first = self.time is None
        dt = 0.0 if first else t - self.time
        gap = dt > c.max_gap
        if gap:
            # The travelled path during a blind interval is not observable.
            # Preserve continuity with a constant-speed extrapolation and
            # advertise large uncertainty; do not jump back to the origin.
            self.distance += self.velocity * dt
            self.sigma_s += c.max_speed * dt
            self.sigma_v = c.max_speed
            self.actuator = 0.0
            self.bias = 0.0
            self.last_good = None
            dt = 0.0
        self.time = t
        valid_command = command is not None and command_age <= c.command_timeout
        target = command / 15.0 if valid_command else 0.0
        previous_v = self.velocity
        if dt:
            self.actuator += (target - self.actuator) * (1.0 - math.exp(-dt / c.actuator_tau))
        acceleration = self._acceleration(self.velocity)
        predicted_v = min(c.max_speed, max(0.0, self.velocity + acceleration * dt))
        self.sigma_v = min(c.max_speed, math.hypot(self.sigma_v, 0.14 * math.sqrt(dt)))

        candidates = []
        for value, age in ((front, front_age), (rear, rear_age)):
            if value is not None and math.isfinite(value):
                speed = value * c.wheel_speed_scale
                if 0 <= speed <= c.max_speed * 1.5 and age <= c.velocity_timeout:
                    candidates.append(speed)
        disagreement = (len(candidates) == 2 and
                        abs(candidates[0] - candidates[1]) > c.bogie_agreement)
        if disagreement:
            # A single unmatched bogie has no independent witness. Trust the model.
            candidates = []
        measured = sum(candidates) / len(candidates) if candidates else None
        gate = max(c.innovation_floor, 3.0 * self.sigma_v + 0.2)
        trusted = measured is not None and (
            first or (gap and len(candidates) == 2) or abs(measured - predicted_v) <= gate)
        if trusted:
            # Shared wheel slip can fool both channels, so do not collapse uncertainty.
            gain = (1.0 if first or gap else min(0.65, max(0.12,
                    self.sigma_v**2 / (self.sigma_v**2 + 0.12**2))))
            self.velocity = predicted_v + gain * (measured - predicted_v)
            self.sigma_v = max(0.10, self.sigma_v * math.sqrt(1.0 - gain))
            if (self.last_good is not None and valid_command and len(candidates) == 2
                    and dt > 0 and abs(measured - predicted_v) < 0.3):
                old_t, old_v = self.last_good
                span = t - old_t
                if 0.4 <= span <= 1.5 and measured > 0.5:
                    observed_a = (measured - old_v) / span
                    if abs(observed_a) <= 3.5:
                        residual = max(-0.3, min(0.3, observed_a - acceleration))
                        self.bias = max(-c.max_bias, min(c.max_bias,
                                        self.bias + c.learning_rate * residual))
                    self.last_good = (t, measured)
            else:
                self.last_good = (t, measured)
        else:
            self.velocity = predicted_v
            self.last_good = None
        self.distance += 0.5 * (previous_v + self.velocity) * dt
        self.sigma_s += self.sigma_v * dt
        status = ('time_gap' if gap else
                  'stale_command' if not valid_command else
                  'bogie_disagreement' if disagreement else
                  'ok' if trusted else
                  'suspect_wheels' if measured is not None else
                  'missing_wheels')
        return Estimate(t, self.distance, self.velocity, acceleration,
                        self.sigma_v, self.sigma_s, status, len(candidates) if trusted else 0)
