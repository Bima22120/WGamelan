"""Timing and beat alignment utilities."""

from typing import List
import numpy as np


class TimingGrid:
    """Manages musical meter, tempo (BPM), and beat subdivisions."""

    def __init__(self, bpm: float = 120.0, meter_numerator: int = 4, meter_denominator: int = 4):
        self.bpm = max(30.0, min(300.0, bpm))
        self.meter_numerator = meter_numerator
        self.meter_denominator = meter_denominator

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm

    def seconds_to_beats(self, seconds: float) -> float:
        return seconds / self.seconds_per_beat

    def beats_to_seconds(self, beats: float) -> float:
        return beats * self.seconds_per_beat

    def get_grid_times(self, total_seconds: float, subdivision: int = 4) -> np.ndarray:
        """Generate time markers for a given subdivision per beat (e.g. 4 for 16th notes)."""
        dt = self.seconds_per_beat / subdivision
        return np.arange(0.0, total_seconds + dt, dt)

    def snap_to_grid(self, time_seconds: float, subdivision: int = 4) -> float:
        """Snap a given timestamp to the nearest subdivision grid position."""
        dt = self.seconds_per_beat / subdivision
        return round(time_seconds / dt) * dt
