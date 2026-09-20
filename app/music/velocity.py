"""Velocity estimation: mapping audio energy to MIDI dynamics [1, 127]."""

from typing import Literal
import numpy as np


class VelocityEstimator:
    """Calculates strike velocities from raw onset amplitude or RMS energy."""

    def __init__(
        self,
        min_velocity: int = 40,
        max_velocity: int = 127,
        curve: Literal["linear", "logarithmic", "exponential"] = "logarithmic",
    ):
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.curve = curve

    def estimate_from_amplitude(self, amplitude: float, peak_ref: float = 1.0) -> int:
        """Map amplitude ratio to [min_velocity, max_velocity]."""
        norm = np.clip(abs(amplitude) / (peak_ref + 1e-9), 0.0, 1.0)

        if self.curve == "logarithmic":
            # Perceptual loudness mapping
            scaled = np.log1p(norm * 9.0) / np.log(10.0)
        elif self.curve == "exponential":
            scaled = norm ** 2
        else:
            scaled = norm

        vel = round(self.min_velocity + scaled * (self.max_velocity - self.min_velocity))
        return int(np.clip(vel, self.min_velocity, self.max_velocity))

    def estimate_from_audio_window(self, audio_slice: np.ndarray) -> int:
        """Estimate velocity from a short segment of audio around an onset strike."""
        if len(audio_slice) == 0:
            return 80
        rms = np.sqrt(np.mean(audio_slice ** 2))
        return self.estimate_from_amplitude(rms * 2.5)
