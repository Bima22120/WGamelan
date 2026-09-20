"""Base pitch detector abstraction and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PitchTrack:
    """Represents a continuous fundamental frequency (f0) track over time."""
    times: np.ndarray               # Timestamp for each frame in seconds
    frequencies_hz: np.ndarray      # Estimated f0 in Hz (0.0 or NaN for unvoiced)
    voiced_flags: np.ndarray        # Boolean array: True if frame is voiced
    voiced_probs: Optional[np.ndarray] = None  # Confidence score in [0.0, 1.0]

    def __len__(self) -> int:
        return len(self.times)


class BasePitchDetector(ABC):
    """Abstract interface for pitch estimation algorithms."""

    def __init__(self, fmin: float = 65.0, fmax: float = 2093.0, hop_length: int = 256):
        """
        Args:
            fmin: Minimum detectable frequency in Hz (default C2 ~ 65Hz).
            fmax: Maximum detectable frequency in Hz (default C7 ~ 2093Hz).
            hop_length: Hop length in samples between consecutive frames.
        """
        self.fmin = fmin
        self.fmax = fmax
        self.hop_length = hop_length

    @abstractmethod
    def detect(self, audio: np.ndarray, sr: int) -> PitchTrack:
        """Estimate pitch track across time for input audio buffer."""
        pass
