"""Onset detection: finding transient attack points in audio."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


@dataclass
class OnsetResult:
    onset_times: np.ndarray      # Timestamps in seconds
    onset_frames: np.ndarray     # Frame indices
    onset_strengths: np.ndarray  # Relative attack strength [0.0, 1.0]


class OnsetDetector:
    """Detects note strikes and percussive transients."""

    def __init__(self, hop_length: int = 512, delta: float = 0.07, wait: int = 4):
        self.hop_length = hop_length
        self.delta = delta
        self.wait = wait

    def detect(self, audio: np.ndarray, sr: int) -> OnsetResult:
        """Detect onset points and transient strengths."""
        if len(audio) == 0:
            return OnsetResult(np.array([]), np.array([]), np.array([]))

        if librosa is not None:
            try:
                onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=self.hop_length)
                onset_frames = librosa.onset.onset_detect(
                    onset_envelope=onset_env,
                    sr=sr,
                    hop_length=self.hop_length,
                    delta=self.delta,
                    wait=self.wait,
                )
                onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)

                # Normalize strengths
                if len(onset_frames) > 0 and len(onset_env) > 0:
                    max_env = np.max(onset_env) + 1e-6
                    strengths = np.clip(onset_env[np.minimum(onset_frames, len(onset_env) - 1)] / max_env, 0.1, 1.0)
                else:
                    strengths = np.array([])

                return OnsetResult(
                    onset_times=onset_times,
                    onset_frames=onset_frames,
                    onset_strengths=strengths,
                )
            except Exception:
                pass

        return self._detect_energy_flux(audio, sr)

    def _detect_energy_flux(self, audio: np.ndarray, sr: int) -> OnsetResult:
        """Fallback spectral flux / energy difference onset detector."""
        frame_len = 1024
        hop = self.hop_length
        num_frames = max(1, (len(audio) - frame_len) // hop + 1)

        energies = np.zeros(num_frames, dtype=np.float32)
        for i in range(num_frames):
            frame = audio[i * hop:i * hop + frame_len]
            energies[i] = np.sqrt(np.mean(frame ** 2))

        # First order difference (half-wave rectified)
        diff = np.diff(energies, prepend=energies[0])
        diff = np.maximum(0, diff)

        # Simple peak picking
        peaks = []
        thresh = np.mean(diff) + self.delta * (np.max(diff) - np.mean(diff) if np.max(diff) > 0 else 1.0)
        last_peak = -self.wait

        for i in range(1, len(diff) - 1):
            if diff[i] > thresh and diff[i] > diff[i - 1] and diff[i] >= diff[i + 1]:
                if i - last_peak >= self.wait:
                    peaks.append(i)
                    last_peak = i

        peak_frames = np.array(peaks, dtype=int)
        peak_times = peak_frames * hop / sr
        max_diff = np.max(diff) + 1e-6 if len(diff) > 0 else 1.0
        strengths = np.clip(diff[peak_frames] / max_diff, 0.1, 1.0) if len(peak_frames) > 0 else np.array([])

        return OnsetResult(
            onset_times=peak_times,
            onset_frames=peak_frames,
            onset_strengths=strengths,
        )
