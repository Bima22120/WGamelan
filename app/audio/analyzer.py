"""Audio feature analyzer for pitch/onset context and spectral properties."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


@dataclass
class AudioSummary:
    duration_sec: float
    sample_rate: int
    num_samples: int
    peak_amplitude: float
    rms_energy: float
    estimated_tempo_bpm: Optional[float] = None
    spectral_centroid_mean: Optional[float] = None


class AudioAnalyzer:
    """Extracts summary and spectral features from audio buffers."""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def summarize(self, audio: np.ndarray) -> AudioSummary:
        """Compute basic statistical summary of the audio."""
        num_samples = len(audio)
        duration = num_samples / self.sample_rate if self.sample_rate > 0 else 0.0
        peak = float(np.max(np.abs(audio))) if num_samples > 0 else 0.0
        rms = float(np.sqrt(np.mean(audio ** 2))) if num_samples > 0 else 0.0

        tempo = None
        centroid_mean = None

        if librosa is not None and num_samples > self.sample_rate:
            try:
                tempo_val, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
                tempo = float(np.atleast_1d(tempo_val)[0])
                centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
                centroid_mean = float(np.mean(centroid))
            except Exception:
                pass

        return AudioSummary(
            duration_sec=duration,
            sample_rate=self.sample_rate,
            num_samples=num_samples,
            peak_amplitude=peak,
            rms_energy=rms,
            estimated_tempo_bpm=tempo,
            spectral_centroid_mean=centroid_mean,
        )

    def compute_energy_envelope(self, audio: np.ndarray, frame_length: int = 1024, hop_length: int = 256) -> np.ndarray:
        """Compute frame-level RMS energy curve."""
        if len(audio) < frame_length:
            return np.array([float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0])
        num_frames = (len(audio) - frame_length) // hop_length + 1
        rms = np.zeros(num_frames, dtype=np.float32)
        for i in range(num_frames):
            start = i * hop_length
            rms[i] = np.sqrt(np.mean(audio[start:start + frame_length] ** 2))
        return rms
