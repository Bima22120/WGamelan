"""Probabilistic YIN (pYIN) and autocorrelation-based pitch detector."""

import numpy as np
from app.pitch.base import BasePitchDetector, PitchTrack

try:
    import librosa
except ImportError:
    librosa = None


class PYINDetector(BasePitchDetector):
    """pYIN pitch detector with pure numpy fallback."""

    def __init__(
        self,
        fmin: float = 65.0,
        fmax: float = 1500.0,
        hop_length: int = 256,
        frame_length: int = 2048,
        threshold: float = 0.1,
    ):
        super().__init__(fmin=fmin, fmax=fmax, hop_length=hop_length)
        self.frame_length = frame_length
        self.threshold = threshold

    def detect(self, audio: np.ndarray, sr: int) -> PitchTrack:
        if len(audio) < self.frame_length:
            # Pad audio if too short
            pad_width = self.frame_length - len(audio)
            audio = np.pad(audio, (0, pad_width), mode="constant")

        if librosa is not None:
            try:
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    sr=sr,
                    frame_length=self.frame_length,
                    hop_length=self.hop_length,
                    fill_na=0.0,
                )
                times = librosa.times_like(f0, sr=sr, hop_length=self.hop_length)
                return PitchTrack(
                    times=times,
                    frequencies_hz=np.nan_to_num(f0, nan=0.0),
                    voiced_flags=voiced_flag,
                    voiced_probs=voiced_probs,
                )
            except Exception:
                pass  # Fall through to fallback detector

        return self._detect_autocorr(audio, sr)

    def _detect_autocorr(self, audio: np.ndarray, sr: int) -> PitchTrack:
        """Autocorrelation fallback f0 estimation."""
        num_frames = max(1, (len(audio) - self.frame_length) // self.hop_length + 1)
        times = np.arange(num_frames) * self.hop_length / sr
        f0 = np.zeros(num_frames, dtype=np.float32)
        voiced_flags = np.zeros(num_frames, dtype=bool)
        voiced_probs = np.zeros(num_frames, dtype=np.float32)

        min_lag = int(sr / self.fmax)
        max_lag = int(sr / self.fmin)

        for i in range(num_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.frame_length]
            if len(frame) < self.frame_length:
                break

            energy = np.sum(frame ** 2)
            if energy < 1e-4:
                continue

            # Normalized autocorrelation
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2:]
            if corr[0] <= 0:
                continue
            norm_corr = corr / corr[0]

            search_range = norm_corr[min_lag:max_lag]
            if len(search_range) == 0:
                continue

            peak_idx = np.argmax(search_range) + min_lag
            peak_val = norm_corr[peak_idx]

            if peak_val > 0.4:  # Voiced threshold
                pitch_hz = sr / peak_idx
                if self.fmin <= pitch_hz <= self.fmax:
                    f0[i] = pitch_hz
                    voiced_flags[i] = True
                    voiced_probs[i] = min(1.0, float(peak_val))

        return PitchTrack(
            times=times,
            frequencies_hz=f0,
            voiced_flags=voiced_flags,
            voiced_probs=voiced_probs,
        )
