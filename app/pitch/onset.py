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
    """Detects note strikes and percussive transients while suppressing sustained guitar distortion, delay, and feedback."""

    def __init__(
        self,
        hop_length: int = 512,
        delta: float = 0.15,
        wait: int = 4,
        min_strength: float = 0.15,
    ):
        self.hop_length = hop_length
        self.delta = delta
        self.wait = wait
        self.min_strength = min_strength

    def detect(
        self,
        audio: np.ndarray,
        sr: int,
        use_hpss: bool = True,
        percussive_audio: Optional[np.ndarray] = None,
    ) -> OnsetResult:
        """Detect transient attack points and strengths.

        Args:
            audio: Source input audio.
            sr: Sample rate.
            use_hpss: If True, extract percussive transients via HPSS before onset detection
                      to suppress sustained distortion, delay tails, and hum.
            percussive_audio: Optional pre-separated percussive component.
        """
        if len(audio) == 0:
            return OnsetResult(np.array([]), np.array([]), np.array([]))

        # Determine target audio for onset detection (percussive component)
        target_audio = audio
        is_percussive_channel = False
        if percussive_audio is not None and len(percussive_audio) > 0:
            target_audio = percussive_audio
            is_percussive_channel = True
        elif use_hpss:
            from app.audio.preprocessing import separate_hpss
            _, p_audio = separate_hpss(audio, sr)
            if np.std(p_audio) > 1e-5:
                target_audio = p_audio
                is_percussive_channel = True

        # Check relative energy of percussive component to original audio
        if is_percussive_channel and len(audio) > 0:
            audio_rms = float(np.sqrt(np.mean(audio ** 2)))
            target_rms = float(np.sqrt(np.mean(target_audio ** 2)))
            target_peak = float(np.max(np.abs(target_audio)))
            audio_peak = float(np.max(np.abs(audio)))

            rel_rms = target_rms / max(audio_rms, 1e-6)
            rel_peak = target_peak / max(audio_peak, 1e-6)

            # If percussive RMS energy ratio is negligible (< 8% RMS), signal is purely sustained harmonic (distortion, hum, feedback)
            if rel_rms < 0.08:
                return OnsetResult(np.array([], dtype=np.float32), np.array([], dtype=int), np.array([], dtype=np.float32))

        if librosa is not None:
            try:
                onset_env = librosa.onset.onset_strength(
                    y=target_audio, sr=sr, hop_length=self.hop_length
                )
                onset_frames = librosa.onset.onset_detect(
                    onset_envelope=onset_env,
                    sr=sr,
                    hop_length=self.hop_length,
                    delta=self.delta,
                    wait=self.wait,
                )
                onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)

                # Normalize strengths and filter low-energy delay/distortion artifacts
                if len(onset_frames) > 0 and len(onset_env) > 0:
                    max_env = np.max(onset_env) + 1e-6
                    raw_strengths = onset_env[np.minimum(onset_frames, len(onset_env) - 1)] / max_env
                    
                    # Filter out weak spurious onsets below min_strength threshold
                    valid_mask = raw_strengths >= self.min_strength
                    if np.any(valid_mask):
                        onset_times = onset_times[valid_mask]
                        onset_frames = onset_frames[valid_mask]
                        strengths = np.clip(raw_strengths[valid_mask], 0.1, 1.0)
                    else:
                        onset_times = np.array([])
                        onset_frames = np.array([])
                        strengths = np.array([])
                else:
                    strengths = np.array([])

                return OnsetResult(
                    onset_times=onset_times,
                    onset_frames=onset_frames,
                    onset_strengths=strengths,
                )
            except Exception:
                pass

        return self._detect_spectral_flux(target_audio, sr)

    def _detect_spectral_flux(self, audio: np.ndarray, sr: int) -> OnsetResult:
        """Fallback spectral flux onset detector with STFT half-wave rectification."""
        frame_len = 1024
        hop = self.hop_length

        try:
            from scipy import signal
            _, _, Zxx = signal.stft(audio, fs=sr, nperseg=frame_len, noverlap=frame_len - hop)
            S = np.abs(Zxx)  # (freq_bins, num_frames)
            # Spectral flux: positive half-wave rectified spectral differences
            diff = np.diff(S, axis=1, prepend=S[:, :1])
            spectral_flux = np.sum(np.maximum(0, diff), axis=0)
        except Exception:
            num_frames = max(1, (len(audio) - frame_len) // hop + 1)
            energies = np.zeros(num_frames, dtype=np.float32)
            for i in range(num_frames):
                frame = audio[i * hop:i * hop + frame_len]
                energies[i] = np.sqrt(np.mean(frame ** 2))
            spectral_flux = np.maximum(0, np.diff(energies, prepend=energies[0]))

        # Adaptive peak picking
        peaks = []
        max_sf = np.max(spectral_flux) if len(spectral_flux) > 0 else 1.0
        mean_sf = np.mean(spectral_flux) if len(spectral_flux) > 0 else 0.0
        thresh = mean_sf + self.delta * (max_sf - mean_sf if max_sf > 0 else 1.0)
        last_peak = -self.wait

        for i in range(1, len(spectral_flux) - 1):
            if spectral_flux[i] > thresh and spectral_flux[i] > spectral_flux[i - 1] and spectral_flux[i] >= spectral_flux[i + 1]:
                if i - last_peak >= self.wait:
                    # Filter low-energy peaks
                    rel_strength = spectral_flux[i] / (max_sf + 1e-6)
                    if rel_strength >= self.min_strength:
                        peaks.append(i)
                        last_peak = i

        peak_frames = np.array(peaks, dtype=int)
        peak_times = peak_frames * hop / sr
        max_diff = np.max(spectral_flux) + 1e-6 if len(spectral_flux) > 0 else 1.0
        strengths = np.clip(spectral_flux[peak_frames] / max_diff, 0.1, 1.0) if len(peak_frames) > 0 else np.array([])

        return OnsetResult(
            onset_times=peak_times,
            onset_frames=peak_frames,
            onset_strengths=strengths,
        )

