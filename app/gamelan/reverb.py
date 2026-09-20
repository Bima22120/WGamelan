"""Acoustic Pendopo reverb: Schroeder architecture with comb and allpass filters.

Simulates the characteristic warm, spacious acoustic of a Javanese *Pendopo* —
a semi-open pavilion with carved teak wood pillars, marble/tiled floor, and a
high pitched roof — in which Gamelan ensembles traditionally perform.

Acoustic profile:
  - Pre-delay  : ~12 ms (distance between instruments and wooden pavilion pillars)
  - RT60       : ~1.2 s (warm natural decay, wood + open air absorption)
  - Character  : Warm low-mids, gentle high-frequency roll-off
  - Stereo     : Decorrelated late reflections for immersive stereo width
"""

from typing import Tuple
import numpy as np


def _comb_filter(signal: np.ndarray, delay_samples: int, feedback: float, damping: float) -> np.ndarray:
    """Feedback comb filter with one-pole lowpass damping."""
    n = len(signal)
    out = np.zeros(n, dtype=np.float32)
    buf = np.zeros(delay_samples, dtype=np.float32)
    filter_state = 0.0
    idx = 0

    for i in range(n):
        delayed = buf[idx]
        filter_state = delayed * (1.0 - damping) + filter_state * damping
        val = signal[i] + filter_state * feedback
        buf[idx] = val
        out[i] = delayed
        idx = (idx + 1) % delay_samples

    return out


def _allpass_filter(signal: np.ndarray, delay_samples: int, gain: float = 0.5) -> np.ndarray:
    """Schroeder allpass filter for echo diffusion."""
    n = len(signal)
    out = np.zeros(n, dtype=np.float32)
    buf = np.zeros(delay_samples, dtype=np.float32)
    idx = 0

    for i in range(n):
        delayed = buf[idx]
        v = signal[i] + delayed * gain
        out[i] = -signal[i] * gain + delayed
        buf[idx] = v
        idx = (idx + 1) % delay_samples

    return out


class PendopoReverb:
    """Acoustic reverb processor modeling a traditional Javanese Pendopo hall."""

    # Default delay lengths in samples at 22050 Hz (prime-like to avoid flutter)
    BASE_COMB_DELAYS = [1116, 1188, 1277, 1356]  # ~50-61 ms
    BASE_ALLPASS_DELAYS = [225, 341]             # ~10-15 ms

    def __init__(
        self,
        sr: int = 22050,
        rt60: float = 1.2,
        damping: float = 0.35,
        wet_level: float = 0.28,
        dry_level: float = 0.85,
        pre_delay_ms: float = 12.0,
        stereo_spread: float = 0.7,
    ):
        self.sr = sr
        self.rt60 = max(0.2, rt60)
        self.damping = float(np.clip(damping, 0.0, 0.9))
        self.wet_level = wet_level
        self.dry_level = dry_level
        self.pre_delay_samples = max(0, int((pre_delay_ms / 1000.0) * sr))
        self.stereo_spread = float(np.clip(stereo_spread, 0.0, 1.0))

        # Scale delay lengths according to sample rate relative to 22050
        scale = sr / 22050.0
        self.comb_delays_l = [int(d * scale) for d in self.BASE_COMB_DELAYS]
        # Offset delays slightly for right channel for rich stereo decorrelation
        offsets = [23, -19, 31, -27]
        self.comb_delays_r = [int((d + off) * scale) for d, off in zip(self.BASE_COMB_DELAYS, offsets)]

        self.allpass_delays = [int(d * scale) for d in self.BASE_ALLPASS_DELAYS]

        # Calculate feedback coefficients from RT60: gain = 10^(-3 * delay_sec / RT60)
        self.feedback_l = [
            float(10.0 ** (-3.0 * (d / self.sr) / self.rt60))
            for d in self.comb_delays_l
        ]
        self.feedback_r = [
            float(10.0 ** (-3.0 * (d / self.sr) / self.rt60))
            for d in self.comb_delays_r
        ]

    def process(self, audio: np.ndarray, stereo: bool = True) -> np.ndarray:
        """Apply Pendopo reverb to audio.

        Args:
            audio: 1D mono or 2D stereo numpy array.
            stereo: If True, outputs (N, 2) stereo array; else (N,) mono.

        Returns:
            Reverberated audio array (float32).
        """
        if len(audio) == 0:
            return np.zeros((0, 2) if stereo else 0, dtype=np.float32)

        # Convert input to 1D mono for reverb tank
        if audio.ndim == 2:
            dry_mono = np.mean(audio, axis=1).astype(np.float32)
            dry_orig = audio.astype(np.float32)
        else:
            dry_mono = np.asarray(audio, dtype=np.float32)
            dry_orig = dry_mono

        # Pre-delay
        if self.pre_delay_samples > 0:
            delayed_input = np.pad(dry_mono, (self.pre_delay_samples, 0), mode="constant")[:len(dry_mono)]
        else:
            delayed_input = dry_mono

        # Parallel comb filters for Left
        wet_l = np.zeros_like(delayed_input)
        for d, fb in zip(self.comb_delays_l, self.feedback_l):
            wet_l += _comb_filter(delayed_input, d, fb, self.damping)
        wet_l /= len(self.comb_delays_l)

        # Series allpass filters for Left
        for ap_d in self.allpass_delays:
            wet_l = _allpass_filter(wet_l, ap_d, gain=0.5)

        if not stereo:
            out = self.dry_level * dry_mono + self.wet_level * wet_l
            peak = np.max(np.abs(out))
            if peak > 0.98:
                out = (out / peak) * 0.98
            return out.astype(np.float32)

        # Parallel comb filters for Right (decorrelated)
        wet_r = np.zeros_like(delayed_input)
        for d, fb in zip(self.comb_delays_r, self.feedback_r):
            wet_r += _comb_filter(delayed_input, d, fb, self.damping)
        wet_r /= len(self.comb_delays_r)

        # Series allpass filters for Right
        for ap_d in self.allpass_delays:
            wet_r = _allpass_filter(wet_r, ap_d, gain=0.5)

        # Cross-mix slightly for natural stereo width
        spread = self.stereo_spread
        wet_left_channel = wet_l * (0.5 + 0.5 * spread) + wet_r * (0.5 - 0.5 * spread)
        wet_right_channel = wet_r * (0.5 + 0.5 * spread) + wet_l * (0.5 - 0.5 * spread)

        # Mix with dry
        if dry_orig.ndim == 2 and dry_orig.shape[1] == 2:
            out_l = self.dry_level * dry_orig[:, 0] + self.wet_level * wet_left_channel
            out_r = self.dry_level * dry_orig[:, 1] + self.wet_level * wet_right_channel
        else:
            out_l = self.dry_level * dry_mono + self.wet_level * wet_left_channel
            out_r = self.dry_level * dry_mono + self.wet_level * wet_right_channel

        out_stereo = np.column_stack((out_l, out_r)).astype(np.float32)

        # Limiter protection
        peak = np.max(np.abs(out_stereo))
        if peak > 0.98:
            out_stereo = (out_stereo / peak) * 0.98

        return out_stereo
