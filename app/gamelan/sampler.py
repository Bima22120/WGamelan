"""Gamelan audio sampler and modal physical synthesis engine."""

from typing import Optional, Dict
import numpy as np
from app.gamelan.envelope import GamelanEnvelope
from app.gamelan.instrument import GamelanInstrument, get_instrument


class GamelanSampler:
    """Renders authentic metallophone sound via modal synthesis and sample playback."""

    def __init__(self, instrument: Optional[GamelanInstrument] = None, sr: int = 22050):
        self.instrument = instrument or get_instrument("saron")
        self.sr = sr
        self.envelope = GamelanEnvelope(
            sr=sr,
            decay_sec=self.instrument.decay_time_sec,
        )

    def synthesize_note(
        self,
        freq_hz: float,
        duration_sec: float,
        velocity: int = 100,
        damped: bool = False,
        damp_time_sec: float = 0.0,
    ) -> np.ndarray:
        """Synthesize a metallophone or gong strike using modal physical parameters."""
        if freq_hz <= 0:
            return np.zeros(int(duration_sec * self.sr), dtype=np.float32)

        is_gong = (self.instrument.instrument_type == "large_gong") or (freq_hz < 90.0)

        if is_gong:
            # Large suspended bronze gong partial ratios & slow decay
            partials = [1.0, 1.452, 2.118, 3.140]
            decay_rates = [1.0, 1.8, 2.8, 4.5]
            total_dur = max(duration_sec, 4.0)
            attack_sec = 0.030  # Soft cloth-wrapped mallet attack
        else:
            # Metallophone bar partial ratios (Euler-Bernoulli clamped-free / free-free bar modes)
            partials = [1.0, 2.756, 5.404, 8.933]
            decay_rates = [1.0, 2.5, 4.0, 6.5]
            total_dur = max(duration_sec, 0.3)
            attack_sec = 0.004

        vel_norm = np.clip(velocity / 127.0, 0.1, 1.0)
        weights = [
            1.0,
            0.45 * self.instrument.brightness * vel_norm,
            0.20 * self.instrument.brightness * (vel_norm ** 1.5),
            0.08 * self.instrument.brightness * (vel_norm ** 2.0),
        ]

        num_samples = int(total_dur * self.sr)
        t = np.linspace(0, total_dur, num_samples, endpoint=False)

        signal = np.zeros(num_samples, dtype=np.float32)
        base_decay = max(self.instrument.decay_time_sec, 4.5 if is_gong else 0.8)

        for p_ratio, d_rate, w in zip(partials, decay_rates, weights):
            mode_freq = freq_hz * p_ratio
            if mode_freq < self.sr / 2.0:
                mode_decay = base_decay / d_rate
                mode_env = np.exp(-t / (mode_decay / 3.0))
                phase = np.random.uniform(0, 2 * np.pi)
                signal += (w * mode_env * np.sin(2 * np.pi * mode_freq * t + phase)).astype(np.float32)

        # Apply characteristic acoustic beating (Ombak) for Gong Ageng
        if is_gong:
            ombak_rate_hz = 2.8  # Typical Javanese gong beating rate ~2.5 - 3.5 Hz
            ombak = 1.0 + 0.20 * np.sin(2 * np.pi * ombak_rate_hz * t).astype(np.float32)
            signal *= ombak

        # Apply mallet strike attack
        attack_len = max(1, int(attack_sec * self.sr))
        signal[:attack_len] *= np.linspace(0.0, 1.0, attack_len)

        # Apply damping if requested (with smooth natural release)
        if damped and damp_time_sec < total_dur:
            damp_idx = max(attack_len, int(damp_time_sec * self.sr))
            if damp_idx < num_samples:
                damp_t = t[damp_idx:] - t[damp_idx]
                # Softer release curve (0.10s) instead of harsh 0.03s cutoff
                signal[damp_idx:] *= np.exp(-damp_t / 0.10)

        # Scale by velocity
        return signal * vel_norm
