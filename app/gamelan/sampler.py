"""Gamelan audio sampler and modal physical synthesis engine."""

from typing import Optional, Dict
import numpy as np
from app.gamelan.envelope import GamelanEnvelope
from app.gamelan.instrument import GamelanInstrument, get_instrument
from app.gamelan.sample_manager import SampleManager


try:
    from scipy.signal import lfilter
except ImportError:
    lfilter = None


class GamelanSampler:
    """Renders authentic metallophone sound via advanced physical modeling and sample playback."""

    def __init__(
        self,
        instrument: Optional[GamelanInstrument] = None,
        sr: int = 22050,
        sample_manager: Optional[SampleManager] = None,
    ):
        self.instrument = instrument or get_instrument("saron")
        self.sr = sr
        self.sample_manager = sample_manager or SampleManager(sr=sr)
        self.envelope = GamelanEnvelope(
            sr=sr,
            decay_sec=self.instrument.decay_time_sec,
        )

    def _generate_mallet_transient(
        self,
        duration_sec: float,
        hardness: float,
        is_gong: bool,
    ) -> np.ndarray:
        """Generate physical mallet strike impact transient (tabuh click/thump)."""
        num_samples = int(duration_sec * self.sr)
        out = np.zeros(num_samples, dtype=np.float32)

        transient_ms = 35.0 if is_gong else (12.0 if hardness > 0.65 else 20.0)
        trans_len = max(1, min(num_samples, int((transient_ms / 1000.0) * self.sr)))
        t_trans = np.linspace(0, transient_ms / 1000.0, trans_len, endpoint=False)

        # White noise pulse shaped by fast exponential decay
        decay_rate = 100.0 if hardness > 0.65 else (50.0 if not is_gong else 30.0)
        noise = np.random.uniform(-1.0, 1.0, trans_len).astype(np.float32)
        env = np.exp(-decay_rate * t_trans).astype(np.float32)

        # Center frequency of mallet impact
        if is_gong:
            fc = 160.0  # Padded cloth mallet thud
        elif hardness > 0.65:
            fc = 3200.0  # Water buffalo horn mallet click (Saron/Peking)
        else:
            fc = 950.0   # Wood mallet knock (Demung/Bonang)

        # Modulate with impact resonance
        impact_tone = np.sin(2 * np.pi * fc * t_trans).astype(np.float32)
        transient = (0.6 * noise + 0.4 * impact_tone) * env
        out[:trans_len] = transient * hardness
        return out

    def _apply_bumbung_resonator(self, signal: np.ndarray, f0: float, q: float = 6.5) -> np.ndarray:
        """Simulate acoustic bamboo tube resonator cavity (Bumbung) underneath the bar."""
        if f0 >= self.sr / 2.5 or f0 <= 30.0:
            return signal

        n = len(signal)
        # 2nd-order bandpass resonator (Helmholtz/tube acoustic model)
        r = np.exp(-np.pi * (f0 / q) / self.sr)
        theta = 2.0 * np.pi * f0 / self.sr
        a1 = -2.0 * r * np.cos(theta)
        a2 = r * r
        gain = (1.0 - r) * 1.5

        if lfilter is not None:
            out = lfilter([gain], [1.0, a1, a2], signal).astype(np.float32)
        else:
            out = np.zeros(n, dtype=np.float32)
            y1 = 0.0
            y2 = 0.0
            for i in range(n):
                y = signal[i] * gain - a1 * y1 - a2 * y2
                y2 = y1
                y1 = y
                out[i] = y

        # Blend dry bar vibration with warm bamboo chamber resonance
        return 0.70 * signal + 0.30 * out

    def synthesize_note(
        self,
        freq_hz: float,
        duration_sec: float,
        velocity: int = 100,
        damped: bool = False,
        damp_time_sec: float = 0.0,
    ) -> np.ndarray:
        """Synthesize a bronze key strike using advanced physical modeling or sample lookup."""
        if freq_hz <= 0:
            return np.zeros(int(duration_sec * self.sr), dtype=np.float32)

        # 1. Check if real recorded WAV sample is available in SampleManager
        if self.sample_manager:
            sample_audio = self.sample_manager.get_sample_by_freq(freq_hz)
            if sample_audio is not None and len(sample_audio) > 0:
                vel_norm = np.clip(velocity / 127.0, 0.1, 1.0)
                target_len = int(max(duration_sec, 0.3) * self.sr)
                if len(sample_audio) >= target_len:
                    res = sample_audio[:target_len].copy()
                else:
                    res = np.pad(sample_audio, (0, target_len - len(sample_audio)), mode="constant")
                return (res * vel_norm).astype(np.float32)

        # 2. Advanced Physical Modeling Synthesis
        is_gong = (self.instrument.instrument_type == "large_gong") or (freq_hz < 90.0)
        is_kettle = (self.instrument.instrument_type == "gong_chime")

        if is_gong:
            # Large suspended bronze gong partial ratios & slow decay
            partials = [1.0, 1.452, 2.118, 3.140]
            decay_rates = [1.0, 1.8, 2.8, 4.5]
            weights = [1.0, 0.45, 0.22, 0.10]
            total_dur = max(duration_sec, 4.5)
            hardness = 0.30
            ombak_rate = 2.8
        elif is_kettle:
            # Bossed gong chimes (Bonang)
            partials = [1.0, 1.480, 2.160, 2.880, 3.750]
            decay_rates = [1.0, 1.6, 2.4, 3.6, 4.8]
            weights = [1.0, 0.45, 0.28, 0.15, 0.08]
            total_dur = max(duration_sec, 1.8)
            hardness = self.instrument.strike_hardness
            ombak_rate = 3.6
        else:
            # Bronze metallophone bar modes (Saron, Demung, Peking)
            # Fundamental + Golden ratio mode (1.618) + Euler-Bernoulli flexural modes
            partials = [1.0, 1.618, 2.756, 3.885, 5.404, 8.933]
            decay_rates = [1.0, 1.7, 2.5, 3.3, 4.2, 6.5]
            bright = self.instrument.brightness
            weights = [
                1.0,
                0.32 * bright,
                0.40 * bright,
                0.18 * bright,
                0.16 * bright,
                0.07 * bright,
            ]
            total_dur = max(duration_sec, 0.45)
            hardness = self.instrument.strike_hardness
            ombak_rate = 3.2

        vel_norm = np.clip(velocity / 127.0, 0.1, 1.0)
        num_samples = int(total_dur * self.sr)
        t = np.linspace(0, total_dur, num_samples, endpoint=False)

        # Dynamic strike pitch sag (bar tension stretching slightly sharp at impact)
        pitch_sag_cents = (18.0 if not is_gong else 8.0) * np.exp(-t / 0.025)
        inst_f0 = freq_hz * (2.0 ** (pitch_sag_cents / 1200.0))
        phase_acc = 2.0 * np.pi * np.cumsum(inst_f0) / self.sr

        signal = np.zeros(num_samples, dtype=np.float32)
        base_decay = max(self.instrument.decay_time_sec, 4.8 if is_gong else 0.9)

        # Modal partials synthesis
        for p_ratio, d_rate, w in zip(partials, decay_rates, weights):
            mode_freq = freq_hz * p_ratio
            if mode_freq < self.sr / 2.0:
                mode_decay = base_decay / d_rate
                mode_env = np.exp(-t / (mode_decay / 3.0)).astype(np.float32)
                # Apply non-linear phase modulation
                inst_phase = phase_acc * p_ratio + np.random.uniform(0, 2 * np.pi)
                signal += (w * mode_env * np.sin(inst_phase)).astype(np.float32)

        # Mallet impact transient (tabuh strike click/thump)
        transient = self._generate_mallet_transient(total_dur, hardness=hardness, is_gong=is_gong)
        signal += transient * (0.35 * vel_norm)

        # Acoustic Bamboo Chamber Resonator (Bumbung)
        if not is_gong:
            signal = self._apply_bumbung_resonator(signal, f0=freq_hz, q=6.5)

        # Acoustic Ombak (beating wave of Javanese tuning)
        ombak_depth = 0.22 if is_gong else (0.12 if is_kettle else 0.08)
        ombak = 1.0 + ombak_depth * np.sin(2 * np.pi * ombak_rate * t).astype(np.float32)
        signal *= ombak

        # Smooth attack envelope
        attack_sec = 0.030 if is_gong else 0.003
        attack_len = max(1, int(attack_sec * self.sr))
        signal[:attack_len] *= np.linspace(0.0, 1.0, attack_len, dtype=np.float32)

        # Damping behavior
        if damped and damp_time_sec < total_dur:
            damp_idx = max(attack_len, int(damp_time_sec * self.sr))
            if damp_idx < num_samples:
                damp_t = t[damp_idx:] - t[damp_idx]
                signal[damp_idx:] *= np.exp(-damp_t / 0.12)

        # Scale by velocity and return normalized float32
        peak = np.max(np.abs(signal))
        if peak > 0.0:
            signal = (signal / peak) * vel_norm

        return signal.astype(np.float32)
