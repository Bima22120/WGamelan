"""Acoustic amplitude envelope for struck metallophone keys."""

import numpy as np


class GamelanEnvelope:
    """Generates natural bronze / iron bar struck envelopes with damping."""

    def __init__(
        self,
        sr: int = 22050,
        attack_sec: float = 0.005,
        decay_sec: float = 1.4,
        damped_decay_sec: float = 0.08,
    ):
        self.sr = sr
        self.attack_sec = attack_sec
        self.decay_sec = decay_sec
        self.damped_decay_sec = damped_decay_sec

    def generate(self, duration_sec: float, damped: bool = False, damp_time_sec: float = 0.0) -> np.ndarray:
        """Create an amplitude envelope curve of length duration_sec."""
        num_samples = int(duration_sec * self.sr)
        if num_samples <= 0:
            return np.array([], dtype=np.float32)

        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        attack_samples = max(1, int(self.attack_sec * self.sr))

        env = np.zeros(num_samples, dtype=np.float32)

        # 1. Linear or concave attack
        if attack_samples > 0:
            env[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)

        # 2. Exponential natural decay
        decay_tau = self.decay_sec / 3.0  # Reach ~5% at decay_sec
        env[attack_samples:] = np.exp(-(t[attack_samples:] - self.attack_sec) / decay_tau)

        # 3. Apply damping (mathet) if active
        if damped and damp_time_sec < duration_sec:
            damp_idx = max(attack_samples, int(damp_time_sec * self.sr))
            if damp_idx < num_samples:
                damp_t = t[damp_idx:] - t[damp_idx]
                damp_tau = self.damped_decay_sec / 3.0
                damp_curve = np.exp(-damp_t / damp_tau)
                env[damp_idx:] *= damp_curve

        return np.clip(env, 0.0, 1.0).astype(np.float32)
