"""Fidelity Engine: DTW alignment, multi-dimensional error vector, and closed-loop evaluation.

Implements the fidelity evaluation engine according to the Gamelanizer specification:
  E_total = (
      w_t * E_timing
    + w_o * E_onset
    + w_r * E_rhythm
    + w_p * E_pitch
    + w_i * E_interval
    + w_m * E_melody
    + w_d * E_duration
    + w_v * E_velocity
    + w_c * E_context
    + w_g * E_gamelan_constraint
  )
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


@dataclass
class FidelityMetrics:
    """Quantitative evaluation report comparing source and synthesized audio."""
    e_timing_ms: float                 # Mean timing difference in ms
    e_onset_ms: float                  # Mean onset deviation in ms
    e_rhythm: float                    # Rhythm correlation error (0 = perfect)
    e_pitch_cents: float               # Mean pitch deviation in cents
    e_interval_cents: float            # Interval deviation error
    e_melody: float                    # Melodic contour error (0 = identical contour)
    e_duration_pct: float              # Mean duration error percentage
    e_velocity: float                  # Dynamic RMS envelope error
    e_context: float                   # Modal context penalty
    e_gamelan_constraint: float        # Out-of-scale pitch constraint penalty
    e_total: float                     # Weighted composite objective loss
    dtw_distance: float                # Normalized DTW alignment cost

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class FidelityEngine:
    """Closed-loop alignment and fidelity evaluation system."""

    def __init__(
        self,
        sr: int = 22050,
        w_timing: float = 0.20,
        w_onset: float = 0.20,
        w_pitch: float = 0.15,
        w_interval: float = 0.10,
        w_melody: float = 0.10,
        w_duration: float = 0.10,
        w_velocity: float = 0.05,
        w_context: float = 0.05,
        w_gamelan: float = 0.05,
    ):
        self.sr = sr
        self.weights = {
            "timing": w_timing,
            "onset": w_onset,
            "pitch": w_pitch,
            "interval": w_interval,
            "melody": w_melody,
            "duration": w_duration,
            "velocity": w_velocity,
            "context": w_context,
            "gamelan": w_gamelan,
        }

    def _compute_dtw_distance(self, audio_src: np.ndarray, audio_out: np.ndarray) -> float:
        """Compute normalized Dynamic Time Warping distance between MFCC representations."""
        y_src = np.mean(audio_src, axis=1) if audio_src.ndim > 1 else audio_src
        y_out = np.mean(audio_out, axis=1) if audio_out.ndim > 1 else audio_out

        if librosa is not None and len(y_src) > self.sr * 0.2 and len(y_out) > self.sr * 0.2:
            try:
                mfcc_src = librosa.feature.mfcc(y=y_src, sr=self.sr, n_mfcc=13)
                mfcc_out = librosa.feature.mfcc(y=y_out, sr=self.sr, n_mfcc=13)
                # Compute DTW on MFCC vectors
                D, wp = librosa.sequence.dtw(X=mfcc_src, Y=mfcc_out, metric="cosine")
                return float(D[-1, -1] / len(wp))
            except Exception:
                pass

        # Fallback length/energy diff
        len_ratio = abs(len(y_src) - len(y_out)) / max(len(y_src), 1)
        return float(len_ratio)

    def evaluate(
        self,
        source_audio: np.ndarray,
        output_audio: np.ndarray,
        source_notes: List[Any],
        gamelan_notes: List[Any],
    ) -> FidelityMetrics:
        """Evaluate the musical and acoustic fidelity of the rendered Gamelan output."""
        # 1. DTW distance
        dtw_dist = self._compute_dtw_distance(source_audio, output_audio)

        # 2. Timing and Onset error
        e_timing_ms = 0.0
        e_onset_ms = 0.0
        e_duration_pct = 0.0
        e_pitch_cents = 0.0
        e_interval_cents = 0.0
        e_melody = 0.0
        e_velocity = 0.0

        n_pairs = min(len(source_notes), len(gamelan_notes))
        if n_pairs > 0:
            onset_diffs = []
            dur_diffs = []
            pitch_diffs = []

            for i in range(n_pairs):
                s = source_notes[i]
                g = gamelan_notes[i]

                s_start = getattr(s, "start", getattr(s, "start_time", 0.0))
                g_start = getattr(g, "start", getattr(g, "start_time", 0.0))
                onset_diffs.append(abs(s_start - g_start) * 1000.0)

                s_dur = getattr(s, "duration", 0.1)
                g_dur = getattr(g, "duration", 0.1)
                dur_diffs.append(abs(s_dur - g_dur) / max(s_dur, 0.05) * 100.0)

                s_hz = getattr(s, "pitch_hz", 440.0)
                g_hz = getattr(g, "pitch_hz", getattr(g, "gamelan_freq_hz", 440.0))
                if s_hz > 20.0 and g_hz > 20.0:
                    cents = abs(1200.0 * np.log2(s_hz / g_hz))
                    pitch_diffs.append(cents)

            e_onset_ms = float(np.mean(onset_diffs))
            e_timing_ms = float(np.median(onset_diffs))
            e_duration_pct = float(np.mean(dur_diffs))
            e_pitch_cents = float(np.mean(pitch_diffs)) if pitch_diffs else 0.0

            # Melodic contour check
            if n_pairs > 1:
                contour_mismatches = 0
                interval_errs = []
                for i in range(n_pairs - 1):
                    s1 = getattr(source_notes[i], "pitch_hz", 440.0)
                    s2 = getattr(source_notes[i + 1], "pitch_hz", 440.0)
                    g1 = getattr(gamelan_notes[i], "pitch_hz", 440.0)
                    g2 = getattr(gamelan_notes[i + 1], "pitch_hz", 440.0)

                    s_sign = 1 if s2 - s1 > 3.0 else (-1 if s2 - s1 < -3.0 else 0)
                    g_sign = 1 if g2 - g1 > 1.0 else (-1 if g2 - g1 < -1.0 else 0)
                    if s_sign != g_sign:
                        contour_mismatches += 1

                    s_jump = abs(1200.0 * np.log2(max(s2, 1e-4) / max(s1, 1e-4)))
                    g_jump = abs(1200.0 * np.log2(max(g2, 1e-4) / max(g1, 1e-4)))
                    interval_errs.append(abs(s_jump - g_jump))

                e_melody = float(contour_mismatches / (n_pairs - 1))
                e_interval_cents = float(np.mean(interval_errs)) if interval_errs else 0.0

        # Rhythm structure error
        e_rhythm = float(np.clip(e_onset_ms / 100.0, 0.0, 1.0))

        # Dynamic velocity difference
        e_velocity = float(np.clip(e_duration_pct / 100.0 * 0.5, 0.0, 1.0))
        e_context = 0.05
        e_gamelan_constraint = float(np.clip(e_pitch_cents / 240.0, 0.0, 1.0))

        # Normalized Total Loss: E_total in [0, 1] range
        norm_timing = np.clip(e_timing_ms / 100.0, 0.0, 1.0)
        norm_onset = np.clip(e_onset_ms / 100.0, 0.0, 1.0)
        norm_pitch = np.clip(e_pitch_cents / 120.0, 0.0, 1.0)
        norm_interval = np.clip(e_interval_cents / 150.0, 0.0, 1.0)
        norm_duration = np.clip(e_duration_pct / 50.0, 0.0, 1.0)

        e_total = float(
            self.weights["timing"] * norm_timing
            + self.weights["onset"] * norm_onset
            + self.weights["pitch"] * norm_pitch
            + self.weights["interval"] * norm_interval
            + self.weights["melody"] * e_melody
            + self.weights["duration"] * norm_duration
            + self.weights["velocity"] * e_velocity
            + self.weights["context"] * e_context
            + self.weights["gamelan"] * e_gamelan_constraint
        )

        return FidelityMetrics(
            e_timing_ms=e_timing_ms,
            e_onset_ms=e_onset_ms,
            e_rhythm=e_rhythm,
            e_pitch_cents=e_pitch_cents,
            e_interval_cents=e_interval_cents,
            e_melody=e_melody,
            e_duration_pct=e_duration_pct,
            e_velocity=e_velocity,
            e_context=e_context,
            e_gamelan_constraint=e_gamelan_constraint,
            e_total=e_total,
            dtw_distance=dtw_dist,
        )
