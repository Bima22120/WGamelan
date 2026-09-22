"""Context-Aware Gamelan Sequence Mapper with Contour & Interval Optimization.

Replaces naive point-by-point nearest-neighbor mapping with global
sequence optimization using Dynamic Programming (Viterbi Search):
    G* = argmin_G E(S, G)

Preserves:
    - Pitch contour direction: sign(source_delta) == sign(gamelan_delta)
    - Relative musical intervals
    - Timing integrity: timing_mode = PRESERVE (default)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import math
import numpy as np

from app.music.source_score import NoteEvent
from app.gamelan.scale import GamelanScale
from app.gamelan.tuning import GamelanPitch


@dataclass
class GamelanNoteEvent:
    """Note event mapped to specific Gamelan pitch and articulation."""
    degree: str                         # Gamelan scale degree (e.g. '1', '2', '3', '5', '6')
    pitch_hz: float                     # Tuned Gamelan frequency in Hz
    start: float                        # Start time in seconds
    duration: float                     # Duration in seconds
    velocity: int = 100                 # Velocity [1, 127]
    instrument: str = "saron"           # Target instrument
    register: int = 0                   # Octave register (-1, 0, 1)
    articulation: str = "open"          # 'open', 'damped', 'staccato'
    cents_deviation: float = 0.0        # Deviation from source in cents
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def start_time(self) -> float:
        return self.start

    @property
    def end_time(self) -> float:
        return self.end

    @property
    def gamelan_freq_hz(self) -> float:
        return self.pitch_hz

    @property
    def gamelan_note(self) -> str:
        return self.degree


class ContextAwareGamelanMapper:
    """Sequence-level optimizer for mapping source melodies to Gamelan scales."""

    def __init__(
        self,
        scale: GamelanScale,
        timing_mode: str = "PRESERVE",
        w_pitch: float = 1.0,
        w_contour: float = 2.5,
        w_interval: float = 1.2,
        w_range: float = 1.5,
        candidates_per_note: int = 5,
    ):
        self.scale = scale
        self.timing_mode = timing_mode.upper()
        self.w_pitch = w_pitch
        self.w_contour = w_contour
        self.w_interval = w_interval
        self.w_range = w_range
        self.k_candidates = candidates_per_note

    def _get_note_candidates(self, source_hz: float) -> List[GamelanPitch]:
        """Retrieve the top-K closest Gamelan scale pitches for a given source Hz."""
        if source_hz <= 20.0 or not self.scale.pitches:
            return [self.scale.pitches[0]] if self.scale.pitches else []

        # Sort all pitches in the scale by cent distance
        def cent_distance(gp: GamelanPitch) -> float:
            return abs(1200.0 * math.log2(source_hz / gp.freq_hz))

        sorted_pitches = sorted(self.scale.pitches, key=cent_distance)
        return sorted_pitches[:self.k_candidates]

    def map_sequence(
        self,
        source_notes: List[NoteEvent],
        beat_grid: Optional[List[float]] = None,
        instrument_name: str = "saron",
        target_register: int = 0,
    ) -> List[GamelanNoteEvent]:
        """Perform Dynamic Programming (Viterbi) search to find the optimal Gamelan melody.

        Minimizes:
            E_total = w_pitch * E_pitch + w_contour * E_contour + w_interval * E_interval
        """
        if not source_notes:
            return []

        N = len(source_notes)
        # 1. Generate candidates for each source note
        candidate_matrix: List[List[GamelanPitch]] = []
        for sn in source_notes:
            cands = self._get_note_candidates(sn.pitch_hz)
            candidate_matrix.append(cands)

        # 2. Forward DP table
        # dp[i][k]: minimum accumulated cost to assign candidate k to note i
        dp = [np.full(len(candidate_matrix[i]), float("inf"), dtype=np.float64) for i in range(N)]
        parent = [np.zeros(len(candidate_matrix[i]), dtype=np.int32) for i in range(N)]

        # Base case (first note)
        for k, cand in enumerate(candidate_matrix[0]):
            cents_err = abs(1200.0 * math.log2(source_notes[0].pitch_hz / cand.freq_hz))
            range_err = abs(cand.octave - target_register) * 150.0
            dp[0][k] = self.w_pitch * (cents_err ** 1.5) + self.w_range * range_err

        # Recurrence relation
        for i in range(1, N):
            src_prev = source_notes[i - 1]
            src_curr = source_notes[i]
            src_delta_hz = src_curr.pitch_hz - src_prev.pitch_hz
            src_sign = 1 if src_delta_hz > 5.0 else (-1 if src_delta_hz < -5.0 else 0)
            src_cents_jump = 1200.0 * math.log2(src_curr.pitch_hz / max(src_prev.pitch_hz, 1e-5))

            for k_curr, cand_curr in enumerate(candidate_matrix[i]):
                # State cost for current candidate
                cents_err = abs(1200.0 * math.log2(src_curr.pitch_hz / cand_curr.freq_hz))
                range_err = abs(cand_curr.octave - target_register) * 150.0
                state_cost = self.w_pitch * (cents_err ** 1.5) + self.w_range * range_err

                best_prev_cost = float("inf")
                best_parent_idx = 0

                for k_prev, cand_prev in enumerate(candidate_matrix[i - 1]):
                    # Transition cost
                    gam_delta_hz = cand_curr.freq_hz - cand_prev.freq_hz
                    gam_sign = 1 if gam_delta_hz > 1.0 else (-1 if gam_delta_hz < -1.0 else 0)
                    gam_cents_jump = 1200.0 * math.log2(cand_curr.freq_hz / cand_prev.freq_hz)

                    # Contour penalty: heavy penalty if contour direction is inverted
                    contour_penalty = 0.0
                    if src_sign != 0 and gam_sign != 0 and src_sign != gam_sign:
                        contour_penalty = 600.0  # Big penalty for reversing melody direction
                    elif src_sign != gam_sign:
                        contour_penalty = 180.0  # Flattened or created artificial step

                    # Interval step distortion
                    interval_err = abs(src_cents_jump - gam_cents_jump)

                    trans_cost = self.w_contour * contour_penalty + self.w_interval * interval_err
                    total_trans = dp[i - 1][k_prev] + trans_cost

                    if total_trans < best_prev_cost:
                        best_prev_cost = total_trans
                        best_parent_idx = k_prev

                dp[i][k_curr] = best_prev_cost + state_cost
                parent[i][k_curr] = best_parent_idx

        # 3. Backtrack best sequence G*
        best_end_k = int(np.argmin(dp[N - 1]))
        chosen_candidates: List[GamelanPitch] = [candidate_matrix[N - 1][best_end_k]]
        curr_k = best_end_k

        for i in range(N - 1, 0, -1):
            curr_k = parent[i][curr_k]
            chosen_candidates.append(candidate_matrix[i - 1][curr_k])

        chosen_candidates.reverse()

        # 4. Construct GamelanNoteEvent sequence with timing mode
        gamelan_events: List[GamelanNoteEvent] = []
        for i, (src, cand) in enumerate(zip(source_notes, chosen_candidates)):
            cents_diff = 1200.0 * math.log2(src.pitch_hz / cand.freq_hz)

            start_t = src.start
            dur_t = src.duration

            # Timing constraint resolution
            if self.timing_mode == "HARD_QUANTIZE" and beat_grid:
                closest_beat = min(beat_grid, key=lambda b: abs(b - start_t))
                start_t = closest_beat
            elif self.timing_mode == "SOFT_QUANTIZE" and beat_grid:
                closest_beat = min(beat_grid, key=lambda b: abs(b - start_t))
                start_t = start_t + 0.35 * (closest_beat - start_t)

            gamelan_events.append(
                GamelanNoteEvent(
                    degree=cand.degree,
                    pitch_hz=cand.freq_hz,
                    start=start_t,
                    duration=dur_t,
                    velocity=src.velocity,
                    instrument=instrument_name,
                    register=cand.octave,
                    articulation="open",
                    cents_deviation=cents_diff,
                    metadata={
                        "solfege": cand.solfege,
                        "traditional_name": cand.traditional_name,
                        "source_pitch_hz": src.pitch_hz,
                        "phrase_id": src.phrase_id,
                        "original_duration": src.duration,
                    },
                )
            )

        return gamelan_events
