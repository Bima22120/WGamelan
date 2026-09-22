"""Instrument Activity Analyzer — derives orchestration statistics from per-instrument events.

From the raw NoteEvent lists extracted by ReferenceEventExtractor, this module computes:
  - Density model: notes per beat per instrument
  - Activity model: fraction of beats where instrument is active
  - Register model: mean and std of pitch frequencies per instrument
  - Interaction graph: anti-correlation (interlocking) between instrument pairs
  - Dynamic model: activity curve across song sections

These statistics are assembled into an OrchestrationProfile by OrchestrationLearner.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.music.source_score import NoteEvent


# ---------------------------------------------------------------------------
# Role inference heuristics
# ---------------------------------------------------------------------------

_KNOWN_ROLES: Dict[str, str] = {
    "gong": "structural",
    "kenong": "structural",
    "kempul": "structural",
    "ketuk": "structural",
    "kendang": "rhythm",
    "saron": "skeleton",
    "demung": "skeleton",
    "slenthem": "skeleton",
    "bonang": "elaboration",
    "bonang barung": "elaboration",
    "bonang panerus": "elaboration",
    "peking": "elaboration",
    "gambang": "elaboration",
    "rebab": "elaboration",
    "gender": "elaboration",
}


def _infer_role(
    instrument: str,
    density: float,
    mean_hz: float,
    total_events: int,
    total_beats: float,
) -> str:
    """Heuristic role inference if instrument is not in known map."""
    known = _KNOWN_ROLES.get(instrument.lower())
    if known:
        return known

    if density < 0.15 or total_beats == 0:
        return "structural"
    if density > 1.8:
        return "elaboration"
    if mean_hz < 150.0:
        return "structural"
    return "skeleton"


# ---------------------------------------------------------------------------
# InstrumentActivityAnalyzer
# ---------------------------------------------------------------------------

class InstrumentActivityAnalyzer:
    """Derives orchestration statistics from per-instrument NoteEvent lists.

    Usage:
        analyzer = InstrumentActivityAnalyzer(tempo_bpm=120.0, total_duration_sec=64.0)
        stats = analyzer.analyze(events_per_instrument)
    """

    def __init__(
        self,
        tempo_bpm: float = 120.0,
        total_duration_sec: float = 60.0,
        n_sections: int = 10,
    ):
        self.bpm = max(40.0, tempo_bpm)
        self.total_duration = total_duration_sec
        self.n_sections = n_sections
        self.beat_duration_sec = 60.0 / self.bpm
        self.total_beats = self.total_duration / self.beat_duration_sec

    def analyze(
        self, events_per_instrument: Dict[str, List[NoteEvent]]
    ) -> "ActivityStats":
        """Compute all orchestration statistics from extracted events.

        Returns:
            ActivityStats: Contains all computed models ready for OrchestrationProfile.
        """
        density_model: Dict[str, float] = {}
        activity_model: Dict[str, float] = {}
        register_model: Dict[str, Tuple[float, float]] = {}
        instrument_roles: Dict[str, str] = {}
        interaction_graph: Dict[str, Dict[str, float]] = {}
        dynamic_model: List[float] = [0.0] * self.n_sections

        total_beats = max(1.0, self.total_beats)

        # --- Per-instrument statistics ---
        onset_series: Dict[str, np.ndarray] = {}  # onset times per instrument

        for instrument, notes in events_per_instrument.items():
            if not notes:
                density_model[instrument] = 0.0
                activity_model[instrument] = 0.0
                register_model[instrument] = (220.0, 50.0)
                instrument_roles[instrument] = _KNOWN_ROLES.get(instrument.lower(), "skeleton")
                continue

            onsets = np.array([n.start for n in notes], dtype=np.float64)
            onset_series[instrument] = onsets
            pitches = np.array([n.pitch_hz for n in notes if n.pitch_hz > 20.0], dtype=np.float64)

            # Density: notes / total_beats
            density = float(len(notes)) / total_beats
            density_model[instrument] = round(density, 4)

            # Activity: fraction of beats that have at least 1 note within ±half-beat
            n_beats_active = self._count_active_beats(onsets)
            activity = float(n_beats_active) / total_beats
            activity_model[instrument] = float(np.clip(activity, 0.0, 1.0))

            # Register
            mean_hz = float(np.mean(pitches)) if len(pitches) > 0 else 220.0
            std_hz = float(np.std(pitches)) if len(pitches) > 1 else 50.0
            register_model[instrument] = (round(mean_hz, 2), round(std_hz, 2))

            # Role inference
            instrument_roles[instrument] = _infer_role(
                instrument, density, mean_hz, len(notes), total_beats
            )

        # --- Dynamic model: per-section activity ---
        section_dur = self.total_duration / self.n_sections
        all_onsets_combined = np.sort(np.concatenate(
            [s for s in onset_series.values() if len(s) > 0]
        )) if onset_series else np.array([])

        for sec_idx in range(self.n_sections):
            t_lo = sec_idx * section_dur
            t_hi = t_lo + section_dur
            if len(all_onsets_combined) > 0:
                in_section = ((all_onsets_combined >= t_lo) & (all_onsets_combined < t_hi)).sum()
                beats_in_section = max(1.0, section_dur / self.beat_duration_sec)
                dynamic_model[sec_idx] = float(np.clip(in_section / beats_in_section / max(1, len(onset_series)), 0.0, 1.0))
            else:
                dynamic_model[sec_idx] = 0.5

        # Normalize dynamic model to [0.4, 1.0] range
        dyn = np.array(dynamic_model, dtype=np.float64)
        if dyn.max() > dyn.min():
            dyn = 0.4 + 0.6 * (dyn - dyn.min()) / (dyn.max() - dyn.min())
        else:
            dyn = np.full_like(dyn, 0.75)
        dynamic_model = [round(float(x), 3) for x in dyn]

        # --- Interaction graph: anti-correlation between pairs ---
        instrument_list = list(onset_series.keys())
        for inst_a in instrument_list:
            interaction_graph[inst_a] = {}
            for inst_b in instrument_list:
                if inst_a == inst_b:
                    continue
                corr = self._compute_anticorrelation(
                    onset_series[inst_a], onset_series[inst_b]
                )
                if corr > 0.1:
                    interaction_graph[inst_a][inst_b] = round(corr, 3)

        return ActivityStats(
            density_model=density_model,
            activity_model=activity_model,
            register_model=register_model,
            instrument_roles=instrument_roles,
            interaction_graph=interaction_graph,
            dynamic_model=dynamic_model,
        )

    def _count_active_beats(self, onsets: np.ndarray) -> int:
        """Count how many beats have at least one note onset within half-beat tolerance."""
        if len(onsets) == 0:
            return 0
        half_beat = self.beat_duration_sec * 0.5
        total_b = int(self.total_beats) + 1
        count = 0
        for b in range(total_b):
            t_beat = b * self.beat_duration_sec
            if np.any((onsets >= t_beat - half_beat) & (onsets < t_beat + half_beat)):
                count += 1
        return count

    def _compute_anticorrelation(
        self, onsets_a: np.ndarray, onsets_b: np.ndarray
    ) -> float:
        """Compute interlocking degree between two onset series.

        Returns a value in [0, 1] where 1 = perfect interlocking (no shared beats).
        """
        if len(onsets_a) == 0 or len(onsets_b) == 0:
            return 0.0

        half_beat = self.beat_duration_sec * 0.4
        shared = 0
        for t in onsets_a:
            if np.any(np.abs(onsets_b - t) < half_beat):
                shared += 1

        # Fraction of a's onsets that are NOT shared with b
        fraction_exclusive = 1.0 - (shared / len(onsets_a))
        return float(np.clip(fraction_exclusive * 0.8, 0.0, 1.0))


# ---------------------------------------------------------------------------
# ActivityStats return container
# ---------------------------------------------------------------------------

class ActivityStats:
    """Container for computed orchestration statistics."""

    def __init__(
        self,
        density_model: Dict[str, float],
        activity_model: Dict[str, float],
        register_model: Dict[str, Tuple[float, float]],
        instrument_roles: Dict[str, str],
        interaction_graph: Dict[str, Dict[str, float]],
        dynamic_model: List[float],
    ):
        self.density_model = density_model
        self.activity_model = activity_model
        self.register_model = register_model
        self.instrument_roles = instrument_roles
        self.interaction_graph = interaction_graph
        self.dynamic_model = dynamic_model
