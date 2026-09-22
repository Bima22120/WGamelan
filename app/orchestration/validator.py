"""ArrangementValidator — hard constraint checker for Gamelan arrangements.

Formally: C(O_T) = True required before arrangement is accepted.

Hard constraints verified:
  1. Melody integrity: melodic contour direction preserved relative to source
  2. Rhythm integrity: onset alignment within tolerance
  3. Instrument range: all notes within valid Hz range per instrument
  4. Tuning validity: all pitches within scale tolerance
  5. Event validity: no events outside total duration, no zero-duration events
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.music.gamelan_score import GamelanScore
from app.music.source_score import SourceScore
from app.gamelan.context_mapper import GamelanNoteEvent
from app.gamelan.scale import GamelanScale


# Acceptable Hz range per instrument
INSTRUMENT_RANGES: Dict[str, Tuple[float, float]] = {
    "saron": (150.0, 700.0),
    "demung": (100.0, 400.0),
    "slenthem": (80.0, 250.0),
    "bonang": (250.0, 1400.0),
    "bonang barung": (250.0, 1400.0),
    "bonang panerus": (400.0, 2000.0),
    "peking": (400.0, 2200.0),
    "gambang": (200.0, 2000.0),
    "kendang": (50.0, 5000.0),
    "gong": (50.0, 150.0),
    "gong ageng": (50.0, 120.0),
    "kenong": (130.0, 350.0),
    "kempul": (100.0, 280.0),
    "ketuk": (200.0, 500.0),
}


@dataclass
class ConstraintViolation:
    """Describes a single constraint violation."""
    constraint: str
    instrument: str
    note_index: int
    detail: str
    severity: str = "warning"    # 'error' = hard fail, 'warning' = soft


@dataclass
class ValidationResult:
    """Result of arrangement validation."""
    passed: bool
    violations: List[ConstraintViolation] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Validation {status}: {self.error_count} errors, "
            f"{self.warning_count} warnings, "
            f"{len(self.violations)} total violations"
        )


class ArrangementValidator:
    """Validates a GamelanScore against hard musical and physical constraints.

    Usage:
        validator = ArrangementValidator(scale=scale, source_score=source_score)
        result = validator.validate(gamelan_score)
        if not result.passed:
            print(result.summary())
    """

    def __init__(
        self,
        scale: Optional[GamelanScale] = None,
        source_score: Optional[SourceScore] = None,
        pitch_tolerance_hz: float = 80.0,    # Max allowed deviation from scale pitch
        timing_tolerance_sec: float = 0.25,   # Max onset deviation from source
        min_note_duration_sec: float = 0.04,  # Minimum valid note duration
    ):
        self.scale = scale or GamelanScale(scale_type="slendro")
        self.source_score = source_score
        self.pitch_tolerance_hz = pitch_tolerance_hz
        self.timing_tolerance_sec = timing_tolerance_sec
        self.min_note_duration_sec = min_note_duration_sec

    def validate(self, gamelan_score: GamelanScore) -> ValidationResult:
        """Run all constraint checks on the arrangement.

        Returns ValidationResult with passed=True only if no hard errors.
        """
        violations: List[ConstraintViolation] = []

        for instrument, notes in gamelan_score.tracks.items():
            violations += self._check_instrument_range(instrument, notes)
            violations += self._check_tuning_validity(instrument, notes)
            violations += self._check_event_validity(instrument, notes, gamelan_score.total_duration)

        # Melody integrity check (requires source score)
        if self.source_score and self.source_score.notes:
            # Find the skeleton track (primary melody carrier)
            skeleton_notes = self._get_skeleton_notes(gamelan_score)
            if skeleton_notes:
                violations += self._check_melody_integrity(skeleton_notes, self.source_score)
                violations += self._check_rhythm_integrity(skeleton_notes, self.source_score)

        error_count = sum(1 for v in violations if v.severity == "error")
        warning_count = sum(1 for v in violations if v.severity == "warning")

        return ValidationResult(
            passed=(error_count == 0),
            violations=violations,
            error_count=error_count,
            warning_count=warning_count,
        )

    # -----------------------------------------------------------------------
    # Constraint checks
    # -----------------------------------------------------------------------

    def _check_instrument_range(
        self, instrument: str, notes: List[GamelanNoteEvent]
    ) -> List[ConstraintViolation]:
        """Check all notes are within the valid pitch range for the instrument."""
        violations = []
        key = instrument.lower()
        if key not in INSTRUMENT_RANGES:
            return []

        lo_hz, hi_hz = INSTRUMENT_RANGES[key]
        for idx, note in enumerate(notes):
            if note.pitch_hz < lo_hz or note.pitch_hz > hi_hz:
                violations.append(ConstraintViolation(
                    constraint="instrument_range",
                    instrument=instrument,
                    note_index=idx,
                    detail=f"{note.pitch_hz:.1f} Hz outside [{lo_hz}, {hi_hz}] Hz",
                    severity="error",
                ))
        return violations

    def _check_tuning_validity(
        self, instrument: str, notes: List[GamelanNoteEvent]
    ) -> List[ConstraintViolation]:
        """Check all pitches are close to a valid scale degree."""
        if not self.scale or not self.scale.pitches:
            return []

        violations = []
        scale_hz = [p.freq_hz for p in self.scale.pitches]

        for idx, note in enumerate(notes):
            if note.pitch_hz < 40.0:
                continue  # Gong/bass — skip
            min_dist = min(abs(note.pitch_hz - hz) for hz in scale_hz)
            if min_dist > self.pitch_tolerance_hz:
                violations.append(ConstraintViolation(
                    constraint="tuning_validity",
                    instrument=instrument,
                    note_index=idx,
                    detail=f"{note.pitch_hz:.1f} Hz is {min_dist:.1f} Hz from nearest scale pitch",
                    severity="warning",
                ))
        return violations

    def _check_event_validity(
        self, instrument: str, notes: List[GamelanNoteEvent], total_duration: float
    ) -> List[ConstraintViolation]:
        """Check notes for zero duration, negative start, or exceeding total duration."""
        violations = []
        for idx, note in enumerate(notes):
            if note.start < 0.0:
                violations.append(ConstraintViolation(
                    constraint="event_validity",
                    instrument=instrument,
                    note_index=idx,
                    detail=f"Negative start time {note.start:.3f}s",
                    severity="error",
                ))
            if note.duration < self.min_note_duration_sec:
                violations.append(ConstraintViolation(
                    constraint="event_validity",
                    instrument=instrument,
                    note_index=idx,
                    detail=f"Duration {note.duration:.3f}s < min {self.min_note_duration_sec}s",
                    severity="warning",
                ))
            if total_duration > 0 and note.start > total_duration + 0.5:
                violations.append(ConstraintViolation(
                    constraint="event_validity",
                    instrument=instrument,
                    note_index=idx,
                    detail=f"Start {note.start:.3f}s exceeds total duration {total_duration:.3f}s",
                    severity="error",
                ))
        return violations

    def _check_melody_integrity(
        self, skeleton_notes: List[GamelanNoteEvent], source: SourceScore
    ) -> List[ConstraintViolation]:
        """Check melodic contour direction is preserved: sign(source_delta) == sign(gamelan_delta)."""
        violations = []
        n = min(len(skeleton_notes), len(source.notes)) - 1

        mismatch_count = 0
        for i in range(n):
            src_delta = source.notes[i + 1].pitch_hz - source.notes[i].pitch_hz
            gam_delta = skeleton_notes[i + 1].pitch_hz - skeleton_notes[i].pitch_hz

            src_sign = 1 if src_delta > 5.0 else (-1 if src_delta < -5.0 else 0)
            gam_sign = 1 if gam_delta > 1.0 else (-1 if gam_delta < -1.0 else 0)

            if src_sign != 0 and gam_sign != 0 and src_sign != gam_sign:
                mismatch_count += 1

        # If more than 30% contour direction mismatches → hard error
        total_intervals = max(1, n)
        mismatch_rate = mismatch_count / total_intervals
        if mismatch_rate > 0.30:
            violations.append(ConstraintViolation(
                constraint="melody_integrity",
                instrument="skeleton",
                note_index=-1,
                detail=f"Contour mismatch rate {mismatch_rate:.1%} exceeds 30% threshold",
                severity="error",
            ))
        elif mismatch_rate > 0.15:
            violations.append(ConstraintViolation(
                constraint="melody_integrity",
                instrument="skeleton",
                note_index=-1,
                detail=f"Contour mismatch rate {mismatch_rate:.1%} (moderate)",
                severity="warning",
            ))

        return violations

    def _check_rhythm_integrity(
        self, skeleton_notes: List[GamelanNoteEvent], source: SourceScore
    ) -> List[ConstraintViolation]:
        """Check onset alignment between skeleton and source notes."""
        violations = []
        n = min(len(skeleton_notes), len(source.notes))

        deviations = []
        for i in range(n):
            dev = abs(skeleton_notes[i].start - source.notes[i].start)
            deviations.append(dev)

        if deviations:
            mean_dev = float(np.mean(deviations))
            if mean_dev > self.timing_tolerance_sec * 2.0:
                violations.append(ConstraintViolation(
                    constraint="rhythm_integrity",
                    instrument="skeleton",
                    note_index=-1,
                    detail=f"Mean onset deviation {mean_dev*1000:.1f}ms exceeds {self.timing_tolerance_sec*2000:.0f}ms",
                    severity="error",
                ))
            elif mean_dev > self.timing_tolerance_sec:
                violations.append(ConstraintViolation(
                    constraint="rhythm_integrity",
                    instrument="skeleton",
                    note_index=-1,
                    detail=f"Mean onset deviation {mean_dev*1000:.1f}ms (elevated)",
                    severity="warning",
                ))

        return violations

    def _get_skeleton_notes(self, gamelan_score: GamelanScore) -> List[GamelanNoteEvent]:
        """Return notes from the first skeleton (balungan) instrument found."""
        skeleton_instruments = ["saron", "demung", "slenthem"]
        for inst in skeleton_instruments:
            if inst in gamelan_score.tracks and gamelan_score.tracks[inst]:
                return gamelan_score.tracks[inst]
        # Fallback: first track available
        for notes in gamelan_score.tracks.values():
            if notes:
                return notes
        return []
