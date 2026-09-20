"""Pitch and onset detection module."""

from app.pitch.base import BasePitchDetector, PitchTrack
from app.pitch.pyin_detector import PYINDetector
from app.pitch.onset import OnsetDetector, OnsetResult
from app.pitch.postprocess import (
    smooth_pitch_track,
    correct_octave_errors,
    segment_notes,
)

__all__ = [
    "BasePitchDetector",
    "PitchTrack",
    "PYINDetector",
    "OnsetDetector",
    "OnsetResult",
    "smooth_pitch_track",
    "correct_octave_errors",
    "segment_notes",
]
