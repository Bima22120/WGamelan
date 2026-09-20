"""Music representation and notation module."""

from app.music.note import Note
from app.music.events import NoteEvent, EventType, Track, Score
from app.music.timing import TimingGrid
from app.music.velocity import VelocityEstimator
from app.music.tuning import (
    hz_to_midi,
    midi_to_hz,
    hz_to_nearest_midi,
    cents_deviation,
    midi_to_note_name,
    hz_to_pitch_info,
)
from app.music.quantizer import NoteQuantizer

__all__ = [
    "Note",
    "NoteEvent",
    "EventType",
    "Track",
    "Score",
    "TimingGrid",
    "VelocityEstimator",
    "hz_to_midi",
    "midi_to_hz",
    "hz_to_nearest_midi",
    "cents_deviation",
    "midi_to_note_name",
    "hz_to_pitch_info",
    "NoteQuantizer",
]
