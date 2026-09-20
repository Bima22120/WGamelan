"""Rhythmic quantizer: aligning notes to musical time subdivisions."""

from typing import List
from app.music.note import Note
from app.music.timing import TimingGrid


class NoteQuantizer:
    """Quantizes note onset times and durations to musical grid."""

    def __init__(self, bpm: float = 120.0, subdivision: int = 4, min_duration_subdiv: float = 0.5):
        """
        Args:
            bpm: Tempo in beats per minute.
            subdivision: Grid division per quarter beat (4 = 16th notes, 2 = 8th notes).
            min_duration_subdiv: Minimum allowed note duration expressed in subdivisions.
        """
        self.grid = TimingGrid(bpm=bpm)
        self.subdivision = subdivision
        self.min_duration_subdiv = min_duration_subdiv

    def quantize_note(self, note: Note) -> Note:
        """Snap a single note's start time and duration to grid."""
        unit_sec = self.grid.seconds_per_beat / self.subdivision
        new_start = round(note.start_time / unit_sec) * unit_sec

        min_dur = unit_sec * self.min_duration_subdiv
        raw_dur = max(note.duration, min_dur)
        new_dur = max(min_dur, round(raw_dur / unit_sec) * unit_sec)

        return Note(
            start_time=max(0.0, new_start),
            duration=new_dur,
            pitch_hz=note.pitch_hz,
            midi_pitch=note.midi_pitch,
            velocity=note.velocity,
            gamelan_note=note.gamelan_note,
            gamelan_freq_hz=note.gamelan_freq_hz,
            metadata=dict(note.metadata),
        )

    def quantize_notes(self, notes: List[Note]) -> List[Note]:
        """Quantize a list of notes and sort chronologically."""
        quantized = [self.quantize_note(n) for n in notes]
        quantized.sort(key=lambda n: n.start_time)
        return quantized
