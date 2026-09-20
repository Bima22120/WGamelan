"""Musical event representation: NoteEvent, Track, and Score containers."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from app.music.note import Note

try:
    import pretty_midi
except ImportError:
    pretty_midi = None


class EventType(Enum):
    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"


@dataclass
class NoteEvent:
    time: float
    event_type: EventType
    note: Note
    velocity: int = 100


@dataclass
class Track:
    """A single musical track (e.g. Saron Barung part)."""
    name: str = "Gamelan"
    instrument: str = "saron"
    notes: List[Note] = field(default_factory=list)

    def add_note(self, note: Note) -> None:
        self.notes.append(note)
        self.notes.sort(key=lambda n: n.start_time)

    def get_events(self) -> List[NoteEvent]:
        """Convert notes to interleaved NoteOn and NoteOff events."""
        events: List[NoteEvent] = []
        for n in self.notes:
            events.append(NoteEvent(time=n.start_time, event_type=EventType.NOTE_ON, note=n, velocity=n.velocity))
            events.append(NoteEvent(time=n.end_time, event_type=EventType.NOTE_OFF, note=n, velocity=0))
        events.sort(key=lambda e: e.time)
        return events


@dataclass
class Score:
    """Full musical arrangement containing one or more tracks."""
    tracks: List[Track] = field(default_factory=list)
    tempo_bpm: float = 120.0

    def add_track(self, track: Track) -> None:
        self.tracks.append(track)

    @property
    def total_duration(self) -> float:
        if not self.tracks:
            return 0.0
        max_end = 0.0
        for track in self.tracks:
            for note in track.notes:
                if note.end_time > max_end:
                    max_end = note.end_time
        return max_end

    def to_pretty_midi(self) -> Optional["pretty_midi.PrettyMIDI"]:
        """Convert score to PrettyMIDI representation for saving or inspection."""
        if pretty_midi is None:
            return None

        pm = pretty_midi.PrettyMIDI(initial_tempo=self.tempo_bpm)
        for trk in self.tracks:
            # Use Glockenspiel program (9) or Vibraphone (11) as bell/metallophone approximation
            prog = pretty_midi.instrument_name_to_program("Glockenspiel")
            inst = pretty_midi.Instrument(program=prog, name=trk.name)
            for n in trk.notes:
                pm_note = pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.midi_pitch,
                    start=n.start_time,
                    end=max(n.start_time + 0.05, n.end_time),
                )
                inst.notes.append(pm_note)
            pm.instruments.append(inst)
        return pm

    def save_midi(self, filepath: str) -> None:
        """Export score to standard MIDI file."""
        pm = self.to_pretty_midi()
        if pm is not None:
            pm.write(filepath)
        else:
            raise RuntimeError("pretty_midi is not installed; cannot export MIDI file.")
