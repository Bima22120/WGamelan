"""Musical representation models: NoteEvent, Phrase, and SourceScore container.

Implements the WHAT representation of the musical input according to the
Gamelanizer architectural specification.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class NoteEvent:
    """Detailed musical note event reconstructed from audio analysis."""
    start: float                       # Start time in seconds
    end: float                         # End time in seconds
    duration: float                     # Duration in seconds
    pitch_hz: float                    # Fundamental frequency in Hz
    midi_pitch: int = 60               # 12-TET MIDI approximation
    velocity: int = 100                # MIDI velocity [1, 127]
    pitch_curve: Optional[np.ndarray] = None        # Sub-note pitch contour
    pitch_confidence: float = 1.0      # f0 detection confidence [0, 1]
    onset_strength: float = 1.0        # Normalized onset attack strength
    beat_index: int = 0                # Index within beat grid
    bar_index: int = 0                 # Musical measure / bar index
    phrase_id: int = 0                 # Phrase segment identifier
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def start_time(self) -> float:
        """Alias for compatibility with existing renderer."""
        return self.start

    @property
    def end_time(self) -> float:
        return self.end


@dataclass
class Phrase:
    """Musical phrase consisting of an ordered sequence of NoteEvents."""
    phrase_id: int
    start_time: float
    end_time: float
    notes: List[NoteEvent] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_notes(self) -> int:
        return len(self.notes)


@dataclass
class SourceScore:
    """Full musical reconstruction of the source performance."""
    notes: List[NoteEvent] = field(default_factory=list)
    phrases: List[Phrase] = field(default_factory=list)
    tempo_bpm: float = 120.0
    beat_times: List[float] = field(default_factory=list)
    time_signature: str = "4/4"
    total_duration: float = 0.0

    def get_pitch_contour(self) -> List[float]:
        """Return list of fundamental frequencies in chronological order."""
        return [n.pitch_hz for n in self.notes if n.pitch_hz > 20.0]

    def get_onsets(self) -> List[float]:
        """Return list of note onset times."""
        return [n.start for n in self.notes]


def segment_phrases(notes: List[NoteEvent], pause_threshold_sec: float = 0.65) -> List[Phrase]:
    """Segment an ordered sequence of NoteEvents into musical phrases based on pauses."""
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: n.start)
    phrases: List[Phrase] = []
    current_phrase_notes: List[NoteEvent] = []
    current_phrase_id = 0

    for i, note in enumerate(sorted_notes):
        current_phrase_notes.append(note)
        note.phrase_id = current_phrase_id

        is_last = (i == len(sorted_notes) - 1)
        if is_last:
            p_start = current_phrase_notes[0].start
            p_end = current_phrase_notes[-1].end
            phrases.append(Phrase(phrase_id=current_phrase_id, start_time=p_start, end_time=p_end, notes=list(current_phrase_notes)))
        else:
            next_start = sorted_notes[i + 1].start
            gap = next_start - note.end
            if gap >= pause_threshold_sec:
                p_start = current_phrase_notes[0].start
                p_end = current_phrase_notes[-1].end
                phrases.append(Phrase(phrase_id=current_phrase_id, start_time=p_start, end_time=p_end, notes=list(current_phrase_notes)))
                current_phrase_id += 1
                current_phrase_notes = []

    return phrases
