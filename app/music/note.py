"""Data structure representing a musical note."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Note:
    """Musical note data entity with both Western and Gamelan representations."""
    start_time: float               # Start time in seconds
    duration: float                 # Duration in seconds
    pitch_hz: float                 # Detected frequency in Hz
    midi_pitch: int = 60            # Closest Western 12-TET MIDI note number
    velocity: int = 100             # Dynamic velocity [1, 127]
    gamelan_note: Optional[str] = None      # Gamelan note label (e.g. '1', '2', '3', '5', '6')
    gamelan_freq_hz: Optional[float] = None # Tuned Gamelan target frequency in Hz
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    @property
    def western_name(self) -> str:
        """Return Western note name like 'C4', 'A4'."""
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (self.midi_pitch // 12) - 1
        note_idx = self.midi_pitch % 12
        return f"{names[note_idx]}{octave}"

    def __repr__(self) -> str:
        gam_str = f", Gamelan={self.gamelan_note} ({self.gamelan_freq_hz:.1f}Hz)" if self.gamelan_note else ""
        return (
            f"Note(start={self.start_time:.2f}s, dur={self.duration:.2f}s, "
            f"hz={self.pitch_hz:.1f}Hz, MIDI={self.midi_pitch} [{self.western_name}], "
            f"vel={self.velocity}{gam_str})"
        )
