"""Gamelan performance styles, Irama meters, and ornamentation rules."""

from typing import List
from app.music.note import Note


class PerformanceStyle:
    """Applies traditional Javanese phrasing and articulation rules."""

    def __init__(self, irama_level: int = 1, apply_damping: bool = True):
        """
        Args:
            irama_level: 1 (Lancar), 2 (Tanggung), 3 (Dadi).
            apply_damping: If True, damps prior note when subsequent note strikes.
        """
        self.irama_level = irama_level
        self.apply_damping = apply_damping

    def apply(self, notes: List[Note]) -> List[Note]:
        """Apply performance phrasing, duration adjustment, and damping windows."""
        if not notes:
            return []

        styled: List[Note] = []
        for i, curr in enumerate(notes):
            # Check if there is a following note to determine damping time
            if i < len(notes) - 1:
                next_note = notes[i + 1]
                gap = next_note.start_time - curr.start_time
                if self.apply_damping and gap > 0:
                    # Saron player dampens right when the next note is struck
                    effective_dur = min(curr.duration, gap + 0.05)
                else:
                    effective_dur = curr.duration
            else:
                effective_dur = curr.duration

            metadata = dict(curr.metadata)
            metadata["irama_level"] = self.irama_level
            metadata["damped"] = self.apply_damping

            styled.append(
                Note(
                    start_time=curr.start_time,
                    duration=effective_dur,
                    pitch_hz=curr.pitch_hz,
                    midi_pitch=curr.midi_pitch,
                    velocity=curr.velocity,
                    gamelan_note=curr.gamelan_note,
                    gamelan_freq_hz=curr.gamelan_freq_hz,
                    metadata=metadata,
                )
            )

        return styled
