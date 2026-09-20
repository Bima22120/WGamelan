"""Pitch mapping: converting continuous frequencies or Western MIDI notes to Gamelan scales."""

import math
from typing import List, Tuple
from app.music.note import Note
from app.gamelan.scale import GamelanScale
from app.gamelan.tuning import GamelanPitch


class PitchMapper:
    """Maps continuous Hz and Western MIDI pitches to Gamelan tones."""

    def __init__(self, scale: GamelanScale):
        self.scale = scale

    def map_frequency(self, freq_hz: float) -> Tuple[GamelanPitch, float]:
        """Find the nearest Gamelan pitch and the deviation in cents."""
        if freq_hz <= 0:
            pitch = self.scale.pitches[0]
            return pitch, 0.0

        closest = self.scale.find_closest(freq_hz)
        cents_diff = 1200.0 * math.log2(freq_hz / closest.freq_hz)
        return closest, cents_diff

    def map_note(self, note: Note) -> Note:
        """Assign closest Gamelan degree and tuned frequency to a Note object."""
        gam_pitch, cents_diff = self.map_frequency(note.pitch_hz)

        # Approximate closest Western MIDI for rendering or MIDI preview
        from app.music.tuning import hz_to_nearest_midi
        midi_approx = hz_to_nearest_midi(gam_pitch.freq_hz)

        metadata = dict(note.metadata)
        metadata["gamelan_solfege"] = gam_pitch.solfege
        metadata["gamelan_cents_deviation"] = cents_diff
        metadata["gamelan_octave"] = gam_pitch.octave

        return Note(
            start_time=note.start_time,
            duration=note.duration,
            pitch_hz=note.pitch_hz,
            midi_pitch=midi_approx,
            velocity=note.velocity,
            gamelan_note=gam_pitch.degree,
            gamelan_freq_hz=gam_pitch.freq_hz,
            metadata=metadata,
        )

    def map_notes(self, notes: List[Note]) -> List[Note]:
        """Map a list of notes to Gamelan scale tones."""
        return [self.map_note(n) for n in notes]
