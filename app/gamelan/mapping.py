"""Pitch mapping: converting continuous frequencies or Western MIDI notes to Gamelan scales."""

import math
from typing import List, Tuple, Optional
from app.music.note import Note
from app.gamelan.scale import GamelanScale
from app.gamelan.tuning import GamelanPitch


def calculate_scale_deviation(freq_hz: float, scale: GamelanScale) -> float:
    """Calculate absolute difference in cents to closest scale tone."""
    if freq_hz <= 0 or not scale.pitches:
        return 0.0
    closest = scale.find_closest(freq_hz)
    return abs(1200.0 * math.log2(freq_hz / closest.freq_hz))


def find_optimal_transposition(
    notes: List[Note],
    scale: GamelanScale,
    search_range_semitones: int = 6,
) -> Tuple[float, float]:
    """Find the transposition shift (in semitones) that minimizes total weighted cent error.

    Args:
        notes: List of detected Note objects.
        scale: Target Gamelan scale.
        search_range_semitones: Range of semitones to explore (-range to +range).

    Returns:
        (best_shift_semitones, minimum_weighted_cent_error)
    """
    valid_notes = [n for n in notes if n.pitch_hz > 40.0 and n.duration > 0]
    if not valid_notes:
        return 0.0, 0.0

    total_weight = sum(n.duration for n in valid_notes)
    if total_weight <= 0:
        return 0.0, 0.0

    # For full 12-TET chromatic diatonic scale, preserve exact original pitch/key
    if "diatonic" in scale.scale_type and (not scale.pathet or "chromatic" in scale.pathet):
        return 0.0, 0.0

    best_shift = 0.0
    best_error = float("inf")


    # Search through semitone offsets [-6, +6]
    for semitone_shift in range(-search_range_semitones, search_range_semitones + 1):
        ratio = 2.0 ** (semitone_shift / 12.0)
        weighted_sq_err = 0.0

        for n in valid_notes:
            shifted_hz = n.pitch_hz * ratio
            dev = calculate_scale_deviation(shifted_hz, scale)
            # Quadratic penalty penalizes harsh out-of-scale outliers more strongly
            weighted_sq_err += n.duration * (dev ** 2)

        avg_rms_error = math.sqrt(weighted_sq_err / total_weight)
        if avg_rms_error < best_error:
            best_error = avg_rms_error
            best_shift = float(semitone_shift)

    return best_shift, best_error


class PitchMapper:
    """Maps continuous Hz and Western MIDI pitches to Gamelan tones with harmonic key alignment."""

    def __init__(
        self,
        scale: GamelanScale,
        auto_align_key: bool = True,
        transposition_semitones: float = 0.0,
    ):
        self.scale = scale
        self.auto_align_key = auto_align_key
        self.transposition_semitones = transposition_semitones
        self.applied_transposition: float = transposition_semitones

    def map_frequency(self, freq_hz: float, shift_semitones: float = 0.0) -> Tuple[GamelanPitch, float, float]:
        """Find the nearest Gamelan pitch and deviation in cents after transposition.

        Returns:
            (closest_gamelan_pitch, cents_deviation, shifted_frequency_hz)
        """
        if freq_hz <= 0:
            pitch = self.scale.pitches[0]
            return pitch, 0.0, 0.0

        shifted_freq = freq_hz * (2.0 ** (shift_semitones / 12.0))
        closest = self.scale.find_closest(shifted_freq)
        cents_diff = 1200.0 * math.log2(shifted_freq / closest.freq_hz)
        return closest, cents_diff, shifted_freq

    def map_note(self, note: Note, shift_semitones: float = 0.0) -> Note:
        """Assign closest Gamelan degree and tuned frequency to a Note object."""
        gam_pitch, cents_diff, shifted_hz = self.map_frequency(note.pitch_hz, shift_semitones=shift_semitones)

        from app.music.tuning import hz_to_nearest_midi
        midi_approx = hz_to_nearest_midi(gam_pitch.freq_hz)

        metadata = dict(note.metadata)
        metadata["gamelan_solfege"] = gam_pitch.solfege
        metadata["gamelan_cents_deviation"] = cents_diff
        metadata["gamelan_octave"] = gam_pitch.octave
        metadata["transposition_applied"] = shift_semitones
        metadata["shifted_pitch_hz"] = shifted_hz

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
        """Map a list of notes to Gamelan scale tones, optimizing key alignment if enabled."""
        if not notes:
            return []

        shift = self.transposition_semitones
        if self.auto_align_key:
            opt_shift, min_err = find_optimal_transposition(notes, self.scale)
            shift = opt_shift
            self.applied_transposition = shift
        else:
            self.applied_transposition = shift

        return [self.map_note(n, shift_semitones=shift) for n in notes]
