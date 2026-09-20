"""Western 12-TET tuning mathematics and conversion functions."""

import math
from typing import Tuple

A4_FREQ = 440.0
A4_MIDI = 69
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def hz_to_midi(freq_hz: float) -> float:
    """Convert frequency in Hz to fractional MIDI pitch number."""
    if freq_hz <= 0:
        return 0.0
    return 69.0 + 12.0 * math.log2(freq_hz / A4_FREQ)


def midi_to_hz(midi_note: float) -> float:
    """Convert MIDI note number (can be fractional) to frequency in Hz."""
    return A4_FREQ * (2.0 ** ((midi_note - A4_MIDI) / 12.0))


def hz_to_nearest_midi(freq_hz: float) -> int:
    """Convert frequency in Hz to closest integer MIDI note [0, 127]."""
    if freq_hz <= 0:
        return 0
    midi = round(hz_to_midi(freq_hz))
    return max(0, min(127, int(midi)))


def cents_deviation(freq_hz: float, ref_hz: float) -> float:
    """Compute difference between two frequencies in cents (100 cents = 1 semitone)."""
    if freq_hz <= 0 or ref_hz <= 0:
        return 0.0
    return 1200.0 * math.log2(freq_hz / ref_hz)


def midi_to_note_name(midi: int) -> str:
    """Convert integer MIDI note to standard scientific pitch notation (e.g. 60 -> 'C4')."""
    midi = max(0, min(127, midi))
    octave = (midi // 12) - 1
    name = NOTE_NAMES[midi % 12]
    return f"{name}{octave}"


def hz_to_pitch_info(freq_hz: float) -> Tuple[int, str, float]:
    """Return (nearest_midi, note_name, cents_offset)."""
    exact_midi = hz_to_midi(freq_hz)
    nearest_midi = max(0, min(127, round(exact_midi)))
    offset_cents = (exact_midi - nearest_midi) * 100.0
    return nearest_midi, midi_to_note_name(nearest_midi), offset_cents
