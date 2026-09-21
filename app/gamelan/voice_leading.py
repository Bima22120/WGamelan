"""Gamelan harmonic voice-leading: register assignment and cadence detection."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
from app.music.note import Note
from app.gamelan.scale import GamelanScale

REGISTER_BOUNDS = {
    "gong":   (45.0,   90.0),
    "kenong": (90.0,  200.0),
    "demung": (135.0, 285.0),
    "saron":  (270.0, 560.0),
    "bonang": (380.0, 900.0),
    "peking": (540.0, 1200.0),
}


class OctaveAssigner:
    """Transposes a frequency into the target instrument register by octave shifts."""

    def __init__(self, scale=None):
        self.scale = scale or GamelanScale()

    def assign(self, freq_hz: float, instrument: str) -> float:
        if freq_hz <= 0.0:
            return freq_hz
        key = instrument.lower()
        for role in REGISTER_BOUNDS:
            if role in key:
                key = role
                break
        else:
            return freq_hz
        lo, hi = REGISTER_BOUNDS[key]
        result = freq_hz
        for _ in range(8):
            if result >= lo:
                break
            result *= 2.0
        for _ in range(8):
            if result <= hi:
                break
            result /= 2.0
        return float(result)

    def assign_note(self, note: Note, instrument: str) -> Note:
        freq = note.gamelan_freq_hz or note.pitch_hz
        new_freq = self.assign(freq, instrument)
        if new_freq == freq:
            return note
        metadata = dict(note.metadata)
        metadata["register_transposed"] = True
        metadata["original_gamelan_freq_hz"] = freq
        return Note(
            start_time=note.start_time,
            duration=note.duration,
            pitch_hz=note.pitch_hz,
            midi_pitch=note.midi_pitch,
            velocity=note.velocity,
            gamelan_note=note.gamelan_note,
            gamelan_freq_hz=new_freq,
            metadata=metadata,
        )


@dataclass
class PhraseInfo:
    start_time: float
    end_time: float
    seleh_freq_hz: float
    seleh_degree: str
    phrase_index: int


class CadenceDetector:
    """Detects phrase boundaries and seleh (cadence) notes in a Gamelan melody."""

    def __init__(self, min_gap_sec: float = 0.55, long_note_sec: float = 1.2):
        self.min_gap_sec = min_gap_sec
        self.long_note_sec = long_note_sec

    def detect_phrases(self, notes: List[Note]) -> List[PhraseInfo]:
        if not notes:
            return []
        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        phrases = []
        phrase_start = sorted_notes[0].start_time
        phrase_idx = 0
        for i, note in enumerate(sorted_notes):
            is_last = (i == len(sorted_notes) - 1)
            is_long = note.duration >= self.long_note_sec
            if is_last:
                gap = float("inf")
            else:
                gap = sorted_notes[i + 1].start_time - note.end_time
            boundary = is_last or (gap >= self.min_gap_sec) or is_long
            if boundary:
                seleh_freq = note.gamelan_freq_hz or note.pitch_hz
                phrases.append(PhraseInfo(
                    start_time=phrase_start,
                    end_time=note.end_time,
                    seleh_freq_hz=seleh_freq,
                    seleh_degree=note.gamelan_note or "1",
                    phrase_index=phrase_idx,
                ))
                phrase_idx += 1
                if not is_last:
                    phrase_start = sorted_notes[i + 1].start_time
        return phrases

    def get_seleh_at(self, time: float, phrases: List[PhraseInfo]):
        for ph in phrases:
            if ph.start_time <= time <= ph.end_time + 0.5:
                return ph
        return None


def assign_registers(saron_notes, bonang_notes, gong_notes, scale=None):
    """Ensure each instrument layer is in its canonical register."""
    assigner = OctaveAssigner(scale=scale)
    adjusted_bonang = [
        assigner.assign_note(n, instrument=n.metadata.get("instrument", "bonang"))
        for n in bonang_notes
    ]
    adjusted_gong = [
        assigner.assign_note(n, instrument=n.metadata.get("instrument", "gong"))
        for n in gong_notes
    ]
    return saron_notes, adjusted_bonang, adjusted_gong
