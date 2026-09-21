"""Bonang embellishment and interlocking pattern generator.

In Javanese Karawitan, the Bonang Barung and Bonang Panerus are leading
melodic elaboration instruments (Panerusan). They weave continuous, syncopated
chime patterns (Pipilan & Gembyangan) in an upper octave register around the core
Balungan melody:
  - Upper Octave Register: Bonang sounds 1 octave above Saron to prevent timbral
    clashing and add brilliant crystalline shimmer.
  - Mipil (Interlocking): Weaves between the previous note and current target note
    to create rich polyphonic texture rather than naive pitch duplication.
  - Gembyangan: Striking sustained tones with delicate dynamic fills.
"""

from typing import List, Optional
import numpy as np
from app.music.note import Note
from app.gamelan.scale import GamelanScale


class BonangEmbellishmentLayer:
    """Generates authentic Bonang chime patterns accompanying the lead melody."""

    def __init__(self, scale: Optional[GamelanScale] = None, bpm: float = 120.0):
        self.scale = scale or GamelanScale(scale_type="slendro")
        self.bpm = bpm if bpm > 30 else 120.0

    def _to_bonang_register(self, freq: float) -> float:
        """Shift pitch into optimal upper-register Bonang kettle chime sweet spot (420–1150 Hz)."""
        if freq <= 0.0:
            return 523.25
        f = freq
        # Elevate to upper octave register (1 octave above Saron)
        if f < 420.0:
            f *= 2.0
        while f > 1250.0:
            f /= 2.0
        while f < 380.0:
            f *= 2.0
        return float(f)

    def generate(self, melody_notes: List[Note]) -> List[Note]:
        """Generate a Bonang chime track from the lead notes.

        Args:
            melody_notes: List of transcribed lead notes (Balungan).

        Returns:
            List of Note objects representing the Bonang accompaniment part.
        """
        if not melody_notes:
            return []

        sorted_notes = sorted(melody_notes, key=lambda n: n.start_time)
        bonang_notes: List[Note] = []

        beat_sec = 60.0 / self.bpm
        # Bonang subdivisions typically play in 8th notes (half beat)
        subdivision_sec = max(0.16, min(0.35, beat_sec / 2.0))

        for i, note in enumerate(sorted_notes):
            raw_freq = note.gamelan_freq_hz or note.pitch_hz
            if raw_freq <= 20.0:
                continue

            curr_bonang_f = self._to_bonang_register(raw_freq)
            dur = note.duration
            t_start = note.start_time

            # 1. Authentic Mipil Interlocking:
            # Weave anticipation from previous note's pitch into target note
            if i > 0:
                prev_note = sorted_notes[i - 1]
                prev_raw_f = prev_note.gamelan_freq_hz or prev_note.pitch_hz
                if prev_raw_f > 20.0:
                    prev_bonang_f = self._to_bonang_register(prev_raw_f)
                    gap = t_start - prev_note.start_time
                    if gap >= subdivision_sec * 1.4:
                        anticipation_time = t_start - subdivision_sec
                        if anticipation_time >= prev_note.start_time + 0.05:
                            # Play previous note on the upbeat anticipation stroke (Pipilan)
                            bonang_notes.append(
                                Note(
                                    start_time=anticipation_time,
                                    duration=subdivision_sec * 0.85,
                                    pitch_hz=prev_bonang_f,
                                    velocity=int(np.clip(note.velocity * 0.75, 40, 85)),
                                    metadata={
                                        "instrument": "bonang",
                                        "role": "bonang_mipil_anticipation",
                                        "damped": False,
                                    },
                                )
                            )

            # 2. Main target hit: Upper-octave resonant chime stroke
            bonang_notes.append(
                Note(
                    start_time=t_start,
                    duration=max(dur * 0.75, subdivision_sec),
                    pitch_hz=curr_bonang_f,
                    velocity=int(np.clip(note.velocity * 0.88, 50, 95)),
                    metadata={
                        "instrument": "bonang",
                        "role": "bonang_main_chime",
                        "damped": False,
                    },
                )
            )

            # 3. Mipil rhythmic continuation / Gembyangan shimmer:
            # If the note duration is long (held note >= 2 subdivisions),
            # Bonang plays a crisp rhythmic elaboration on the 2nd half of the beat
            if dur >= subdivision_sec * 2.0:
                fill_time = t_start + subdivision_sec
                bonang_notes.append(
                    Note(
                        start_time=fill_time,
                        duration=subdivision_sec * 0.85,
                        pitch_hz=curr_bonang_f,
                        velocity=int(np.clip(note.velocity * 0.70, 38, 80)),
                        metadata={
                            "instrument": "bonang",
                            "role": "bonang_shimmer_fill",
                            "damped": False,
                        },
                    )
                )

        bonang_notes.sort(key=lambda n: n.start_time)
        return bonang_notes

