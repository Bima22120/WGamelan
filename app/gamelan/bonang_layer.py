"""Bonang embellishment and interlocking pattern generator.

In Javanese Karawitan, the Bonang Barung and Bonang Panerus are leading
melodic elaboration instruments (Panerusan). They weave continuous, syncopated
chime patterns (Pipilan & Gembyangan) around the core Balungan melody:
  - Pipilan: Alternating and anticipating core notes in rapid, interlocking chimes.
  - Gembyangan: Striking octaves together to provide harmonic shimmer.

This module automatically generates a traditional Bonang chime track from
the transcribed lead melody notes, giving the music the rich, full-ensemble
texture heard in authentic Gamelan performances and AI song generators.
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
        subdivision_sec = max(0.18, beat_sec / 2.0)

        for i, note in enumerate(sorted_notes):
            freq = note.gamelan_freq_hz or note.pitch_hz
            if freq <= 20.0:
                continue

            dur = note.duration
            t_start = note.start_time

            # 1. Pipilan technique: Anticipation stroke before the balungan target
            # Play an offbeat anticipation 1 subdivision before (if space permits)
            if i > 0:
                prev_note = sorted_notes[i - 1]
                prev_freq = prev_note.gamelan_freq_hz or prev_note.pitch_hz
                gap = t_start - prev_note.start_time
                if gap >= subdivision_sec * 1.5:
                    anticipation_time = t_start - subdivision_sec
                    if anticipation_time >= prev_note.end_time - 0.05:
                        bonang_notes.append(
                            Note(
                                start_time=anticipation_time,
                                duration=subdivision_sec * 0.9,
                                pitch_hz=freq,
                                velocity=int(np.clip(note.velocity * 0.85, 45, 105)),
                                metadata={
                                    "instrument": "bonang",
                                    "role": "bonang_anticipation",
                                    "damped": False,
                                },
                            )
                        )

            # 2. Main target hit: Gembyangan / chime stroke
            bonang_notes.append(
                Note(
                    start_time=t_start,
                    duration=max(dur * 0.8, subdivision_sec),
                    pitch_hz=freq,
                    velocity=int(np.clip(note.velocity * 0.95, 55, 115)),
                    metadata={
                        "instrument": "bonang",
                        "role": "bonang_main",
                        "damped": False,
                    },
                )
            )

            # 3. Sustained fill: If note duration is long (>= 2 subdivisions),
            # Bonang plays a rhythmic repeated chime on the second half of the beat
            if dur >= subdivision_sec * 2.0:
                fill_time = t_start + subdivision_sec
                bonang_notes.append(
                    Note(
                        start_time=fill_time,
                        duration=subdivision_sec * 0.9,
                        pitch_hz=freq,
                        velocity=int(np.clip(note.velocity * 0.80, 45, 100)),
                        metadata={
                            "instrument": "bonang",
                            "role": "bonang_fill",
                            "damped": False,
                        },
                    )
                )

        bonang_notes.sort(key=lambda n: n.start_time)
        return bonang_notes
