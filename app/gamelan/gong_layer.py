"""Automatic Gamelan colotomic punctuation and Gong layer generator.

In traditional Javanese Karawitan, the musical form (Gending) is cyclical
and anchored by structural punctuation instruments (Colotomy):
  1. Gong Ageng: The deepest, most sacred instrument, sounded at the end of
     each cyclical period (Gongan) and at the final cadence of the piece.
  2. Kenong / Kempul: Secondary punctuation marking internal cycle subdivisions
     (Kenongan), providing rhythmic weight and harmonic grounding.

This module analyzes the melodic note sequence, tempo, and phrase pauses,
and automatically generates an authentic Gong and Kenong accompaniment track.
"""

from typing import List, Optional
import numpy as np
from app.music.note import Note
from app.gamelan.scale import GamelanScale
from app.gamelan.instrument import get_instrument


class GongPunctuationLayer:
    """Generates structural Gong Ageng and Kenong punctuation for a melody."""

    def __init__(
        self,
        scale: Optional[GamelanScale] = None,
        bpm: float = 120.0,
        add_kenong: bool = True,
        gongan_beats: int = 16,
    ):
        self.scale = scale or GamelanScale(scale_type="slendro")
        self.bpm = bpm if bpm > 30 else 120.0
        self.add_kenong = add_kenong
        self.gongan_beats = gongan_beats

    def _get_default_root_freq(self) -> float:
        """Derive default root frequency from scale root."""
        if self.scale and self.scale.pitches:
            root_candidates = [p.freq_hz for p in self.scale.pitches if p.degree in ("1", "2") and p.octave == 0]
            root_freq = root_candidates[0] if root_candidates else self.scale.pitches[0].freq_hz
        else:
            root_freq = 261.63
        return float(root_freq)

    def _to_gong_register(self, freq: float) -> float:
        """Shift frequency into deep resonant sub-bass Gong Ageng register (45–85 Hz)."""
        if freq <= 0.0:
            freq = self._get_default_root_freq()
        f = freq
        while f > 85.0:
            f /= 2.0
        while f < 45.0:
            f *= 2.0
        return float(f)

    def _to_kenong_register(self, freq: float) -> float:
        """Shift frequency into warm mid-bass Kenong register (95–180 Hz)."""
        if freq <= 0.0:
            freq = self._get_default_root_freq()
        f = freq
        while f > 180.0:
            f /= 2.0
        while f < 95.0:
            f *= 2.0
        return float(f)

    def _find_seleh_freq(self, target_time: float, sorted_notes: List[Note]) -> float:
        """Find the active or cadence note (Nada Seleh) around target_time to avoid harmonic clash."""
        candidates = []
        for n in sorted_notes:
            freq = n.gamelan_freq_hz or n.pitch_hz
            if freq <= 20.0:
                continue
            # Note actively sounding at target_time
            if n.start_time <= target_time <= n.end_time:
                candidates.append((0.0, freq))
            # Note that concluded just before target_time (within 0.8s)
            elif 0.0 < (target_time - n.end_time) <= 0.8:
                candidates.append((target_time - n.end_time, freq))
            # Note starting immediately after target_time (within 0.3s)
            elif 0.0 < (n.start_time - target_time) <= 0.3:
                candidates.append((n.start_time - target_time, freq))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

        return self._get_default_root_freq()

    def generate(self, melody_notes: List[Note]) -> List[Note]:
        """Generate colotomic punctuation notes matching the melody and harmonic seleh points.

        Args:
            melody_notes: List of transcribed lead notes.

        Returns:
            List of Note objects representing the Gong & Kenong accompaniment track.
        """
        if not melody_notes:
            return []

        sorted_notes = sorted(melody_notes, key=lambda n: n.start_time)
        first_start = sorted_notes[0].start_time
        last_end = max(n.end_time for n in sorted_notes)
        total_duration = last_end - first_start

        sec_per_beat = 60.0 / self.bpm
        gongan_duration = self.gongan_beats * sec_per_beat  # e.g., 16 beats = 8.0s at 120bpm

        # Identify natural phrase pause points (silence >= 0.65s)
        phrase_pause_times: List[float] = []
        for i in range(len(sorted_notes) - 1):
            gap = sorted_notes[i + 1].start_time - sorted_notes[i].end_time
            if gap >= 0.65:
                # Phrase ends at previous note end
                phrase_pause_times.append(sorted_notes[i].end_time)

        gong_times: List[float] = []

        # 1. Periodic gongan strikes
        if total_duration >= gongan_duration:
            num_cycles = int(np.floor(total_duration / gongan_duration))
            for c in range(1, num_cycles + 1):
                t = first_start + c * gongan_duration
                if t < last_end - 1.0:
                    gong_times.append(t)

        # 2. Add phrase pauses if not already close to an existing gong
        for pt in phrase_pause_times:
            if not any(abs(pt - gt) < 2.0 for gt in gong_times) and pt < last_end - 1.0:
                gong_times.append(pt)

        # 3. Final cadence gong (always sounded at the end of the song)
        final_gong_time = last_end
        if not any(abs(final_gong_time - gt) < 1.5 for gt in gong_times):
            gong_times.append(final_gong_time)

        gong_times.sort()

        gong_notes: List[Note] = []

        # Generate Gong Ageng notes with Harmonic Seleh Pitch Alignment
        for gt in gong_times:
            seleh_hz = self._find_seleh_freq(gt, sorted_notes)
            aligned_gong_freq = self._to_gong_register(seleh_hz)

            gong_notes.append(
                Note(
                    start_time=gt,
                    duration=4.5,
                    pitch_hz=aligned_gong_freq,
                    velocity=115,
                    metadata={
                        "instrument": "gong",
                        "role": "colotomy_gong",
                        "harmonic_seleh_aligned": True,
                        "damped": False,
                    },
                )
            )

        # Generate Kenong notes at cycle midpoints if requested
        if self.add_kenong and gongan_duration >= 4.0:
            kenong_interval = gongan_duration / 2.0
            t_curr = first_start + kenong_interval
            while t_curr < last_end:
                # Avoid clashing within 1.2s of a Gong Ageng
                if not any(abs(t_curr - gt) < 1.2 for gt in gong_times):
                    k_seleh_hz = self._find_seleh_freq(t_curr, sorted_notes)
                    aligned_kenong_freq = self._to_kenong_register(k_seleh_hz)

                    gong_notes.append(
                        Note(
                            start_time=t_curr,
                            duration=2.0,
                            pitch_hz=aligned_kenong_freq,
                            velocity=95,
                            metadata={
                                "instrument": "kenong",
                                "role": "colotomy_kenong",
                                "harmonic_seleh_aligned": True,
                                "damped": False,
                            },
                        )
                    )
                t_curr += kenong_interval

        gong_notes.sort(key=lambda n: n.start_time)
        return gong_notes

