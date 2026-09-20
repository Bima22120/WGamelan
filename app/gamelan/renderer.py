"""Gamelan audio renderer: synthesizes notes and composites into a continuous audio buffer."""

from typing import List, Optional
import numpy as np
from app.music.note import Note
from app.gamelan.sampler import GamelanSampler
from app.gamelan.instrument import GamelanInstrument, get_instrument


class GamelanRenderer:
    """Renders a sequence of Gamelan notes into a composite audio wave."""

    def __init__(self, instrument: Optional[GamelanInstrument] = None, sr: int = 22050):
        self.sr = sr
        self.instrument = instrument or get_instrument("saron")
        self.sampler = GamelanSampler(instrument=self.instrument, sr=sr)

    def render(self, notes: List[Note], pad_end_sec: float = 1.0) -> np.ndarray:
        """Render notes into a single-channel 1D numpy array."""
        if not notes:
            return np.zeros(int(self.sr * pad_end_sec), dtype=np.float32)

        total_duration = max(n.end_time for n in notes) + self.instrument.decay_time_sec + pad_end_sec
        num_samples = int(total_duration * self.sr)
        master_buffer = np.zeros(num_samples, dtype=np.float32)

        for note in notes:
            freq = note.gamelan_freq_hz or note.pitch_hz
            if freq <= 20.0:
                continue

            # Synthesize note
            damped = note.metadata.get("damped", self.instrument.damping_enabled)
            note_audio = self.sampler.synthesize_note(
                freq_hz=freq,
                duration_sec=note.duration,
                velocity=note.velocity,
                damped=damped,
                damp_time_sec=note.duration,
            )

            start_idx = int(note.start_time * self.sr)
            end_idx = min(start_idx + len(note_audio), num_samples)
            slice_len = end_idx - start_idx

            if slice_len > 0:
                master_buffer[start_idx:end_idx] += note_audio[:slice_len]

        # Prevent clipping by normalizing if peak > 0.95
        peak = np.max(np.abs(master_buffer))
        if peak > 0.95:
            master_buffer = (master_buffer / peak) * 0.95

        return master_buffer
