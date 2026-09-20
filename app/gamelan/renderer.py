from typing import List, Optional
import numpy as np
from app.music.note import Note
from app.gamelan.sampler import GamelanSampler
from app.gamelan.instrument import GamelanInstrument, get_instrument
from app.gamelan.reverb import PendopoReverb


class GamelanRenderer:
    """Renders a sequence of Gamelan notes into a composite audio wave."""

    def __init__(
        self,
        instrument: Optional[GamelanInstrument] = None,
        sr: int = 22050,
        legato: bool = True,
        reverb: bool = False,
        stereo: bool = False,
    ):
        self.sr = sr
        self.instrument = instrument or get_instrument("saron")
        self.sampler = GamelanSampler(instrument=self.instrument, sr=sr)
        self.gong_sampler = GamelanSampler(instrument=get_instrument("gong"), sr=sr)
        self.legato = legato
        self.reverb_enabled = reverb
        self.stereo = stereo
        self.reverb_processor = PendopoReverb(sr=sr)

    def render(
        self,
        notes: List[Note],
        pad_end_sec: float = 1.5,
        legato: Optional[bool] = None,
        reverb: Optional[bool] = None,
        stereo: Optional[bool] = None,
    ) -> np.ndarray:
        """Render notes into a composite audio array (mono 1D or stereo 2D)."""
        use_legato = self.legato if legato is None else legato
        use_reverb = self.reverb_enabled if reverb is None else reverb
        use_stereo = self.stereo if stereo is None else stereo

        if not notes:
            num_zeros = int(self.sr * pad_end_sec)
            return np.zeros((num_zeros, 2) if (use_reverb and use_stereo) else num_zeros, dtype=np.float32)

        # Allow extra headroom for natural gong & reverb decay
        max_decay = max(self.instrument.decay_time_sec, 4.5)
        total_duration = max(n.end_time for n in notes) + max_decay + pad_end_sec
        num_samples = int(total_duration * self.sr)
        master_buffer = np.zeros(num_samples, dtype=np.float32)

        for note in notes:
            freq = note.gamelan_freq_hz or note.pitch_hz
            if freq <= 20.0:
                continue

            inst_meta = note.metadata.get("instrument", "").lower()
            is_gong = ("gong" in inst_meta or "kenong" in inst_meta or freq < 90.0)
            active_sampler = self.gong_sampler if is_gong else self.sampler

            # Determine damping behavior
            if is_gong or use_legato:
                # In legato mode or gong, allow natural ring-out so notes overlap harmoniously
                damped = False
                damp_time = note.duration
                duration_to_synth = max(note.duration * 1.5, 0.6 if not is_gong else 4.5)
            else:
                damped = note.metadata.get("damped", self.instrument.damping_enabled)
                damp_time = note.duration
                duration_to_synth = note.duration

            note_audio = active_sampler.synthesize_note(
                freq_hz=freq,
                duration_sec=duration_to_synth,
                velocity=note.velocity,
                damped=damped,
                damp_time_sec=damp_time,
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

        # Apply Pendopo Reverb if enabled
        if use_reverb:
            return self.reverb_processor.process(master_buffer, stereo=use_stereo)

        return master_buffer
