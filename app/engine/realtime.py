"""Real-time streaming audio engine for interactive live conversion."""

from typing import Optional, Callable
import numpy as np
from app.gamelan.scale import GamelanScale
from app.gamelan.instrument import get_instrument
from app.gamelan.sampler import GamelanSampler
from app.pitch.pyin_detector import PYINDetector
from app.engine.event_queue import EventQueue


class RealtimeEngine:
    """Handles real-time chunked audio input and on-the-fly Gamelan sound synthesis."""

    def __init__(
        self,
        scale_name: str = "slendro",
        instrument_name: str = "saron",
        sample_rate: int = 22050,
        buffer_size: int = 1024,
    ):
        self.sr = sample_rate
        self.buffer_size = buffer_size
        self.scale = GamelanScale(scale_type=scale_name)
        self.instrument = get_instrument(instrument_name)
        self.sampler = GamelanSampler(instrument=self.instrument, sr=self.sr)
        self.detector = PYINDetector(hop_length=128)
        self.event_queue = EventQueue()
        self._running = False
        self._last_pitch_hz: float = 0.0

    def process_chunk(self, in_chunk: np.ndarray) -> np.ndarray:
        """Process a single incoming buffer chunk and return synthetic audio."""
        if len(in_chunk) == 0:
            return np.zeros(self.buffer_size, dtype=np.float32)

        track = self.detector.detect(in_chunk, self.sr)
        voiced = track.voiced_flags
        f0 = track.frequencies_hz

        out_chunk = np.zeros(len(in_chunk), dtype=np.float32)

        if np.any(voiced):
            active_f0 = f0[voiced]
            median_f0 = float(np.median(active_f0))
            if median_f0 > 40.0:
                gam_pitch = self.scale.find_closest(median_f0)
                # Only trigger new note if pitch differs significantly
                if abs(gam_pitch.freq_hz - self._last_pitch_hz) > 5.0:
                    self._last_pitch_hz = gam_pitch.freq_hz
                    synth = self.sampler.synthesize_note(
                        freq_hz=gam_pitch.freq_hz,
                        duration_sec=len(in_chunk) / self.sr,
                        velocity=100,
                    )
                    slice_len = min(len(out_chunk), len(synth))
                    out_chunk[:slice_len] = synth[:slice_len]

        return out_chunk
