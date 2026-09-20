"""Master audio exporter and signal formatting engine."""

import os
from typing import Literal
import numpy as np
from app.audio.loader import save_audio


class EngineRenderer:
    """Handles audio master bus processing, clipping protection, and file output."""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def export(
        self,
        audio_buffer: np.ndarray,
        output_filepath: str,
        channels: Literal["mono", "stereo"] = "mono",
        pan: float = 0.0,
        gain_db: float = 0.0,
    ) -> str:
        """Export audio buffer to disk with gain and spatial formatting."""
        data = np.asarray(audio_buffer, dtype=np.float32)

        # Apply gain
        if gain_db != 0.0:
            scale = 10.0 ** (gain_db / 20.0)
            data = data * scale

        # Soft clip limiter
        data = np.tanh(data)

        if channels == "stereo":
            # Pan between -1.0 (left) and +1.0 (right)
            pan_clamped = max(-1.0, min(1.0, pan))
            left_gain = np.cos((pan_clamped + 1.0) * np.pi / 4.0)
            right_gain = np.sin((pan_clamped + 1.0) * np.pi / 4.0)
            stereo_data = np.column_stack((data * left_gain, data * right_gain))
            return save_audio(output_filepath, stereo_data, sr=self.sample_rate)

        return save_audio(output_filepath, data, sr=self.sample_rate)
