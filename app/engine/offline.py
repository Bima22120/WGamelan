"""Offline file processor for batch converting audio files to Gamelan."""

import os
from typing import Optional, Callable
from app.audio.loader import load_audio
from app.engine.pipeline import GamelanizerPipeline, PipelineResult
from app.engine.renderer import EngineRenderer


class OfflineProcessor:
    """Processes audio files on disk and saves rendered Gamelan audio and optional MIDI."""

    def __init__(
        self,
        scale_name: str = "slendro",
        pathet: Optional[str] = None,
        instrument_name: str = "saron",
        sample_rate: int = 22050,
        quantize: bool = True,
    ):
        self.pipeline = GamelanizerPipeline(
            scale_name=scale_name,
            pathet=pathet,
            instrument_name=instrument_name,
            sample_rate=sample_rate,
            quantize=quantize,
        )
        self.engine_renderer = EngineRenderer(sample_rate=sample_rate)

    def process_file(
        self,
        input_path: str,
        output_wav_path: str,
        output_midi_path: Optional[str] = None,
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ) -> PipelineResult:
        """Process an input audio file and write output files."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Load input audio
        audio, _ = load_audio(input_path, target_sr=self.pipeline.sr, mono=True)

        # Process through pipeline
        result = self.pipeline.process(audio, progress_cb=progress_cb)

        # Save audio WAV
        self.engine_renderer.export(result.audio_output, output_wav_path)

        # Optional MIDI export
        if output_midi_path:
            try:
                result.score.save_midi(output_midi_path)
            except Exception as e:
                print(f"[Warning] Failed to export MIDI: {e}")

        return result
