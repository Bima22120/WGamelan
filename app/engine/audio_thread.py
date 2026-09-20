"""Background thread worker for asynchronous audio processing."""

import threading
from typing import Callable, Optional, Any
from app.engine.pipeline import GamelanizerPipeline, PipelineResult
from app.engine.offline import OfflineProcessor


class AudioWorkerThread(threading.Thread):
    """Executes audio conversion asynchronously in a background thread."""

    def __init__(
        self,
        processor: OfflineProcessor,
        input_path: str,
        output_wav_path: str,
        output_midi_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        completion_callback: Optional[Callable[[Optional[PipelineResult], Optional[Exception]], None]] = None,
    ):
        super().__init__(daemon=True)
        self.processor = processor
        self.input_path = input_path
        self.output_wav_path = output_wav_path
        self.output_midi_path = output_midi_path
        self.progress_callback = progress_callback
        self.completion_callback = completion_callback
        self.result: Optional[PipelineResult] = None
        self.error: Optional[Exception] = None

    def run(self) -> None:
        try:
            self.result = self.processor.process_file(
                input_path=self.input_path,
                output_wav_path=self.output_wav_path,
                output_midi_path=self.output_midi_path,
                progress_cb=self.progress_callback,
            )
            if self.completion_callback:
                self.completion_callback(self.result, None)
        except Exception as e:
            self.error = e
            if self.completion_callback:
                self.completion_callback(None, e)
