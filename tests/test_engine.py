"""Tests for engine pipelines and offline batch conversion."""

import os
import tempfile
import unittest
import numpy as np

from app.engine.pipeline import GamelanizerPipeline
from app.engine.offline import OfflineProcessor
from app.engine.renderer import EngineRenderer
from app.engine.event_queue import EventQueue
from app.engine.latency import LatencyTracker


class TestEngine(unittest.TestCase):

    def setUp(self):
        self.sr = 22050
        t = np.linspace(0, 0.4, int(self.sr * 0.4), endpoint=False)
        n1 = (0.8 * np.sin(2 * np.pi * 270.0 * t)).astype(np.float32)
        n2 = (0.8 * np.sin(2 * np.pi * 310.0 * t)).astype(np.float32)
        self.synthetic_melody = np.concatenate([n1, n2])

    def test_event_queue(self):
        q = EventQueue()
        self.assertTrue(q.empty())
        q.put("note_1")
        self.assertEqual(q.qsize(), 1)
        item = q.get()
        self.assertEqual(item, "note_1")
        self.assertTrue(q.empty())

    def test_latency_tracker(self):
        tracker = LatencyTracker(sample_rate=22050, buffer_size=512)
        self.assertTrue(np.isclose(tracker.buffer_latency_ms, (512 / 22050) * 1000.0))
        tracker.start_timer()
        elapsed = tracker.stop_timer()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_engine_renderer(self):
        renderer = EngineRenderer(sample_rate=22050)
        buf = np.sin(np.linspace(0, 10, 2205)).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        try:
            saved = renderer.export(buf, out_path, channels="stereo")
            self.assertTrue(os.path.exists(saved))
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_gamelanizer_pipeline(self):
        pipeline = GamelanizerPipeline(scale_name="slendro", instrument_name="saron", sample_rate=self.sr)
        result = pipeline.process(self.synthetic_melody)

        self.assertGreater(len(result.audio_output), 0)
        self.assertEqual(result.sample_rate, self.sr)
        self.assertGreater(result.summary.duration_sec, 0.7)

    def test_offline_processor(self):
        from app.audio.loader import save_audio

        with tempfile.NamedTemporaryFile(suffix="_in.wav", delete=False) as f_in, \
             tempfile.NamedTemporaryFile(suffix="_out.wav", delete=False) as f_out, \
             tempfile.NamedTemporaryFile(suffix="_out.mid", delete=False) as f_mid:
            in_path = f_in.name
            out_wav = f_out.name
            out_mid = f_mid.name

        try:
            save_audio(in_path, self.synthetic_melody, sr=self.sr)
            processor = OfflineProcessor(scale_name="slendro", instrument_name="saron", sample_rate=self.sr)
            result = processor.process_file(
                input_path=in_path,
                output_wav_path=out_wav,
                output_midi_path=out_mid,
            )
            self.assertTrue(os.path.exists(out_wav))
            self.assertGreater(os.path.getsize(out_wav), 100)
            self.assertTrue(os.path.exists(out_mid))
        finally:
            for p in [in_path, out_wav, out_mid]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass


if __name__ == "__main__":
    unittest.main()
