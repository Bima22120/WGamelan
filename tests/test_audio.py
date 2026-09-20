"""Tests for audio loading, preprocessing, and analysis."""

import os
import tempfile
import unittest
import numpy as np

from app.audio.loader import load_audio, save_audio
from app.audio.preprocessing import (
    normalize_peak,
    normalize_rms,
    highpass_filter,
    trim_silence,
    preprocess_audio,
)
from app.audio.analyzer import AudioAnalyzer


class TestAudio(unittest.TestCase):

    def setUp(self):
        self.sr = 22050
        t = np.linspace(0, 0.5, int(self.sr * 0.5), endpoint=False)
        self.audio = (0.8 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    def test_save_and_load_audio(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            saved_path = save_audio(tmp_path, self.audio, sr=self.sr)
            self.assertTrue(os.path.exists(saved_path))

            loaded_audio, loaded_sr = load_audio(saved_path, target_sr=self.sr)
            self.assertEqual(loaded_sr, self.sr)
            self.assertEqual(len(loaded_audio), len(self.audio))
            self.assertLessEqual(np.max(np.abs(loaded_audio)), 1.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_normalize_peak(self):
        data = np.array([0.2, -0.5, 0.1], dtype=np.float32)
        norm = normalize_peak(data, target_peak=0.95)
        self.assertTrue(np.isclose(np.max(np.abs(norm)), 0.95))

    def test_normalize_rms(self):
        data = np.random.uniform(-0.5, 0.5, 1000).astype(np.float32)
        norm = normalize_rms(data, target_rms=0.1)
        calc_rms = np.sqrt(np.mean(norm ** 2))
        self.assertTrue(np.isclose(calc_rms, 0.1, atol=0.02))

    def test_highpass_filter(self):
        dirty = self.audio + 0.3
        filtered = highpass_filter(dirty, sr=self.sr, cutoff_hz=50.0)
        self.assertLess(abs(np.mean(filtered)), 0.05)

    def test_trim_silence(self):
        silence = np.zeros(1000, dtype=np.float32)
        tone = np.sin(np.linspace(0, 10, 2000)).astype(np.float32)
        composite = np.concatenate([silence, tone, silence])
        trimmed, start, end = trim_silence(composite, threshold_db=-30.0)
        self.assertLess(len(trimmed), len(composite))
        self.assertGreater(start, 0)
        self.assertLess(start, 1000)

    def test_audio_analyzer(self):
        analyzer = AudioAnalyzer(sample_rate=self.sr)
        summary = analyzer.summarize(self.audio)
        self.assertGreater(summary.duration_sec, 0.4)
        self.assertEqual(summary.sample_rate, self.sr)
        self.assertGreater(summary.peak_amplitude, 0.5)


if __name__ == "__main__":
    unittest.main()
