"""Unit tests for GongPunctuationLayer module (Solusi 4)."""

import unittest
from app.music.note import Note
from app.gamelan.scale import GamelanScale
from app.gamelan.gong_layer import GongPunctuationLayer


class TestGongLayer(unittest.TestCase):

    def setUp(self):
        self.scale = GamelanScale(scale_type="slendro")
        self.gong_layer = GongPunctuationLayer(scale=self.scale, bpm=120.0, add_kenong=True, gongan_beats=16)

    def test_gong_frequency_range(self):
        root = self.gong_layer._get_default_root_freq()
        freq = self.gong_layer._to_gong_register(root)
        # Large gong should resonate in deep bass / sub-bass range (45 to 85 Hz)
        self.assertGreaterEqual(freq, 45.0)
        self.assertLessEqual(freq, 85.0)

    def test_kenong_frequency(self):
        root = self.gong_layer._get_default_root_freq()
        gong_freq = self.gong_layer._to_gong_register(root)
        kenong_freq = self.gong_layer._to_kenong_register(root)
        self.assertGreaterEqual(kenong_freq, 95.0)
        self.assertLessEqual(kenong_freq, 180.0)
        self.assertAlmostEqual(kenong_freq, gong_freq * 2.0)

    def test_generate_cadence_gong(self):
        # A melody that ends at 5.0 seconds
        melody = [
            Note(start_time=0.0, duration=1.0, pitch_hz=270.0),
            Note(start_time=1.0, duration=1.5, pitch_hz=310.0),
            Note(start_time=3.0, duration=2.0, pitch_hz=350.0),  # ends at 5.0
        ]
        gong_notes = self.gong_layer.generate(melody)
        self.assertGreater(len(gong_notes), 0)

        # There must be a final cadence Gong Ageng at or near the last note's end (5.0s)
        gongs_only = [n for n in gong_notes if n.metadata.get("role") == "colotomy_gong"]
        self.assertGreater(len(gongs_only), 0)
        last_gong = gongs_only[-1]
        self.assertAlmostEqual(last_gong.start_time, 5.0, delta=0.1)

    def test_phrase_pause_gong(self):
        # Melody with a distinct phrase pause: 0.0-2.0s, silence 2.0-3.5s, 3.5-5.0s
        melody = [
            Note(start_time=0.0, duration=2.0, pitch_hz=270.0),  # phrase 1 ends at 2.0s
            Note(start_time=3.5, duration=1.5, pitch_hz=310.0),  # phrase 2 starts at 3.5s (gap 1.5s)
        ]
        gong_notes = self.gong_layer.generate(melody)
        gong_times = [n.start_time for n in gong_notes if n.metadata.get("role") == "colotomy_gong"]

        # Should have a gong around the pause (2.0s) and final cadence (5.0s)
        has_phrase_gong = any(abs(t - 2.0) < 0.2 for t in gong_times)
        self.assertTrue(has_phrase_gong)

    def test_empty_melody(self):
        gong_notes = self.gong_layer.generate([])
        self.assertEqual(len(gong_notes), 0)


if __name__ == "__main__":
    unittest.main()
