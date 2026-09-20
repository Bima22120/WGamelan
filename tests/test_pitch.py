"""Tests for pitch tracking and onset detection."""

import unittest
import numpy as np

from app.pitch.pyin_detector import PYINDetector
from app.pitch.onset import OnsetDetector
from app.pitch.postprocess import smooth_pitch_track, correct_octave_errors, segment_notes


class TestPitch(unittest.TestCase):

    def setUp(self):
        self.sr = 22050
        t1 = np.linspace(0, 0.3, int(self.sr * 0.3), endpoint=False)
        tone1 = (0.8 * np.sin(2 * np.pi * 300.0 * t1)).astype(np.float32)

        t2 = np.linspace(0, 0.3, int(self.sr * 0.3), endpoint=False)
        tone2 = (0.8 * np.sin(2 * np.pi * 440.0 * t2)).astype(np.float32)

        gap = np.zeros(int(self.sr * 0.05), dtype=np.float32)
        self.seq = np.concatenate([tone1, gap, tone2])

    def test_pitch_detection(self):
        detector = PYINDetector(fmin=100.0, fmax=800.0, hop_length=256)
        track = detector.detect(self.seq, sr=self.sr)

        self.assertGreater(len(track), 0)
        self.assertEqual(len(track.frequencies_hz), len(track.times))
        voiced_pitches = track.frequencies_hz[track.voiced_flags]
        self.assertGreater(len(voiced_pitches), 0)
        median_p = np.median(voiced_pitches)
        self.assertTrue(250.0 < median_p < 500.0)

    def test_onset_detection(self):
        detector = OnsetDetector(hop_length=256)
        res = detector.detect(self.seq, sr=self.sr)
        self.assertGreaterEqual(len(res.onset_times), 1)

    def test_smooth_pitch_track(self):
        pitches = np.array([440.0, 440.0, 880.0, 440.0, 440.0], dtype=np.float32)
        smoothed = smooth_pitch_track(pitches, filter_size=3)
        self.assertEqual(smoothed[2], 440.0)

    def test_correct_octave_errors(self):
        pitches = np.array([220.0, 440.0, 220.0], dtype=np.float32)
        corrected = correct_octave_errors(pitches)
        self.assertTrue(np.isclose(corrected[1], 220.0))

    def test_segment_notes(self):
        times = np.linspace(0, 1.0, 50)
        pitch_hz = np.full(50, 310.0, dtype=np.float32)
        voiced = np.ones(50, dtype=bool)
        onsets = np.array([0.0, 0.5])

        notes = segment_notes(times, pitch_hz, voiced, onsets, min_duration=0.1)
        self.assertGreaterEqual(len(notes), 1)
        self.assertTrue(np.isclose(notes[0][2], 310.0, atol=5.0))


if __name__ == "__main__":
    unittest.main()
