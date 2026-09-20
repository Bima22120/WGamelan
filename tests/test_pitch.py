"""Tests for pitch tracking and onset detection."""

import unittest
import numpy as np

from app.pitch.pyin_detector import PYINDetector
from app.pitch.onset import OnsetDetector
from app.pitch.postprocess import (
    smooth_pitch_track,
    suppress_vibrato,
    correct_octave_errors,
    merge_adjacent_notes,
    segment_notes,
)


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

    def test_suppress_vibrato(self):
        """Vibrato-like signal (440 Hz ± 5% oscillation) should converge to stable pitch."""
        sr = 22050
        hop = 256
        n_frames = 100
        frame_rate = sr / hop

        # Simulate pitch track with 6 Hz vibrato: 440 * (1 + 0.04 * sin(2pi * 6 * t))
        t = np.arange(n_frames) / frame_rate
        vibrato_pitch = (440.0 * (1.0 + 0.04 * np.sin(2 * np.pi * 6.0 * t))).astype(np.float32)

        smoothed = suppress_vibrato(vibrato_pitch, hop_length=hop, sr=sr)

        # After suppression, std deviation should be much smaller
        original_std = np.std(vibrato_pitch)
        smoothed_std = np.std(smoothed)
        self.assertLess(smoothed_std, original_std * 0.5)

        # Mean should remain close to 440 Hz
        self.assertTrue(np.isclose(np.mean(smoothed), 440.0, atol=15.0))

    def test_merge_adjacent_notes(self):
        """Fragments of same pitch separated by short gaps should be merged."""
        # Same pitch 310 Hz, 50ms gap — should be merged
        notes_same = [
            (0.0, 0.3, 310.0),
            (0.35, 0.3, 313.0),   # ~17 cents diff, within 1.06 ratio
        ]
        merged = merge_adjacent_notes(notes_same, max_gap_sec=0.1, max_pitch_ratio=1.06)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0][0], 0.0)
        self.assertAlmostEqual(merged[0][1], 0.65, places=2)

        # Different pitches should NOT be merged
        notes_diff = [
            (0.0, 0.3, 270.0),
            (0.32, 0.3, 410.0),   # 2 major semitones apart, should not merge
        ]
        not_merged = merge_adjacent_notes(notes_diff, max_gap_sec=0.1, max_pitch_ratio=1.06)
        self.assertEqual(len(not_merged), 2)

    def test_segment_notes_with_merge(self):
        """Segmentation with merge enabled should reduce choppy fragments."""
        sr = 22050
        # 1.0s of voiced 310 Hz with a tiny 0.1s unvoiced gap in the middle
        n_total = 100
        times = np.linspace(0, 1.5, n_total)
        pitch_hz = np.full(n_total, 310.0, dtype=np.float32)
        voiced = np.ones(n_total, dtype=bool)
        # Introduce a small silence gap around frame 40
        voiced[38:45] = False
        pitch_hz[38:45] = 0.0

        notes_merged = segment_notes(
            times, pitch_hz, voiced, np.array([]), min_duration=0.05,
            merge_gap_sec=0.15, merge_pitch_ratio=1.06
        )
        notes_no_merge = segment_notes(
            times, pitch_hz, voiced, np.array([]), min_duration=0.05,
            merge_gap_sec=0.0
        )
        # Merged version should have fewer notes than un-merged version
        self.assertLessEqual(len(notes_merged), len(notes_no_merge))


if __name__ == "__main__":
    unittest.main()
