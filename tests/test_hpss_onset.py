"""Unit tests for HPSS separation and electric guitar effect onset suppression."""

import unittest
import numpy as np

from app.audio.preprocessing import separate_hpss
from app.pitch.onset import OnsetDetector


class TestHPSSAndOnset(unittest.TestCase):
    def setUp(self):
        self.sr = 22050
        self.duration = 2.0
        self.t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)

    def test_separate_hpss_shapes(self):
        """Verify that HPSS separation maintains array dimensions and data types."""
        audio = np.sin(2 * np.pi * 440.0 * self.t).astype(np.float32)
        h, p = separate_hpss(audio, self.sr)
        self.assertEqual(len(h), len(audio))
        self.assertEqual(len(p), len(audio))
        self.assertEqual(h.dtype, np.float32)
        self.assertEqual(p.dtype, np.float32)

    def test_distortion_sustain_rejection(self):
        """Verify that continuous sustained distortion does not generate false onset attacks."""
        # Create a sustained distorted guitar signal: 440 Hz + heavy harmonics + noise
        fundamental = np.sin(2 * np.pi * 440.0 * self.t)
        h2 = 0.5 * np.sin(2 * np.pi * 880.0 * self.t)
        h3 = 0.3 * np.sin(2 * np.pi * 1320.0 * self.t)
        distortion_sustain = np.clip((fundamental + h2 + h3) * 3.0, -0.9, 0.9).astype(np.float32)

        detector = OnsetDetector(hop_length=512)
        result = detector.detect(distortion_sustain, sr=self.sr, use_hpss=True)

        # A steady sustained note with distortion should have very few or zero false transient onsets
        self.assertLessEqual(len(result.onset_times), 2)

    def test_pick_attack_detection_with_hpss(self):
        """Verify that sharp transient pick strikes are reliably detected despite background sustain."""
        # Create 2 sharp impulses (pick attacks) at t=0.3s and t=1.2s over sustained tone
        signal_audio = 0.1 * np.sin(2 * np.pi * 330.0 * self.t)
        
        # Add transient bursts at t=0.3s and t=1.2s
        idx1 = int(0.3 * self.sr)
        idx2 = int(1.2 * self.sr)
        burst_len = int(0.02 * self.sr)
        
        signal_audio[idx1:idx1 + burst_len] += 0.8 * np.random.randn(burst_len)
        signal_audio[idx2:idx2 + burst_len] += 0.8 * np.random.randn(burst_len)
        signal_audio = signal_audio.astype(np.float32)

        detector = OnsetDetector(hop_length=512)
        result = detector.detect(signal_audio, sr=self.sr, use_hpss=True)

        self.assertGreaterEqual(len(result.onset_times), 1)
        # Check that detected times are near 0.3s or 1.2s
        near_03 = any(abs(t - 0.3) < 0.15 for t in result.onset_times)
        near_12 = any(abs(t - 1.2) < 0.15 for t in result.onset_times)
        self.assertTrue(near_03 or near_12)


if __name__ == "__main__":
    unittest.main()
