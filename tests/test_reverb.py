"""Unit tests for PendopoReverb module (Solusi 3)."""

import unittest
import numpy as np
from app.gamelan.reverb import PendopoReverb


class TestPendopoReverb(unittest.TestCase):

    def setUp(self):
        self.sr = 22050
        self.reverb = PendopoReverb(sr=self.sr, rt60=1.0, wet_level=0.3, dry_level=0.8)

    def test_impulse_response_stereo(self):
        # Create an impulse signal: single spike of 1.0 at t=0
        impulse = np.zeros(int(self.sr * 1.5), dtype=np.float32)
        impulse[0] = 1.0

        out = self.reverb.process(impulse, stereo=True)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[1], 2)
        self.assertEqual(len(out), len(impulse))

        # Check that energy decays smoothly into the tail
        early_energy = np.mean(np.abs(out[:int(self.sr * 0.2), 0]))
        tail_energy = np.mean(np.abs(out[int(self.sr * 0.8):, 0]))
        self.assertGreater(early_energy, tail_energy)
        self.assertGreater(tail_energy, 0.0)

    def test_reverb_mono(self):
        signal = np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.5, int(self.sr * 0.5))).astype(np.float32)
        out = self.reverb.process(signal, stereo=False)

        self.assertEqual(out.ndim, 1)
        self.assertEqual(len(out), len(signal))
        self.assertLessEqual(np.max(np.abs(out)), 1.0)

    def test_empty_input(self):
        empty = np.array([], dtype=np.float32)
        out_stereo = self.reverb.process(empty, stereo=True)
        self.assertEqual(len(out_stereo), 0)

        out_mono = self.reverb.process(empty, stereo=False)
        self.assertEqual(len(out_mono), 0)

    def test_stereo_decorrelation(self):
        # Left and right channels should have slight decorrelation for spatial width
        impulse = np.zeros(int(self.sr * 0.5), dtype=np.float32)
        impulse[0] = 1.0
        out = self.reverb.process(impulse, stereo=True)

        left = out[:, 0]
        right = out[:, 1]
        diff = np.max(np.abs(left - right))
        self.assertGreater(diff, 0.01)  # Distinct stereo reflections


if __name__ == "__main__":
    unittest.main()
