"""Unit tests for BonangEmbellishmentLayer module."""

import unittest
from app.music.note import Note
from app.gamelan.scale import GamelanScale
from app.gamelan.bonang_layer import BonangEmbellishmentLayer


class TestBonangLayer(unittest.TestCase):

    def setUp(self):
        self.scale = GamelanScale(scale_type="slendro")
        self.bonang = BonangEmbellishmentLayer(scale=self.scale, bpm=120.0)

    def test_generate_bonang_notes(self):
        melody = [
            Note(start_time=0.0, duration=0.8, pitch_hz=270.0),
            Note(start_time=1.0, duration=0.8, pitch_hz=310.0),
        ]
        bonang_notes = self.bonang.generate(melody)
        self.assertGreater(len(bonang_notes), 0)

        # Each note in bonang_notes should have instrument 'bonang'
        for n in bonang_notes:
            self.assertEqual(n.metadata.get("instrument"), "bonang")
            self.assertGreater(n.velocity, 0)
            self.assertGreater(n.duration, 0)

    def test_empty_melody(self):
        bonang_notes = self.bonang.generate([])
        self.assertEqual(len(bonang_notes), 0)


if __name__ == "__main__":
    unittest.main()
