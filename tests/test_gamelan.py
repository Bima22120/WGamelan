"""Tests for Gamelan tuning, scales, instruments, mapping, and synthesis."""

import unittest
import numpy as np

from app.music.note import Note
from app.gamelan.tuning import build_tuning_table, SLENDRO_BASE_FREQS, PELOG_BASE_FREQS
from app.gamelan.scale import GamelanScale
from app.gamelan.instrument import get_instrument, INSTRUMENTS
from app.gamelan.mapping import PitchMapper
from app.gamelan.envelope import GamelanEnvelope
from app.gamelan.renderer import GamelanRenderer


class TestGamelan(unittest.TestCase):

    def test_slendro_and_pelog_tuning(self):
        slendro_table = build_tuning_table("slendro", octaves=[0])
        self.assertEqual(len(slendro_table), len(SLENDRO_BASE_FREQS))

        pelog_table = build_tuning_table("pelog", octaves=[0])
        self.assertEqual(len(pelog_table), len(PELOG_BASE_FREQS))

    def test_scale_modes(self):
        scale_slendro = GamelanScale(scale_type="slendro")
        self.assertGreater(len(scale_slendro.pitches), 0)

        # 275 Hz should be closest to Slendro 1 (Ji ~ 270 Hz)
        closest = scale_slendro.find_closest(275.0)
        self.assertEqual(closest.degree, "1")
        self.assertEqual(closest.solfege, "Ji")

    def test_instrument_registry(self):
        saron = get_instrument("saron")
        self.assertEqual(saron.name, "Saron Barung")
        self.assertTrue(saron.damping_enabled)

        demung = get_instrument("demung")
        self.assertEqual(demung.octave_shift, -1)

    def test_pitch_mapper(self):
        scale = GamelanScale(scale_type="slendro")
        mapper = PitchMapper(scale=scale)

        n = Note(start_time=0.0, duration=0.5, pitch_hz=312.0)
        mapped = mapper.map_note(n)
        self.assertEqual(mapped.gamelan_note, "2")
        self.assertEqual(mapped.metadata["gamelan_solfege"], "Ro")
        self.assertEqual(mapped.gamelan_freq_hz, 310.0)

    def test_gamelan_envelope(self):
        env_gen = GamelanEnvelope(sr=22050, attack_sec=0.005, decay_sec=1.0)
        env = env_gen.generate(duration_sec=0.5)
        self.assertEqual(len(env), int(22050 * 0.5))
        self.assertLessEqual(np.max(env), 1.0)
        self.assertLess(env[-1], env[100])

    def test_gamelan_sampler_and_renderer(self):
        renderer = GamelanRenderer(instrument=get_instrument("saron"), sr=22050)
        notes = [
            Note(start_time=0.0, duration=0.3, pitch_hz=270.0, gamelan_freq_hz=270.0, gamelan_note="1"),
            Note(start_time=0.3, duration=0.3, pitch_hz=310.0, gamelan_freq_hz=310.0, gamelan_note="2"),
        ]
        audio = renderer.render(notes)
        self.assertGreater(len(audio), 0)
        self.assertGreater(np.max(np.abs(audio)), 0.01)
        self.assertLessEqual(np.max(np.abs(audio)), 1.0)

    def test_optimal_transposition(self):
        from app.gamelan.mapping import find_optimal_transposition
        scale = GamelanScale(scale_type="slendro")

        # Let's create notes intentionally shifted by +2 semitones from Slendro tones:
        # e.g. 270 * 2^(2/12) and 310 * 2^(2/12)
        shift_expected = -2.0
        n1 = Note(start_time=0.0, duration=1.0, pitch_hz=270.0 * (2.0 ** (2.0 / 12.0)))
        n2 = Note(start_time=1.0, duration=1.0, pitch_hz=310.0 * (2.0 ** (2.0 / 12.0)))

        best_shift, min_err = find_optimal_transposition([n1, n2], scale)
        self.assertEqual(best_shift, shift_expected)
        self.assertLess(min_err, 15.0)  # Error should be near zero after shifting -2 st

    def test_pitch_mapper_auto_key(self):
        scale = GamelanScale(scale_type="slendro")
        mapper = PitchMapper(scale=scale, auto_align_key=True)

        n = Note(start_time=0.0, duration=1.0, pitch_hz=270.0 * (2.0 ** (2.0 / 12.0)))
        mapped = mapper.map_notes([n])
        self.assertEqual(mapper.applied_transposition, -2.0)
        self.assertEqual(mapped[0].gamelan_note, "1")


if __name__ == "__main__":
    unittest.main()
