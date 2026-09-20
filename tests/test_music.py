"""Tests for musical notes, timing, events, and quantizer."""

import unittest
from app.music.note import Note
from app.music.events import Track, Score
from app.music.timing import TimingGrid
from app.music.velocity import VelocityEstimator
from app.music.tuning import hz_to_midi, midi_to_hz, hz_to_nearest_midi, cents_deviation, midi_to_note_name
from app.music.quantizer import NoteQuantizer


class TestMusic(unittest.TestCase):

    def test_note_representation(self):
        n = Note(start_time=1.0, duration=0.5, pitch_hz=440.0, midi_pitch=69, velocity=100)
        self.assertEqual(n.end_time, 1.5)
        self.assertEqual(n.western_name, "A4")

    def test_tuning_conversions(self):
        # A4 = 440 Hz -> MIDI 69
        self.assertEqual(round(hz_to_midi(440.0)), 69)
        self.assertEqual(round(midi_to_hz(69)), 440)
        self.assertEqual(hz_to_nearest_midi(440.0), 69)
        self.assertEqual(midi_to_note_name(60), "C4")

        # Cents difference of octave is 1200 cents
        cents = cents_deviation(880.0, 440.0)
        self.assertEqual(round(cents), 1200)

    def test_timing_grid(self):
        grid = TimingGrid(bpm=120.0)
        self.assertEqual(grid.seconds_per_beat, 0.5)
        self.assertEqual(grid.seconds_to_beats(1.0), 2.0)
        self.assertEqual(grid.beats_to_seconds(2.0), 1.0)

    def test_velocity_estimator(self):
        estimator = VelocityEstimator(min_velocity=40, max_velocity=127)
        vel_loud = estimator.estimate_from_amplitude(1.0)
        vel_quiet = estimator.estimate_from_amplitude(0.05)
        self.assertEqual(vel_loud, 127)
        self.assertLess(vel_quiet, vel_loud)
        self.assertGreaterEqual(vel_quiet, 40)

    def test_quantizer(self):
        quantizer = NoteQuantizer(bpm=120.0, subdivision=4)
        n = Note(start_time=0.13, duration=0.24, pitch_hz=440.0)
        qn = quantizer.quantize_note(n)
        self.assertEqual(round(qn.start_time, 3), 0.125)

    def test_score_and_track(self):
        track = Track(name="Saron")
        track.add_note(Note(start_time=0.0, duration=0.5, pitch_hz=270.0, midi_pitch=61))
        track.add_note(Note(start_time=0.5, duration=0.5, pitch_hz=310.0, midi_pitch=63))

        score = Score(tracks=[track], tempo_bpm=120.0)
        self.assertEqual(score.total_duration, 1.0)
        pm = score.to_pretty_midi()
        self.assertIsNotNone(pm)
        self.assertEqual(len(pm.instruments), 1)
        self.assertEqual(len(pm.instruments[0].notes), 2)


if __name__ == "__main__":
    unittest.main()
