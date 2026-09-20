"""Main processing pipeline: coordinates audio analysis, pitch mapping, and synthesis."""

from dataclasses import dataclass
from typing import List, Optional, Callable
import numpy as np

from app.audio.loader import load_audio
from app.audio.preprocessing import preprocess_audio
from app.audio.analyzer import AudioAnalyzer, AudioSummary
from app.pitch.pyin_detector import PYINDetector
from app.pitch.onset import OnsetDetector
from app.pitch.postprocess import smooth_pitch_track, suppress_vibrato, correct_octave_errors, segment_notes
from app.music.note import Note
from app.music.velocity import VelocityEstimator
from app.music.quantizer import NoteQuantizer
from app.music.events import Track, Score
from app.gamelan.scale import GamelanScale
from app.gamelan.instrument import get_instrument, GamelanInstrument
from app.gamelan.mapping import PitchMapper
from app.gamelan.performance import PerformanceStyle
from app.gamelan.renderer import GamelanRenderer
from app.gamelan.gong_layer import GongPunctuationLayer
from app.gamelan.bonang_layer import BonangEmbellishmentLayer


@dataclass
class PipelineResult:
    audio_output: np.ndarray
    sample_rate: int
    detected_notes: List[Note]
    mapped_notes: List[Note]
    score: Score
    summary: AudioSummary
    transposition_applied: float = 0.0
    gong_notes: List[Note] = None
    bonang_notes: List[Note] = None

    def __post_init__(self):
        if self.gong_notes is None:
            self.gong_notes = []
        if self.bonang_notes is None:
            self.bonang_notes = []


class GamelanizerPipeline:
    """Unified audio-to-gamelan conversion engine."""

    def __init__(
        self,
        scale_name: str = "slendro",
        pathet: Optional[str] = None,
        instrument_name: str = "saron",
        sample_rate: int = 22050,
        quantize: bool = True,
        bpm: float = 120.0,
        auto_key: bool = True,
        transpose: float = 0.0,
        legato: bool = True,
        reverb: bool = True,
        add_gong: bool = True,
        add_bonang: bool = True,
        stereo: bool = True,
    ):
        self.sr = sample_rate
        self.scale = GamelanScale(scale_type=scale_name, pathet=pathet)
        self.instrument = get_instrument(instrument_name)
        self.quantize = quantize
        self.bpm = bpm
        self.auto_key = auto_key
        self.transpose = transpose
        self.legato = legato
        self.reverb = reverb
        self.add_gong = add_gong
        self.add_bonang = add_bonang
        self.stereo = stereo

        # Subsystems
        self.analyzer = AudioAnalyzer(sample_rate=self.sr)
        self.pitch_detector = PYINDetector(hop_length=256)
        self.onset_detector = OnsetDetector(hop_length=512)
        self.velocity_estimator = VelocityEstimator()
        self.quantizer = NoteQuantizer(bpm=self.bpm, subdivision=4)
        self.mapper = PitchMapper(
            scale=self.scale,
            auto_align_key=self.auto_key,
            transposition_semitones=self.transpose,
        )
        self.performance = PerformanceStyle(irama_level=1, apply_damping=self.instrument.damping_enabled)
        self.renderer = GamelanRenderer(
            instrument=self.instrument,
            sr=self.sr,
            legato=self.legato,
            reverb=self.reverb,
            stereo=self.stereo,
        )
        self.gong_layer = GongPunctuationLayer(scale=self.scale, bpm=self.bpm)
        self.bonang_layer = BonangEmbellishmentLayer(scale=self.scale, bpm=self.bpm)

    def process(
        self,
        audio_input: np.ndarray,
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ) -> PipelineResult:
        """Run the conversion pipeline on an in-memory audio array."""
        def report(msg: str, pct: float):
            if progress_cb:
                progress_cb(msg, pct)

        # 1. Preprocessing
        report("Preprocessing audio...", 0.1)
        clean_audio = preprocess_audio(audio_input, sr=self.sr)
        summary = self.analyzer.summarize(clean_audio)
        if summary.estimated_tempo_bpm and summary.estimated_tempo_bpm > 40:
            self.bpm = summary.estimated_tempo_bpm
            self.quantizer = NoteQuantizer(bpm=self.bpm)

        # 2. Pitch & Onset Detection
        report("Detecting pitch track...", 0.3)
        pitch_track = self.pitch_detector.detect(clean_audio, sr=self.sr)

        report("Detecting onsets...", 0.45)
        onset_res = self.onset_detector.detect(clean_audio, sr=self.sr)

        # 3. Post-process pitch & segment notes
        report("Smoothing & vibrato suppression...", 0.52)
        smoothed_f0 = smooth_pitch_track(pitch_track.frequencies_hz)
        smoothed_f0 = suppress_vibrato(
            smoothed_f0,
            hop_length=self.pitch_detector.hop_length,
            sr=self.sr,
        )
        smoothed_f0 = correct_octave_errors(smoothed_f0)

        report("Segmenting & merging notes...", 0.58)
        raw_segments = segment_notes(
            times=pitch_track.times,
            pitch_hz=smoothed_f0,
            voiced_flags=pitch_track.voiced_flags,
            onset_times=onset_res.onset_times,
            merge_gap_sec=0.15,
            merge_pitch_ratio=1.06,
        )

        # Build Note objects
        notes: List[Note] = []
        hop_s = self.pitch_detector.hop_length / self.sr
        for start_t, dur, f0 in raw_segments:
            idx = int(start_t * self.sr)
            window = clean_audio[idx:idx + int(0.05 * self.sr)]
            vel = self.velocity_estimator.estimate_from_audio_window(window)
            notes.append(
                Note(
                    start_time=start_t,
                    duration=dur,
                    pitch_hz=f0,
                    velocity=vel,
                )
            )

        # 4. Optional Quantization
        if self.quantize and notes:
            report("Quantizing rhythm...", 0.65)
            notes = self.quantizer.quantize_notes(notes)

        # 5. Gamelan Pitch Mapping & Articulation
        report("Mapping to Gamelan scale...", 0.75)
        mapped_notes = self.mapper.map_notes(notes)
        styled_notes = self.performance.apply(mapped_notes)
        if self.mapper.applied_transposition != 0.0:
            report(f"Applied Key Transposition: {self.mapper.applied_transposition:+.1f} semitones", 0.8)

        # 6. Colotomic Gong Punctuation Layer (Solusi 4)
        gong_notes: List[Note] = []
        if self.add_gong and styled_notes:
            report("Generating Gong & Kenong layer...", 0.82)
            self.gong_layer.bpm = self.bpm
            gong_notes = self.gong_layer.generate(styled_notes)

        # 7. Bonang Embellishment Layer (Full Ensemble Karawitan)
        bonang_notes: List[Note] = []
        if self.add_bonang and styled_notes and self.instrument.name.lower() != "bonang barung":
            report("Arranging Bonang chime ensemble layer...", 0.86)
            self.bonang_layer.bpm = self.bpm
            bonang_notes = self.bonang_layer.generate(styled_notes)

        # 8. Build Multi-Track Score
        lead_track = Track(name=self.instrument.name, instrument=self.instrument.name, notes=styled_notes)
        tracks = [lead_track]
        if bonang_notes:
            bonang_track = Track(name="Bonang Barung", instrument="bonang", notes=bonang_notes)
            tracks.append(bonang_track)
        if gong_notes:
            gong_track = Track(name="Gong Ageng & Kenong", instrument="gong", notes=gong_notes)
            tracks.append(gong_track)
        score = Score(tracks=tracks, tempo_bpm=self.bpm)

        # 9. Render Audio (Physical Modeling + Legato + Pendopo Reverb)
        report("Synthesizing Full Gamelan Ensemble (Physical Modeling + Reverb)...", 0.90)
        all_notes = styled_notes + bonang_notes + gong_notes
        rendered_audio = self.renderer.render(
            all_notes,
            legato=self.legato,
            reverb=self.reverb,
            stereo=self.stereo,
        )

        report("Complete!", 1.0)
        return PipelineResult(
            audio_output=rendered_audio,
            sample_rate=self.sr,
            detected_notes=notes,
            mapped_notes=styled_notes,
            gong_notes=gong_notes,
            bonang_notes=bonang_notes,
            score=score,
            summary=summary,
            transposition_applied=self.mapper.applied_transposition,
        )
