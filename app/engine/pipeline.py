"""Main processing pipeline: coordinates audio analysis, pitch mapping, synthesis, and fidelity evaluation.

Supports two orchestration modes:
  1. Rule-Based (default): GongPunctuationLayer + BonangEmbellishmentLayer (backward-compatible)
  2. Reference-Based: OrchestrationLearner + OrchestrationTransfer (when reference_audio provided)

Data flow (Reference-Based mode):
    SOURCE AUDIO → AudioAnalysis → SourceScore
    REFERENCE AUDIO → OrchestrationLearner → OrchestrationProfile
    SourceScore + OrchestrationProfile → OrchestrationTransfer → GamelanScore
    GamelanScore → PerformanceEngine → Renderer → GamelanAudio
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import numpy as np

from app.audio.loader import load_audio
from app.audio.preprocessing import preprocess_audio
from app.audio.analyzer import AudioAnalyzer, AudioSummary
from app.pitch.pyin_detector import PYINDetector
from app.pitch.onset import OnsetDetector
from app.pitch.postprocess import smooth_pitch_track, suppress_vibrato, correct_octave_errors, segment_notes
from app.music.note import Note
from app.music.source_score import NoteEvent, SourceScore, segment_phrases
from app.music.gamelan_score import GamelanScore
from app.music.velocity import VelocityEstimator
from app.music.quantizer import NoteQuantizer
from app.music.events import Track, Score
from app.gamelan.scale import GamelanScale
from app.gamelan.instrument import get_instrument, GamelanInstrument
from app.gamelan.context_mapper import ContextAwareGamelanMapper, GamelanNoteEvent
from app.gamelan.performance_model import PerformanceTranslationEngine, PerformanceEvent
from app.gamelan.performance import PerformanceStyle
from app.gamelan.renderer import GamelanRenderer
from app.gamelan.gong_layer import GongPunctuationLayer
from app.gamelan.bonang_layer import BonangEmbellishmentLayer
from app.fidelity.evaluator import FidelityEngine, FidelityMetrics
from app.orchestration.orchestration_profile import OrchestrationProfile
from app.orchestration.learner import OrchestrationLearner
from app.orchestration.transfer_engine import OrchestrationTransfer
from app.orchestration.validator import ArrangementValidator


@dataclass
class PipelineResult:
    audio_output: np.ndarray
    sample_rate: int
    detected_notes: List[Note]
    mapped_notes: List[Note]
    score: Score
    summary: AudioSummary
    transposition_applied: float = 0.0
    gong_notes: List[Note] = field(default_factory=list)
    bonang_notes: List[Note] = field(default_factory=list)
    source_score: Optional[SourceScore] = None
    fidelity_metrics: Optional[FidelityMetrics] = None
    orchestration_profile: Optional[OrchestrationProfile] = None
    gamelan_score: Optional[GamelanScore] = None
    orchestration_mode: str = "rule_based"     # 'rule_based' or 'reference_based'

    def __post_init__(self):
        if self.gong_notes is None:
            self.gong_notes = []
        if self.bonang_notes is None:
            self.bonang_notes = []


class GamelanizerPipeline:
    """Unified audio-to-gamelan conversion engine with closed-loop fidelity evaluation.

    Supports two orchestration modes:
      - Rule-Based (default): deterministic colotomic layers (backward-compatible)
      - Reference-Based: learns HOW a reference gamelan ensemble plays, then transfers
        that orchestration behavior to any source song.

    Reference-based mode activated by passing reference_audio to process().
    """

    def __init__(
        self,
        scale_name: str = "slendro",
        pathet: Optional[str] = None,
        instrument_name: str = "saron",
        sample_rate: int = 22050,
        quantize: bool = False,
        timing_mode: str = "PRESERVE",
        bpm: float = 120.0,
        auto_key: bool = True,
        transpose: float = 0.0,
        legato: bool = True,
        reverb: bool = True,
        add_gong: bool = True,
        add_bonang: bool = True,
        stereo: bool = True,
        orchestration_profile: Optional[OrchestrationProfile] = None,
    ):
        self.sr = sample_rate
        self.scale = GamelanScale(scale_type=scale_name, pathet=pathet)
        self.instrument = get_instrument(instrument_name)
        self.quantize = quantize
        self.timing_mode = "SOFT_QUANTIZE" if quantize else timing_mode
        self.bpm = bpm
        self.auto_key = auto_key
        self.transpose = transpose
        self.legato = legato
        self.reverb = reverb
        self.add_gong = add_gong
        self.add_bonang = add_bonang
        self.stereo = stereo

        # Core Analysis & Reconstruction Subsystems
        self.analyzer = AudioAnalyzer(sample_rate=self.sr)
        self.pitch_detector = PYINDetector(hop_length=256)
        self.onset_detector = OnsetDetector(hop_length=512)
        self.velocity_estimator = VelocityEstimator()
        self.quantizer = NoteQuantizer(bpm=self.bpm, subdivision=4)

        # Context-Aware Sequence Mapping & Performance Translation
        self.context_mapper = ContextAwareGamelanMapper(
            scale=self.scale,
            timing_mode=self.timing_mode,
            w_pitch=1.0,
            w_contour=2.5,
            w_interval=1.2,
        )
        self.performance_engine = PerformanceTranslationEngine(
            instrument=self.instrument,
            apply_damping=self.instrument.damping_enabled,
        )
        self.performance = PerformanceStyle(irama_level=1, apply_damping=self.instrument.damping_enabled)

        # Physical Modeling & Colotomic Synthesis Subsystems
        self.renderer = GamelanRenderer(
            instrument=self.instrument,
            sr=self.sr,
            legato=self.legato,
            reverb=self.reverb,
            stereo=self.stereo,
        )
        self.gong_layer = GongPunctuationLayer(scale=self.scale, bpm=self.bpm)
        self.bonang_layer = BonangEmbellishmentLayer(scale=self.scale, bpm=self.bpm)

        # Reference-Based Orchestration Transfer
        self._orchestration_profile: Optional[OrchestrationProfile] = orchestration_profile
        self._orchestration_learner = OrchestrationLearner(profile_dir="data/orchestration_profiles")
        self._orchestration_transfer = OrchestrationTransfer(
            scale=self.scale,
            timing_mode=self.timing_mode,
        )

        # Closed-Loop Fidelity Engine
        self.fidelity_engine = FidelityEngine(sr=self.sr)

    def process(
        self,
        audio_input: np.ndarray,
        progress_cb: Optional[Callable[[str, float], None]] = None,
        reference_audio: Optional[np.ndarray] = None,
        reference_name: str = "reference_gamelan",
    ) -> PipelineResult:
        """Run the end-to-end musical reconstruction and conversion pipeline.

        Args:
            audio_input: Source audio (guitar / any instrument) to convert.
            progress_cb: Optional progress callback (message, fraction 0–1).
            reference_audio: Optional reference gamelan audio. If provided, the
                             pipeline learns the orchestration from it and applies
                             that orchestration to the source song (Reference-Based mode).
                             If None, rule-based gong + bonang layers are used.
            reference_name: Name for saving the learned orchestration profile.
        """
        def report(msg: str, pct: float):
            if progress_cb:
                progress_cb(msg, pct)

        # 1. Preprocessing (mono, resample, highpass, peak normalize, HPSS separation)
        report("Preprocessing audio & HPSS separation...", 0.08)
        clean_audio = preprocess_audio(audio_input, sr=self.sr)
        from app.audio.preprocessing import separate_hpss
        harmonic_audio, percussive_audio = separate_hpss(clean_audio, sr=self.sr)

        summary = self.analyzer.summarize(clean_audio)
        if summary.estimated_tempo_bpm and summary.estimated_tempo_bpm > 40:
            self.bpm = summary.estimated_tempo_bpm
            self.quantizer = NoteQuantizer(bpm=self.bpm)
            self.gong_layer.bpm = self.bpm
            self.bonang_layer.bpm = self.bpm

        # 2. Feature Extraction (f0 pitch from harmonic/clean, transient onsets from percussive)
        report("Extracting f0 pitch contour and percussive onsets...", 0.25)
        pitch_audio = harmonic_audio if np.std(harmonic_audio) > 1e-4 else clean_audio
        pitch_track = self.pitch_detector.detect(pitch_audio, sr=self.sr)
        onset_res = self.onset_detector.detect(
            clean_audio, sr=self.sr, use_hpss=False, percussive_audio=percussive_audio
        )

        # 3. Post-process pitch (smoothing, vibrato suppression, octave correction)
        report("Smoothing & vibrato suppression...", 0.40)
        smoothed_f0 = smooth_pitch_track(pitch_track.frequencies_hz)
        smoothed_f0 = suppress_vibrato(
            smoothed_f0,
            hop_length=self.pitch_detector.hop_length,
            sr=self.sr,
        )
        smoothed_f0 = correct_octave_errors(smoothed_f0)

        # 4. Musical Reconstruction: Note Segmentation & SourceScore
        report("Reconstructing musical score & phrases...", 0.52)
        raw_segments = segment_notes(
            times=pitch_track.times,
            pitch_hz=smoothed_f0,
            voiced_flags=pitch_track.voiced_flags,
            onset_times=onset_res.onset_times,
            merge_gap_sec=0.15,
            merge_pitch_ratio=1.06,
        )

        source_note_events: List[NoteEvent] = []
        detected_notes: List[Note] = []

        for start_t, dur, f0 in raw_segments:
            idx = int(start_t * self.sr)
            window = clean_audio[idx:idx + int(0.05 * self.sr)]
            vel = self.velocity_estimator.estimate_from_audio_window(window)

            note_event = NoteEvent(
                start=start_t,
                end=start_t + dur,
                duration=dur,
                pitch_hz=f0,
                velocity=vel,
            )
            source_note_events.append(note_event)

            detected_notes.append(
                Note(
                    start_time=start_t,
                    duration=dur,
                    pitch_hz=f0,
                    velocity=vel,
                )
            )

        # Phrase segmentation
        phrases = segment_phrases(source_note_events, pause_threshold_sec=0.65)
        source_score = SourceScore(
            notes=source_note_events,
            phrases=phrases,
            tempo_bpm=self.bpm,
            beat_times=list(onset_res.onset_times),
            total_duration=len(clean_audio) / self.sr,
        )

        # 5. Context-Aware Gamelan Sequence Mapping (Contour & Interval Optimization)
        report(f"Context-Aware Sequence Mapping to {self.scale.scale_type.capitalize()}...", 0.68)
        gamelan_events = self.context_mapper.map_sequence(
            source_notes=source_note_events,
            beat_grid=list(onset_res.onset_times),
            instrument_name=self.instrument.name,
            target_register=self.instrument.octave_shift,
        )

        # Convert to standard Note list for performance styling & renderer
        mapped_notes: List[Note] = []
        for ge in gamelan_events:
            n = Note(
                start_time=ge.start,
                duration=ge.duration,
                pitch_hz=ge.pitch_hz,
                velocity=ge.velocity,
                gamelan_note=ge.degree,
                gamelan_freq_hz=ge.pitch_hz,
                metadata=dict(ge.metadata),
            )
            mapped_notes.append(n)

        # 6. Performance Translation Layer (Physical Damping & Phrasing)
        styled_notes = self.performance.apply(mapped_notes)

        # =================================================================
        # 7. ORCHESTRATION LAYER
        # Branch A: Reference-Based (if reference_audio or pre-loaded profile)
        # Branch B: Rule-Based fallback (original behavior, backward-compat)
        # =================================================================
        orchestration_profile: Optional[OrchestrationProfile] = None
        gamelan_score_obj: Optional[GamelanScore] = None
        orchestration_mode = "rule_based"
        gong_notes: List[Note] = []
        bonang_notes: List[Note] = []

        use_reference = (reference_audio is not None) or (self._orchestration_profile is not None)

        if use_reference:
            # --- Reference-Based Orchestration ---
            report("Reference-Based Orchestration Transfer...", 0.72)

            # 7a. Learn profile from reference audio (if not pre-loaded)
            if self._orchestration_profile is not None:
                orchestration_profile = self._orchestration_profile
            elif reference_audio is not None:
                report("Learning orchestration from reference audio...", 0.73)
                orchestration_profile = self._orchestration_learner.fit(
                    reference_audio=reference_audio,
                    sr=self.sr,
                    name=reference_name,
                    laras=self.scale.scale_type,
                    auto_save=True,
                )

            # 7b. Transfer orchestration to source score
            if orchestration_profile is not None:
                report("Applying orchestration transfer...", 0.78)
                gamelan_score_obj = self._orchestration_transfer.apply(
                    source_score=source_score,
                    profile=orchestration_profile,
                )

                # 7c. Validate arrangement
                validator = ArrangementValidator(
                    scale=self.scale,
                    source_score=source_score,
                )
                val_result = validator.validate(gamelan_score_obj)
                if not val_result.passed:
                    # Log errors but continue (soft handling)
                    report(f"Orchestration validation: {val_result.error_count} errors", 0.79)

                # 7d. Convert GamelanScore tracks to Note lists for renderer
                report("Flattening orchestration score to tracks...", 0.82)
                for inst_name, g_notes in gamelan_score_obj.tracks.items():
                    track_notes = [
                        Note(
                            start_time=gn.start,
                            duration=gn.duration,
                            pitch_hz=gn.pitch_hz,
                            velocity=gn.velocity,
                            gamelan_note=gn.degree,
                            gamelan_freq_hz=gn.pitch_hz,
                            metadata={"instrument": inst_name, **gn.metadata},
                        )
                        for gn in g_notes
                    ]
                    role = gamelan_score_obj.role_of(inst_name)
                    if role in ("skeleton",):
                        styled_notes = track_notes  # Override skeleton with transferred version
                    elif role == "elaboration":
                        bonang_notes = track_notes
                    elif role == "structural":
                        gong_notes.extend(track_notes)

                orchestration_mode = "reference_based"

        else:
            # --- Rule-Based Orchestration (original behavior) ---
            if self.add_gong and styled_notes:
                report("Generating Harmonic Seleh Gong & Kenong punctuation...", 0.78)
                gong_notes = self.gong_layer.generate(styled_notes)

            if self.add_bonang and styled_notes and self.instrument.name.lower() != "bonang barung":
                report("Arranging Bonang Chime Embellishment Layer...", 0.84)
                bonang_notes = self.bonang_layer.generate(styled_notes)

        # 8. Multi-Track Score Container
        lead_track = Track(name=self.instrument.name, instrument=self.instrument.name, notes=styled_notes)
        tracks = [lead_track]
        if bonang_notes:
            bonang_track = Track(name="Bonang Barung", instrument="bonang", notes=bonang_notes)
            tracks.append(bonang_track)
        if gong_notes:
            gong_track = Track(name="Gong Ageng & Kenong", instrument="gong", notes=gong_notes)
            tracks.append(gong_track)
        score = Score(tracks=tracks, tempo_bpm=self.bpm)

        # 10. Hybrid Physical Modeling Synthesis + Acoustic Mastering (HPF + Exciter)
        report("Synthesizing Full Gamelan Ensemble with Acoustic Mastering...", 0.90)
        all_notes = styled_notes + bonang_notes + gong_notes
        input_dur = len(clean_audio) / self.sr

        rendered_audio = self.renderer.render(
            all_notes,
            legato=self.legato,
            reverb=self.reverb,
            stereo=self.stereo,
            target_duration_sec=input_dur,
        )

        # 11. Closed-Loop Fidelity Engine: DTW Alignment & Error Vector
        report("Evaluating DTW alignment & Multi-Dimensional Error Vector...", 0.96)
        fidelity_metrics = self.fidelity_engine.evaluate(
            source_audio=clean_audio,
            output_audio=rendered_audio,
            source_notes=source_note_events,
            gamelan_notes=styled_notes,
        )

        report("Complete!", 1.0)
        return PipelineResult(
            audio_output=rendered_audio,
            sample_rate=self.sr,
            detected_notes=detected_notes,
            mapped_notes=styled_notes,
            gong_notes=gong_notes,
            bonang_notes=bonang_notes,
            score=score,
            summary=summary,
            source_score=source_score,
            fidelity_metrics=fidelity_metrics,
            transposition_applied=0.0,
            orchestration_profile=orchestration_profile,
            gamelan_score=gamelan_score_obj,
            orchestration_mode=orchestration_mode,
        )

