"""OrchestrationLearner — learns orchestration behavior from reference gamelan audio.

API:
    profile = OrchestrationLearner().fit(reference_audio, sr=22050)
    profile.save("data/orchestration_profiles/mature_gamelan.json")

The learned profile captures HOW instruments cooperate, NOT WHAT music is played.
It can be reused to orchestrate any source song in the same orchestration style.

Design principle:
    REFERENCE AUDIO → teaches: HOW instruments cooperate
    SOURCE SONG     → supplies: WHAT music must be played
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

from app.orchestration.orchestration_profile import OrchestrationProfile, PhraseRule, CadenceRule
from app.orchestration.reference_extractor import ReferenceEventExtractor
from app.orchestration.activity_analyzer import InstrumentActivityAnalyzer
from app.music.source_score import NoteEvent


_DEFAULT_PROFILE_DIR = os.path.join("data", "orchestration_profiles")


class OrchestrationLearner:
    """Learns orchestration behavior from a reference gamelan audio recording.

    Usage:
        learner = OrchestrationLearner()
        profile = learner.fit(reference_audio, sr=22050, name="my_gamelan")
        profile.save("data/orchestration_profiles/my_gamelan.json")
    """

    def __init__(
        self,
        instruments: Optional[List[str]] = None,
        hop_length: int = 256,
        estimate_pitch: bool = True,
        profile_dir: str = _DEFAULT_PROFILE_DIR,
    ):
        self.instruments = instruments
        self.hop_length = hop_length
        self.estimate_pitch = estimate_pitch
        self.profile_dir = profile_dir

    def fit(
        self,
        reference_audio: np.ndarray,
        sr: int = 22050,
        name: str = "reference_gamelan",
        laras: str = "slendro",
        pathet: Optional[str] = None,
        auto_save: bool = True,
    ) -> OrchestrationProfile:
        """Learn orchestration profile from reference gamelan audio.

        Args:
            reference_audio: Mono or stereo float32 audio array
            sr: Sample rate
            name: Profile name for persistence
            laras: Scale type ('slendro' or 'pelog')
            pathet: Modal context (optional)
            auto_save: If True, save profile to data/orchestration_profiles/

        Returns:
            OrchestrationProfile with learned orchestration behavior
        """
        # Convert to mono
        if reference_audio.ndim > 1:
            audio = np.mean(reference_audio, axis=1).astype(np.float32)
        else:
            audio = reference_audio.astype(np.float32)

        total_duration = len(audio) / sr

        # --- Tempo estimation ---
        tempo_bpm = self._estimate_tempo(audio, sr)

        # --- Step 1: Extract per-instrument note events ---
        extractor = ReferenceEventExtractor(
            instruments=self.instruments,
            hop_length=self.hop_length,
            estimate_pitch=self.estimate_pitch,
        )
        events_per_instrument: Dict[str, List[NoteEvent]] = extractor.extract(audio, sr=sr)

        # --- Step 2: Compute activity statistics ---
        analyzer = InstrumentActivityAnalyzer(
            tempo_bpm=tempo_bpm,
            total_duration_sec=total_duration,
            n_sections=10,
        )
        stats = analyzer.analyze(events_per_instrument)

        # --- Step 3: Infer phrase rules from dynamic model ---
        phrase_rules = self._infer_phrase_rules(stats.dynamic_model, stats.density_model)

        # --- Step 4: Infer cadence rules from reference pitch patterns ---
        cadence_rules = self._infer_cadence_rules(events_per_instrument)

        # --- Step 5: Build state machine from phrase type sequence ---
        state_machine = self._build_state_machine(stats.dynamic_model)

        # --- Step 6: Assemble OrchestrationProfile ---
        profile = OrchestrationProfile(
            name=name,
            laras=laras,
            pathet=pathet,
            instrument_roles=stats.instrument_roles,
            density_model=stats.density_model,
            activity_model=stats.activity_model,
            register_model=stats.register_model,
            interaction_graph=stats.interaction_graph,
            state_machine=state_machine,
            phrase_rules=phrase_rules,
            cadence_rules=cadence_rules,
            dynamic_model=stats.dynamic_model,
            metadata={
                "reference_duration_sec": round(total_duration, 2),
                "reference_tempo_bpm": round(tempo_bpm, 2),
                "n_instruments_detected": len([k for k, v in events_per_instrument.items() if v]),
                "sample_rate": sr,
            },
        )

        # Auto-save to data/orchestration_profiles/
        if auto_save:
            os.makedirs(self.profile_dir, exist_ok=True)
            save_path = os.path.join(self.profile_dir, f"{name}.json")
            profile.save(save_path)

        return profile

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _estimate_tempo(self, audio: np.ndarray, sr: int) -> float:
        """Estimate BPM from reference audio."""
        if librosa is None:
            return 120.0
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, hop_length=512)
            # librosa may return array
            tempo_val = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
            return float(np.clip(tempo_val, 40.0, 240.0))
        except Exception:
            return 120.0

    def _infer_phrase_rules(
        self,
        dynamic_model: List[float],
        density_model: Dict[str, float],
    ) -> Dict[str, PhraseRule]:
        """Infer phrase-type orchestration rules from the dynamic curve.

        Maps sections of the song to phrase types based on energy trajectory.
        """
        if not dynamic_model:
            return OrchestrationProfile().phrase_rules

        dyn = np.array(dynamic_model)
        max_dyn = float(dyn.max()) if dyn.max() > 0 else 1.0

        # Determine which instruments are "core" (activity > 0.3)
        core_instruments = [k for k, v in density_model.items() if v > 0.3]

        def _rule(phrase_type: str, dyn_level: float) -> PhraseRule:
            # Scale density by relative dynamic
            return PhraseRule(
                phrase_type=phrase_type,
                density_multiplier=round(float(dyn_level / max(max_dyn, 0.01)), 2),
                active_instruments=core_instruments,
                elaboration_enabled=(dyn_level > 0.55),
                interlocking_enabled=(dyn_level > 0.65),
                dynamic_level=round(float(dyn_level), 2),
                gong_cadence=True,
            )

        intro_dyn = float(np.mean(dyn[:2])) if len(dyn) >= 2 else 0.5
        outro_dyn = float(np.mean(dyn[-2:])) if len(dyn) >= 2 else 0.6
        peak_dyn = float(dyn.max())
        mid_dyn = float(np.mean(dyn[3:7])) if len(dyn) >= 7 else float(np.mean(dyn))

        return {
            "intro": _rule("intro", intro_dyn * 0.7),
            "verse": _rule("verse", mid_dyn * 0.85),
            "build": _rule("build", (mid_dyn + peak_dyn) / 2),
            "climax": _rule("climax", peak_dyn),
            "break": _rule("break", intro_dyn * 0.5),
            "outro": _rule("outro", outro_dyn),
        }

    def _infer_cadence_rules(
        self,
        events_per_instrument: Dict[str, List[NoteEvent]],
    ) -> List[CadenceRule]:
        """Infer cadential patterns from structural instrument activity.

        Uses gong/kenong onset positions to identify seleh points.
        """
        cadence_rules: List[CadenceRule] = []

        # Find the most common "structural" pitches
        structural = ["gong", "kenong", "kempul"]
        structural_pitches: List[float] = []
        for inst in structural:
            for note in events_per_instrument.get(inst, []):
                if note.pitch_hz > 20.0:
                    structural_pitches.append(note.pitch_hz)

        if structural_pitches:
            # Use most common pitch cluster as the "seleh" target
            pitches_arr = np.array(structural_pitches)
            # Cluster into up to 3 groups (simplified)
            unique_approx = []
            for p in sorted(pitches_arr):
                if not unique_approx or abs(p - unique_approx[-1]) > 30.0:
                    unique_approx.append(p)

            for i, hz in enumerate(unique_approx[:3]):
                cadence_rules.append(CadenceRule(
                    final_degree=str(i + 1),     # Simplified degree label
                    approach_degrees=[str(j + 1) for j in range(max(0, i - 1), min(5, i + 2)) if j != i],
                    seleh_instruments=["gong", "kenong"],
                    gong_weight=1.0 if i == 0 else 0.6,
                ))

        # Fallback default
        if not cadence_rules:
            cadence_rules = [
                CadenceRule("6", approach_degrees=["1", "5"], seleh_instruments=["gong", "kenong"], gong_weight=1.0),
                CadenceRule("2", approach_degrees=["3", "1"], seleh_instruments=["kenong"], gong_weight=0.5),
            ]

        return cadence_rules

    def _build_state_machine(self, dynamic_model: List[float]) -> Dict[str, Dict[str, float]]:
        """Build a phrase state machine from the dynamic energy curve.

        Identifies transitions: intro → verse → build → climax → outro
        based on the energy trajectory.
        """
        if not dynamic_model:
            return OrchestrationProfile().state_machine

        dyn = np.array(dynamic_model)
        n = len(dyn)

        # Identify peak and where it occurs
        peak_idx = int(np.argmax(dyn))
        peak_frac = peak_idx / max(n - 1, 1)

        if peak_frac < 0.35:
            # Early peak: quick climax
            return {
                "intro": {"verse": 1.0},
                "verse": {"climax": 0.6, "build": 0.4},
                "build": {"climax": 1.0},
                "climax": {"outro": 1.0},
                "outro": {"end": 1.0},
            }
        elif peak_frac > 0.70:
            # Late peak: build towards end
            return {
                "intro": {"verse": 1.0},
                "verse": {"verse": 0.5, "build": 0.5},
                "build": {"climax": 1.0},
                "climax": {"outro": 1.0},
                "outro": {"end": 1.0},
            }
        else:
            # Mid peak: standard ABA structure
            return {
                "intro": {"verse": 0.8, "build": 0.2},
                "verse": {"build": 0.5, "climax": 0.3, "verse": 0.2},
                "build": {"climax": 0.7, "verse": 0.3},
                "climax": {"verse": 0.5, "outro": 0.5},
                "outro": {"end": 1.0},
            }
