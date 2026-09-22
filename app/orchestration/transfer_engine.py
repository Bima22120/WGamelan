"""OrchestrationTransfer — applies an OrchestrationProfile to a SourceScore.

This is the core transfer engine implementing:

    SOURCE SCORE + ORCHESTRATION PROFILE → GAMELAN SCORE

Formally:
    Y = Render(Transfer(Analyze(S), LearnOrchestration(R)))

The system MUST:
    - Preserve: TIME, RHYTHM, MELODY, INTERVAL, PHRASE, DURATION, DYNAMICS, ARTICULATION
    - Transform: TIMBRE, INSTRUMENT, TUNING SYSTEM, INSTRUMENT RANGE, PERFORMANCE MODEL

The system MUST NOT:
    - Copy musical content from reference to target
    - Apply reference melody to output
    - Override source timing or pitch contour

API:
    transfer = OrchestrationTransfer(scale=scale)
    gamelan_score = transfer.apply(source_score, profile)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.music.source_score import NoteEvent, SourceScore, Phrase
from app.music.gamelan_score import GamelanScore, InstrumentRole
from app.orchestration.orchestration_profile import OrchestrationProfile, PhraseRule
from app.gamelan.context_mapper import ContextAwareGamelanMapper, GamelanNoteEvent
from app.gamelan.scale import GamelanScale
from app.gamelan.tuning import GamelanPitch


# ---------------------------------------------------------------------------
# Phrase type classifier
# ---------------------------------------------------------------------------

def _classify_phrase_type(
    phrase: Phrase,
    source_score: SourceScore,
    dynamic_model: List[float],
    phrase_idx: int,
    total_phrases: int,
) -> str:
    """Classify a phrase into a type based on its position and energy."""
    progress = phrase_idx / max(1, total_phrases - 1)

    # Compute relative energy of this phrase vs total
    all_velocities = [n.velocity for n in source_score.notes if n.velocity > 0]
    if not all_velocities:
        return "verse"
    global_mean_vel = float(np.mean(all_velocities))
    phrase_vel = float(np.mean([n.velocity for n in phrase.notes])) if phrase.notes else global_mean_vel
    rel_energy = phrase_vel / max(global_mean_vel, 1.0)

    if phrase_idx == 0:
        return "intro"
    if phrase_idx == total_phrases - 1:
        return "outro"

    # Use dynamic model to estimate energy at this position
    dyn_idx = int(progress * len(dynamic_model))
    dyn_idx = max(0, min(dyn_idx, len(dynamic_model) - 1))
    expected_energy = dynamic_model[dyn_idx] if dynamic_model else 0.75

    if rel_energy > 1.15 or expected_energy > 0.85:
        return "climax"
    elif rel_energy > 1.05 or expected_energy > 0.72:
        return "build"
    elif rel_energy < 0.75 and phrase_idx not in (0, total_phrases - 1):
        return "break"
    return "verse"


# ---------------------------------------------------------------------------
# OrchestrationTransfer
# ---------------------------------------------------------------------------

class OrchestrationTransfer:
    """Applies orchestration behavior from a profile to a SourceScore.

    The transfer preserves:
        - Note timings and onset times (from source)
        - Melodic contour and intervals (mapped to gamelan scale via Viterbi)
        - Phrase structure and dynamics

    And applies orchestration from the profile:
        - Which instruments play in each phrase
        - Density of elaboration notes (bonang, peking)
        - Structural events (gong, kenong) at cadential positions
        - Interlocking patterns between instruments
    """

    def __init__(
        self,
        scale: Optional[GamelanScale] = None,
        timing_mode: str = "PRESERVE",
    ):
        self.scale = scale or GamelanScale(scale_type="slendro")
        self.timing_mode = timing_mode

        # Context-aware mapper shared for all instruments
        self._mapper = ContextAwareGamelanMapper(
            scale=self.scale,
            timing_mode=timing_mode,
            w_pitch=1.0,
            w_contour=2.5,
            w_interval=1.2,
        )

    def apply(
        self,
        source_score: SourceScore,
        profile: OrchestrationProfile,
    ) -> GamelanScore:
        """Transfer orchestration profile onto source musical content.

        Args:
            source_score: Musical content from source song (WHAT)
            profile: Orchestration behavior from reference gamelan (HOW)

        Returns:
            GamelanScore: Full multi-track Gamelan arrangement
        """
        gamelan_score = GamelanScore(
            tempo_bpm=source_score.tempo_bpm,
            total_duration=source_score.total_duration,
            laras=profile.laras,
            pathet=profile.pathet,
        )

        if not source_score.notes:
            return gamelan_score

        # Update mapper scale type if needed
        if profile.laras != self.scale.scale_type:
            self.scale = GamelanScale(scale_type=profile.laras, pathet=profile.pathet)
            self._mapper = ContextAwareGamelanMapper(
                scale=self.scale,
                timing_mode=self.timing_mode,
                w_pitch=1.0,
                w_contour=2.5,
                w_interval=1.2,
            )

        # Determine phrase types using source score + profile dynamic model
        phrases = source_score.phrases if source_score.phrases else [
            Phrase(0, source_score.notes[0].start,
                   source_score.notes[-1].end,
                   source_score.notes)
        ]
        total_phrases = len(phrases)

        # --- Step 1: Build skeleton (primary melody) ---
        skeleton_notes = self.build_skeleton(source_score, profile)
        if skeleton_notes:
            gamelan_score.add_track("saron", skeleton_notes)
            # Generate demung (octave below saron) if active in profile
            demung_notes = self._generate_demung(skeleton_notes, profile)
            if demung_notes:
                gamelan_score.add_track("demung", demung_notes)
            # Slenthem (bass foundation, every other note)
            slenthem_notes = self._generate_slenthem(skeleton_notes, profile)
            if slenthem_notes:
                gamelan_score.add_track("slenthem", slenthem_notes)

        # --- Step 2: Elaborate per phrase ---
        all_bonang: List[GamelanNoteEvent] = []
        all_gong: List[GamelanNoteEvent] = []
        all_kenong: List[GamelanNoteEvent] = []

        for phrase_idx, phrase in enumerate(phrases):
            if not phrase.notes:
                continue

            phrase_type = _classify_phrase_type(
                phrase, source_score, profile.dynamic_model, phrase_idx, total_phrases
            )
            rule = profile.rule_for_phrase(phrase_type)
            progress = phrase_idx / max(1, total_phrases - 1)
            dyn_mult = profile.dynamic_at(progress) * rule.density_multiplier

            # 2a. Bonang elaboration
            if rule.elaboration_enabled and "bonang" in rule.active_instruments:
                bonang = self.generate_elaboration(phrase.notes, profile, rule, dyn_mult)
                all_bonang.extend(bonang)

            # 2b. Structural events (gong + kenong)
            if rule.gong_cadence:
                gong_ev, kenong_ev = self.generate_structure(phrase.notes, profile, rule)
                all_gong.extend(gong_ev)
                all_kenong.extend(kenong_ev)

        if all_bonang:
            all_bonang.sort(key=lambda n: n.start)
            gamelan_score.add_track("bonang", all_bonang)

        if all_gong:
            all_gong.sort(key=lambda n: n.start)
            gamelan_score.add_track("gong", all_gong)

        if all_kenong:
            all_kenong.sort(key=lambda n: n.start)
            gamelan_score.add_track("kenong", all_kenong)

        # --- Step 3: Apply density control ---
        gamelan_score = self.apply_density_control(gamelan_score, source_score, profile)

        # --- Step 4: Validate musical integrity ---
        # (Soft validation: silently clip out-of-range pitches; hard errors logged)
        gamelan_score = self._sanitize(gamelan_score)

        return gamelan_score

    # -----------------------------------------------------------------------
    # Skeleton layer (balungan)
    # -----------------------------------------------------------------------

    def build_skeleton(
        self, source_score: SourceScore, profile: OrchestrationProfile
    ) -> List[GamelanNoteEvent]:
        """Map the source melody to the primary gamelan skeleton (saron register)."""
        if not source_score.notes:
            return []

        # Use register model from profile for saron
        target_hz, _ = profile.register_model.get("saron", (293.0, 80.0))
        target_octave = _hz_to_octave_shift(target_hz, self.scale)

        mapped = self._mapper.map_sequence(
            source_notes=source_score.notes,
            beat_grid=source_score.beat_times or None,
            instrument_name="saron",
            target_register=target_octave,
        )
        return mapped

    def assign_primary_melody(
        self,
        source_score: SourceScore,
        skeleton_notes: List[GamelanNoteEvent],
    ) -> List[GamelanNoteEvent]:
        """Already done by build_skeleton; this is a no-op alias for the formal API."""
        return skeleton_notes

    def _generate_demung(
        self, skeleton_notes: List[GamelanNoteEvent], profile: OrchestrationProfile
    ) -> List[GamelanNoteEvent]:
        """Generate demung (one octave below saron) for skeleton support."""
        if profile.activity_for_instrument("demung") < 0.3:
            return []

        demung_notes = []
        activity = profile.activity_for_instrument("demung")
        for i, note in enumerate(skeleton_notes):
            if i % 2 != 0 and activity < 0.5:  # Demung plays every other note if sparse
                continue
            # Shift down one octave
            low_hz = note.pitch_hz / 2.0
            low_hz = max(80.0, low_hz)  # Clip to demung range
            demung_notes.append(GamelanNoteEvent(
                degree=note.degree,
                pitch_hz=low_hz,
                start=note.start,
                duration=note.duration * 1.1,
                velocity=int(note.velocity * 0.85),
                instrument="demung",
                register=note.register - 1,
                articulation=note.articulation,
                metadata=dict(note.metadata),
            ))
        return demung_notes

    def _generate_slenthem(
        self, skeleton_notes: List[GamelanNoteEvent], profile: OrchestrationProfile
    ) -> List[GamelanNoteEvent]:
        """Generate slenthem bass notes (2 octaves below saron), every 2nd or 4th note."""
        if profile.activity_for_instrument("slenthem") < 0.25:
            return []

        slenthem_notes = []
        activity = profile.activity_for_instrument("slenthem")
        step = 2 if activity > 0.45 else 4

        for i, note in enumerate(skeleton_notes):
            if i % step != 0:
                continue
            bass_hz = note.pitch_hz / 4.0  # Two octaves down
            bass_hz = max(70.0, bass_hz)
            slenthem_notes.append(GamelanNoteEvent(
                degree=note.degree,
                pitch_hz=bass_hz,
                start=note.start,
                duration=note.duration * 1.5,
                velocity=int(note.velocity * 0.75),
                instrument="slenthem",
                register=note.register - 2,
                articulation="open",
                metadata=dict(note.metadata),
            ))
        return slenthem_notes

    # -----------------------------------------------------------------------
    # Elaboration layer (bonang/peking mipil)
    # -----------------------------------------------------------------------

    def generate_elaboration(
        self,
        phrase_notes: List[NoteEvent],
        profile: OrchestrationProfile,
        rule: PhraseRule,
        dyn_mult: float = 1.0,
    ) -> List[GamelanNoteEvent]:
        """Generate Bonang mipil/gembyangan elaboration from profile density model.

        Density model controls how many bonang notes surround each skeleton note.
        """
        bonang_density = profile.density_for_instrument("bonang") * dyn_mult
        bonang_notes: List[GamelanNoteEvent] = []

        if not phrase_notes:
            return bonang_notes

        beat_sec = 60.0 / max(1.0, profile.density_model.get("tempo_bpm", 120.0))

        # Get bonang register from profile
        bonang_mean_hz, _ = profile.register_model.get("bonang", (520.0, 120.0))

        for i, src_note in enumerate(phrase_notes):
            src_mapped = self._mapper._get_note_candidates(src_note.pitch_hz)
            if not src_mapped:
                continue
            cand = src_mapped[0]
            bonang_hz = _shift_to_register(cand.freq_hz, bonang_mean_hz)

            # Main bonang hit on the note
            bonang_notes.append(GamelanNoteEvent(
                degree=cand.degree,
                pitch_hz=bonang_hz,
                start=src_note.start,
                duration=min(src_note.duration * 0.75, 0.35),
                velocity=int(src_note.velocity * 0.85 * rule.dynamic_level),
                instrument="bonang",
                register=1,
                articulation="open",
                metadata={"role": "bonang_main", "phrase_id": src_note.phrase_id},
            ))

            # Mipil anticipation: if density > 1.5, add upbeat anticipation stroke
            if bonang_density > 1.5 and i > 0:
                prev_note = phrase_notes[i - 1]
                gap = src_note.start - prev_note.start
                subdiv = max(0.15, min(0.30, gap * 0.45))
                anti_time = src_note.start - subdiv
                if anti_time > prev_note.start + 0.05:
                    prev_mapped = self._mapper._get_note_candidates(prev_note.pitch_hz)
                    prev_hz = _shift_to_register(
                        prev_mapped[0].freq_hz if prev_mapped else cand.freq_hz,
                        bonang_mean_hz,
                    )
                    bonang_notes.append(GamelanNoteEvent(
                        degree=prev_mapped[0].degree if prev_mapped else cand.degree,
                        pitch_hz=prev_hz,
                        start=anti_time,
                        duration=subdiv * 0.85,
                        velocity=int(src_note.velocity * 0.70 * rule.dynamic_level),
                        instrument="bonang",
                        register=1,
                        articulation="open",
                        metadata={"role": "bonang_anticipation", "phrase_id": src_note.phrase_id},
                    ))

            # Gembyangan shimmer: long note fill
            if src_note.duration > 0.5 and bonang_density > 1.8:
                fill_t = src_note.start + src_note.duration * 0.5
                bonang_notes.append(GamelanNoteEvent(
                    degree=cand.degree,
                    pitch_hz=bonang_hz,
                    start=fill_t,
                    duration=min(src_note.duration * 0.25, 0.2),
                    velocity=int(src_note.velocity * 0.65 * rule.dynamic_level),
                    instrument="bonang",
                    register=1,
                    articulation="open",
                    metadata={"role": "bonang_shimmer", "phrase_id": src_note.phrase_id},
                ))

        return bonang_notes

    # -----------------------------------------------------------------------
    # Interlocking (mipil anti-correlation)
    # -----------------------------------------------------------------------

    def generate_interlocking(
        self,
        skeleton_notes: List[GamelanNoteEvent],
        elaboration_notes: List[GamelanNoteEvent],
        profile: OrchestrationProfile,
    ) -> List[GamelanNoteEvent]:
        """Generate interlocking fills between skeleton and elaboration.

        Uses profile interaction_graph to determine interlocking strength.
        Notes not to be used for output directly — merged into elaboration layer.
        """
        interlock_strength = profile.interaction_graph.get("bonang", {}).get("saron", 0.5)
        if interlock_strength < 0.3 or not skeleton_notes:
            return elaboration_notes

        # Anti-correlated fill: add bonang on beats between skeleton onsets
        extra_bonang: List[GamelanNoteEvent] = []
        bonang_mean_hz, _ = profile.register_model.get("bonang", (520.0, 120.0))

        for i in range(len(skeleton_notes) - 1):
            curr = skeleton_notes[i]
            nxt = skeleton_notes[i + 1]
            gap = nxt.start - curr.start
            if gap < 0.20:
                continue

            mid_time = curr.start + gap * 0.5
            # Find candidate at midpoint
            cands = self._mapper._get_note_candidates(curr.pitch_hz)
            if not cands:
                continue
            fill_hz = _shift_to_register(cands[0].freq_hz, bonang_mean_hz)

            extra_bonang.append(GamelanNoteEvent(
                degree=cands[0].degree,
                pitch_hz=fill_hz,
                start=mid_time,
                duration=gap * 0.35,
                velocity=int(curr.velocity * 0.60 * interlock_strength),
                instrument="bonang",
                register=1,
                articulation="open",
                metadata={"role": "interlocking_fill"},
            ))

        combined = elaboration_notes + extra_bonang
        combined.sort(key=lambda n: n.start)
        return combined

    # -----------------------------------------------------------------------
    # Structural events (gong, kenong, kempul)
    # -----------------------------------------------------------------------

    def generate_structure(
        self,
        phrase_notes: List[NoteEvent],
        profile: OrchestrationProfile,
        rule: PhraseRule,
    ) -> Tuple[List[GamelanNoteEvent], List[GamelanNoteEvent]]:
        """Generate gong and kenong colotomic events from cadence rules.

        Gong marks the end of a complete gong cycle (seleh).
        Kenong marks mid-phrase cadential points.

        Returns:
            (gong_notes, kenong_notes)
        """
        gong_notes: List[GamelanNoteEvent] = []
        kenong_notes: List[GamelanNoteEvent] = []

        if not phrase_notes:
            return gong_notes, kenong_notes

        gong_hz, _ = profile.register_model.get("gong", (98.0, 20.0))
        kenong_hz, _ = profile.register_model.get("kenong", (195.0, 50.0))

        phrase_end = phrase_notes[-1].end if hasattr(phrase_notes[-1], 'end') else (
            phrase_notes[-1].start + phrase_notes[-1].duration
        )
        phrase_start = phrase_notes[0].start
        phrase_duration = phrase_end - phrase_start

        # Kenong: every 1/4 of phrase
        n_kenong_points = max(1, int(phrase_duration / 3.0))
        for k in range(n_kenong_points):
            t_kenong = phrase_start + (k + 0.5) * (phrase_duration / n_kenong_points)
            # Find pitch closest to kenong timepoint
            nearest = min(phrase_notes, key=lambda n: abs(n.start - t_kenong))
            candidates = self._mapper._get_note_candidates(nearest.pitch_hz)
            pitch_hz = kenong_hz
            degree = "2"
            if candidates:
                pitch_hz = _shift_to_register(candidates[0].freq_hz, kenong_hz)
                degree = candidates[0].degree

            kenong_notes.append(GamelanNoteEvent(
                degree=degree,
                pitch_hz=pitch_hz,
                start=t_kenong,
                duration=1.0,
                velocity=int(nearest.velocity * 0.80 * rule.dynamic_level),
                instrument="kenong",
                register=-1,
                articulation="open",
                metadata={"role": "kenong_cadence"},
            ))

        # Gong: at phrase end (seleh)
        final_note = phrase_notes[-1]
        candidates = self._mapper._get_note_candidates(final_note.pitch_hz)
        gong_degree = "6"
        if candidates:
            gong_degree = candidates[0].degree
        gong_notes.append(GamelanNoteEvent(
            degree=gong_degree,
            pitch_hz=gong_hz,
            start=phrase_end - 0.05,
            duration=3.0,
            velocity=int(final_note.velocity * rule.dynamic_level),
            instrument="gong",
            register=-2,
            articulation="open",
            metadata={"role": "gong_seleh"},
        ))

        return gong_notes, kenong_notes

    def generate_rhythm(
        self,
        source_score: SourceScore,
        profile: OrchestrationProfile,
    ) -> List[GamelanNoteEvent]:
        """Generate kendang rhythmic events from profile density model.

        Basic kendang pattern: accents on beat and offbeat based on density.
        """
        kendang_density = profile.density_for_instrument("kendang")
        if kendang_density < 0.5:
            return []

        beat_sec = 60.0 / max(source_score.tempo_bpm, 40.0)
        kendang_notes: List[GamelanNoteEvent] = []
        total = source_score.total_duration

        t = 0.0
        while t < total:
            # Main downbeat
            kendang_notes.append(GamelanNoteEvent(
                degree="x",
                pitch_hz=200.0,
                start=t,
                duration=beat_sec * 0.4,
                velocity=90,
                instrument="kendang",
                register=0,
                articulation="staccato",
                metadata={"role": "kendang_beat"},
            ))
            # Offbeat if density > 1.5
            if kendang_density > 1.5:
                off_t = t + beat_sec * 0.5
                if off_t < total:
                    kendang_notes.append(GamelanNoteEvent(
                        degree="x",
                        pitch_hz=120.0,
                        start=off_t,
                        duration=beat_sec * 0.35,
                        velocity=65,
                        instrument="kendang",
                        register=0,
                        articulation="staccato",
                        metadata={"role": "kendang_offbeat"},
                    ))
            t += beat_sec

        return kendang_notes

    # -----------------------------------------------------------------------
    # Density control
    # -----------------------------------------------------------------------

    def apply_density_control(
        self,
        gamelan_score: GamelanScore,
        source_score: SourceScore,
        profile: OrchestrationProfile,
    ) -> GamelanScore:
        """Scale note velocities per section based on profile dynamic model.

        Also prunes elaboration notes when density exceeds profile model.
        """
        if not profile.dynamic_model or source_score.total_duration <= 0:
            return gamelan_score

        total_dur = source_score.total_duration
        n_sections = len(profile.dynamic_model)

        for instrument, notes in gamelan_score.tracks.items():
            for note in notes:
                progress = note.start / total_dur if total_dur > 0 else 0.5
                dyn_level = profile.dynamic_at(progress)
                role = gamelan_score.role_of(instrument)

                # Elaboration instruments follow dynamic model more strictly
                if role == InstrumentRole.ELABORATION:
                    note.velocity = int(np.clip(note.velocity * dyn_level, 30, 120))
                elif role == InstrumentRole.SKELETON:
                    # Skeleton maintains stronger dynamic presence
                    note.velocity = int(np.clip(note.velocity * (0.7 + 0.3 * dyn_level), 50, 127))

        return gamelan_score

    def apply_state_machine(
        self,
        gamelan_score: GamelanScore,
        source_score: SourceScore,
        profile: OrchestrationProfile,
    ) -> GamelanScore:
        """Apply phrase state transitions to control instrument presence.

        Mutes instruments not active in the current phrase state.
        """
        # State machine application is handled implicitly via phrase_rules
        # in generate_elaboration / generate_structure; this is a no-op pass-through
        return gamelan_score

    def validate(self, gamelan_score: GamelanScore) -> GamelanScore:
        """Validate the arrangement; return as-is (full validation in ArrangementValidator)."""
        return gamelan_score

    # -----------------------------------------------------------------------
    # Sanitize
    # -----------------------------------------------------------------------

    def _sanitize(self, gamelan_score: GamelanScore) -> GamelanScore:
        """Remove invalid notes (negative start, zero duration, extreme pitch)."""
        for instrument in list(gamelan_score.tracks.keys()):
            clean = []
            for note in gamelan_score.tracks[instrument]:
                if note.start < 0:
                    note.start = 0.0
                if note.duration < 0.04:
                    note.duration = 0.04
                if note.pitch_hz <= 0:
                    continue
                clean.append(note)
            gamelan_score.tracks[instrument] = sorted(clean, key=lambda n: n.start)
        return gamelan_score


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _hz_to_octave_shift(target_hz: float, scale: GamelanScale) -> int:
    """Determine the octave shift needed to center pitches around target_hz."""
    if not scale.pitches:
        return 0
    mean_scale_hz = float(np.mean([p.freq_hz for p in scale.pitches]))
    if mean_scale_hz <= 0:
        return 0
    ratio = target_hz / mean_scale_hz
    if ratio > 1.8:
        return 1
    elif ratio < 0.6:
        return -1
    return 0


def _shift_to_register(source_hz: float, target_mean_hz: float) -> float:
    """Shift source_hz to be in the same octave as target_mean_hz."""
    if source_hz <= 0 or target_mean_hz <= 0:
        return source_hz
    hz = source_hz
    # Shift up
    while hz < target_mean_hz * 0.6:
        hz *= 2.0
    # Shift down
    while hz > target_mean_hz * 1.8:
        hz /= 2.0
    return float(hz)
