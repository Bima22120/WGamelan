"""OrchestrationProfile — reusable orchestration artifact learned from reference audio.

Encodes HOW a gamelan ensemble accompanies music, separated from WHAT music
is being played. Can be persisted as JSON and reloaded.

Usage:
    profile = OrchestrationLearner().fit(reference_audio, sr=22050)
    profile.save("data/orchestration_profiles/mature_gamelan.json")

    # Later:
    profile = OrchestrationProfile.load("data/orchestration_profiles/mature_gamelan.json")
    gamelan_score = OrchestrationTransfer().apply(source_score, profile)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Phrase-level orchestration rules
# ---------------------------------------------------------------------------

@dataclass
class PhraseRule:
    """Orchestration behavior for a given musical phrase type."""
    phrase_type: str                              # 'intro', 'verse', 'build', 'climax', 'break', 'outro'
    density_multiplier: float = 1.0              # Scale overall note density
    active_instruments: List[str] = field(default_factory=list)  # Which instruments play
    elaboration_enabled: bool = True             # Bonang/peking elaboration on
    interlocking_enabled: bool = True            # Interlocking patterns on
    dynamic_level: float = 0.8                   # Velocity multiplier [0, 1]
    gong_cadence: bool = False                   # Gong punctuation at phrase end


@dataclass
class CadenceRule:
    """Rule for cadential (seleh) patterns at phrase endings."""
    final_degree: str                            # Target ending scale degree
    approach_degrees: List[str] = field(default_factory=list)  # Penultimate degrees allowed
    seleh_instruments: List[str] = field(default_factory=list)  # Instruments that mark seleh
    gong_weight: float = 1.0                     # How strongly to signal gong cadence


@dataclass
class OrnamentRule:
    """Rule for melodic ornamentation (cengkok, wiled)."""
    pitch_degree: str                            # Degree that triggers ornament
    ornament_type: str = "none"                  # 'gregel', 'mbesut', 'none'
    probability: float = 0.0                     # How often to apply


# ---------------------------------------------------------------------------
# OrchestrationProfile
# ---------------------------------------------------------------------------

@dataclass
class OrchestrationProfile:
    """Reusable artifact encoding the orchestration behavior of a reference gamelan.

    This is the OUTPUT of OrchestrationLearner.fit() and the INPUT to
    OrchestrationTransfer.apply(). It encodes HOW instruments cooperate,
    NOT WHAT musical content is played.

    Attributes:
        name: Human-readable identifier
        laras: Scale type learned from reference ('slendro' or 'pelog')
        instrument_roles: Mapping instrument → role ('skeleton', 'elaboration', etc.)
        density_model: Mean notes-per-beat per instrument
        activity_model: Per-instrument activity curve normalized to [0, 1] per section
        register_model: (mean_hz, std_hz) per instrument
        interaction_graph: Correlation matrix between instruments
        state_machine: Phrase state transitions {from_state: {to_state: probability}}
        phrase_rules: Per phrase-type orchestration rules
        cadence_rules: List of cadential patterns
        ornament_rules: List of ornamentation rules
        dynamic_model: Density curve across song progression
        metadata: Arbitrary extra info
    """
    name: str = "default_profile"
    laras: str = "slendro"
    pathet: Optional[str] = None

    instrument_roles: Dict[str, str] = field(default_factory=lambda: {
        "saron": "skeleton",
        "demung": "skeleton",
        "slenthem": "skeleton",
        "bonang": "elaboration",
        "peking": "elaboration",
        "kendang": "rhythm",
        "gong": "structural",
        "kenong": "structural",
        "kempul": "structural",
    })

    # Learned density: notes per beat per instrument (from reference analysis)
    density_model: Dict[str, float] = field(default_factory=lambda: {
        "saron": 1.0,
        "demung": 0.5,
        "slenthem": 0.5,
        "bonang": 2.0,
        "peking": 2.5,
        "kendang": 1.5,
        "gong": 0.08,
        "kenong": 0.25,
        "kempul": 0.16,
    })

    # Activity model: per-instrument fraction of beats active (from reference)
    activity_model: Dict[str, float] = field(default_factory=lambda: {
        "saron": 0.90,
        "demung": 0.70,
        "slenthem": 0.55,
        "bonang": 0.80,
        "peking": 0.75,
        "kendang": 0.65,
        "gong": 0.05,
        "kenong": 0.18,
        "kempul": 0.12,
    })

    # Register model: (mean_hz, std_hz) per instrument
    register_model: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "saron": (293.0, 80.0),
        "demung": (195.0, 60.0),
        "slenthem": (147.0, 40.0),
        "bonang": (520.0, 120.0),
        "peking": (587.0, 130.0),
        "gong": (98.0, 20.0),
        "kenong": (195.0, 50.0),
        "kempul": (146.0, 30.0),
    })

    # Interaction graph: anti-correlation between instrument pairs (0=none, 1=strong interlocking)
    interaction_graph: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "bonang": {"saron": 0.6, "peking": 0.3},
        "peking": {"bonang": 0.3, "saron": 0.8},
        "saron": {"bonang": 0.6, "demung": 0.2},
    })

    # State machine: phrase-type → transition probabilities
    state_machine: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "intro": {"verse": 0.8, "build": 0.2},
        "verse": {"verse": 0.5, "build": 0.3, "chorus": 0.2},
        "build": {"chorus": 0.7, "climax": 0.3},
        "chorus": {"verse": 0.4, "build": 0.3, "outro": 0.3},
        "climax": {"outro": 0.6, "verse": 0.4},
        "outro": {"end": 1.0},
    })

    # Phrase-level orchestration rules
    phrase_rules: Dict[str, PhraseRule] = field(default_factory=lambda: {
        "intro": PhraseRule("intro", density_multiplier=0.6, dynamic_level=0.55,
                            active_instruments=["saron", "slenthem", "gong"],
                            elaboration_enabled=False, interlocking_enabled=False, gong_cadence=True),
        "verse": PhraseRule("verse", density_multiplier=1.0, dynamic_level=0.75,
                            active_instruments=["saron", "demung", "bonang", "kenong", "gong"],
                            elaboration_enabled=True, interlocking_enabled=True, gong_cadence=True),
        "build": PhraseRule("build", density_multiplier=1.3, dynamic_level=0.85,
                            active_instruments=["saron", "demung", "bonang", "peking", "kenong", "gong"],
                            elaboration_enabled=True, interlocking_enabled=True, gong_cadence=True),
        "climax": PhraseRule("climax", density_multiplier=1.5, dynamic_level=0.95,
                             active_instruments=["saron", "demung", "slenthem", "bonang", "peking", "kendang", "gong", "kenong"],
                             elaboration_enabled=True, interlocking_enabled=True, gong_cadence=True),
        "break": PhraseRule("break", density_multiplier=0.4, dynamic_level=0.5,
                            active_instruments=["saron", "gong"],
                            elaboration_enabled=False, interlocking_enabled=False, gong_cadence=True),
        "outro": PhraseRule("outro", density_multiplier=0.7, dynamic_level=0.65,
                            active_instruments=["saron", "slenthem", "bonang", "gong"],
                            elaboration_enabled=True, interlocking_enabled=False, gong_cadence=True),
    })

    # Cadential patterns (learned from reference)
    cadence_rules: List[CadenceRule] = field(default_factory=lambda: [
        CadenceRule("6", approach_degrees=["1", "5"], seleh_instruments=["gong", "kenong"], gong_weight=1.0),
        CadenceRule("2", approach_degrees=["3", "1"], seleh_instruments=["kenong", "kempul"], gong_weight=0.6),
        CadenceRule("5", approach_degrees=["6", "3"], seleh_instruments=["kenong"], gong_weight=0.5),
    ])

    # Ornamentation rules
    ornament_rules: List[OrnamentRule] = field(default_factory=list)

    # Dynamic model: activity multiplier per 10% progression through song
    dynamic_model: List[float] = field(default_factory=lambda: [
        0.55, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95, 0.90, 0.80, 0.70
    ])

    metadata: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist OrchestrationProfile to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "name": self.name,
            "laras": self.laras,
            "pathet": self.pathet,
            "instrument_roles": self.instrument_roles,
            "density_model": self.density_model,
            "activity_model": self.activity_model,
            "register_model": {k: list(v) for k, v in self.register_model.items()},
            "interaction_graph": self.interaction_graph,
            "state_machine": self.state_machine,
            "phrase_rules": {
                k: {
                    "phrase_type": v.phrase_type,
                    "density_multiplier": v.density_multiplier,
                    "active_instruments": v.active_instruments,
                    "elaboration_enabled": v.elaboration_enabled,
                    "interlocking_enabled": v.interlocking_enabled,
                    "dynamic_level": v.dynamic_level,
                    "gong_cadence": v.gong_cadence,
                }
                for k, v in self.phrase_rules.items()
            },
            "cadence_rules": [
                {
                    "final_degree": r.final_degree,
                    "approach_degrees": r.approach_degrees,
                    "seleh_instruments": r.seleh_instruments,
                    "gong_weight": r.gong_weight,
                }
                for r in self.cadence_rules
            ],
            "dynamic_model": self.dynamic_model,
            "metadata": self.metadata,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "OrchestrationProfile":
        """Load an OrchestrationProfile from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        profile = cls(
            name=data.get("name", "loaded_profile"),
            laras=data.get("laras", "slendro"),
            pathet=data.get("pathet"),
            instrument_roles=data.get("instrument_roles", {}),
            density_model=data.get("density_model", {}),
            activity_model=data.get("activity_model", {}),
            register_model={k: tuple(v) for k, v in data.get("register_model", {}).items()},
            interaction_graph=data.get("interaction_graph", {}),
            state_machine=data.get("state_machine", {}),
            dynamic_model=data.get("dynamic_model", []),
            metadata=data.get("metadata", {}),
        )

        # Reconstruct phrase_rules
        for k, v in data.get("phrase_rules", {}).items():
            profile.phrase_rules[k] = PhraseRule(**v)

        # Reconstruct cadence_rules
        profile.cadence_rules = [
            CadenceRule(**r) for r in data.get("cadence_rules", [])
        ]

        return profile

    def rule_for_phrase(self, phrase_type: str) -> PhraseRule:
        """Return PhraseRule for given type, falling back to 'verse' default."""
        return self.phrase_rules.get(phrase_type, self.phrase_rules.get("verse", PhraseRule("verse")))

    def density_for_instrument(self, instrument: str) -> float:
        """Return learned density (notes/beat) for instrument."""
        key = instrument.lower()
        return self.density_model.get(key, 1.0)

    def activity_for_instrument(self, instrument: str) -> float:
        """Return learned activity fraction [0,1] for instrument."""
        key = instrument.lower()
        return self.activity_model.get(key, 0.7)

    def dynamic_at(self, progress: float) -> float:
        """Return dynamic level at a given song progress [0.0, 1.0]."""
        if not self.dynamic_model:
            return 0.8
        idx = int(progress * len(self.dynamic_model))
        idx = max(0, min(idx, len(self.dynamic_model) - 1))
        return float(self.dynamic_model[idx])
