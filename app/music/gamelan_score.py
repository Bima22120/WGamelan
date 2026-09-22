"""Multi-track Gamelan Score container.

Represents the complete musical arrangement after orchestration transfer,
before physical synthesis. Separates WHAT (musical content) from HOW (performance).

Data flow:
    SourceScore
        ↓
    OrchestrationProfile
        ↓
    GamelanScore         ← this module
        ↓
    PerformanceScore
        ↓
    Audio
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json

from app.gamelan.context_mapper import GamelanNoteEvent


# ---------------------------------------------------------------------------
# Instrument role taxonomy
# ---------------------------------------------------------------------------

class InstrumentRole:
    """Roles in Javanese gamelan ensembles."""
    SKELETON = "skeleton"          # Balungan: demung, saron, slenthem
    ELABORATION = "elaboration"    # Panerusan: bonang, peking, gambang, rebab
    RHYTHM = "rhythm"              # Rhythmic: kendang
    STRUCTURAL = "structural"      # Colotomic: gong ageng, kempul, kenong, ketuk
    ORNAMENT = "ornament"          # Decorative fills


# ---------------------------------------------------------------------------
# GamelanScore
# ---------------------------------------------------------------------------

@dataclass
class GamelanScore:
    """Complete multi-track arrangement of a source song in gamelan instrumentation.

    Attributes:
        tracks: Ordered dict mapping instrument name → list of GamelanNoteEvent
        tempo_bpm: Detected tempo of the arrangement
        total_duration: Total duration in seconds
        laras: Scale type used ('slendro' or 'pelog')
        pathet: Modal context if specified
        metadata: Arbitrary annotation dict
    """
    tracks: Dict[str, List[GamelanNoteEvent]] = field(default_factory=dict)
    tempo_bpm: float = 120.0
    total_duration: float = 0.0
    laras: str = "slendro"
    pathet: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Role groupings
    _ROLE_MAP: Dict[str, str] = field(default_factory=lambda: {
        "saron": InstrumentRole.SKELETON,
        "demung": InstrumentRole.SKELETON,
        "slenthem": InstrumentRole.SKELETON,
        "bonang": InstrumentRole.ELABORATION,
        "bonang barung": InstrumentRole.ELABORATION,
        "bonang panerus": InstrumentRole.ELABORATION,
        "peking": InstrumentRole.ELABORATION,
        "kendang": InstrumentRole.RHYTHM,
        "gong": InstrumentRole.STRUCTURAL,
        "gong ageng": InstrumentRole.STRUCTURAL,
        "kempul": InstrumentRole.STRUCTURAL,
        "kenong": InstrumentRole.STRUCTURAL,
        "ketuk": InstrumentRole.STRUCTURAL,
    })

    def add_track(self, instrument: str, notes: List[GamelanNoteEvent]) -> None:
        """Add or replace a track for the given instrument."""
        self.tracks[instrument] = sorted(notes, key=lambda n: n.start)

    def get_track(self, instrument: str) -> List[GamelanNoteEvent]:
        """Return notes for instrument, empty list if not present."""
        return self.tracks.get(instrument, [])

    def all_notes(self) -> List[GamelanNoteEvent]:
        """Return all notes across all tracks, sorted by onset time."""
        all_n: List[GamelanNoteEvent] = []
        for notes in self.tracks.values():
            all_n.extend(notes)
        return sorted(all_n, key=lambda n: n.start)

    @property
    def instrument_names(self) -> List[str]:
        return list(self.tracks.keys())

    @property
    def total_notes(self) -> int:
        return sum(len(v) for v in self.tracks.values())

    def role_of(self, instrument: str) -> str:
        """Return the orchestration role of the given instrument."""
        key = instrument.lower()
        return self._ROLE_MAP.get(key, InstrumentRole.SKELETON)

    def skeleton_tracks(self) -> Dict[str, List[GamelanNoteEvent]]:
        """Return only skeleton (balungan) instrument tracks."""
        return {k: v for k, v in self.tracks.items()
                if self.role_of(k) == InstrumentRole.SKELETON}

    def elaboration_tracks(self) -> Dict[str, List[GamelanNoteEvent]]:
        """Return only elaboration (panerusan) instrument tracks."""
        return {k: v for k, v in self.tracks.items()
                if self.role_of(k) == InstrumentRole.ELABORATION}

    def structural_tracks(self) -> Dict[str, List[GamelanNoteEvent]]:
        """Return only structural (colotomic) instrument tracks."""
        return {k: v for k, v in self.tracks.items()
                if self.role_of(k) == InstrumentRole.STRUCTURAL}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict (for debugging / inspection)."""
        result: Dict[str, Any] = {
            "tempo_bpm": self.tempo_bpm,
            "total_duration": self.total_duration,
            "laras": self.laras,
            "pathet": self.pathet,
            "metadata": self.metadata,
            "tracks": {},
        }
        for inst, notes in self.tracks.items():
            result["tracks"][inst] = [
                {
                    "start": n.start,
                    "duration": n.duration,
                    "pitch_hz": n.pitch_hz,
                    "degree": n.degree,
                    "velocity": n.velocity,
                    "register": n.register,
                    "articulation": n.articulation,
                }
                for n in notes
            ]
        return result

    def save_json(self, path: str) -> None:
        """Persist GamelanScore to JSON for inspection."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
