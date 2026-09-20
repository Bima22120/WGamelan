"""Gamelan scale definitions: Laras Slendro, Laras Pelog, and Pathet modes."""

from typing import List, Optional
from app.gamelan.tuning import GamelanPitch, build_tuning_table

PATHET_NOTES = {
    # Slendro modes
    "slendro_nem": ["2", "3", "5", "6", "1"],
    "slendro_sanga": ["5", "6", "1", "2", "3"],
    "slendro_manyura": ["6", "1", "2", "3", "5"],
    # Pelog modes
    "pelog_lima": ["1", "2", "3", "5", "6"],
    "pelog_nem": ["2", "3", "5", "6", "1"],
    "pelog_barang": ["2", "3", "5", "6", "7"],
    "pelog_full": ["1", "2", "3", "4", "5", "6", "7"],
}


class GamelanScale:
    """Encapsulates a specific Gamelan tuning system and optional Pathet mode."""

    def __init__(self, scale_type: str = "slendro", pathet: Optional[str] = None):
        self.scale_type = scale_type.lower()
        self.pathet = pathet.lower() if pathet else None
        self._pitches: List[GamelanPitch] = build_tuning_table(self.scale_type, octaves=[-1, 0, 1])

        # Filter by pathet if specified
        if self.pathet and self.pathet in PATHET_NOTES:
            allowed = set(PATHET_NOTES[self.pathet])
            self._pitches = [p for p in self._pitches if p.degree in allowed]

    @property
    def pitches(self) -> List[GamelanPitch]:
        return self._pitches

    def find_closest(self, freq_hz: float) -> GamelanPitch:
        """Find the Gamelan pitch with the minimum frequency difference."""
        if not self._pitches:
            raise ValueError("Scale contains no pitches.")
        return min(self._pitches, key=lambda p: abs(p.freq_hz - freq_hz))

    def __repr__(self) -> str:
        mode_str = f" ({self.pathet})" if self.pathet else ""
        return f"GamelanScale({self.scale_type.capitalize()}{mode_str}, {len(self._pitches)} tones)"
