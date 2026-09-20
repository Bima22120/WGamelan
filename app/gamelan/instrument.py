"""Gamelan instrument definitions and acoustic profiles."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class GamelanInstrument:
    name: str
    instrument_type: str        # 'metallophone', 'gong_chime', 'large_gong'
    octave_shift: int           # -1 (Demung), 0 (Barung), 1 (Peking)
    decay_time_sec: float       # Natural resonance decay time
    damping_enabled: bool       # Whether player damps the previous key (mathet)
    strike_hardness: float      # Mallet hardness [0.1 = soft wood/felt, 1.0 = hard horn]
    brightness: float           # Overtone / inharmonic metallic partial balance


INSTRUMENTS: Dict[str, GamelanInstrument] = {
    "saron": GamelanInstrument(
        name="Saron Barung",
        instrument_type="metallophone",
        octave_shift=0,
        decay_time_sec=1.4,
        damping_enabled=True,
        strike_hardness=0.7,
        brightness=0.8,
    ),
    "demung": GamelanInstrument(
        name="Saron Demung",
        instrument_type="metallophone",
        octave_shift=-1,
        decay_time_sec=2.2,
        damping_enabled=True,
        strike_hardness=0.5,
        brightness=0.6,
    ),
    "peking": GamelanInstrument(
        name="Saron Peking",
        instrument_type="metallophone",
        octave_shift=1,
        decay_time_sec=0.9,
        damping_enabled=False,
        strike_hardness=0.9,
        brightness=1.0,
    ),
    "bonang": GamelanInstrument(
        name="Bonang Barung",
        instrument_type="gong_chime",
        octave_shift=0,
        decay_time_sec=1.8,
        damping_enabled=False,
        strike_hardness=0.6,
        brightness=0.75,
    ),
    "gong": GamelanInstrument(
        name="Gong Ageng",
        instrument_type="large_gong",
        octave_shift=-2,
        decay_time_sec=5.0,
        damping_enabled=False,
        strike_hardness=0.3,
        brightness=0.4,
    ),
}


def get_instrument(name: str) -> GamelanInstrument:
    """Retrieve an instrument profile by name (defaults to Saron Barung)."""
    key = name.lower()
    for k, inst in INSTRUMENTS.items():
        if k in key:
            return inst
    return INSTRUMENTS["saron"]
