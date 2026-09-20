"""Gamelan tuning systems: Laras Slendro and Laras Pelog frequency tables and intervals."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math


@dataclass
class GamelanPitch:
    degree: str         # '1', '2', '3', '4', '5', '6', '7'
    solfege: str        # 'Ji', 'Ro', 'Lu', 'Pat', 'Ma', 'Nem', 'Pi'
    traditional_name: str
    freq_hz: float
    octave: int         # 0=tengah (middle), -1=rendah (low), +1=tinggi (high)
    cents_from_base: float


# Standard reference frequencies for middle octave (Saron Barung register)
# Reference: Kunst (1973), Hood (1977), Surjodiningrat (1972)
SLENDRO_BASE_FREQS = {
    "1": 270.0,  # Ji (Panunggal) ~ C#4
    "2": 310.0,  # Ro (Gulu) ~ D#4
    "3": 355.0,  # Lu (Dhadha) ~ F4
    "5": 410.0,  # Ma (Lima) ~ G#4
    "6": 470.0,  # Nem ~ A#4
}

PELOG_BASE_FREQS = {
    "1": 262.0,  # Ji (Panunggal) ~ C4
    "2": 285.0,  # Ro (Gulu) ~ D4
    "3": 335.0,  # Lu (Dhadha) ~ E4
    "4": 360.0,  # Pat (Pelog) ~ F4
    "5": 390.0,  # Ma (Lima) ~ G4
    "6": 425.0,  # Nem ~ G#4
    "7": 495.0,  # Pi (Barang) ~ B4
}

SOLFEGE_NAMES = {
    "1": ("Ji", "Panunggal/Barang"),
    "2": ("Ro", "Gulu"),
    "3": ("Lu", "Dhadha"),
    "4": ("Pat", "Pelog"),
    "5": ("Ma", "Lima"),
    "6": ("Nem", "Nem"),
    "7": ("Pi", "Barang"),
}


def build_tuning_table(scale_type: str = "slendro", octaves: List[int] = [-1, 0, 1]) -> List[GamelanPitch]:
    """Generate pitch tables across specified octaves.

    Args:
        scale_type: 'slendro' or 'pelog'.
        octaves: List of octave shifts (-1=low/Demung, 0=middle/Barung, 1=high/Peking).

    Returns:
        List of GamelanPitch instances ordered by frequency.
    """
    scale_type = scale_type.lower()
    base_dict = SLENDRO_BASE_FREQS if "slendro" in scale_type else PELOG_BASE_FREQS
    base_root = base_dict["1"]

    pitches: List[GamelanPitch] = []
    for oct_shift in octaves:
        mult = 2.0 ** oct_shift
        for deg, base_f in base_dict.items():
            freq = base_f * mult
            solfege, name = SOLFEGE_NAMES.get(deg, (deg, deg))
            cents = 1200.0 * math.log2(freq / base_root)
            pitches.append(
                GamelanPitch(
                    degree=deg,
                    solfege=solfege,
                    traditional_name=name,
                    freq_hz=freq,
                    octave=oct_shift,
                    cents_from_base=cents,
                )
            )

    pitches.sort(key=lambda p: p.freq_hz)
    return pitches
