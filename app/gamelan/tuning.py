"""Gamelan tuning systems: Laras Slendro, Laras Pelog, and Hybrid Diatonic frequency tables."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math


@dataclass
class GamelanPitch:
    degree: str         # '1', '2', '3', '4', '5', '6', '7', etc.
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

# Hybrid Diatonic / 12-TET tuning with authentic Gamelan physical timbre & solfege
# Aligns 1:1 with Western guitar, piano, and pop music to eliminate dissonant clashes
DIATONIC_BASE_FREQS = {
    "1": 261.63,   # C4 (Ji)
    "1#": 277.18,  # C#4
    "2": 293.66,   # D4 (Ro)
    "2#": 311.13,  # D#4
    "3": 329.63,   # E4 (Lu)
    "4": 349.23,   # F4 (Pat)
    "4#": 369.99,  # F#4
    "5": 392.00,   # G4 (Ma)
    "5#": 415.30,  # G#4
    "6": 440.00,   # Nem (A4)
    "6#": 466.16,  # A#4
    "7": 493.88,   # Pi (Barang) ~ B4
}

SOLFEGE_NAMES = {
    "1": ("Ji", "Panunggal/Barang"),
    "1#": ("Ji-seling", "Panunggal Miring"),
    "2": ("Ro", "Gulu"),
    "2#": ("Ro-seling", "Gulu Miring"),
    "3": ("Lu", "Dhadha"),
    "4": ("Pat", "Pelog"),
    "4#": ("Pat-seling", "Pelog Miring"),
    "5": ("Ma", "Lima"),
    "5#": ("Ma-seling", "Lima Miring"),
    "6": ("Nem", "Nem"),
    "6#": ("Nem-seling", "Nem Miring"),
    "7": ("Pi", "Barang"),
}


def build_tuning_table(scale_type: str = "slendro", octaves: List[int] = [-1, 0, 1]) -> List[GamelanPitch]:
    """Generate pitch tables across specified octaves.

    Args:
        scale_type: 'slendro', 'pelog', or 'diatonic'.
        octaves: List of octave shifts (-2=Demung rendah, -1=Demung, 0=Barung, 1=Peking, 2=Peking tinggi).

    Returns:
        List of GamelanPitch instances ordered by frequency.
    """
    scale_type = scale_type.lower()
    if "diatonic" in scale_type or "hybrid" in scale_type:
        base_dict = DIATONIC_BASE_FREQS
    elif "pelog" in scale_type:
        base_dict = PELOG_BASE_FREQS
    else:
        base_dict = SLENDRO_BASE_FREQS

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

