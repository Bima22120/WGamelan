"""Audio module: loading, preprocessing, and analysis."""

from app.audio.loader import load_audio, save_audio
from app.audio.preprocessing import (
    normalize_peak,
    normalize_rms,
    highpass_filter,
    trim_silence,
    preprocess_audio,
)
from app.audio.analyzer import AudioAnalyzer, AudioSummary

__all__ = [
    "load_audio",
    "save_audio",
    "normalize_peak",
    "normalize_rms",
    "highpass_filter",
    "trim_silence",
    "preprocess_audio",
    "AudioAnalyzer",
    "AudioSummary",
]
