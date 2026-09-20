"""Audio loading and saving utilities for Gamelanizer."""

import os
from typing import Tuple, Optional
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import scipy.io.wavfile as wavfile
    import scipy.signal as signal
except ImportError:
    wavfile = None
    signal = None

try:
    import librosa
except ImportError:
    librosa = None


def load_audio(filepath: str, target_sr: Optional[int] = 22050, mono: bool = True) -> Tuple[np.ndarray, int]:
    """Load an audio file into a normalized floating-point numpy array.

    Args:
        filepath: Path to the audio file (wav, flac, ogg, etc.).
        target_sr: Target sample rate in Hz. If None, preserves original rate.
        mono: If True, downmixes multi-channel audio to mono.

    Returns:
        A tuple of (audio_array, sample_rate). Audio is float32 in [-1.0, 1.0].
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    if sf is not None:
        audio, sr = sf.read(filepath, dtype="float32", always_2d=False)
    elif wavfile is not None and filepath.lower().endswith(".wav"):
        sr, raw = wavfile.read(filepath)
        if raw.dtype == np.int16:
            audio = raw.astype(np.float32) / 32768.0
        elif raw.dtype == np.int32:
            audio = raw.astype(np.float32) / 2147483648.0
        elif raw.dtype == np.uint8:
            audio = (raw.astype(np.float32) - 128.0) / 128.0
        else:
            audio = raw.astype(np.float32)
    elif librosa is not None:
        audio, sr = librosa.load(filepath, sr=target_sr, mono=mono)
        return audio.astype(np.float32), sr
    else:
        raise RuntimeError("No suitable audio library available (soundfile, scipy, or librosa required).")

    # Downmix to mono if needed
    if mono and audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if required
    if target_sr is not None and sr != target_sr:
        if librosa is not None:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        elif signal is not None:
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples).astype(np.float32)
        sr = target_sr

    return np.ascontiguousarray(audio, dtype=np.float32), sr


def save_audio(filepath: str, audio: np.ndarray, sr: int = 22050, subtype: str = "PCM_16") -> str:
    """Save an audio numpy array to disk as a WAV file.

    Args:
        filepath: Target output path.
        audio: 1D or 2D numpy array of audio samples.
        sr: Sampling rate in Hz.
        subtype: Soundfile subtype (e.g., 'PCM_16', 'PCM_24', 'FLOAT').

    Returns:
        Absolute path to saved file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)

    # Prevent clipping
    max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if max_val > 1.0:
        audio = audio / max_val

    if sf is not None:
        sf.write(filepath, audio, sr, subtype=subtype)
    elif wavfile is not None:
        int_data = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        wavfile.write(filepath, sr, int_data)
    else:
        raise RuntimeError("Neither soundfile nor scipy is available to save audio.")

    return os.path.abspath(filepath)
