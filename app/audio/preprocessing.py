"""Audio preprocessing utilities: normalization, filtering, trimming, and denoising."""

from typing import Tuple
import numpy as np

try:
    from scipy import signal
except ImportError:
    signal = None


def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio amplitude to a target peak value."""
    peak = np.max(np.abs(audio))
    if peak <= 1e-7:
        return audio.copy()
    return (audio / peak) * target_peak


def normalize_rms(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normalize audio so its Root Mean Square (RMS) matches target_rms."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms <= 1e-7:
        return audio.copy()
    gain = target_rms / rms
    normalized = audio * gain
    # Clip any momentary peaks above 1.0 to avoid harsh digital distortion
    return np.clip(normalized, -1.0, 1.0)


def highpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 50.0, order: int = 4) -> np.ndarray:
    """Apply a Butterworth high-pass filter to eliminate sub-bass rumble and DC offset."""
    if signal is None:
        return audio - np.mean(audio)  # Basic DC offset removal fallback
    nyquist = 0.5 * sr
    normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
    b, a = signal.butter(order, normalized_cutoff, btype="high", analog=False)
    return signal.filtfilt(b, a, audio).astype(np.float32)


def trim_silence(audio: np.ndarray, threshold_db: float = -40.0, frame_length: int = 1024, hop_length: int = 256) -> Tuple[np.ndarray, int, int]:
    """Trim leading and trailing silence based on energy threshold in dB.

    Returns:
        (trimmed_audio, start_sample, end_sample)
    """
    if len(audio) == 0:
        return audio, 0, 0

    # Calculate frame energies
    num_frames = max(1, (len(audio) - frame_length) // hop_length + 1)
    energies = []
    for i in range(num_frames):
        start = i * hop_length
        frame = audio[start:start + frame_length]
        rms = np.sqrt(np.mean(frame ** 2)) + 1e-9
        db = 20 * np.log10(rms)
        energies.append(db)
    energies = np.array(energies)

    voiced_frames = np.where(energies >= threshold_db)[0]
    if len(voiced_frames) == 0:
        return audio, 0, len(audio)

    start_sample = voiced_frames[0] * hop_length
    end_sample = min(len(audio), (voiced_frames[-1] + 1) * hop_length + frame_length)
    return audio[start_sample:end_sample], start_sample, end_sample


def preprocess_audio(audio: np.ndarray, sr: int, target_peak: float = 0.95, highpass_cutoff: float = 50.0) -> np.ndarray:
    """Standard full preprocessing chain: DC/Highpass filtering and peak normalization."""
    filtered = highpass_filter(audio, sr, cutoff_hz=highpass_cutoff)
    normalized = normalize_peak(filtered, target_peak=target_peak)
    return np.ascontiguousarray(normalized, dtype=np.float32)
