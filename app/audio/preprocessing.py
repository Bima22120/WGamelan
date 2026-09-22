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


def separate_hpss(audio: np.ndarray, sr: int, margin: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Separate audio signal into Harmonic and Percussive components (HPSS).

    Harmonic component isolates sustained guitar notes, overdrive, feedback, hum, and delay tails.
    Percussive component isolates transient pick strikes and percussive attacks.

    Args:
        audio: 1D float32 audio numpy array.
        sr: Sample rate.
        margin: Separation margin coefficient.

    Returns:
        (harmonic_audio, percussive_audio)
    """
    if len(audio) == 0:
        return audio.copy(), audio.copy()

    try:
        import librosa
        h, p = librosa.effects.hpss(y=audio, margin=margin)
        return h.astype(np.float32), p.astype(np.float32)
    except Exception:
        pass

    # NumPy / SciPy STFT fallback for HPSS
    if signal is None:
        return audio.copy(), audio.copy()

    try:
        from scipy import ndimage
        nperseg = 1024
        noverlap = 768
        f, t, Zxx = signal.stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
        S = np.abs(Zxx)

        # Median filter along time (harmonic) and frequency (percussive)
        H_mag = ndimage.median_filter(S, size=(1, 15)) if ndimage else S
        P_mag = ndimage.median_filter(S, size=(15, 1)) if ndimage else S

        eps = 1e-6
        H_power = H_mag ** 2
        P_power = P_mag ** 2
        total_power = H_power + P_power + eps

        mask_h = H_power / total_power
        mask_p = P_power / total_power

        Zh = Zxx * mask_h
        Zp = Zxx * mask_p

        _, harmonic_audio = signal.istft(Zh, fs=sr, nperseg=nperseg, noverlap=noverlap)
        _, percussive_audio = signal.istft(Zp, fs=sr, nperseg=nperseg, noverlap=noverlap)

        # Truncate / pad to match original audio length
        min_len_h = min(len(audio), len(harmonic_audio))
        min_len_p = min(len(audio), len(percussive_audio))

        h_out = np.zeros_like(audio)
        p_out = np.zeros_like(audio)
        h_out[:min_len_h] = harmonic_audio[:min_len_h]
        p_out[:min_len_p] = percussive_audio[:min_len_p]

        return h_out.astype(np.float32), p_out.astype(np.float32)
    except Exception:
        return audio.copy(), audio.copy()

