"""Reference Event Extractor — extracts per-instrument events from reference gamelan audio.

Uses frequency-band separation and harmonic/percussive decomposition to isolate
each instrument family and extract note events.

Instrument frequency bands (approximate):
    gong ageng:   50–130 Hz
    slenthem:     100–200 Hz
    demung:       150–300 Hz
    saron:        200–600 Hz
    bonang:       350–1200 Hz
    peking:       500–2000 Hz
    kendang:      wideband transient (80–4000 Hz)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

from app.music.source_score import NoteEvent


# ---------------------------------------------------------------------------
# Frequency band definitions per instrument family
# ---------------------------------------------------------------------------

INSTRUMENT_BANDS: Dict[str, Tuple[float, float]] = {
    "gong":       (50.0,   140.0),
    "slenthem":   (90.0,   220.0),
    "demung":     (140.0,  320.0),
    "saron":      (180.0,  650.0),
    "bonang":     (320.0,  1300.0),
    "peking":     (480.0,  2200.0),
    "kendang":    (70.0,   4000.0),   # rhythmic band (transient-heavy)
}


def _bandpass(audio: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> np.ndarray:
    """Apply a simple FFT-based bandpass filter."""
    n = len(audio)
    fft = np.fft.rfft(audio, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    fft_filtered = fft * mask
    return np.fft.irfft(fft_filtered, n=n).astype(np.float32)


def _extract_onset_events(
    band_audio: np.ndarray,
    sr: int,
    hop_length: int = 256,
    min_note_sec: float = 0.08,
    min_gap_sec: float = 0.06,
) -> List[Tuple[float, float, float]]:
    """Extract (onset_sec, duration_sec, rms_energy) from a band-filtered signal.

    Returns list of (start, duration, energy) tuples.
    """
    if librosa is None or np.max(np.abs(band_audio)) < 1e-6:
        return []

    try:
        onset_frames = librosa.onset.onset_detect(
            y=band_audio,
            sr=sr,
            hop_length=hop_length,
            backtrack=True,
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    except Exception:
        return []

    if len(onset_times) == 0:
        return []

    # Compute per-frame RMS energy
    frame_rms = librosa.feature.rms(y=band_audio, hop_length=hop_length)[0]
    frame_times = librosa.frames_to_time(np.arange(len(frame_rms)), sr=sr, hop_length=hop_length)

    events: List[Tuple[float, float, float]] = []
    for i, t_start in enumerate(onset_times):
        t_end = onset_times[i + 1] if i + 1 < len(onset_times) else len(band_audio) / sr
        duration = max(min_note_sec, t_end - t_start - min_gap_sec)

        # Compute mean RMS within window
        mask = (frame_times >= t_start) & (frame_times < t_end)
        energy = float(np.mean(frame_rms[mask])) if mask.any() else 0.0

        events.append((float(t_start), float(duration), energy))

    return events


def _estimate_pitch_in_band(
    band_audio: np.ndarray,
    sr: int,
    onset_sec: float,
    duration_sec: float,
    lo_hz: float,
    hi_hz: float,
) -> float:
    """Estimate dominant pitch within a time window using autocorrelation."""
    if librosa is None:
        return float((lo_hz + hi_hz) / 2.0)

    start_sample = int(onset_sec * sr)
    end_sample = int((onset_sec + min(duration_sec, 0.3)) * sr)
    segment = band_audio[start_sample:end_sample]

    if len(segment) < 256 or np.max(np.abs(segment)) < 1e-7:
        return float((lo_hz + hi_hz) / 2.0)

    try:
        f0_arr, voiced, _ = librosa.pyin(
            segment,
            fmin=lo_hz,
            fmax=hi_hz,
            sr=sr,
            hop_length=128,
        )
        valid = f0_arr[voiced & ~np.isnan(f0_arr)]
        if len(valid) > 0:
            return float(np.median(valid))
    except Exception:
        pass

    return float((lo_hz + hi_hz) / 2.0)


# ---------------------------------------------------------------------------
# ReferenceEventExtractor
# ---------------------------------------------------------------------------

class ReferenceEventExtractor:
    """Extracts NoteEvent lists per instrument family from a reference gamelan recording.

    Each instrument is isolated via frequency-band filtering. Note events are
    extracted using onset detection per band.

    Usage:
        extractor = ReferenceEventExtractor()
        events_per_instrument = extractor.extract(audio, sr=22050)
        # Returns: Dict[str, List[NoteEvent]]
    """

    def __init__(
        self,
        instruments: Optional[List[str]] = None,
        hop_length: int = 256,
        min_note_sec: float = 0.08,
        estimate_pitch: bool = True,
    ):
        self.instruments = instruments or list(INSTRUMENT_BANDS.keys())
        self.hop_length = hop_length
        self.min_note_sec = min_note_sec
        self.estimate_pitch = estimate_pitch

    def extract(
        self,
        audio: np.ndarray,
        sr: int = 22050,
    ) -> Dict[str, List[NoteEvent]]:
        """Extract per-instrument NoteEvent lists from reference audio.

        Args:
            audio: Mono float32 audio array
            sr: Sample rate

        Returns:
            Dict mapping instrument name → List[NoteEvent]
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)

        # Harmonic/percussive separation: kendang uses percussive component
        harmonic = audio
        percussive = audio
        if librosa is not None:
            try:
                harmonic, percussive = librosa.effects.hpss(audio, margin=3.0)
            except Exception:
                pass

        result: Dict[str, List[NoteEvent]] = {}

        for instrument in self.instruments:
            if instrument not in INSTRUMENT_BANDS:
                continue

            lo_hz, hi_hz = INSTRUMENT_BANDS[instrument]

            # Kendang uses percussive component; others use harmonic
            source = percussive if instrument == "kendang" else harmonic

            # Band-filter the appropriate signal
            band = _bandpass(source, sr, lo_hz, hi_hz)

            # Extract onset events
            raw_events = _extract_onset_events(
                band,
                sr,
                hop_length=self.hop_length,
                min_note_sec=self.min_note_sec,
            )

            note_events: List[NoteEvent] = []
            for start_t, dur, energy in raw_events:
                if energy < 1e-6:
                    continue

                pitch_hz = float((lo_hz + hi_hz) / 2.0)
                if self.estimate_pitch and instrument != "kendang":
                    pitch_hz = _estimate_pitch_in_band(band, sr, start_t, dur, lo_hz, hi_hz)

                # MIDI approximation
                midi_pitch = 69
                if pitch_hz > 20.0:
                    midi_pitch = int(round(12.0 * math.log2(pitch_hz / 440.0) + 69))
                    midi_pitch = max(0, min(127, midi_pitch))

                # Velocity from energy (normalized RMS → MIDI velocity)
                velocity = int(np.clip(energy * 800.0, 30, 120))

                note_events.append(NoteEvent(
                    start=start_t,
                    end=start_t + dur,
                    duration=dur,
                    pitch_hz=pitch_hz,
                    midi_pitch=midi_pitch,
                    velocity=velocity,
                    pitch_confidence=0.6,
                    onset_strength=float(np.clip(energy * 400.0, 0.1, 1.0)),
                    metadata={"instrument": instrument, "band_lo": lo_hz, "band_hi": hi_hz},
                ))

            result[instrument] = note_events

        return result
