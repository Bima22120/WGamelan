"""Pitch post-processing: vibrato smoothing, octave correction, note merging, and segmentation."""

from typing import List, Tuple
import numpy as np

try:
    from scipy.ndimage import median_filter, uniform_filter1d
except ImportError:
    median_filter = None
    uniform_filter1d = None


# ─────────────────────────────────────────────────────────
# 1. Pitch Smoothing
# ─────────────────────────────────────────────────────────

def smooth_pitch_track(pitch_hz: np.ndarray, filter_size: int = 5) -> np.ndarray:
    """Apply median filter to remove transient pitch glitches while preserving steady steps."""
    if len(pitch_hz) < filter_size:
        return pitch_hz.copy()
    if median_filter is not None:
        return median_filter(pitch_hz, size=filter_size)
    # Pure numpy fallback
    padded = np.pad(pitch_hz, filter_size // 2, mode="edge")
    out = np.zeros_like(pitch_hz)
    for i in range(len(pitch_hz)):
        out[i] = np.median(padded[i:i + filter_size])
    return out


def suppress_vibrato(
    pitch_hz: np.ndarray,
    hop_length: int = 256,
    sr: int = 22050,
    vibrato_rate_hz: float = 6.0,
    smoothing_window_ms: float = 80.0,
) -> np.ndarray:
    """Suppress vocal and instrumental vibrato by applying a running median over a time window.

    Vibrato is a periodic pitch oscillation (typically 5–8 Hz for voice).
    A running median spanning ~one full vibrato cycle will flatten it into
    a stable pitch center, which maps cleanly to a single Gamelan bilah.

    Args:
        pitch_hz: Raw f0 track (0 = unvoiced).
        hop_length: Hop length used during detection (samples per frame).
        sr: Sample rate.
        vibrato_rate_hz: Approximate vibrato oscillation rate in Hz.
        smoothing_window_ms: Additional smoothing window in ms (default 80ms ≈ half cycle at 6Hz).

    Returns:
        Vibrato-suppressed pitch array.
    """
    if len(pitch_hz) == 0:
        return pitch_hz.copy()

    frame_rate = sr / hop_length  # frames per second
    # Window spanning one vibrato period
    period_frames = int(frame_rate / vibrato_rate_hz)
    # Additional smoothing window
    extra_frames = int((smoothing_window_ms / 1000.0) * frame_rate)
    window = max(3, period_frames + extra_frames)

    # Ensure window is odd for median filter symmetry
    if window % 2 == 0:
        window += 1

    # Work only on voiced sections to avoid blending zeros with voiced pitches
    out = pitch_hz.copy().astype(np.float32)
    voiced = out > 0.0

    if np.any(voiced):
        # Temporarily replace unvoiced (0.0) with NaN for interpolation
        temp = np.where(voiced, out, np.nan)
        # Forward-fill NaN to prevent voiced regions bleeding silence into median
        mask = np.isnan(temp)
        idx = np.where(~mask, np.arange(len(temp)), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = temp[idx]

        # Apply wide median filter to suppress periodic vibrato
        if median_filter is not None:
            smoothed = median_filter(filled, size=window)
        else:
            padded = np.pad(filled, window // 2, mode="edge")
            smoothed = np.array([
                np.median(padded[i:i + window]) for i in range(len(filled))
            ], dtype=np.float32)

        # Restore unvoiced regions to 0
        out = np.where(voiced, smoothed, 0.0).astype(np.float32)

    return out


# ─────────────────────────────────────────────────────────
# 2. Octave Error Correction
# ─────────────────────────────────────────────────────────

def correct_octave_errors(pitch_hz: np.ndarray, max_jump_ratio: float = 1.85) -> np.ndarray:
    """Fix sudden octave doubling or halving glitches in the f0 track."""
    out = pitch_hz.copy()
    for i in range(1, len(out) - 1):
        prev_p = out[i - 1]
        curr_p = out[i]
        next_p = out[i + 1]

        if prev_p > 0 and curr_p > 0 and next_p > 0:
            if 1.8 < curr_p / prev_p < 2.2 and 1.8 < curr_p / next_p < 2.2:
                out[i] = curr_p / 2.0
            elif 0.45 < curr_p / prev_p < 0.55 and 0.45 < curr_p / next_p < 0.55:
                out[i] = curr_p * 2.0
    return out


# ─────────────────────────────────────────────────────────
# 3. Note Merging (eliminates choppy gaps between syllables)
# ─────────────────────────────────────────────────────────

def merge_adjacent_notes(
    notes: List[Tuple[float, float, float]],
    max_gap_sec: float = 0.15,
    max_pitch_ratio: float = 1.06,
) -> List[Tuple[float, float, float]]:
    """Merge consecutive notes of similar pitch separated by tiny silences.

    When a vocalist sustains a vowel (e.g., *"aaa..."* or *"ooo..."*),
    brief micro-silences between consonants or breath moments can slice
    a single note into several short fragments.  This function re-glues
    them into one continuous note.

    Rules:
    - Gap between note_end and next note_start <= max_gap_sec.
    - Pitch ratio between the two notes <= max_pitch_ratio (≈ a minor 2nd,
      ~100 cents — ensuring we merge the same sustained pitch, not melodic
      leaps).

    Args:
        notes: List of (start_time, duration, pitch_hz) tuples.
        max_gap_sec: Maximum silence gap to bridge (default 0.15s).
        max_pitch_ratio: Maximum allowed pitch deviation ratio (default 1.06 ≈ 100 cents).

    Returns:
        Merged note list.
    """
    if not notes:
        return []

    merged: List[Tuple[float, float, float]] = [notes[0]]

    for curr in notes[1:]:
        prev = merged[-1]
        prev_end = prev[0] + prev[1]
        gap = curr[0] - prev_end

        # Same-pitch similarity: compare in log (cents) space
        if prev[2] > 0 and curr[2] > 0:
            pitch_ratio = max(prev[2], curr[2]) / min(prev[2], curr[2])
        else:
            pitch_ratio = float("inf")

        if gap <= max_gap_sec and pitch_ratio <= max_pitch_ratio:
            # Merge: extend previous note's duration and re-compute pitch as
            # weighted average by duration
            new_dur = curr[0] + curr[1] - prev[0]
            w_prev = prev[1] / (prev[1] + curr[1])
            w_curr = curr[1] / (prev[1] + curr[1])
            avg_pitch = prev[2] * w_prev + curr[2] * w_curr
            merged[-1] = (prev[0], new_dur, avg_pitch)
        else:
            merged.append(curr)

    return merged


# ─────────────────────────────────────────────────────────
# 4. Note Segmentation (with vibrato suppression integrated)
# ─────────────────────────────────────────────────────────

def segment_notes(
    times: np.ndarray,
    pitch_hz: np.ndarray,
    voiced_flags: np.ndarray,
    onset_times: np.ndarray,
    min_duration: float = 0.08,
    merge_gap_sec: float = 0.15,
    merge_pitch_ratio: float = 1.06,
) -> List[Tuple[float, float, float]]:
    """Segment pitch track into discrete notes, then merge similar adjacent fragments.

    Args:
        times: Array of time positions.
        pitch_hz: Vibrato-suppressed f0 track.
        voiced_flags: Boolean array indicating voiced frames.
        onset_times: List of detected transient onset times.
        min_duration: Minimum valid note length in seconds.
        merge_gap_sec: Max silence gap for note merging (0 = disable merging).
        merge_pitch_ratio: Max pitch ratio for note merging.

    Returns:
        List of tuples: (start_time, duration, median_pitch_hz).
    """
    notes: List[Tuple[float, float, float]] = []
    if len(times) == 0:
        return notes

    dt = times[1] - times[0] if len(times) > 1 else 0.02
    in_note = False
    note_start_idx = 0

    for i in range(len(times)):
        curr_time = times[i]
        is_onset = any(abs(curr_time - ot) < (dt * 0.75) for ot in onset_times)
        is_voiced = voiced_flags[i] and pitch_hz[i] > 30.0

        if not in_note:
            if is_voiced:
                in_note = True
                note_start_idx = i
        else:
            if not is_voiced or is_onset:
                start_t = times[note_start_idx]
                dur = curr_time - start_t
                if dur >= min_duration:
                    segment_pitches = pitch_hz[note_start_idx:i]
                    valid_p = segment_pitches[segment_pitches > 0]
                    if len(valid_p) > 0:
                        median_hz = float(np.median(valid_p))
                        notes.append((start_t, dur, median_hz))

                if is_voiced and is_onset:
                    note_start_idx = i
                    in_note = True
                else:
                    in_note = False

    # Close trailing note
    if in_note:
        start_t = times[note_start_idx]
        dur = times[-1] - start_t
        if dur >= min_duration:
            segment_pitches = pitch_hz[note_start_idx:]
            valid_p = segment_pitches[segment_pitches > 0]
            if len(valid_p) > 0:
                notes.append((start_t, dur, float(np.median(valid_p))))

    # Merge fragments that belong to the same sustained note
    if merge_gap_sec > 0 and len(notes) > 1:
        notes = merge_adjacent_notes(notes, max_gap_sec=merge_gap_sec, max_pitch_ratio=merge_pitch_ratio)

    return notes
