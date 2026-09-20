"""Pitch post-processing: median filtering, octave jump correction, and note segmentation."""

from typing import List, Tuple
import numpy as np

try:
    from scipy.ndimage import median_filter
except ImportError:
    median_filter = None


def smooth_pitch_track(pitch_hz: np.ndarray, filter_size: int = 5) -> np.ndarray:
    """Apply median filter to remove transient pitch glitches while preserving steady steps."""
    if len(pitch_hz) < filter_size:
        return pitch_hz.copy()
    if median_filter is not None:
        return median_filter(pitch_hz, size=filter_size)

    # Pure numpy median filter
    padded = np.pad(pitch_hz, filter_size // 2, mode="edge")
    out = np.zeros_like(pitch_hz)
    for i in range(len(pitch_hz)):
        out[i] = np.median(padded[i:i + filter_size])
    return out


def correct_octave_errors(pitch_hz: np.ndarray, max_jump_ratio: float = 1.85) -> np.ndarray:
    """Fix sudden octave doubling or halving glitches."""
    out = pitch_hz.copy()
    for i in range(1, len(out) - 1):
        prev_p = out[i - 1]
        curr_p = out[i]
        next_p = out[i + 1]

        if prev_p > 0 and curr_p > 0 and next_p > 0:
            # Check if current pitch jumped up by ~octave while neighbors agree
            if 1.8 < curr_p / prev_p < 2.2 and 1.8 < curr_p / next_p < 2.2:
                out[i] = curr_p / 2.0
            # Check if current pitch dropped by ~half while neighbors agree
            elif 0.45 < curr_p / prev_p < 0.55 and 0.45 < curr_p / next_p < 0.55:
                out[i] = curr_p * 2.0
    return out


def segment_notes(
    times: np.ndarray,
    pitch_hz: np.ndarray,
    voiced_flags: np.ndarray,
    onset_times: np.ndarray,
    min_duration: float = 0.08,
) -> List[Tuple[float, float, float]]:
    """Segment pitch track into discrete notes bounded by onsets and unvoiced gaps.

    Args:
        times: Array of time positions.
        pitch_hz: Array of smoothed fundamental frequencies.
        voiced_flags: Boolean array of voiced frames.
        onset_times: List of detected transient onset times.
        min_duration: Minimum valid note length in seconds.

    Returns:
        List of tuples: (start_time, duration, median_pitch_hz).
    """
    notes: List[Tuple[float, float, float]] = []
    if len(times) == 0:
        return notes

    # Combine onset boundaries with unvoiced transitions
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
            # Note ends if voice ceases or a new onset triggers
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

    return notes
