"""Gamelan note renderer with adaptive mathet damping and per-note reverb tails.

Improvements implemented:
  - Solusi 1 (Mathet Adaptif): Notes are sorted by start time and the renderer
    automatically computes each note's actual ring-out window based on the next
    note's onset, then applies a short exponential fade-out to simulate the
    traditional Saron/Demung "mathet" damping technique.
  - Solusi 2 (Duration Scaling): The synthesised duration is scaled
    proportionally to the original detected duration so that short guitar
    strokes map to short gamelan tones and long strokes map to naturally-
    ringing tones.
  - Solusi 3 (Per-note reverb tail): Instead of one global reverb pass on the
    mixed buffer, each note is given an independent convolution reverb tail
    that is mixed into the master buffer, preventing earlier notes' reverb
    from bleeding into later notes.
"""

from typing import List, Optional
import numpy as np
from app.music.note import Note
from app.gamelan.sampler import GamelanSampler
from app.gamelan.instrument import GamelanInstrument, get_instrument
from app.gamelan.reverb import PendopoReverb


# ---------------------------------------------------------------------------
# Solusi 1 – Mathet (adaptive damping) helper
# ---------------------------------------------------------------------------

def _apply_mathet_fade(
    audio: np.ndarray,
    fade_start_idx: int,
    sr: int,
    fade_ms: float = 80.0,
) -> np.ndarray:
    """Apply a short exponential fade-out starting at *fade_start_idx*.

    Mimics the Saron/Demung player pressing the key with the left hand
    (mathet) to mute the previous note the instant a new note is struck.

    Args:
        audio: Synthesised note audio (1D float32, modified in-place).
        fade_start_idx: Sample index at which damping begins.
        sr: Sample rate.
        fade_ms: Length of the exponential fade window in milliseconds.

    Returns:
        The same audio buffer with the fade applied.
    """
    n = len(audio)
    if fade_start_idx >= n:
        return audio

    fade_len = max(1, int((fade_ms / 1000.0) * sr))
    fade_end = min(n, fade_start_idx + fade_len)

    # Exponential fade: starts immediately below 1.0 and drops to ~0.7 % at t=1.
    # Use linspace starting just above 0 so the very first sample is attenuated.
    t_fade = np.linspace(0.05, 1.0, fade_end - fade_start_idx, endpoint=True, dtype=np.float32)
    fade_env = np.exp(-5.0 * t_fade)
    audio[fade_start_idx:fade_end] *= fade_env

    # Silence everything after the fade window
    if fade_end < n:
        audio[fade_end:] = 0.0

    return audio


# ---------------------------------------------------------------------------
# Solusi 3 – Per-note reverb tail helper
# ---------------------------------------------------------------------------

def _note_reverb_mono(
    audio: np.ndarray,
    reverb_processor: "PendopoReverb",
) -> np.ndarray:
    """Convolve a single note's audio with the reverb IR (returns mono).

    Builds a short impulse response by feeding a dirac impulse through the
    reverb tank, then fast-convolves the note audio against it.  This gives
    each note an independent reverb tail instead of a single global pass that
    leaks reverb energy across note boundaries.

    Args:
        audio: Mono note audio (1D float32).
        reverb_processor: Pre-configured PendopoReverb instance.

    Returns:
        Reverberated mono audio (1D float32), length = len(audio) + IR_len - 1.
    """
    if len(audio) == 0:
        return audio

    sr = reverb_processor.sr
    rt60 = reverb_processor.rt60
    note_dur = len(audio) / sr

    # IR length capped at rt60 * 1.5 for efficiency (max 5 s)
    ir_len = int(min(sr * (note_dur + rt60 * 1.5), sr * 5.0))
    impulse = np.zeros(ir_len, dtype=np.float32)
    impulse[0] = 1.0
    ir_mono = reverb_processor.process(impulse, stereo=False).astype(np.float32)

    # Fast convolution (scipy if available, else numpy)
    try:
        from scipy.signal import fftconvolve
        wet = fftconvolve(audio, ir_mono, mode="full").astype(np.float32)
    except ImportError:
        wet = np.convolve(audio, ir_mono).astype(np.float32)

    # Blend dry + wet into output buffer
    dry_len = len(audio)
    result = np.zeros(len(wet), dtype=np.float32)
    result[:dry_len] = (
        reverb_processor.dry_level * audio
        + reverb_processor.wet_level * wet[:dry_len]
    )
    if len(wet) > dry_len:
        result[dry_len:] = reverb_processor.wet_level * wet[dry_len:]

    peak = np.max(np.abs(result))
    if peak > 0.98:
        result = (result / peak) * 0.98

    return result


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

class GamelanRenderer:
    """Renders a sequence of Gamelan notes into a composite audio wave.

    Key behaviours
    --------------
    - **Mathet Adaptif (Solusi 1)**: When a new melodic note begins, the
      previous melodic note is automatically faded out at the onset of the new
      note, replicating the damping hand technique of Saron/Demung players.
    - **Duration Scaling (Solusi 2)**: The synthesised length of each note is
      derived from the original detected duration so that short guitar
      articulations map to short gamelan tones and long held notes map to
      fuller ring-out.  A per-instrument headroom multiplier is still applied
      (e.g. Gong always gets extra decay).
    - **Per-note Reverb (Solusi 3)**: Each note is individually convolved with
      a short reverb impulse response before being mixed into the master
      buffer, preventing reverb energy from one note contaminating the next.
    """

    # Length (ms) of the exponential fade applied when a new note triggers mathet
    MATHET_FADE_MS: float = 80.0

    # Even with mathet, let at least this fraction of the instrument decay ring
    MIN_RINGOUT_FRACTION: float = 0.25

    def __init__(
        self,
        instrument: Optional[GamelanInstrument] = None,
        sr: int = 22050,
        legato: bool = True,
        reverb: bool = False,
        stereo: bool = False,
    ):
        self.sr = sr
        self.instrument = instrument or get_instrument("saron")
        self.sampler = GamelanSampler(instrument=self.instrument, sr=sr)
        self.bonang_sampler = GamelanSampler(instrument=get_instrument("bonang"), sr=sr)
        self.gong_sampler = GamelanSampler(instrument=get_instrument("gong"), sr=sr)
        self.legato = legato
        self.reverb_enabled = reverb
        self.stereo = stereo
        self.reverb_processor = PendopoReverb(sr=sr)

    # ------------------------------------------------------------------
    # Solusi 1 + 2: compute synth duration and mathet point
    # ------------------------------------------------------------------

    def _compute_synth_duration(
        self,
        note: Note,
        next_onset: Optional[float],
        is_gong: bool,
        is_bonang: bool,
    ) -> tuple:
        """Return (synth_dur_sec, apply_mathet, mathet_sample_idx).

        Implements Solusi 1 + Solusi 2 jointly:

        * The note is allowed to ring naturally for up to its *detected*
          duration from the guitar (Solusi 2 – respect original articulation).
          Uses ``metadata["original_duration"]`` if set by PerformanceStyle,
          otherwise falls back to ``note.duration``.
        * If a following melodic note starts before the natural decay ends,
          a mathet fade is scheduled at the next note's onset (Solusi 1).
        * Gong notes always ring to their full natural decay (traditional).
        """
        instr_decay = self.instrument.decay_time_sec

        if is_gong:
            # Gong rings freely regardless of context
            return max(note.duration, 4.5), False, 0

        # Solusi 2: use original detected guitar duration as the ring-out basis.
        # PerformanceStyle may shorten note.duration for phrasing purposes, so
        # original_duration reflects the true articulation from the source audio.
        original_dur = note.metadata.get("original_duration", note.duration)
        natural_ringout = original_dur + instr_decay * 0.25

        if next_onset is not None:
            gap_to_next = next_onset - note.start_time

            # Mathet fires at the instant the next note is struck
            mathet_at_sec = gap_to_next

            # Enforce a minimum ring-out so fast passages don't sound clipped
            min_ringout = instr_decay * self.MIN_RINGOUT_FRACTION
            mathet_at_sec = max(mathet_at_sec, min_ringout)

            fade_sec = self.MATHET_FADE_MS / 1000.0
            synth_dur = mathet_at_sec + fade_sec + 0.05  # tiny silence tail
            apply_mathet = mathet_at_sec < natural_ringout
            mathet_sample = int(mathet_at_sec * self.sr)
        else:
            # Last melodic note or isolated note: let it decay naturally
            synth_dur = natural_ringout
            apply_mathet = False
            mathet_sample = 0

        synth_dur = max(synth_dur, 0.15)
        return synth_dur, apply_mathet, mathet_sample


    # ------------------------------------------------------------------
    # Main render entry point
    # ------------------------------------------------------------------

    def render(
        self,
        notes: List[Note],
        pad_end_sec: float = 1.5,
        legato: Optional[bool] = None,
        reverb: Optional[bool] = None,
        stereo: Optional[bool] = None,
        target_duration_sec: Optional[float] = None,
    ) -> np.ndarray:
        """Render notes into a composite audio array (mono 1D or stereo 2D).

        Processing order:
            1. Sort notes by start time.
            2. Per note: compute adaptive synth duration + mathet point (S1 & S2).
            3. Synthesise note audio.
            4. Apply mathet fade if required (Solusi 1).
            5. Per-note reverb convolution (Solusi 3, when reverb enabled).
            6. Accumulate into master buffer.
            7. Strict reference duration locking (prevents duration mismatch).
            8. Acoustic mastering chain (HPF 85Hz + 2–8 kHz Harmonic Exciter).
            9. Global normalisation + optional stereo widening.
        """
        use_legato = self.legato if legato is None else legato
        use_reverb = self.reverb_enabled if reverb is None else reverb
        use_stereo = self.stereo if stereo is None else stereo

        if not notes:
            total_sec = target_duration_sec if target_duration_sec else pad_end_sec
            num_zeros = int(self.sr * total_sec)
            if use_reverb and use_stereo:
                return np.zeros((num_zeros, 2), dtype=np.float32)
            return np.zeros(num_zeros, dtype=np.float32)

        # Sort chronologically (guard against unsorted input)
        sorted_notes = sorted(notes, key=lambda n: n.start_time)

        # Allocate master buffer with headroom for reverb tails
        max_decay = max(self.instrument.decay_time_sec, 4.5)
        rt60_headroom = self.reverb_processor.rt60 * 1.5 if use_reverb else 0.0
        total_duration = (
            max(n.end_time for n in sorted_notes)
            + max_decay
            + rt60_headroom
            + pad_end_sec
        )
        num_samples = int(total_duration * self.sr)
        master_buffer = np.zeros(num_samples, dtype=np.float32)

        for note_idx, note in enumerate(sorted_notes):
            freq = note.gamelan_freq_hz or note.pitch_hz
            if freq <= 20.0:
                continue

            inst_meta = note.metadata.get("instrument", "").lower()
            is_gong = ("gong" in inst_meta or "kenong" in inst_meta or freq < 90.0)
            is_bonang = ("bonang" in inst_meta)

            if is_bonang:
                active_sampler = self.bonang_sampler
            elif is_gong:
                active_sampler = self.gong_sampler
            else:
                active_sampler = self.sampler

            # -----------------------------------------------------------
            # Solusi 1 & 2: find next melodic note onset, compute timings
            # -----------------------------------------------------------
            next_onset: Optional[float] = None
            if not is_gong and not is_bonang and use_legato:
                for future in sorted_notes[note_idx + 1:]:
                    f_meta = future.metadata.get("instrument", "").lower()
                    f_freq = future.gamelan_freq_hz or future.pitch_hz
                    f_is_gong = "gong" in f_meta or "kenong" in f_meta or f_freq < 90.0
                    f_is_bonang = "bonang" in f_meta
                    if not f_is_gong and not f_is_bonang:
                        next_onset = future.start_time
                        break

            synth_dur, apply_mathet, mathet_sample = self._compute_synth_duration(
                note, next_onset, is_gong, is_bonang
            )

            # Instrument-level damping flag (only relevant in non-legato mode)
            damped = (
                note.metadata.get("damped", self.instrument.damping_enabled)
                if not use_legato else False
            )

            note_audio = active_sampler.synthesize_note(
                freq_hz=freq,
                duration_sec=synth_dur,
                velocity=note.velocity,
                damped=damped,
                damp_time_sec=note.duration,
            )

            # -----------------------------------------------------------
            # Solusi 1: apply mathet fade at next-note onset
            # -----------------------------------------------------------
            if apply_mathet and mathet_sample > 0:
                note_audio = _apply_mathet_fade(
                    note_audio,
                    mathet_sample,
                    self.sr,
                    fade_ms=self.MATHET_FADE_MS,
                )

            # -----------------------------------------------------------
            # Solusi 3: per-note reverb convolution
            # -----------------------------------------------------------
            if use_reverb:
                note_audio = _note_reverb_mono(note_audio, self.reverb_processor)

            # Accumulate into master buffer
            start_idx = int(note.start_time * self.sr)
            end_idx = min(start_idx + len(note_audio), num_samples)
            slice_len = end_idx - start_idx
            if slice_len > 0:
                master_buffer[start_idx:end_idx] += note_audio[:slice_len]

        # -----------------------------------------------------------
        # Section 1 Correction: Strict Target Duration Alignment
        # Locks output length to reference audio timeline (prevents tail stretch)
        # -----------------------------------------------------------
        if target_duration_sec is not None and target_duration_sec > 0:
            target_samples = int(target_duration_sec * self.sr)
            if len(master_buffer) > target_samples:
                fade_len = min(int(0.25 * self.sr), target_samples)
                fade_start = target_samples - fade_len
                fade_curve = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                master_buffer[fade_start:target_samples] *= fade_curve
                master_buffer = master_buffer[:target_samples]
            elif len(master_buffer) < target_samples:
                master_buffer = np.pad(master_buffer, (0, target_samples - len(master_buffer)))

        # Stereo widening pass if requested
        if use_reverb and use_stereo:
            out_audio = self.reverb_processor.process(master_buffer, stereo=True)
        else:
            out_audio = master_buffer

        # -----------------------------------------------------------
        # Section 4 & 5 Correction: Acoustic Mastering Chain
        # HPF 85 Hz (cuts 0-200 Hz sub mud) + Harmonic Exciter (restores 2-8 kHz)
        # -----------------------------------------------------------
        from app.audio.harmonic_exciter import AcousticMasteringChain
        mastering = AcousticMasteringChain(
            sr=self.sr,
            hpf_cutoff_hz=85.0,
            exciter_drive=1.5,
            exciter_wet=0.35,
            air_gain_db=3.0,
        )
        out_audio = mastering.process(out_audio)

        # Global normalisation guard
        peak = np.max(np.abs(out_audio))
        if peak > 0.95:
            out_audio = (out_audio / peak) * 0.95

        return out_audio

