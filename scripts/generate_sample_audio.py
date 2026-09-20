"""Helper script to synthesize acoustic guitar demo audio for testing."""

import os
import sys
import numpy as np

# Ensure root workspace is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.audio.loader import save_audio


def karplus_strong(freq: float, duration: float, sr: int = 22050) -> np.ndarray:
    """Synthesize plucked string sound using Karplus-Strong algorithm."""
    n_samples = int(duration * sr)
    period = max(2, int(sr / freq))
    buf = np.random.uniform(-1.0, 1.0, period).astype(np.float32)
    out = np.zeros(n_samples, dtype=np.float32)
    decay = 0.991
    idx = 0
    for i in range(n_samples):
        val = buf[idx]
        out[i] = val
        next_val = (val + buf[(idx + 1) % period]) * 0.5 * decay
        buf[idx] = next_val
        idx = (idx + 1) % period
    return out


def generate_demo_guitar(output_path: str = "data/input/guitar.wav", sr: int = 22050) -> str:
    """Generate an acoustic arpeggio melody."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Melodic notes: C4 (261.6), E4 (329.6), G4 (392.0), A4 (440.0), C5 (523.2), G4, E4, C4
    melody = [261.63, 329.63, 392.00, 440.00, 523.25, 392.00, 329.63, 261.63]
    segments = []
    for freq in melody:
        segments.append(karplus_strong(freq, duration=0.45, sr=sr))

    full_audio = np.concatenate(segments)
    saved = save_audio(output_path, full_audio, sr=sr)
    print(f"Demo guitar audio saved to: {saved}")
    return saved


if __name__ == "__main__":
    generate_demo_guitar()
