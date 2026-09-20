"""Sample manager: loading, caching, and querying recorded Gamelan WAV samples."""

import os
import json
from typing import Dict, Optional, Tuple
import numpy as np
from app.audio.loader import load_audio


class SampleManager:
    """Manages on-disk audio samples and provides pitch lookup with fallbacks."""

    def __init__(self, samples_dir: str = "data/samples/saron", sr: int = 22050):
        self.samples_dir = samples_dir
        self.sr = sr
        self.metadata: Dict = {}
        self.cache: Dict[str, np.ndarray] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        meta_path = os.path.join(self.samples_dir, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}

    def get_sample(self, note_name: str, tuning: str = "slendro") -> Optional[np.ndarray]:
        """Retrieve cached or loaded sample audio buffer for a given note name."""
        cache_key = f"{tuning}_{note_name}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Look in metadata for matching entry
        for item in self.metadata.get("samples", []):
            if str(item.get("note")) == str(note_name) and item.get("tuning") == tuning:
                fn = item.get("filename")
                if fn:
                    fp = os.path.join(self.samples_dir, fn)
                    if os.path.exists(fp):
                        audio, _ = load_audio(fp, target_sr=self.sr, mono=True)
                        self.cache[cache_key] = audio
                        return audio

    def get_sample_by_freq(
        self,
        freq_hz: float,
        tuning: str = "slendro",
        max_deviation_cents: float = 60.0,
    ) -> Optional[np.ndarray]:
        """Retrieve sample matching a target frequency within deviation threshold."""
        if freq_hz <= 0 or not self.metadata.get("samples"):
            return None

        best_sample = None
        best_diff_cents = float("inf")

        for item in self.metadata.get("samples", []):
            if tuning and item.get("tuning") != tuning:
                continue
            sample_f0 = float(item.get("frequency_hz", 0.0))
            if sample_f0 <= 0:
                continue

            diff_cents = abs(1200.0 * np.log2(freq_hz / sample_f0))
            if diff_cents < best_diff_cents:
                best_diff_cents = diff_cents
                best_sample = item

        if best_sample and best_diff_cents <= max_deviation_cents:
            note_name = str(best_sample.get("note"))
            item_tuning = best_sample.get("tuning", tuning)
            return self.get_sample(note_name, tuning=item_tuning)

        return None
