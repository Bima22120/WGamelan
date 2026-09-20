"""File selection helper for CLI and GUI environments."""

import os
from typing import Optional, List

SUPPORTED_AUDIO_EXTS = [".wav", ".mp3", ".flac", ".ogg", ".aiff", ".m4a"]


class FileSelector:
    """Manages file browsing, path resolution, and extension validation."""

    @staticmethod
    def is_valid_audio(filepath: str) -> bool:
        if not os.path.isfile(filepath):
            return False
        ext = os.path.splitext(filepath)[1].lower()
        return ext in SUPPORTED_AUDIO_EXTS

    @staticmethod
    def get_default_output_path(input_path: str, suffix: str = "_gamelan.wav", output_dir: str = "output") -> str:
        """Derive standard output path in the output directory."""
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{base_name}{suffix}")

    @staticmethod
    def browse_open_file(title: str = "Select Input Audio File") -> Optional[str]:
        """Try GUI file dialog (tkinter or PyQt), returning filepath if chosen."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title=title,
                filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg"), ("All Files", "*.*")],
            )
            root.destroy()
            return path if path else None
        except Exception:
            return None
