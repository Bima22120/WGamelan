"""Progress monitoring and reporting utilities for CLI and GUI interfaces."""

import sys
import time
from typing import Optional, Callable


class ProgressBar:
    """Terminal progress bar with step messages and percentage calculation."""

    def __init__(self, total_steps: int = 100, bar_length: int = 30, description: str = "Processing"):
        self.total_steps = total_steps
        self.bar_length = bar_length
        self.description = description
        self.start_time = time.time()

    def update(self, message: str, fraction: float) -> None:
        """fraction in range [0.0, 1.0]."""
        pct = max(0.0, min(1.0, fraction))
        filled = int(self.bar_length * pct)
        bar = "=" * filled + (">" if filled < self.bar_length else "") + " " * (self.bar_length - filled - (1 if filled < self.bar_length else 0))
        pct_display = int(pct * 100)
        sys.stdout.write(f"\r{self.description} [{bar}] {pct_display}% - {message:<30}")
        sys.stdout.flush()
        if pct >= 1.0:
            sys.stdout.write("\n")
            sys.stdout.flush()


def console_progress_callback(message: str, fraction: float) -> None:
    """Simple functional callback that prints formatted progress in console."""
    pct = int(fraction * 100)
    print(f"[{pct:3d}%] {message}")
