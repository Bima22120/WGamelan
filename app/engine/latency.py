"""Latency calculation and timing compensation utilities."""

import time
from typing import Optional


class LatencyTracker:
    """Measures and calculates latency profiles for audio buffers and event processing."""

    def __init__(self, sample_rate: int = 22050, buffer_size: int = 512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self._start_time: Optional[float] = None
        self._last_elapsed: float = 0.0

    @property
    def buffer_latency_ms(self) -> float:
        """Intrinsic audio buffer latency in milliseconds."""
        return (self.buffer_size / self.sample_rate) * 1000.0

    def start_timer(self) -> None:
        self._start_time = time.perf_counter()

    def stop_timer(self) -> float:
        if self._start_time is None:
            return 0.0
        self._last_elapsed = (time.perf_counter() - self._start_time) * 1000.0
        self._start_time = None
        return self._last_elapsed

    @property
    def last_latency_ms(self) -> float:
        return self._last_elapsed
