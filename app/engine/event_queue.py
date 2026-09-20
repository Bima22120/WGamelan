"""Thread-safe event queue for inter-thread communication."""

import queue
from typing import Any, Optional


class EventQueue:
    """Thread-safe wrapper around standard library queue with utility helpers."""

    def __init__(self, maxsize: int = 0):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> None:
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        return self._queue.get(block=block, timeout=timeout)

    def get_nowait(self) -> Optional[Any]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()

    def clear(self) -> None:
        with self._queue.mutex:
            self._queue.queue.clear()
