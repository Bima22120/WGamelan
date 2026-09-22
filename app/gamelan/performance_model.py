"""Performance translation engine: separating WHAT (Note) from HOW (Performance).

Implements the performance translation layer according to the Gamelanizer specification:
  - Instrument physical model
  - Mallet attack hardness
  - Damping (mathet) rules
  - Microtiming adjustments
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from app.gamelan.context_mapper import GamelanNoteEvent
from app.gamelan.instrument import GamelanInstrument, get_instrument


@dataclass
class PerformanceEvent:
    """Actionable performance instruction ready for the synthesis engine."""
    pitch_hz: float
    start_time: float
    duration: float
    velocity: int = 100
    instrument_name: str = "saron"
    attack_hardness: float = 0.7
    damping_mode: str = "mathet"       # 'mathet', 'natural', 'staccato'
    damp_time_sec: float = 0.0
    microtiming_sec: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def effective_start(self) -> float:
        return max(0.0, self.start_time + self.microtiming_sec)


class PerformanceTranslationEngine:
    """Translates GamelanNoteEvent list into physically-informed PerformanceEvents."""

    def __init__(self, instrument: Optional[GamelanInstrument] = None, apply_damping: bool = True):
        self.instrument = instrument or get_instrument("saron")
        self.apply_damping = apply_damping

    def translate(self, notes: List[GamelanNoteEvent]) -> List[PerformanceEvent]:
        """Convert musical note events into performance events with physical articulation."""
        if not notes:
            return []

        sorted_notes = sorted(notes, key=lambda n: n.start)
        events: List[PerformanceEvent] = []

        is_gong = ("gong" in self.instrument.instrument_type or "large_gong" in self.instrument.instrument_type)
        is_kettle = ("gong_chime" in self.instrument.instrument_type or "bonang" in self.instrument.name.lower())

        for i, curr in enumerate(sorted_notes):
            # Compute damping window (mathet)
            damp_mode = "natural"
            damp_time = curr.duration

            if not is_gong and self.apply_damping and i < len(sorted_notes) - 1:
                next_n = sorted_notes[i + 1]
                gap = next_n.start - curr.start
                if gap > 0:
                    damp_mode = "mathet"
                    damp_time = gap

            # Dynamic attack hardness: harder strike at higher velocities
            norm_vel = curr.velocity / 127.0
            attack = float(self.instrument.strike_hardness * (0.8 + 0.4 * norm_vel))

            # Microtiming offset: natural subtle humanization (~-3 to +5 ms)
            microtiming = float(np.random.uniform(-0.003, 0.005)) if not is_gong else 0.0

            events.append(
                PerformanceEvent(
                    pitch_hz=curr.pitch_hz,
                    start_time=curr.start,
                    duration=curr.duration,
                    velocity=curr.velocity,
                    instrument_name=self.instrument.name,
                    attack_hardness=attack,
                    damping_mode=damp_mode,
                    damp_time_sec=damp_time,
                    microtiming_sec=microtiming,
                    metadata=dict(curr.metadata),
                )
            )

        return events
