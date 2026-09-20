"""Control configuration parameters and options manager."""

from dataclasses import dataclass, asdict
from typing import Literal, Dict, Any


@dataclass
class EngineControls:
    """Settings controller for the Gamelanizer engine."""
    scale: Literal["slendro", "pelog"] = "slendro"
    pathet: str = ""
    instrument: Literal["saron", "demung", "peking", "bonang", "gong"] = "saron"
    quantize: bool = True
    bpm: float = 120.0
    irama_level: int = 1
    sample_rate: int = 22050
    export_midi: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineControls":
        valid_keys = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_keys)
