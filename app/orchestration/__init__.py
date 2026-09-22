"""Orchestration module — Reference-Based Gamelan Orchestration Transfer.

This module implements the core of the GAMELANIZER architecture:

    REFERENCE GAMELAN AUDIO
        ↓
    ORCHESTRATION ANALYSIS (OrchestrationLearner)
        ↓
    ORCHESTRATION TEMPLATE (OrchestrationProfile)
        ↓
    SOURCE SONG MUSICAL ANALYSIS
        ↓
    ORCHESTRATION TRANSFER (OrchestrationTransfer)
        ↓
    GAMELAN SCORE
        ↓
    GAMELAN PERFORMANCE
        ↓
    GAMELAN AUDIO

Primary invariant: OUTPUT_INSTRUMENT_FAMILY == GAMELAN always.
"""

from app.orchestration.orchestration_profile import OrchestrationProfile
from app.orchestration.learner import OrchestrationLearner
from app.orchestration.transfer_engine import OrchestrationTransfer
from app.orchestration.validator import ArrangementValidator

__all__ = [
    "OrchestrationProfile",
    "OrchestrationLearner",
    "OrchestrationTransfer",
    "ArrangementValidator",
]
