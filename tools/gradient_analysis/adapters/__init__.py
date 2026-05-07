"""Model-specific adapters for gradient analysis modules.

Phase 2 introduces HiP-AD and VAD adapters authored in parallel against a
single Protocol so the interface does not inherit either model's
assumptions (spec §4.1, R5 mitigation).
"""
from .base import GradientAnalysisAdapter, TemporalSnapshot

__all__ = ["GradientAnalysisAdapter", "TemporalSnapshot"]
