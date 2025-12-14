"""
Learning Module

Provides access to federated learning services via API calls to SwarmBridge.

REFACTORED: Replaces direct Flower FL integration with unified FL service client.
"""

from .fl_service_client import (
    FederatedLearningServiceClient,
    FLRoundConfig,
    FLRoundStatus,
)

__all__ = [
    "FederatedLearningServiceClient",
    "FLRoundConfig",
    "FLRoundStatus",
]
