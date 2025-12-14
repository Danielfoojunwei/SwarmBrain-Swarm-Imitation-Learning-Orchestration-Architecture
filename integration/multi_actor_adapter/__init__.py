"""
Multi-Actor Integration for SwarmBrain

Integrates the SwarmBridge Multi-Actor Swarm Imitation Learning Architecture
with SwarmBrain for cooperative humanoid robot coordination.
"""

from .csa_engine import (
    ActorIntent,
    CoordinationLatent,
    CoordinationMode,
    CSAConfig,
    CSAEngine,
    MultiActorAction,
    MultiActorObservation,
    RoleType,
    TaskPhase,
)
from .multi_actor_coordinator import MultiActorCoordinator, MultiActorTask
from .swarm_bridge import SwarmBridge, SwarmRoundConfig

__all__ = [
    "CSAEngine",
    "CSAConfig",
    "MultiActorObservation",
    "MultiActorAction",
    "CoordinationLatent",
    "RoleType",
    "ActorIntent",
    "CoordinationMode",
    "TaskPhase",
    "MultiActorCoordinator",
    "MultiActorTask",
    "SwarmBridge",
    "SwarmRoundConfig",
]
