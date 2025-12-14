"""
Coordination Module

Standardized role assignment and coordination primitives aligned with
SwarmBridge Multi-Actor and Dynamical platforms.
"""

from .standardized_roles import (
    CoordinationConfig,
    CoordinationPrimitive,
    CoordinationPrimitiveExecutor,
    RoleAssignment,
    RoleType,
    StandardizedRoleAssigner,
)

__all__ = [
    "RoleType",
    "CoordinationPrimitive",
    "RoleAssignment",
    "CoordinationConfig",
    "StandardizedRoleAssigner",
    "CoordinationPrimitiveExecutor",
]
