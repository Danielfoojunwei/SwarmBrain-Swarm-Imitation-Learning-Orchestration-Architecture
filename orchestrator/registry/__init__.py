"""
Registry Module

Provides access to unified skill and artifact registries.
"""

from .skill_registry_client import (
    SkillDefinition,
    SkillType,
    UnifiedSkillRegistryClient,
)

__all__ = [
    "SkillDefinition",
    "SkillType",
    "UnifiedSkillRegistryClient",
]
