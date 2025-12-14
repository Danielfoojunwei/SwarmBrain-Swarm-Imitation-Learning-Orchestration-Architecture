"""
Unified Skill Registry Client

Provides access to skill definitions, metadata, and versions from the
centralized skill registry (CSA Registry for multi-actor, Dynamical for single-actor).

This client abstracts away local skill management and fetches skill information
from external services.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class SkillType(str, Enum):
    """Type of skill"""
    SINGLE_ACTOR = "single_actor"  # Dynamical single-actor skill
    MULTI_ACTOR = "multi_actor"  # CSA multi-actor skill
    HYBRID = "hybrid"  # Can be used in both modes


@dataclass
class SkillDefinition:
    """Skill definition from registry"""
    skill_id: str
    skill_name: str
    skill_type: SkillType
    version: str
    description: str

    # Execution requirements
    required_capabilities: List[str]
    min_actors: int
    max_actors: int

    # For multi-actor skills
    roles: Optional[Dict[str, str]] = None  # role_id -> role_type
    coordination_mode: Optional[str] = None

    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    author: Optional[str] = None
    tags: List[str] = None

    # Registry information
    registry_url: Optional[str] = None
    artifact_url: Optional[str] = None


class UnifiedSkillRegistryClient:
    """
    Client for accessing unified skill registry

    Fetches skills from:
    - CSA Registry (multi-actor cooperative skills)
    - Dynamical API (single-actor skills)

    Provides a unified interface for skill discovery and metadata retrieval.
    """

    def __init__(
        self,
        csa_registry_url: str = "http://localhost:8082",
        dynamical_api_url: str = "http://localhost:8085",
        timeout: int = 10,
    ):
        self.csa_registry_url = csa_registry_url.rstrip('/')
        self.dynamical_api_url = dynamical_api_url.rstrip('/')
        self.timeout = timeout

        # Local cache
        self._skill_cache: Dict[str, SkillDefinition] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_refresh: Optional[datetime] = None

        logger.info(f"Initialized UnifiedSkillRegistryClient")
        logger.info(f"  CSA Registry: {self.csa_registry_url}")
        logger.info(f"  Dynamical API: {self.dynamical_api_url}")

    def list_skills(
        self,
        skill_type: Optional[SkillType] = None,
        tags: Optional[List[str]] = None,
        refresh: bool = False,
    ) -> List[SkillDefinition]:
        """
        List all available skills

        Args:
            skill_type: Filter by skill type
            tags: Filter by tags
            refresh: Force refresh from registries

        Returns:
            List of skill definitions
        """
        # Refresh cache if needed
        if refresh or self._should_refresh_cache():
            self._refresh_cache()

        # Filter skills
        skills = list(self._skill_cache.values())

        if skill_type:
            skills = [s for s in skills if s.skill_type == skill_type]

        if tags:
            skills = [s for s in skills if any(tag in (s.tags or []) for tag in tags)]

        return skills

    def get_skill(self, skill_id: str, refresh: bool = False) -> Optional[SkillDefinition]:
        """
        Get skill definition by ID

        Args:
            skill_id: Skill identifier
            refresh: Force refresh from registry

        Returns:
            Skill definition or None if not found
        """
        if refresh or skill_id not in self._skill_cache or self._should_refresh_cache():
            self._refresh_cache()

        return self._skill_cache.get(skill_id)

    def search_skills(
        self,
        query: str,
        skill_type: Optional[SkillType] = None,
    ) -> List[SkillDefinition]:
        """
        Search skills by name or description

        Args:
            query: Search query
            skill_type: Filter by skill type

        Returns:
            List of matching skills
        """
        all_skills = self.list_skills(skill_type=skill_type)

        query_lower = query.lower()
        matches = []

        for skill in all_skills:
            if (query_lower in skill.skill_name.lower() or
                query_lower in skill.description.lower()):
                matches.append(skill)

        return matches

    def get_skill_metadata(self, skill_id: str) -> Dict[str, Any]:
        """Get detailed skill metadata"""
        skill = self.get_skill(skill_id, refresh=True)

        if not skill:
            return {}

        return {
            "skill_id": skill.skill_id,
            "skill_name": skill.skill_name,
            "skill_type": skill.skill_type.value,
            "version": skill.version,
            "description": skill.description,
            "required_capabilities": skill.required_capabilities,
            "min_actors": skill.min_actors,
            "max_actors": skill.max_actors,
            "roles": skill.roles,
            "coordination_mode": skill.coordination_mode,
            "created_at": skill.created_at.isoformat() if skill.created_at else None,
            "updated_at": skill.updated_at.isoformat() if skill.updated_at else None,
            "author": skill.author,
            "tags": skill.tags,
            "registry_url": skill.registry_url,
            "artifact_url": skill.artifact_url,
        }

    def _refresh_cache(self) -> None:
        """Refresh skill cache from all registries"""
        logger.info("Refreshing skill cache from registries")

        new_cache = {}

        # Fetch from CSA Registry (multi-actor skills)
        try:
            csa_skills = self._fetch_csa_skills()
            new_cache.update(csa_skills)
            logger.info(f"Fetched {len(csa_skills)} skills from CSA Registry")
        except Exception as e:
            logger.error(f"Failed to fetch from CSA Registry: {e}")

        # Fetch from Dynamical API (single-actor skills)
        try:
            dynamical_skills = self._fetch_dynamical_skills()
            new_cache.update(dynamical_skills)
            logger.info(f"Fetched {len(dynamical_skills)} skills from Dynamical API")
        except Exception as e:
            logger.error(f"Failed to fetch from Dynamical API: {e}")

        self._skill_cache = new_cache
        self._last_refresh = datetime.utcnow()
        logger.info(f"Total skills in cache: {len(self._skill_cache)}")

    def _fetch_csa_skills(self) -> Dict[str, SkillDefinition]:
        """Fetch multi-actor skills from CSA Registry"""
        skills = {}

        try:
            response = requests.get(
                f"{self.csa_registry_url}/api/v1/csa/list",
                timeout=self.timeout,
            )
            response.raise_for_status()

            csa_list = response.json()

            for csa in csa_list.get("csas", []):
                skill_id = csa.get("csa_id")

                skill = SkillDefinition(
                    skill_id=skill_id,
                    skill_name=csa.get("name", skill_id),
                    skill_type=SkillType.MULTI_ACTOR,
                    version=csa.get("version", "1.0.0"),
                    description=csa.get("description", ""),
                    required_capabilities=csa.get("required_capabilities", []),
                    min_actors=csa.get("num_actors", 2),
                    max_actors=csa.get("num_actors", 2),
                    roles=csa.get("roles", {}),
                    coordination_mode=csa.get("coordination_mode"),
                    created_at=datetime.fromisoformat(csa["created_at"]) if "created_at" in csa else None,
                    updated_at=datetime.fromisoformat(csa["updated_at"]) if "updated_at" in csa else None,
                    author=csa.get("author"),
                    tags=csa.get("tags", []),
                    registry_url=self.csa_registry_url,
                    artifact_url=f"{self.csa_registry_url}/api/v1/csa/{skill_id}/download",
                )

                skills[skill_id] = skill

        except requests.RequestException as e:
            logger.warning(f"CSA Registry unavailable: {e}")
            # Return empty dict if registry is unavailable

        return skills

    def _fetch_dynamical_skills(self) -> Dict[str, SkillDefinition]:
        """Fetch single-actor skills from Dynamical API"""
        skills = {}

        try:
            response = requests.get(
                f"{self.dynamical_api_url}/api/v1/skills/list",
                timeout=self.timeout,
            )
            response.raise_for_status()

            skill_list = response.json()

            for skill_data in skill_list.get("skills", []):
                skill_id = skill_data.get("skill_id")

                skill = SkillDefinition(
                    skill_id=skill_id,
                    skill_name=skill_data.get("name", skill_id),
                    skill_type=SkillType.SINGLE_ACTOR,
                    version=skill_data.get("version", "1.0.0"),
                    description=skill_data.get("description", ""),
                    required_capabilities=skill_data.get("required_capabilities", []),
                    min_actors=1,
                    max_actors=1,
                    roles=None,
                    coordination_mode=None,
                    created_at=datetime.fromisoformat(skill_data["created_at"]) if "created_at" in skill_data else None,
                    updated_at=datetime.fromisoformat(skill_data["updated_at"]) if "updated_at" in skill_data else None,
                    author=skill_data.get("author"),
                    tags=skill_data.get("tags", []),
                    registry_url=self.dynamical_api_url,
                    artifact_url=f"{self.dynamical_api_url}/api/v1/skills/{skill_id}/download",
                )

                skills[skill_id] = skill

        except requests.RequestException as e:
            logger.warning(f"Dynamical API unavailable: {e}")
            # Return empty dict if API is unavailable

        return skills

    def _should_refresh_cache(self) -> bool:
        """Check if cache should be refreshed"""
        if not self._last_refresh:
            return True

        age = (datetime.utcnow() - self._last_refresh).total_seconds()
        return age > self._cache_ttl

    def invalidate_cache(self) -> None:
        """Manually invalidate the cache"""
        self._skill_cache.clear()
        self._last_refresh = None
        logger.info("Skill cache invalidated")
