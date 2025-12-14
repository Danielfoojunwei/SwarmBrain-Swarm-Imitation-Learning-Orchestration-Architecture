"""
CSA Importer Module

Imports Cooperative Skill Artifacts (CSAs) from SwarmBridge training pipeline
and CSA Registry, validates compatibility with Dynamical's MoE format, and
registers them for mission planning.

CSA Flow:
SwarmBridge (training) → CSA Registry → CSA Importer → Skill Registry → Mission Planner
"""

import logging
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from orchestrator.schemas.csa_schema import (
    CooperativeSkillArtifact,
    SkillType,
    EmbeddingType,
    EncryptionScheme,
)

logger = logging.getLogger(__name__)


class CSAImporter:
    """
    Imports and validates Cooperative Skill Artifacts from SwarmBridge/CSA Registry
    """

    def __init__(
        self,
        csa_registry_url: str = "http://localhost:8082",
        swarmbridge_url: str = "http://localhost:8083",
        dynamical_api_url: str = "http://localhost:8085",
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            csa_registry_url: URL of CSA Registry service
            swarmbridge_url: URL of SwarmBridge FL Coordinator
            dynamical_api_url: URL of Dynamical API
            cache_dir: Local cache directory for CSA manifests
        """
        self.csa_registry_url = csa_registry_url.rstrip("/")
        self.swarmbridge_url = swarmbridge_url.rstrip("/")
        self.dynamical_api_url = dynamical_api_url.rstrip("/")

        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".swarmbrain" / "csa_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"CSA Importer initialized with registry={self.csa_registry_url}")

    def fetch_csa_from_registry(self, skill_id: str) -> Optional[CooperativeSkillArtifact]:
        """
        Fetch CSA manifest from CSA Registry

        Args:
            skill_id: Unique skill identifier

        Returns:
            CooperativeSkillArtifact if found, None otherwise
        """
        try:
            url = f"{self.csa_registry_url}/api/v1/skills/{skill_id}"
            response = requests.get(url, timeout=10)

            if response.status_code == 404:
                logger.warning(f"CSA not found in registry: {skill_id}")
                return None

            response.raise_for_status()
            csa_dict = response.json()

            # Deserialize CSA
            csa = CooperativeSkillArtifact.from_dict(csa_dict)

            # Validate
            csa.validate()

            # Cache locally
            self._cache_csa(csa)

            logger.info(f"Fetched CSA from registry: {skill_id} (v{csa.version})")
            return csa

        except requests.RequestException as e:
            logger.error(f"Failed to fetch CSA {skill_id} from registry: {e}")
            # Try loading from cache
            return self._load_from_cache(skill_id)

    def fetch_csa_from_swarmbridge(self, training_job_id: str) -> Optional[CooperativeSkillArtifact]:
        """
        Fetch CSA directly from SwarmBridge after training completes

        Args:
            training_job_id: SwarmBridge training job ID

        Returns:
            CooperativeSkillArtifact if training complete, None otherwise
        """
        try:
            # Check training status
            status_url = f"{self.swarmbridge_url}/api/v1/training/jobs/{training_job_id}/status"
            status_response = requests.get(status_url, timeout=10)
            status_response.raise_for_status()
            status = status_response.json()

            if status["status"] != "completed":
                logger.warning(f"Training job {training_job_id} not completed (status={status['status']})")
                return None

            # Fetch exported CSA
            csa_url = f"{self.swarmbridge_url}/api/v1/training/jobs/{training_job_id}/export_csa"
            csa_response = requests.get(csa_url, timeout=30)
            csa_response.raise_for_status()
            csa_dict = csa_response.json()

            csa = CooperativeSkillArtifact.from_dict(csa_dict)
            csa.validate()

            # Cache and upload to registry
            self._cache_csa(csa)
            self._upload_to_registry(csa)

            logger.info(f"Fetched CSA from SwarmBridge: {csa.skill_id} (training_job={training_job_id})")
            return csa

        except requests.RequestException as e:
            logger.error(f"Failed to fetch CSA from SwarmBridge (job {training_job_id}): {e}")
            return None

    def list_available_csas(self) -> List[Dict[str, Any]]:
        """
        List all available CSAs from registry

        Returns:
            List of CSA metadata (id, name, version, roles, etc.)
        """
        try:
            url = f"{self.csa_registry_url}/api/v1/skills"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            skills = response.json()
            # Filter for multi-actor skills only
            csas = [s for s in skills if s.get("skill_type") == "multi_actor"]

            logger.info(f"Found {len(csas)} CSAs in registry")
            return csas

        except requests.RequestException as e:
            logger.error(f"Failed to list CSAs from registry: {e}")
            return []

    def validate_dynamical_compatibility(self, csa: CooperativeSkillArtifact) -> bool:
        """
        Validate that CSA is compatible with Dynamical's MoE format

        Checks:
        - All role experts use supported embedding types (MOAI)
        - Encryption schemes supported by Dynamical
        - Checkpoint URIs are accessible
        - Architecture types supported by Dynamical

        Args:
            csa: Cooperative Skill Artifact to validate

        Returns:
            True if compatible, raises ValueError otherwise
        """
        # Check embedding types
        supported_embeddings = {EmbeddingType.MOAI_COMPRESSED, EmbeddingType.RAW}
        for role, expert in csa.role_experts.items():
            if expert.embedding_type not in supported_embeddings:
                raise ValueError(
                    f"Role '{role}' uses unsupported embedding type: {expert.embedding_type}. "
                    f"Dynamical supports: {supported_embeddings}"
                )

        # Check encryption schemes
        supported_encryption = {
            EncryptionScheme.OPENFHE_BFV,
            EncryptionScheme.OPENFHE_CKKS,
            EncryptionScheme.NONE,
        }
        for role, expert in csa.role_experts.items():
            if expert.encryption_scheme not in supported_encryption:
                raise ValueError(
                    f"Role '{role}' uses unsupported encryption scheme: {expert.encryption_scheme}. "
                    f"Dynamical supports: {supported_encryption}"
                )

        # Check checkpoint accessibility (optional - just log warnings)
        for role, expert in csa.role_experts.items():
            if not self._check_checkpoint_accessible(expert.checkpoint_uri):
                logger.warning(f"Checkpoint for role '{role}' may not be accessible: {expert.checkpoint_uri}")

        if not self._check_checkpoint_accessible(csa.coordination_encoder.checkpoint_uri):
            logger.warning(
                f"Coordination encoder checkpoint may not be accessible: "
                f"{csa.coordination_encoder.checkpoint_uri}"
            )

        logger.info(f"CSA {csa.skill_id} is compatible with Dynamical")
        return True

    def register_csa_with_dynamical(self, csa: CooperativeSkillArtifact) -> bool:
        """
        Register CSA with Dynamical API for skill execution

        Args:
            csa: Cooperative Skill Artifact to register

        Returns:
            True if registration successful
        """
        try:
            # Validate compatibility first
            self.validate_dynamical_compatibility(csa)

            # Register each role expert as a Dynamical skill
            for role in csa.required_roles:
                expert = csa.role_experts[role]

                # Prepare Dynamical skill payload
                skill_payload = {
                    "skill_id": f"{csa.skill_id}:{role}",  # Unique per role
                    "skill_name": f"{csa.skill_name} ({role})",
                    "skill_type": "multi_actor_role",
                    "role": role,
                    "parent_csa_id": csa.skill_id,
                    "checkpoint_uri": expert.checkpoint_uri,
                    "embedding_type": expert.embedding_type.value,
                    "architecture": expert.architecture.value,
                    "encryption_scheme": expert.encryption_scheme.value,
                    "coordination_encoder_uri": csa.coordination_encoder.checkpoint_uri,
                    "coordination_latent_dim": csa.coordination_encoder.latent_dim,
                    "input_dim": expert.input_dim,
                    "output_dim": expert.output_dim,
                    "version": csa.version,
                }

                # Register with Dynamical API
                url = f"{self.dynamical_api_url}/api/v1/skills/register"
                response = requests.post(url, json=skill_payload, timeout=10)
                response.raise_for_status()

                logger.info(f"Registered CSA role '{role}' with Dynamical: {csa.skill_id}:{role}")

            return True

        except requests.RequestException as e:
            logger.error(f"Failed to register CSA {csa.skill_id} with Dynamical: {e}")
            return False
        except ValueError as e:
            logger.error(f"CSA {csa.skill_id} incompatible with Dynamical: {e}")
            return False

    def import_and_register_csa(self, skill_id: str, source: str = "registry") -> bool:
        """
        Complete import pipeline: fetch CSA, validate, register with Dynamical

        Args:
            skill_id: Skill ID (from registry) or training job ID (from SwarmBridge)
            source: 'registry' or 'swarmbridge'

        Returns:
            True if import successful
        """
        # Fetch CSA
        if source == "registry":
            csa = self.fetch_csa_from_registry(skill_id)
        elif source == "swarmbridge":
            csa = self.fetch_csa_from_swarmbridge(skill_id)
        else:
            logger.error(f"Unknown source: {source}")
            return False

        if not csa:
            logger.error(f"Failed to fetch CSA: {skill_id} (source={source})")
            return False

        # Register with Dynamical
        success = self.register_csa_with_dynamical(csa)

        if success:
            logger.info(f"✅ Successfully imported and registered CSA: {csa.skill_id}")
        else:
            logger.error(f"❌ Failed to import CSA: {skill_id}")

        return success

    # --- Private Helper Methods ---

    def _cache_csa(self, csa: CooperativeSkillArtifact) -> None:
        """Cache CSA manifest locally"""
        cache_file = self.cache_dir / f"{csa.skill_id}.json"
        with open(cache_file, "w") as f:
            json.dump(csa.to_dict(), f, indent=2)
        logger.debug(f"Cached CSA: {cache_file}")

    def _load_from_cache(self, skill_id: str) -> Optional[CooperativeSkillArtifact]:
        """Load CSA from local cache"""
        cache_file = self.cache_dir / f"{skill_id}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                csa_dict = json.load(f)
            csa = CooperativeSkillArtifact.from_dict(csa_dict)
            logger.info(f"Loaded CSA from cache: {skill_id}")
            return csa
        except Exception as e:
            logger.error(f"Failed to load CSA from cache: {e}")
            return None

    def _upload_to_registry(self, csa: CooperativeSkillArtifact) -> bool:
        """Upload CSA to registry (after fetching from SwarmBridge)"""
        try:
            url = f"{self.csa_registry_url}/api/v1/skills"
            response = requests.post(url, json=csa.to_dict(), timeout=10)
            response.raise_for_status()
            logger.info(f"Uploaded CSA to registry: {csa.skill_id}")
            return True
        except requests.RequestException as e:
            logger.warning(f"Failed to upload CSA to registry: {e}")
            return False

    def _check_checkpoint_accessible(self, uri: str) -> bool:
        """Check if checkpoint URI is accessible"""
        # Simple check - just verify format
        if uri.startswith("s3://") or uri.startswith("http://") or uri.startswith("https://"):
            return True  # Assume accessible (full validation requires AWS/HTTP checks)
        elif Path(uri).exists():
            return True
        else:
            return False
