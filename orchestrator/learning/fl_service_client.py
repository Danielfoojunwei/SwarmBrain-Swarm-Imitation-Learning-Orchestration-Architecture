"""
Federated Learning Service Client

Replaces direct Flower FL integration with API calls to the unified
SwarmBridge federated learning coordinator.

This client allows the orchestrator to:
- Trigger training rounds
- Monitor training progress
- Apply aggregated model updates across the fleet
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class FLRoundStatus(str, Enum):
    """Status of a federated learning round"""
    PENDING = "pending"
    ACTIVE = "active"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FLRoundConfig:
    """Configuration for a federated learning round"""
    round_id: str
    learning_mode: str  # "single_actor", "multi_actor", "hybrid"
    privacy_mode: str  # "ldp", "dp_sgd", "he", "fhe", "secure_agg", "none"
    aggregation_strategy: str  # "mean", "trimmed_mean", "median", "krum", "secure_agg"
    min_participants: int = 2
    max_participants: int = 10
    timeout_seconds: int = 3600

    # Privacy parameters
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    noise_multiplier: Optional[float] = None
    clip_norm: Optional[float] = None

    # Multi-actor specific
    csa_base_id: Optional[str] = None
    num_actors: Optional[int] = None


@dataclass
class FLRoundStatus:
    """Status of a federated learning round"""
    round_id: str
    status: FLRoundStatus
    participants: List[str]
    started_at: datetime
    completed_at: Optional[datetime]
    aggregated_metrics: Dict[str, Any]


class FederatedLearningServiceClient:
    """
    Client for unified federated learning service (SwarmBridge)

    Replaces direct Flower FL integration with API calls to the
    centralized FL coordinator.
    """

    def __init__(
        self,
        swarm_bridge_url: str = "http://localhost:8083",
        timeout: int = 30,
    ):
        self.swarm_bridge_url = swarm_bridge_url.rstrip('/')
        self.timeout = timeout

        logger.info(f"FederatedLearningServiceClient initialized")
        logger.info(f"  SwarmBridge URL: {self.swarm_bridge_url}")

    def start_training_round(
        self,
        round_config: FLRoundConfig,
    ) -> Dict[str, Any]:
        """
        Start a new federated learning round

        Args:
            round_config: Round configuration

        Returns:
            Round start response with status and participants
        """
        logger.info(f"Starting FL round: {round_config.round_id}")

        payload = {
            "round_id": round_config.round_id,
            "learning_mode": round_config.learning_mode,
            "privacy_mode": round_config.privacy_mode,
            "aggregation_strategy": round_config.aggregation_strategy,
            "min_participants": round_config.min_participants,
            "max_participants": round_config.max_participants,
            "timeout_seconds": round_config.timeout_seconds,
            "epsilon": round_config.epsilon,
            "delta": round_config.delta,
            "noise_multiplier": round_config.noise_multiplier,
            "clip_norm": round_config.clip_norm,
            "csa_base_id": round_config.csa_base_id,
            "num_actors": round_config.num_actors,
        }

        try:
            response = requests.post(
                f"{self.swarm_bridge_url}/api/v1/rounds/start",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"FL round started successfully: {result.get('status')}")
            return result

        except requests.RequestException as e:
            logger.error(f"Failed to start FL round: {e}")
            raise

    def get_round_status(self, round_id: str) -> Dict[str, Any]:
        """
        Get status of a federated learning round

        Args:
            round_id: Round identifier

        Returns:
            Round status including participants and progress
        """
        try:
            response = requests.get(
                f"{self.swarm_bridge_url}/api/v1/rounds/{round_id}/status",
                timeout=self.timeout,
            )
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            logger.error(f"Failed to get round status: {e}")
            return {"round_id": round_id, "status": "unknown", "error": str(e)}

    def wait_for_round_completion(
        self,
        round_id: str,
        poll_interval: int = 10,
        max_wait: int = 3600,
    ) -> Dict[str, Any]:
        """
        Wait for a federated learning round to complete

        Args:
            round_id: Round identifier
            poll_interval: Seconds between status checks
            max_wait: Maximum seconds to wait

        Returns:
            Final round status
        """
        import time

        logger.info(f"Waiting for FL round {round_id} to complete")

        start_time = time.time()

        while True:
            status = self.get_round_status(round_id)

            if status.get("status") in ["completed", "failed"]:
                logger.info(f"FL round {round_id} finished with status: {status.get('status')}")
                return status

            elapsed = time.time() - start_time
            if elapsed > max_wait:
                logger.warning(f"FL round {round_id} timed out after {max_wait}s")
                return {"round_id": round_id, "status": "timeout"}

            time.sleep(poll_interval)

    def list_active_rounds(self) -> List[Dict[str, Any]]:
        """
        List all active federated learning rounds

        Returns:
            List of active rounds
        """
        try:
            response = requests.get(
                f"{self.swarm_bridge_url}/api/v1/rounds/active",
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            return result.get("rounds", [])

        except requests.RequestException as e:
            logger.error(f"Failed to list active rounds: {e}")
            return []

    def get_site_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about registered FL sites

        Returns:
            Site statistics including counts by type and mode
        """
        try:
            response = requests.get(
                f"{self.swarm_bridge_url}/api/v1/sites/statistics",
                timeout=self.timeout,
            )
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            logger.error(f"Failed to get site statistics: {e}")
            return {}

    def trigger_skill_update(
        self,
        skill_id: str,
        target_robots: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Trigger skill model update on robots after FL round completion

        Args:
            skill_id: Skill to update
            target_robots: List of robot IDs (None = all robots)

        Returns:
            Update status
        """
        logger.info(f"Triggering skill update for: {skill_id}")

        payload = {
            "skill_id": skill_id,
            "target_robots": target_robots or [],
        }

        try:
            response = requests.post(
                f"{self.swarm_bridge_url}/api/v1/skills/{skill_id}/update",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"Skill update triggered: {result.get('status')}")
            return result

        except requests.RequestException as e:
            logger.error(f"Failed to trigger skill update: {e}")
            raise

    def get_aggregated_metrics(
        self,
        round_id: str,
    ) -> Dict[str, Any]:
        """
        Get aggregated metrics from a completed round

        Args:
            round_id: Round identifier

        Returns:
            Aggregated metrics (loss, accuracy, etc.)
        """
        try:
            response = requests.get(
                f"{self.swarm_bridge_url}/api/v1/rounds/{round_id}/metrics",
                timeout=self.timeout,
            )
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            logger.error(f"Failed to get aggregated metrics: {e}")
            return {}

    def cancel_round(self, round_id: str) -> Dict[str, Any]:
        """
        Cancel an active federated learning round

        Args:
            round_id: Round identifier

        Returns:
            Cancellation status
        """
        logger.info(f"Cancelling FL round: {round_id}")

        try:
            response = requests.post(
                f"{self.swarm_bridge_url}/api/v1/rounds/{round_id}/cancel",
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"FL round cancelled: {result.get('status')}")
            return result

        except requests.RequestException as e:
            logger.error(f"Failed to cancel round: {e}")
            raise

    def monitor_training_progress(
        self,
        round_id: str,
    ) -> Dict[str, Any]:
        """
        Get real-time training progress for a round

        Args:
            round_id: Round identifier

        Returns:
            Training progress metrics
        """
        try:
            response = requests.get(
                f"{self.swarm_bridge_url}/api/v1/rounds/{round_id}/progress",
                timeout=self.timeout,
            )
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            logger.error(f"Failed to get training progress: {e}")
            return {"round_id": round_id, "progress": 0, "status": "unknown"}

    def export_csa_from_training(
        self,
        training_job_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Export trained Cooperative Skill Artifact (CSA) from SwarmBridge

        After multi-actor FL training completes, this method retrieves the
        trained CSA in Dynamical-compatible MoE format.

        Args:
            training_job_id: SwarmBridge training job ID

        Returns:
            CSA dictionary in Dynamical-compatible format, or None if not ready

        Raises:
            requests.RequestException: If API call fails
        """
        logger.info(f"Exporting CSA from training job: {training_job_id}")

        try:
            # Check if training is complete
            status_url = f"{self.swarm_bridge_url}/api/v1/training/jobs/{training_job_id}/status"
            status_response = requests.get(status_url, timeout=self.timeout)
            status_response.raise_for_status()

            status_data = status_response.json()

            if status_data.get("status") != "completed":
                logger.warning(
                    f"Training job {training_job_id} not completed (status={status_data.get('status')}). "
                    "Cannot export CSA yet."
                )
                return None

            # Export CSA
            export_url = f"{self.swarm_bridge_url}/api/v1/training/jobs/{training_job_id}/export_csa"
            export_response = requests.get(export_url, timeout=60)  # Longer timeout for export
            export_response.raise_for_status()

            csa_data = export_response.json()

            # Validate CSA has required fields
            required_fields = ["skill_id", "skill_name", "required_roles", "role_experts", "coordination_encoder"]
            missing_fields = [f for f in required_fields if f not in csa_data]

            if missing_fields:
                logger.error(f"Exported CSA missing required fields: {missing_fields}")
                return None

            logger.info(
                f"✅ Successfully exported CSA: {csa_data['skill_id']} "
                f"(roles={csa_data['required_roles']}, version={csa_data.get('version', 'unknown')})"
            )

            return csa_data

        except requests.RequestException as e:
            logger.error(f"Failed to export CSA from training job {training_job_id}: {e}")
            raise

    def upload_csa_to_registry(
        self,
        csa_data: Dict[str, Any],
        csa_registry_url: str = "http://localhost:8082",
    ) -> bool:
        """
        Upload exported CSA to CSA Registry for discovery and deployment

        Args:
            csa_data: CSA dictionary (from export_csa_from_training)
            csa_registry_url: URL of CSA Registry service

        Returns:
            True if upload successful

        Raises:
            requests.RequestException: If upload fails
        """
        skill_id = csa_data.get("skill_id", "unknown")
        logger.info(f"Uploading CSA to registry: {skill_id}")

        try:
            registry_url = f"{csa_registry_url.rstrip('/')}/api/v1/skills"
            response = requests.post(
                registry_url,
                json=csa_data,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"✅ CSA uploaded to registry: {skill_id} (registry_id={result.get('id')})")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to upload CSA {skill_id} to registry: {e}")
            raise

    def complete_training_to_deployment_pipeline(
        self,
        training_job_id: str,
        csa_registry_url: str = "http://localhost:8082",
    ) -> Optional[str]:
        """
        Complete end-to-end pipeline: Training → CSA Export → Registry Upload

        This is a convenience method that orchestrates the full flow from
        completed SwarmBridge training to deployed CSA ready for use.

        Args:
            training_job_id: SwarmBridge training job ID
            csa_registry_url: URL of CSA Registry

        Returns:
            CSA skill_id if successful, None otherwise
        """
        logger.info(f"Starting training-to-deployment pipeline for job: {training_job_id}")

        try:
            # Step 1: Export CSA from SwarmBridge
            csa_data = self.export_csa_from_training(training_job_id)
            if not csa_data:
                logger.error("Failed to export CSA - training may not be complete")
                return None

            skill_id = csa_data["skill_id"]

            # Step 2: Upload to CSA Registry
            upload_success = self.upload_csa_to_registry(csa_data, csa_registry_url)
            if not upload_success:
                logger.error(f"Failed to upload CSA {skill_id} to registry")
                return None

            logger.info(
                f"🎉 Training-to-deployment pipeline complete for {skill_id}! "
                "CSA is now available for mission planning."
            )
            return skill_id

        except Exception as e:
            logger.error(f"Training-to-deployment pipeline failed: {e}")
            return None
