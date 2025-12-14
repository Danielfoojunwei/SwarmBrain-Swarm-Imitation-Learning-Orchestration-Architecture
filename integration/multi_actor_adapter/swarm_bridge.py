"""
SwarmBridge: Unified Federated Learning Coordinator

Bridges SwarmBrain's Flower FL (single-actor) with Multi-Actor's OpenFL (multi-actor)
for unified privacy-preserving swarm learning.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class LearningMode(str, Enum):
    """Federated learning mode"""
    SINGLE_ACTOR = "single_actor"  # Use Flower FL (dynamical_2)
    MULTI_ACTOR = "multi_actor"  # Use OpenFL (multi_actor CSA)
    HYBRID = "hybrid"  # Both modes active


class PrivacyMode(str, Enum):
    """Privacy-preserving mode"""
    NONE = "none"
    LDP = "ldp"  # Local Differential Privacy
    DP_SGD = "dp_sgd"  # Differential Privacy SGD
    HE = "he"  # Homomorphic Encryption
    FHE = "fhe"  # Fully Homomorphic Encryption
    SECURE_AGG = "secure_agg"  # Dropout-resilient secure aggregation (SwarmBrain)


class AggregationStrategy(str, Enum):
    """Aggregation strategy"""
    MEAN = "mean"  # Simple average
    WEIGHTED_MEAN = "weighted_mean"  # Weighted by data size
    TRIMMED_MEAN = "trimmed_mean"  # Remove outliers
    MEDIAN = "median"  # Element-wise median
    KRUM = "krum"  # Byzantine-robust Krum
    SECURE_AGG = "secure_agg"  # SwarmBrain's dropout-resilient


@dataclass
class SwarmRoundConfig:
    """Configuration for a swarm learning round"""
    round_id: str
    learning_mode: LearningMode
    privacy_mode: PrivacyMode
    aggregation_strategy: AggregationStrategy
    min_participants: int = 2
    max_participants: int = 10
    timeout_seconds: int = 3600

    # Privacy parameters
    epsilon: Optional[float] = None  # DP epsilon
    delta: Optional[float] = None  # DP delta
    noise_multiplier: Optional[float] = None  # DP-SGD noise
    clip_norm: Optional[float] = None  # Gradient clipping
    dropout_threshold: Optional[float] = 0.4  # Secure agg threshold

    # Multi-actor specific
    csa_base_id: Optional[str] = None  # Base CSA for multi-actor learning
    num_actors: Optional[int] = None  # Number of actors in cooperative task


@dataclass
class SwarmSite:
    """Federated learning site"""
    site_id: str
    site_type: str  # "factory", "lab", "edge"
    learning_mode: LearningMode
    privacy_preferences: List[PrivacyMode]
    active: bool = True
    last_seen: datetime = datetime.utcnow()
    metadata: Dict[str, Any] = None


class SwarmBridge:
    """
    Unified coordinator for single-actor and multi-actor federated learning

    Integrates:
    - Flower FL (SwarmBrain) for single-actor skills
    - OpenFL (Multi-Actor) for cooperative CSA artifacts
    - Unified privacy mechanisms (LDP, DP-SGD, HE, Secure Aggregation)
    - Cross-mode learning transfer
    """

    def __init__(
        self,
        workspace_dir: str,
        flower_server_address: str = "localhost:8080",
        openfl_coordinator_address: str = "localhost:8081",
        csa_registry_url: str = "http://localhost:8082",
    ):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.flower_server_address = flower_server_address
        self.openfl_coordinator_address = openfl_coordinator_address
        self.csa_registry_url = csa_registry_url

        # Registered sites
        self.sites: Dict[str, SwarmSite] = {}

        # Active rounds
        self.active_rounds: Dict[str, SwarmRoundConfig] = {}

        # Round history
        self.round_history: List[Dict[str, Any]] = []

        logger.info(f"SwarmBridge initialized with workspace: {workspace_dir}")

    async def register_site(
        self,
        site_id: str,
        site_type: str,
        learning_mode: LearningMode,
        privacy_preferences: List[PrivacyMode],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Register a federated learning site

        Args:
            site_id: Unique site identifier
            site_type: Type of site (factory, lab, edge)
            learning_mode: Supported learning mode
            privacy_preferences: Preferred privacy modes
            metadata: Additional site metadata

        Returns:
            Registration response
        """
        site = SwarmSite(
            site_id=site_id,
            site_type=site_type,
            learning_mode=learning_mode,
            privacy_preferences=privacy_preferences,
            active=True,
            last_seen=datetime.utcnow(),
            metadata=metadata or {},
        )

        self.sites[site_id] = site

        logger.info(f"Registered site: {site_id} ({site_type}, {learning_mode.value})")

        return {
            "status": "registered",
            "site_id": site_id,
            "flower_server": self.flower_server_address,
            "openfl_coordinator": self.openfl_coordinator_address,
            "csa_registry": self.csa_registry_url,
        }

    async def start_swarm_round(
        self,
        round_config: SwarmRoundConfig,
    ) -> Dict[str, Any]:
        """
        Start a federated learning round

        Args:
            round_config: Round configuration

        Returns:
            Round initialization response
        """
        # Validate configuration
        eligible_sites = self._get_eligible_sites(round_config)

        if len(eligible_sites) < round_config.min_participants:
            return {
                "status": "failed",
                "reason": f"Insufficient sites: {len(eligible_sites)} < {round_config.min_participants}",
            }

        # Store round
        self.active_rounds[round_config.round_id] = round_config

        # Route to appropriate FL system
        if round_config.learning_mode == LearningMode.SINGLE_ACTOR:
            result = await self._start_flower_round(round_config, eligible_sites)
        elif round_config.learning_mode == LearningMode.MULTI_ACTOR:
            result = await self._start_openfl_round(round_config, eligible_sites)
        elif round_config.learning_mode == LearningMode.HYBRID:
            result = await self._start_hybrid_round(round_config, eligible_sites)
        else:
            return {"status": "failed", "reason": "Unknown learning mode"}

        # Record in history
        self.round_history.append({
            "round_id": round_config.round_id,
            "learning_mode": round_config.learning_mode.value,
            "privacy_mode": round_config.privacy_mode.value,
            "participants": [s.site_id for s in eligible_sites],
            "started_at": datetime.utcnow().isoformat(),
            "result": result,
        })

        return result

    async def _start_flower_round(
        self,
        round_config: SwarmRoundConfig,
        sites: List[SwarmSite],
    ) -> Dict[str, Any]:
        """Start single-actor FL round using Flower"""
        logger.info(f"Starting Flower FL round: {round_config.round_id}")

        # Apply privacy mechanism
        privacy_config = self._get_privacy_config(round_config)

        # In production, integrate with actual Flower server
        # For now, return simulated response
        return {
            "status": "started",
            "round_id": round_config.round_id,
            "fl_system": "flower",
            "participants": [s.site_id for s in sites],
            "privacy_config": privacy_config,
            "server_address": self.flower_server_address,
        }

    async def _start_openfl_round(
        self,
        round_config: SwarmRoundConfig,
        sites: List[SwarmSite],
    ) -> Dict[str, Any]:
        """Start multi-actor FL round using OpenFL"""
        logger.info(f"Starting OpenFL round: {round_config.round_id}")

        # Apply privacy mechanism
        privacy_config = self._get_privacy_config(round_config)

        # Get base CSA
        if not round_config.csa_base_id:
            return {
                "status": "failed",
                "reason": "csa_base_id required for multi-actor learning",
            }

        # In production, integrate with actual OpenFL coordinator
        # For now, return simulated response
        return {
            "status": "started",
            "round_id": round_config.round_id,
            "fl_system": "openfl",
            "participants": [s.site_id for s in sites],
            "privacy_config": privacy_config,
            "csa_base_id": round_config.csa_base_id,
            "coordinator_address": self.openfl_coordinator_address,
        }

    async def _start_hybrid_round(
        self,
        round_config: SwarmRoundConfig,
        sites: List[SwarmSite],
    ) -> Dict[str, Any]:
        """Start hybrid round (both single-actor and multi-actor)"""
        logger.info(f"Starting hybrid FL round: {round_config.round_id}")

        # Partition sites by capability
        single_actor_sites = [s for s in sites if s.learning_mode in [LearningMode.SINGLE_ACTOR, LearningMode.HYBRID]]
        multi_actor_sites = [s for s in sites if s.learning_mode in [LearningMode.MULTI_ACTOR, LearningMode.HYBRID]]

        results = {}

        # Start Flower round if applicable
        if single_actor_sites:
            flower_result = await self._start_flower_round(round_config, single_actor_sites)
            results["flower"] = flower_result

        # Start OpenFL round if applicable
        if multi_actor_sites:
            openfl_result = await self._start_openfl_round(round_config, multi_actor_sites)
            results["openfl"] = openfl_result

        return {
            "status": "started",
            "round_id": round_config.round_id,
            "fl_system": "hybrid",
            "results": results,
        }

    def _get_eligible_sites(self, round_config: SwarmRoundConfig) -> List[SwarmSite]:
        """Get sites eligible for this round"""
        eligible = []

        for site in self.sites.values():
            if not site.active:
                continue

            # Check learning mode compatibility
            if round_config.learning_mode == LearningMode.HYBRID:
                eligible.append(site)
            elif site.learning_mode in [round_config.learning_mode, LearningMode.HYBRID]:
                eligible.append(site)

            # Check privacy mode support
            if round_config.privacy_mode not in site.privacy_preferences:
                if PrivacyMode.NONE not in site.privacy_preferences:
                    logger.warning(f"Site {site.site_id} does not support {round_config.privacy_mode.value}")

        return eligible[:round_config.max_participants]

    def _get_privacy_config(self, round_config: SwarmRoundConfig) -> Dict[str, Any]:
        """Get privacy configuration for round"""
        config = {
            "mode": round_config.privacy_mode.value,
        }

        if round_config.privacy_mode == PrivacyMode.LDP:
            config["epsilon"] = round_config.epsilon or 1.0

        elif round_config.privacy_mode == PrivacyMode.DP_SGD:
            config["epsilon"] = round_config.epsilon or 1.0
            config["delta"] = round_config.delta or 1e-5
            config["noise_multiplier"] = round_config.noise_multiplier or 1.1
            config["clip_norm"] = round_config.clip_norm or 1.0

        elif round_config.privacy_mode == PrivacyMode.SECURE_AGG:
            config["threshold_ratio"] = round_config.dropout_threshold or 0.4

        elif round_config.privacy_mode in [PrivacyMode.HE, PrivacyMode.FHE]:
            config["scheme"] = "CKKS"  # For real-valued weights
            config["poly_modulus_degree"] = 8192

        return config

    async def aggregate_round(
        self,
        round_id: str,
        site_updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Aggregate updates from sites

        Args:
            round_id: Round identifier
            site_updates: Updates from each site {site_id -> update}

        Returns:
            Aggregation result
        """
        if round_id not in self.active_rounds:
            return {"status": "failed", "reason": "Round not found"}

        round_config = self.active_rounds[round_id]

        # Apply aggregation strategy
        if round_config.aggregation_strategy == AggregationStrategy.MEAN:
            aggregated = self._aggregate_mean(site_updates)
        elif round_config.aggregation_strategy == AggregationStrategy.TRIMMED_MEAN:
            aggregated = self._aggregate_trimmed_mean(site_updates, trim_ratio=0.1)
        elif round_config.aggregation_strategy == AggregationStrategy.MEDIAN:
            aggregated = self._aggregate_median(site_updates)
        elif round_config.aggregation_strategy == AggregationStrategy.KRUM:
            aggregated = self._aggregate_krum(site_updates)
        elif round_config.aggregation_strategy == AggregationStrategy.SECURE_AGG:
            aggregated = self._aggregate_secure_agg(site_updates, round_config)
        else:
            aggregated = self._aggregate_mean(site_updates)

        # Clean up round
        del self.active_rounds[round_id]

        return {
            "status": "completed",
            "round_id": round_id,
            "aggregated_update": aggregated,
            "num_participants": len(site_updates),
        }

    def _aggregate_mean(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Simple mean aggregation"""
        # Simplified - in production, aggregate actual model parameters
        return {"aggregation": "mean", "num_updates": len(updates)}

    def _aggregate_trimmed_mean(self, updates: Dict[str, Any], trim_ratio: float = 0.1) -> Dict[str, Any]:
        """Trimmed mean aggregation (Byzantine-robust)"""
        return {"aggregation": "trimmed_mean", "trim_ratio": trim_ratio, "num_updates": len(updates)}

    def _aggregate_median(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Element-wise median aggregation"""
        return {"aggregation": "median", "num_updates": len(updates)}

    def _aggregate_krum(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Krum aggregation (Byzantine-robust)"""
        return {"aggregation": "krum", "num_updates": len(updates)}

    def _aggregate_secure_agg(self, updates: Dict[str, Any], round_config: SwarmRoundConfig) -> Dict[str, Any]:
        """SwarmBrain's dropout-resilient secure aggregation"""
        return {
            "aggregation": "secure_agg",
            "threshold": round_config.dropout_threshold,
            "num_updates": len(updates),
        }

    def get_round_status(self, round_id: str) -> Dict[str, Any]:
        """Get status of a round"""
        if round_id in self.active_rounds:
            config = self.active_rounds[round_id]
            return {
                "round_id": round_id,
                "status": "active",
                "learning_mode": config.learning_mode.value,
                "privacy_mode": config.privacy_mode.value,
            }

        # Check history
        for record in reversed(self.round_history):
            if record["round_id"] == round_id:
                return {
                    "round_id": round_id,
                    "status": "completed",
                    "started_at": record["started_at"],
                    "participants": record["participants"],
                }

        return {"round_id": round_id, "status": "not_found"}

    def get_site_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered sites"""
        total_sites = len(self.sites)
        active_sites = sum(1 for s in self.sites.values() if s.active)

        by_type = {}
        by_mode = {}

        for site in self.sites.values():
            by_type[site.site_type] = by_type.get(site.site_type, 0) + 1
            by_mode[site.learning_mode.value] = by_mode.get(site.learning_mode.value, 0) + 1

        return {
            "total_sites": total_sites,
            "active_sites": active_sites,
            "by_type": by_type,
            "by_learning_mode": by_mode,
            "total_rounds": len(self.round_history),
            "active_rounds": len(self.active_rounds),
        }
