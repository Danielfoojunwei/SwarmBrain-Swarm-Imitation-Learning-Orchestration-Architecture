"""
Cooperative Skill Artifact (CSA) Engine for SwarmBrain

Loads and executes multi-actor cooperative skills packaged as CSA artifacts
from the Multi-Actor repository integration.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RoleType(str, Enum):
    """Actor roles in cooperative tasks"""
    LEADER = "leader"
    FOLLOWER = "follower"
    SPOTTER = "spotter"
    SUPPORT = "support"
    MONITOR = "monitor"
    CUSTOM = "custom"


class ActorIntent(str, Enum):
    """Actor intent types for communication"""
    GRASP = "grasp"
    MOVE = "move"
    WAIT = "wait"
    HANDOFF = "handoff"
    SUPPORT = "support"
    MONITOR = "monitor"


class CoordinationMode(str, Enum):
    """Multi-actor coordination modes"""
    HIERARCHICAL = "hierarchical"  # Leader-follower
    PEER_TO_PEER = "peer_to_peer"  # Equal collaboration
    DYNAMIC = "dynamic"  # Adaptive
    CONSENSUS = "consensus"  # Vote-based


class TaskPhase(str, Enum):
    """Task execution phases (BehaviorTree states)"""
    APPROACH = "approach"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSFER = "transfer"
    PLACE = "place"
    RETREAT = "retreat"
    ABORT = "abort"


@dataclass
class CSAConfig:
    """Configuration for a Cooperative Skill Artifact"""
    csa_id: str
    csa_version: str
    num_actors: int
    roles: Dict[str, RoleType]  # robot_id -> role
    coordination_mode: CoordinationMode
    task_phases: List[TaskPhase]
    safety_envelope: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class MultiActorObservation:
    """Multi-actor observation for coordination"""
    robot_id: str
    role: RoleType
    position: np.ndarray  # [3]
    velocity: np.ndarray  # [3]
    joint_positions: np.ndarray  # [n_joints]
    joint_velocities: np.ndarray  # [n_joints]
    rgb_image: Optional[np.ndarray] = None  # [H, W, 3]
    depth_image: Optional[np.ndarray] = None  # [H, W]
    force_torque: Optional[np.ndarray] = None  # [6]
    actor_positions: Optional[Dict[str, np.ndarray]] = None  # All actor positions
    shared_object_state: Optional[np.ndarray] = None


@dataclass
class CoordinationLatent:
    """Coordination latent representation across all actors"""
    global_latent: torch.Tensor  # [latent_dim]
    pairwise_latents: Dict[Tuple[str, str], torch.Tensor]  # (id1, id2) -> latent
    intent_embeddings: Dict[str, torch.Tensor]  # robot_id -> intent embedding
    coordination_mode: CoordinationMode
    current_phase: TaskPhase


@dataclass
class MultiActorAction:
    """Multi-actor coordinated action"""
    robot_id: str
    action: np.ndarray  # [action_dim]
    intent: ActorIntent
    confidence: float
    requires_coordination: bool
    coordinated_with: List[str]  # List of robot IDs this action coordinates with


class CSAEngine:
    """
    Engine for loading and executing Cooperative Skill Artifacts (CSA)

    Features:
    - Load CSA from Multi-Actor repository format
    - Multi-actor coordination with hierarchical encoding
    - Intent communication and prediction
    - Dynamic role assignment
    - Safety verification
    - BehaviorTree phase state machine
    """

    def __init__(
        self,
        workspace_dir: str,
        device: str = "cpu",
        safety_checks: bool = True,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        self.safety_checks = safety_checks

        # Loaded CSAs
        self.csas: Dict[str, Dict[str, Any]] = {}

        # Active coordination state
        self.coordination_latent: Optional[CoordinationLatent] = None
        self.current_phase: TaskPhase = TaskPhase.APPROACH

        # Safety configuration
        self.min_separation = 0.5  # meters
        self.max_relative_velocity = 0.3  # m/s

        logger.info(f"CSAEngine initialized with workspace: {workspace_dir}")

    def load_csa(self, csa_path: str, csa_id: str) -> CSAConfig:
        """
        Load a Cooperative Skill Artifact from disk

        Args:
            csa_path: Path to CSA tar.gz file or directory
            csa_id: Unique identifier for this CSA

        Returns:
            CSA configuration
        """
        csa_path = Path(csa_path)

        if not csa_path.exists():
            raise FileNotFoundError(f"CSA not found: {csa_path}")

        # Load manifest
        manifest_path = csa_path / "manifest.json" if csa_path.is_dir() else None
        if manifest_path and manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        else:
            # Create default manifest
            manifest = {
                "csa_id": csa_id,
                "version": "1.0.0",
                "num_actors": 2,
                "roles": {},
                "coordination_mode": "hierarchical",
            }

        # Load role adapters
        roles_dir = csa_path / "roles" if csa_path.is_dir() else csa_path
        role_adapters = {}
        if roles_dir.exists():
            for adapter_path in roles_dir.glob("*_adapter.pt"):
                role_name = adapter_path.stem.replace("_adapter", "")
                role_adapters[role_name] = torch.load(adapter_path, map_location=self.device)

        # Load coordination encoder
        coord_encoder_path = csa_path / "coordination_encoder.pt" if csa_path.is_dir() else None
        coordination_encoder = None
        if coord_encoder_path and coord_encoder_path.exists():
            coordination_encoder = torch.load(coord_encoder_path, map_location=self.device)

        # Load BehaviorTree phase machine
        phase_machine_path = csa_path / "phase_machine.xml" if csa_path.is_dir() else None
        phase_machine = None
        if phase_machine_path and phase_machine_path.exists():
            with open(phase_machine_path) as f:
                phase_machine = f.read()

        # Load safety envelope
        safety_path = csa_path / "safety_envelope.json" if csa_path.is_dir() else None
        safety_envelope = {}
        if safety_path and safety_path.exists():
            with open(safety_path) as f:
                safety_envelope = json.load(f)
                self.min_separation = safety_envelope.get("min_separation", self.min_separation)
                self.max_relative_velocity = safety_envelope.get("max_relative_velocity", self.max_relative_velocity)

        # Store CSA
        self.csas[csa_id] = {
            "manifest": manifest,
            "role_adapters": role_adapters,
            "coordination_encoder": coordination_encoder,
            "phase_machine": phase_machine,
            "safety_envelope": safety_envelope,
        }

        # Create config
        config = CSAConfig(
            csa_id=manifest["csa_id"],
            csa_version=manifest.get("version", "1.0.0"),
            num_actors=manifest.get("num_actors", 2),
            roles={k: RoleType(v) for k, v in manifest.get("roles", {}).items()},
            coordination_mode=CoordinationMode(manifest.get("coordination_mode", "hierarchical")),
            task_phases=[TaskPhase(p) for p in manifest.get("task_phases", ["approach", "grasp", "lift", "transfer", "place", "retreat"])],
            safety_envelope=safety_envelope,
            metadata=manifest.get("metadata", {}),
        )

        logger.info(f"Loaded CSA: {csa_id} with {config.num_actors} actors")
        return config

    def execute_multi_actor_skill(
        self,
        csa_id: str,
        observations: Dict[str, MultiActorObservation],
        current_intent: Dict[str, ActorIntent],
    ) -> Dict[str, MultiActorAction]:
        """
        Execute multi-actor cooperative skill

        Args:
            csa_id: ID of the CSA to execute
            observations: Observations from all actors {robot_id -> observation}
            current_intent: Current intent for each actor {robot_id -> intent}

        Returns:
            Coordinated actions for all actors {robot_id -> action}
        """
        if csa_id not in self.csas:
            raise ValueError(f"CSA not loaded: {csa_id}")

        csa = self.csas[csa_id]
        manifest = csa["manifest"]
        num_actors = manifest["num_actors"]

        # 1. Encode coordination latent
        coordination_latent = self._encode_coordination_latent(
            observations, current_intent, csa
        )

        # 2. Update phase based on BehaviorTree state machine
        self.current_phase = self._update_phase(coordination_latent, csa)

        # 3. Compute role-conditioned actions
        actions = {}
        for robot_id, obs in observations.items():
            # Get role-specific adapter
            role = manifest["roles"].get(robot_id, "custom")
            adapter = csa["role_adapters"].get(role)

            if adapter is None:
                logger.warning(f"No adapter found for role: {role}")
                # Return safe default action (zero velocity)
                actions[robot_id] = MultiActorAction(
                    robot_id=robot_id,
                    action=np.zeros(7),  # Default 7-DOF action
                    intent=current_intent.get(robot_id, ActorIntent.WAIT),
                    confidence=0.0,
                    requires_coordination=True,
                    coordinated_with=list(observations.keys()),
                )
                continue

            # Compute action using adapter
            action, confidence = self._compute_action(
                obs, coordination_latent, adapter, role
            )

            # Predict intent for next step
            next_intent = self._predict_next_intent(
                robot_id, coordination_latent, current_intent
            )

            actions[robot_id] = MultiActorAction(
                robot_id=robot_id,
                action=action,
                intent=next_intent,
                confidence=confidence,
                requires_coordination=True,
                coordinated_with=[rid for rid in observations.keys() if rid != robot_id],
            )

        # 4. Safety verification
        if self.safety_checks:
            is_safe, violations = self._verify_safety(observations, actions)
            if not is_safe:
                logger.error(f"Safety violations detected: {violations}")
                # Return emergency stop actions
                return {
                    robot_id: MultiActorAction(
                        robot_id=robot_id,
                        action=np.zeros(7),
                        intent=ActorIntent.WAIT,
                        confidence=0.0,
                        requires_coordination=True,
                        coordinated_with=[],
                    )
                    for robot_id in observations.keys()
                }

        return actions

    def _encode_coordination_latent(
        self,
        observations: Dict[str, MultiActorObservation],
        current_intent: Dict[str, ActorIntent],
        csa: Dict[str, Any],
    ) -> CoordinationLatent:
        """Encode hierarchical coordination latent"""
        # This is a simplified version - in production, use the actual
        # HierarchicalCoordinationEncoder from multi_actor repository

        global_latent = torch.zeros(64).to(self.device)  # Placeholder
        pairwise_latents = {}
        intent_embeddings = {}

        robot_ids = list(observations.keys())
        for i, id1 in enumerate(robot_ids):
            for id2 in robot_ids[i+1:]:
                pairwise_latents[(id1, id2)] = torch.zeros(32).to(self.device)

            # Intent embedding
            intent_embeddings[id1] = torch.zeros(32).to(self.device)

        return CoordinationLatent(
            global_latent=global_latent,
            pairwise_latents=pairwise_latents,
            intent_embeddings=intent_embeddings,
            coordination_mode=CoordinationMode(csa["manifest"].get("coordination_mode", "hierarchical")),
            current_phase=self.current_phase,
        )

    def _update_phase(
        self,
        coordination_latent: CoordinationLatent,
        csa: Dict[str, Any],
    ) -> TaskPhase:
        """Update task phase based on BehaviorTree state machine"""
        # Simplified phase transition logic
        # In production, execute actual BehaviorTree.CPP

        task_phases = csa["manifest"].get("task_phases", [])
        if not task_phases:
            return self.current_phase

        current_idx = task_phases.index(self.current_phase.value) if self.current_phase.value in task_phases else 0

        # Simple progression (in reality, use BT conditions)
        if current_idx < len(task_phases) - 1:
            next_phase = task_phases[current_idx + 1]
            return TaskPhase(next_phase)

        return self.current_phase

    def _compute_action(
        self,
        observation: MultiActorObservation,
        coordination_latent: CoordinationLatent,
        adapter: Dict[str, Any],
        role: str,
    ) -> Tuple[np.ndarray, float]:
        """Compute action using role-specific adapter"""
        # Simplified action computation
        # In production, use actual policy adapter from CSA

        action_dim = 7  # Default 7-DOF action
        action = np.zeros(action_dim)
        confidence = 0.8  # Placeholder confidence

        return action, confidence

    def _predict_next_intent(
        self,
        robot_id: str,
        coordination_latent: CoordinationLatent,
        current_intent: Dict[str, ActorIntent],
    ) -> ActorIntent:
        """Predict next intent for actor"""
        # Simplified intent prediction
        # In production, use IntentCommunicationModule from multi_actor

        return current_intent.get(robot_id, ActorIntent.MOVE)

    def _verify_safety(
        self,
        observations: Dict[str, MultiActorObservation],
        actions: Dict[str, MultiActorAction],
    ) -> Tuple[bool, List[str]]:
        """Verify multi-actor safety constraints"""
        violations = []

        # 1. Check pairwise separation
        robot_ids = list(observations.keys())
        for i, id1 in enumerate(robot_ids):
            for id2 in robot_ids[i+1:]:
                pos1 = observations[id1].position
                pos2 = observations[id2].position
                distance = np.linalg.norm(pos1 - pos2)

                if distance < self.min_separation:
                    violations.append(f"Separation violation: {id1}-{id2} distance={distance:.2f}m")

        # 2. Check relative velocities
        for i, id1 in enumerate(robot_ids):
            for id2 in robot_ids[i+1:]:
                vel1 = observations[id1].velocity
                vel2 = observations[id2].velocity
                rel_vel = np.linalg.norm(vel1 - vel2)

                if rel_vel > self.max_relative_velocity:
                    violations.append(f"Velocity violation: {id1}-{id2} rel_vel={rel_vel:.2f}m/s")

        # 3. Check intent conflicts
        for id1 in robot_ids:
            for id2 in robot_ids:
                if id1 != id2:
                    intent1 = actions[id1].intent
                    intent2 = actions[id2].intent

                    # Example conflict: both trying to grasp simultaneously
                    if intent1 == ActorIntent.GRASP and intent2 == ActorIntent.GRASP:
                        violations.append(f"Intent conflict: {id1} and {id2} both attempting GRASP")

        is_safe = len(violations) == 0
        return is_safe, violations

    def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get current coordination metrics"""
        if self.coordination_latent is None:
            return {}

        return {
            "coordination_mode": self.coordination_latent.coordination_mode.value,
            "current_phase": self.current_phase.value,
            "num_intent_embeddings": len(self.coordination_latent.intent_embeddings),
            "num_pairwise_latents": len(self.coordination_latent.pairwise_latents),
        }
