"""
Multi-Actor Coordinator for SwarmBrain

Coordinates multi-robot cooperative tasks using CSA artifacts
and manages role assignment, intent communication, and formation control.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .csa_engine import (
    ActorIntent,
    CoordinationMode,
    CSAConfig,
    CSAEngine,
    MultiActorAction,
    MultiActorObservation,
    RoleType,
    TaskPhase,
)

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of multi-actor task"""
    PENDING = "pending"
    ASSIGNING_ROLES = "assigning_roles"
    IN_PROGRESS = "in_progress"
    COORDINATING = "coordinating"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class RobotCapabilities:
    """Robot capabilities for role assignment"""
    robot_id: str
    strength: float  # 0.0-1.0
    dexterity: float  # 0.0-1.0
    perception: float  # 0.0-1.0
    speed: float  # 0.0-1.0
    reach: float  # meters
    payload: float  # kg
    available: bool = True
    current_battery: float = 1.0  # 0.0-1.0


@dataclass
class FormationConfig:
    """Formation configuration for multi-actor task"""
    formation_type: str  # "line", "circle", "triangle", "custom"
    separation_distance: float  # meters
    relative_positions: Dict[str, np.ndarray]  # role -> relative position
    orientation: str  # "facing_center", "facing_forward", "custom"


@dataclass
class MultiActorTask:
    """Multi-actor cooperative task"""
    task_id: str
    csa_id: str
    required_roles: Dict[RoleType, int]  # role -> count
    assigned_robots: Dict[str, RoleType]  # robot_id -> role
    formation: Optional[FormationConfig]
    status: TaskStatus
    priority: int = 0
    created_at: datetime = datetime.utcnow()
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None


class MultiActorCoordinator:
    """
    Coordinates multi-actor cooperative tasks in SwarmBrain

    Features:
    - Dynamic role assignment based on capabilities
    - Formation control
    - Intent-based coordination
    - Safety verification
    - Task lifecycle management
    """

    def __init__(
        self,
        csa_engine: CSAEngine,
        capabilities_file: Optional[str] = None,
    ):
        self.csa_engine = csa_engine

        # Robot capabilities
        self.robot_capabilities: Dict[str, RobotCapabilities] = {}
        if capabilities_file:
            self._load_capabilities(capabilities_file)

        # Active tasks
        self.tasks: Dict[str, MultiActorTask] = {}

        # Current observations and intents
        self.current_observations: Dict[str, MultiActorObservation] = {}
        self.current_intents: Dict[str, ActorIntent] = {}

        logger.info("MultiActorCoordinator initialized")

    def register_robot(self, capabilities: RobotCapabilities) -> None:
        """Register a robot with its capabilities"""
        self.robot_capabilities[capabilities.robot_id] = capabilities
        logger.info(f"Registered robot: {capabilities.robot_id}")

    def assign_multi_actor_task(
        self,
        task: MultiActorTask,
    ) -> bool:
        """
        Assign a multi-actor task to robots

        Args:
            task: Multi-actor task to assign

        Returns:
            True if task assigned successfully
        """
        # 1. Check if CSA is loaded
        if task.csa_id not in self.csa_engine.csas:
            logger.error(f"CSA not loaded: {task.csa_id}")
            return False

        # 2. Dynamic role assignment
        assigned_robots = self._assign_roles(task.required_roles)

        if not assigned_robots:
            logger.error(f"Failed to assign roles for task: {task.task_id}")
            task.status = TaskStatus.FAILED
            return False

        task.assigned_robots = assigned_robots
        task.status = TaskStatus.ASSIGNING_ROLES
        task.started_at = datetime.utcnow()

        # 3. Initialize formation if configured
        if task.formation:
            self._initialize_formation(task)

        # 4. Store task
        self.tasks[task.task_id] = task

        logger.info(f"Assigned task {task.task_id} to {len(assigned_robots)} robots")
        return True

    def _assign_roles(
        self,
        required_roles: Dict[RoleType, int],
    ) -> Dict[str, RoleType]:
        """
        Dynamically assign roles to robots based on capabilities

        Args:
            required_roles: Required roles and counts

        Returns:
            Assigned robots {robot_id -> role}
        """
        assigned = {}
        available_robots = {
            rid: caps for rid, caps in self.robot_capabilities.items()
            if caps.available and caps.current_battery > 0.2
        }

        # Assign each role type
        for role, count in required_roles.items():
            # Score each available robot for this role
            scores = {}
            for robot_id, caps in available_robots.items():
                if robot_id not in assigned:
                    score = self._compute_role_score(caps, role)
                    scores[robot_id] = score

            # Assign top-scoring robots
            sorted_robots = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for robot_id, score in sorted_robots[:count]:
                assigned[robot_id] = role

        return assigned

    def _compute_role_score(
        self,
        capabilities: RobotCapabilities,
        role: RoleType,
    ) -> float:
        """Compute capability score for a role"""
        if role == RoleType.LEADER:
            # Leader needs high strength and perception
            return 0.6 * capabilities.strength + 0.4 * capabilities.perception

        elif role == RoleType.FOLLOWER:
            # Follower needs balanced capabilities
            return (
                0.3 * capabilities.strength +
                0.3 * capabilities.dexterity +
                0.2 * capabilities.perception +
                0.2 * capabilities.speed
            )

        elif role == RoleType.SPOTTER:
            # Spotter needs high perception
            return 0.8 * capabilities.perception + 0.2 * capabilities.dexterity

        elif role == RoleType.SUPPORT:
            # Support needs high strength and stability
            return 0.7 * capabilities.strength + 0.3 * capabilities.payload

        elif role == RoleType.MONITOR:
            # Monitor needs perception only
            return capabilities.perception

        else:
            # Default balanced score
            return (
                capabilities.strength +
                capabilities.dexterity +
                capabilities.perception
            ) / 3.0

    def _initialize_formation(self, task: MultiActorTask) -> None:
        """Initialize formation for task"""
        if not task.formation:
            return

        formation = task.formation

        # Compute absolute positions from relative positions
        if formation.formation_type == "line":
            # Arrange robots in a line
            center = np.array([0.0, 0.0, 0.0])
            spacing = formation.separation_distance

            for i, (robot_id, role) in enumerate(task.assigned_robots.items()):
                offset = (i - len(task.assigned_robots) / 2.0) * spacing
                formation.relative_positions[robot_id] = center + np.array([offset, 0.0, 0.0])

        elif formation.formation_type == "circle":
            # Arrange robots in a circle
            center = np.array([0.0, 0.0, 0.0])
            radius = formation.separation_distance

            n = len(task.assigned_robots)
            for i, (robot_id, role) in enumerate(task.assigned_robots.items()):
                angle = 2 * np.pi * i / n
                formation.relative_positions[robot_id] = center + np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0.0
                ])

        elif formation.formation_type == "triangle":
            # Arrange robots in a triangle
            if len(task.assigned_robots) >= 3:
                robot_ids = list(task.assigned_robots.keys())
                formation.relative_positions[robot_ids[0]] = np.array([0.0, 0.0, 0.0])
                formation.relative_positions[robot_ids[1]] = np.array([
                    formation.separation_distance, 0.0, 0.0
                ])
                formation.relative_positions[robot_ids[2]] = np.array([
                    formation.separation_distance / 2.0,
                    formation.separation_distance * np.sqrt(3) / 2.0,
                    0.0
                ])

        logger.info(f"Initialized {formation.formation_type} formation for task {task.task_id}")

    def update_observations(
        self,
        observations: Dict[str, MultiActorObservation],
    ) -> None:
        """Update current observations from all actors"""
        self.current_observations.update(observations)

    def execute_coordination_step(
        self,
        task_id: str,
    ) -> Dict[str, MultiActorAction]:
        """
        Execute one coordination step for a task

        Args:
            task_id: Task ID to execute

        Returns:
            Coordinated actions for all actors
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task not found: {task_id}")

        task = self.tasks[task_id]

        # Get observations for assigned robots
        task_observations = {
            robot_id: self.current_observations[robot_id]
            for robot_id in task.assigned_robots.keys()
            if robot_id in self.current_observations
        }

        if not task_observations:
            logger.warning(f"No observations available for task: {task_id}")
            return {}

        # Execute multi-actor skill
        actions = self.csa_engine.execute_multi_actor_skill(
            task.csa_id,
            task_observations,
            self.current_intents,
        )

        # Update task status
        if task.status == TaskStatus.ASSIGNING_ROLES:
            task.status = TaskStatus.IN_PROGRESS

        # Update intents for next step
        for robot_id, action in actions.items():
            self.current_intents[robot_id] = action.intent

        return actions

    def check_task_completion(self, task_id: str) -> bool:
        """Check if task is completed"""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        # Simple completion check based on phase
        # In production, use actual BehaviorTree completion conditions
        current_phase = self.csa_engine.current_phase

        if current_phase == TaskPhase.RETREAT:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            return True

        return False

    def abort_task(self, task_id: str, reason: str = "") -> None:
        """Abort a task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = TaskStatus.ABORTED
            logger.warning(f"Task {task_id} aborted: {reason}")

            # Send stop commands to all assigned robots
            for robot_id in task.assigned_robots.keys():
                self.current_intents[robot_id] = ActorIntent.WAIT

    def get_coordination_status(self, task_id: str) -> Dict[str, Any]:
        """Get coordination status for a task"""
        if task_id not in self.tasks:
            return {}

        task = self.tasks[task_id]
        metrics = self.csa_engine.get_coordination_metrics()

        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "assigned_robots": task.assigned_robots,
            "current_phase": self.csa_engine.current_phase.value,
            "coordination_metrics": metrics,
            "formation": task.formation.formation_type if task.formation else None,
        }

    def _load_capabilities(self, capabilities_file: str) -> None:
        """Load robot capabilities from file"""
        import json
        from pathlib import Path

        cap_path = Path(capabilities_file)
        if not cap_path.exists():
            logger.warning(f"Capabilities file not found: {capabilities_file}")
            return

        with open(cap_path) as f:
            capabilities_data = json.load(f)

        for robot_data in capabilities_data.get("robots", []):
            caps = RobotCapabilities(
                robot_id=robot_data["robot_id"],
                strength=robot_data.get("strength", 0.5),
                dexterity=robot_data.get("dexterity", 0.5),
                perception=robot_data.get("perception", 0.5),
                speed=robot_data.get("speed", 0.5),
                reach=robot_data.get("reach", 1.0),
                payload=robot_data.get("payload", 10.0),
                available=robot_data.get("available", True),
                current_battery=robot_data.get("battery", 1.0),
            )
            self.robot_capabilities[caps.robot_id] = caps

        logger.info(f"Loaded capabilities for {len(self.robot_capabilities)} robots")
