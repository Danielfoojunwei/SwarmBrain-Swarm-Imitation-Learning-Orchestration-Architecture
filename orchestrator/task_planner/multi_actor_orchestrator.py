"""
Multi-Actor Mission Orchestrator Extension

Extends MissionOrchestrator to support multi-actor cooperative tasks using
CSA artifacts from the Multi-Actor repository integration.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .mission_orchestrator import (
    CoordinationPrimitive,
    MissionOrchestrator,
    Task,
    TaskGraph,
    TaskStatus,
    WorkOrder,
)

# Import multi-actor integration components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "integration"))

from multi_actor_adapter import (
    CSAEngine,
    MultiActorCoordinator,
    MultiActorTask,
    RoleType,
    ActorIntent,
    CoordinationMode,
    FormationConfig,
)

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Type of task"""
    SINGLE_ACTOR = "single_actor"  # Single robot task
    MULTI_ACTOR = "multi_actor"  # Multi-robot cooperative task


@dataclass
class ExtendedTask(Task):
    """Extended task with multi-actor support"""
    task_type: TaskType = TaskType.SINGLE_ACTOR
    required_actors: int = 1
    required_roles: Dict[RoleType, int] = field(default_factory=dict)
    csa_id: Optional[str] = None
    formation: Optional[FormationConfig] = None


class MultiActorMissionOrchestrator(MissionOrchestrator):
    """
    Extended orchestrator with multi-actor cooperative task support

    Integrates:
    - Single-actor tasks (existing SwarmBrain functionality)
    - Multi-actor cooperative tasks (CSA-based)
    - Dynamic role assignment
    - Formation control
    - Intent-based coordination
    """

    def __init__(
        self,
        csa_workspace: str = "./robot_control/skills/csa",
        capabilities_file: Optional[str] = None,
    ):
        super().__init__()

        # Initialize CSA engine
        self.csa_engine = CSAEngine(workspace_dir=csa_workspace)

        # Initialize multi-actor coordinator
        self.multi_actor_coordinator = MultiActorCoordinator(
            csa_engine=self.csa_engine,
            capabilities_file=capabilities_file,
        )

        # Multi-actor task tracking
        self.multi_actor_tasks: Dict[str, MultiActorTask] = {}

        logger.info("MultiActorMissionOrchestrator initialized")

    def load_csa(self, csa_path: str, csa_id: str) -> None:
        """Load a Cooperative Skill Artifact"""
        config = self.csa_engine.load_csa(csa_path, csa_id)
        logger.info(f"Loaded CSA: {csa_id} ({config.num_actors} actors)")

    def create_multi_actor_mission(
        self,
        work_order: WorkOrder,
    ) -> TaskGraph:
        """
        Create a mission with both single-actor and multi-actor tasks

        Args:
            work_order: Work order with mixed task types

        Returns:
            TaskGraph with extended task types
        """
        task_graph = TaskGraph()

        for task_spec in work_order.tasks:
            task_type = TaskType(task_spec.get("task_type", "single_actor"))

            if task_type == TaskType.SINGLE_ACTOR:
                # Create standard single-actor task
                task = Task(
                    task_id=f"{work_order.order_id}_{task_spec['id']}",
                    skill=task_spec["skill"],
                    role=task_spec.get("role", "default"),
                    dependencies=task_spec.get("dependencies", []),
                    coordination=self._parse_coordination(task_spec.get("coordination")),
                    metadata=task_spec.get("metadata", {}),
                )

            elif task_type == TaskType.MULTI_ACTOR:
                # Create multi-actor cooperative task
                required_roles = task_spec.get("required_roles", {})
                required_roles_enum = {
                    RoleType(role): count
                    for role, count in required_roles.items()
                }

                # Parse formation if specified
                formation = None
                if "formation" in task_spec:
                    formation = FormationConfig(
                        formation_type=task_spec["formation"].get("type", "line"),
                        separation_distance=task_spec["formation"].get("separation", 1.0),
                        relative_positions={},
                        orientation=task_spec["formation"].get("orientation", "facing_center"),
                    )

                task = ExtendedTask(
                    task_id=f"{work_order.order_id}_{task_spec['id']}",
                    skill=task_spec["skill"],
                    role=task_spec.get("role", "cooperative"),
                    dependencies=task_spec.get("dependencies", []),
                    coordination=CoordinationPrimitive.RENDEZVOUS,  # Multi-actor tasks require coordination
                    metadata=task_spec.get("metadata", {}),
                    task_type=TaskType.MULTI_ACTOR,
                    required_actors=task_spec.get("required_actors", 2),
                    required_roles=required_roles_enum,
                    csa_id=task_spec.get("csa_id"),
                    formation=formation,
                )

            task_graph.add_task(task)

        # Validate the task graph
        if not task_graph.is_valid():
            raise ValueError(f"Task graph for order {work_order.order_id} contains cycles")

        self.active_missions[work_order.order_id] = task_graph
        logger.info(
            f"Created multi-actor mission {work_order.order_id} with {len(task_graph.tasks)} tasks"
        )

        return task_graph

    async def assign_tasks(self, mission_id: str) -> Dict[str, Any]:
        """
        Assign both single-actor and multi-actor tasks

        Args:
            mission_id: Mission ID to assign tasks for

        Returns:
            Assignment results
        """
        task_graph = self.active_missions.get(mission_id)
        if not task_graph:
            raise ValueError(f"Mission {mission_id} not found")

        ready_tasks = task_graph.get_ready_tasks()

        single_actor_assignments = {}
        multi_actor_assignments = {}

        for task in ready_tasks:
            if isinstance(task, ExtendedTask) and task.task_type == TaskType.MULTI_ACTOR:
                # Assign multi-actor task
                success = await self._assign_multi_actor_task(task)
                if success:
                    multi_actor_assignments[task.task_id] = self.multi_actor_tasks[task.task_id]
            else:
                # Assign single-actor task using parent class logic
                assignment = await super().assign_tasks(mission_id)
                single_actor_assignments.update(assignment)

        return {
            "single_actor": single_actor_assignments,
            "multi_actor": multi_actor_assignments,
        }

    async def _assign_multi_actor_task(self, task: ExtendedTask) -> bool:
        """Assign a multi-actor cooperative task"""
        if not task.csa_id:
            logger.error(f"Multi-actor task {task.task_id} missing csa_id")
            return False

        # Create MultiActorTask for the coordinator
        multi_actor_task = MultiActorTask(
            task_id=task.task_id,
            csa_id=task.csa_id,
            required_roles=task.required_roles,
            assigned_robots={},
            formation=task.formation,
            status="pending",
            priority=task.metadata.get("priority", 0),
            metadata=task.metadata,
        )

        # Assign task to multi-actor coordinator
        success = self.multi_actor_coordinator.assign_multi_actor_task(multi_actor_task)

        if success:
            task.status = TaskStatus.ASSIGNED
            self.multi_actor_tasks[task.task_id] = multi_actor_task
            logger.info(f"Assigned multi-actor task: {task.task_id}")
        else:
            logger.error(f"Failed to assign multi-actor task: {task.task_id}")

        return success

    async def execute_multi_actor_step(
        self,
        task_id: str,
        observations: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute one coordination step for a multi-actor task

        Args:
            task_id: Multi-actor task ID
            observations: Observations from all actors

        Returns:
            Coordinated actions for all actors
        """
        if task_id not in self.multi_actor_tasks:
            raise ValueError(f"Multi-actor task not found: {task_id}")

        # Update observations
        self.multi_actor_coordinator.update_observations(observations)

        # Execute coordination step
        actions = self.multi_actor_coordinator.execute_coordination_step(task_id)

        # Check task completion
        is_complete = self.multi_actor_coordinator.check_task_completion(task_id)
        if is_complete:
            # Update task status in task graph
            for mission_id, task_graph in self.active_missions.items():
                if task_id in task_graph.tasks:
                    task_graph.tasks[task_id].status = TaskStatus.COMPLETED
                    logger.info(f"Multi-actor task completed: {task_id}")

        return {
            "actions": actions,
            "is_complete": is_complete,
        }

    def get_multi_actor_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a multi-actor task"""
        return self.multi_actor_coordinator.get_coordination_status(task_id)

    def get_mission_status_extended(self, mission_id: str) -> Dict[str, Any]:
        """Get extended mission status including multi-actor tasks"""
        base_status = self.get_mission_status(mission_id)

        # Add multi-actor task details
        multi_actor_status = {}
        for task_id, multi_task in self.multi_actor_tasks.items():
            if task_id.startswith(mission_id):
                multi_actor_status[task_id] = {
                    "status": multi_task.status,
                    "assigned_robots": multi_task.assigned_robots,
                    "csa_id": multi_task.csa_id,
                    "coordination": self.get_multi_actor_status(task_id),
                }

        base_status["multi_actor_tasks"] = multi_actor_status
        return base_status

    def register_robot_extended(
        self,
        robot_id: str,
        capabilities: List[str],
        robot_capabilities: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Register robot with extended capabilities for multi-actor tasks

        Args:
            robot_id: Robot ID
            capabilities: List of skills the robot can perform
            robot_capabilities: Capability scores {strength, dexterity, perception, etc.}
            metadata: Additional metadata
        """
        # Register with parent class
        self.register_robot(robot_id, capabilities, metadata)

        # Register with multi-actor coordinator
        from multi_actor_adapter.multi_actor_coordinator import RobotCapabilities

        caps = RobotCapabilities(
            robot_id=robot_id,
            strength=robot_capabilities.get("strength", 0.5),
            dexterity=robot_capabilities.get("dexterity", 0.5),
            perception=robot_capabilities.get("perception", 0.5),
            speed=robot_capabilities.get("speed", 0.5),
            reach=robot_capabilities.get("reach", 1.0),
            payload=robot_capabilities.get("payload", 10.0),
            available=True,
            current_battery=robot_capabilities.get("battery", 1.0),
        )

        self.multi_actor_coordinator.register_robot(caps)
        logger.info(f"Registered robot {robot_id} with multi-actor capabilities")
