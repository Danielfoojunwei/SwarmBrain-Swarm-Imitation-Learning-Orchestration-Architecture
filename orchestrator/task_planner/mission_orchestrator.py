"""
Mission Orchestrator Service

Transforms work orders into task graphs and coordinates multi-robot missions.
Uses behavior trees for workflow modeling and supports Open-RMF integration.

REFACTORED: Now uses unified skill registry instead of local skill management.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from datetime import datetime
import networkx as nx

# Import unified skill registry
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from registry import UnifiedSkillRegistryClient, SkillDefinition, SkillType


class TaskStatus(Enum):
    """Status of a task in the task graph."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class CoordinationPrimitive(Enum):
    """Types of coordination primitives."""
    HANDOVER = "handover"
    MUTEX = "mutex"
    BARRIER = "barrier"
    RENDEZVOUS = "rendezvous"


@dataclass
class Task:
    """Represents a single task in a mission."""
    task_id: str
    skill: str
    role: str
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_robot: Optional[str] = None
    coordination: Optional[CoordinationPrimitive] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkOrder:
    """A work order to be executed by the swarm."""
    order_id: str
    description: str
    tasks: List[Dict[str, Any]]
    priority: int = 1
    deadline: Optional[datetime] = None


class TaskGraph:
    """
    Directed acyclic graph of tasks with dependencies.
    Built using NetworkX for efficient traversal and analysis.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.tasks: Dict[str, Task] = {}

    def add_task(self, task: Task):
        """Add a task to the graph."""
        self.tasks[task.task_id] = task
        self.graph.add_node(task.task_id, task=task)

        # Add edges for dependencies
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                self.graph.add_edge(dep_id, task.task_id)

    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        ready = []
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING:
                deps_satisfied = all(
                    self.tasks[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                    if dep_id in self.tasks
                )
                if deps_satisfied:
                    ready.append(task)
        return ready

    def get_critical_path(self) -> List[str]:
        """Return the critical path (longest path) through the task graph."""
        try:
            return nx.dag_longest_path(self.graph)
        except nx.NetworkXError:
            return []

    def is_valid(self) -> bool:
        """Check if the task graph is a valid DAG."""
        return nx.is_directed_acyclic_graph(self.graph)


class MissionOrchestrator:
    """
    Main orchestrator service that converts work orders into task graphs
    and coordinates robot execution.

    REFACTORED: Uses UnifiedSkillRegistryClient for skill discovery instead
    of local skill management.
    """

    def __init__(
        self,
        csa_registry_url: str = "http://localhost:8082",
        dynamical_api_url: str = "http://localhost:8085",
    ):
        self.logger = logging.getLogger(__name__)
        self.active_missions: Dict[str, TaskGraph] = {}
        self.robot_registry: Dict[str, Dict[str, Any]] = {}

        # Use unified skill registry client
        self.skill_registry_client = UnifiedSkillRegistryClient(
            csa_registry_url=csa_registry_url,
            dynamical_api_url=dynamical_api_url,
        )
        self.logger.info("MissionOrchestrator initialized with unified skill registry")

    def register_robot(
        self,
        robot_id: str,
        capabilities: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register a robot with its capabilities."""
        self.robot_registry[robot_id] = {
            'capabilities': capabilities,
            'status': 'available',
            'current_task': None,
            'metadata': metadata or {}
        }
        self.logger.info(f'Registered robot {robot_id} with capabilities: {capabilities}')

    def list_available_skills(
        self,
        skill_type: Optional[SkillType] = None,
    ) -> List[SkillDefinition]:
        """
        List available skills from unified registry.

        Args:
            skill_type: Filter by skill type (single_actor, multi_actor, hybrid)

        Returns:
            List of available skill definitions
        """
        return self.skill_registry_client.list_skills(skill_type=skill_type)

    def get_skill_info(self, skill_id: str) -> Optional[SkillDefinition]:
        """
        Get skill information from registry.

        Args:
            skill_id: Skill identifier

        Returns:
            Skill definition or None if not found
        """
        return self.skill_registry_client.get_skill(skill_id)

    def create_mission(self, work_order: WorkOrder) -> TaskGraph:
        """
        Convert a work order into an executable task graph.

        REFACTORED: Validates skills against unified registry instead of local storage.

        Args:
            work_order: The work order to plan

        Returns:
            TaskGraph ready for execution
        """
        task_graph = TaskGraph()

        # Create tasks from work order
        for task_spec in work_order.tasks:
            skill_name = task_spec['skill']

            # Validate skill exists in registry
            skill_def = self.skill_registry_client.get_skill(skill_name)
            if not skill_def:
                # Try searching by name
                matches = self.skill_registry_client.search_skills(skill_name)
                if matches:
                    skill_def = matches[0]
                    self.logger.warning(
                        f"Skill '{skill_name}' not found by ID, using '{skill_def.skill_id}' from search"
                    )
                else:
                    raise ValueError(f"Skill '{skill_name}' not found in registry")

            # Store skill metadata in task
            task_metadata = task_spec.get('metadata', {})
            task_metadata['skill_definition'] = skill_def

            task = Task(
                task_id=f"{work_order.order_id}_{task_spec['id']}",
                skill=skill_def.skill_id,  # Use skill ID from registry
                role=task_spec.get('role', 'default'),
                dependencies=task_spec.get('dependencies', []),
                coordination=self._parse_coordination(task_spec.get('coordination')),
                metadata=task_metadata,
            )
            task_graph.add_task(task)

        # Validate the task graph
        if not task_graph.is_valid():
            raise ValueError(f'Task graph for order {work_order.order_id} contains cycles')

        self.active_missions[work_order.order_id] = task_graph
        self.logger.info(
            f'Created mission {work_order.order_id} with {len(task_graph.tasks)} tasks'
        )

        return task_graph

    def _parse_coordination(self, coord_spec: Optional[str]) -> Optional[CoordinationPrimitive]:
        """Parse coordination primitive from specification."""
        if not coord_spec:
            return None
        try:
            return CoordinationPrimitive(coord_spec.lower())
        except ValueError:
            self.logger.warning(f'Unknown coordination primitive: {coord_spec}')
            return None

    async def assign_tasks(self, mission_id: str) -> Dict[str, str]:
        """
        Assign ready tasks to available robots.

        REFACTORED: Uses skill definitions from registry to match robot capabilities.

        Args:
            mission_id: ID of the mission to assign tasks for

        Returns:
            Mapping of task_id to robot_id
        """
        task_graph = self.active_missions.get(mission_id)
        if not task_graph:
            raise ValueError(f'Mission {mission_id} not found')

        ready_tasks = task_graph.get_ready_tasks()
        available_robots = [
            rid for rid, info in self.robot_registry.items()
            if info['status'] == 'available'
        ]

        assignments = {}

        for task in ready_tasks:
            # Get skill definition from task metadata
            skill_def = task.metadata.get('skill_definition')

            if not skill_def:
                # Fallback: fetch from registry
                skill_def = self.skill_registry_client.get_skill(task.skill)

            required_capabilities = skill_def.required_capabilities if skill_def else [task.skill]

            # Find a robot with the required capabilities
            for robot_id in available_robots:
                robot_info = self.robot_registry[robot_id]
                robot_capabilities = set(robot_info['capabilities'])

                # Check if robot has all required capabilities
                if all(cap in robot_capabilities for cap in required_capabilities):
                    # Assign task to robot
                    task.status = TaskStatus.ASSIGNED
                    task.assigned_robot = robot_id
                    robot_info['status'] = 'busy'
                    robot_info['current_task'] = task.task_id
                    assignments[task.task_id] = robot_id

                    self.logger.info(
                        f'Assigned task {task.task_id} ({task.skill}) to robot {robot_id}'
                    )

                    available_robots.remove(robot_id)
                    break

        return assignments

    async def execute_coordination(
        self,
        primitive: CoordinationPrimitive,
        robot_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Execute a coordination primitive across multiple robots.

        Args:
            primitive: The coordination primitive to execute
            robot_ids: IDs of robots participating
            metadata: Additional coordination parameters
        """
        self.logger.info(
            f'Executing {primitive.value} coordination with robots: {robot_ids}'
        )

        if primitive == CoordinationPrimitive.HANDOVER:
            await self._coordinate_handover(robot_ids, metadata)
        elif primitive == CoordinationPrimitive.MUTEX:
            await self._coordinate_mutex(robot_ids, metadata)
        elif primitive == CoordinationPrimitive.BARRIER:
            await self._coordinate_barrier(robot_ids, metadata)
        elif primitive == CoordinationPrimitive.RENDEZVOUS:
            await self._coordinate_rendezvous(robot_ids, metadata)

    async def _coordinate_handover(
        self,
        robot_ids: List[str],
        metadata: Optional[Dict[str, Any]]
    ):
        """Coordinate object handover between robots."""
        # Implement handover coordination logic
        # This would involve:
        # 1. Synchronize approach trajectories
        # 2. Coordinate grasp transfer
        # 3. Verify successful transfer
        pass

    async def _coordinate_mutex(
        self,
        robot_ids: List[str],
        metadata: Optional[Dict[str, Any]]
    ):
        """Coordinate mutual exclusion for shared resources."""
        # Implement mutex logic
        # This would involve:
        # 1. Lock resource for first robot
        # 2. Queue other robots
        # 3. Release and signal next robot
        pass

    async def _coordinate_barrier(
        self,
        robot_ids: List[str],
        metadata: Optional[Dict[str, Any]]
    ):
        """Coordinate barrier synchronization."""
        # Implement barrier sync
        # All robots must reach checkpoint before proceeding
        pass

    async def _coordinate_rendezvous(
        self,
        robot_ids: List[str],
        metadata: Optional[Dict[str, Any]]
    ):
        """Coordinate rendezvous between robots."""
        # Implement rendezvous logic
        # Robots move to meeting point simultaneously
        pass

    def get_mission_status(self, mission_id: str) -> Dict[str, Any]:
        """Get current status of a mission."""
        task_graph = self.active_missions.get(mission_id)
        if not task_graph:
            return {'error': f'Mission {mission_id} not found'}

        status_counts = {}
        for task_status in TaskStatus:
            status_counts[task_status.value] = sum(
                1 for t in task_graph.tasks.values()
                if t.status == task_status
            )

        return {
            'mission_id': mission_id,
            'total_tasks': len(task_graph.tasks),
            'status_breakdown': status_counts,
            'critical_path': task_graph.get_critical_path(),
            'is_valid': task_graph.is_valid()
        }
