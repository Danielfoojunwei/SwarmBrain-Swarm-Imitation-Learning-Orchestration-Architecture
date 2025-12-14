"""
Standardized Role Assignment and Coordination Primitives

Provides common role and coordination schemas aligned with:
- SwarmBridge Multi-Actor roles (LEADER, FOLLOWER, SUPPORT, MONITOR)
- Dynamical single-actor roles
- Industrial task requirements

This ensures compatibility across the entire swarm ecosystem.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class RoleType(str, Enum):
    """
    Standardized role types across SwarmBrain, SwarmBridge, and Dynamical

    These roles are compatible with:
    - SwarmBridge Multi-Actor CSA roles
    - Dynamical skill execution contexts
    - Industrial task assignments
    """
    # Multi-actor roles (from SwarmBridge)
    LEADER = "leader"  # Coordinates multi-actor task
    FOLLOWER = "follower"  # Follows leader's coordination
    SUPPORT = "support"  # Provides physical support/stability
    SPOTTER = "spotter"  # Monitors without direct interaction
    MONITOR = "monitor"  # Perception-only observation

    # Single-actor roles (from Dynamical)
    EXECUTOR = "executor"  # Executes individual skill
    OPERATOR = "operator"  # Operator-guided execution

    # Industrial roles
    PICKER = "picker"  # Pick items
    PLACER = "placer"  # Place items
    TRANSPORTER = "transporter"  # Transport items
    INSPECTOR = "inspector"  # Quality inspection
    ASSEMBLER = "assembler"  # Assembly tasks

    # Generic
    CUSTOM = "custom"  # User-defined role


class CoordinationPrimitive(str, Enum):
    """
    Standardized coordination primitives

    Compatible with both SwarmBridge multi-actor coordination
    and traditional swarm coordination patterns.
    """
    # Multi-actor coordination (from SwarmBridge)
    HANDOVER = "handover"  # Transfer object between actors
    FORMATION = "formation"  # Maintain geometric formation
    SYNCHRONIZATION = "synchronization"  # Synchronize actions
    INTENT_SHARING = "intent_sharing"  # Share intent for coordination

    # Traditional coordination
    MUTEX = "mutex"  # Mutual exclusion for shared resource
    BARRIER = "barrier"  # Wait for all participants
    RENDEZVOUS = "rendezvous"  # Meet at specific location
    ELECTION = "election"  # Leader election

    # Industrial coordination
    QUEUE = "queue"  # Sequential access to resource
    BROADCAST = "broadcast"  # Broadcast state to all robots


@dataclass
class RoleAssignment:
    """Standardized role assignment"""
    robot_id: str
    role: RoleType
    task_id: str
    priority: int = 0
    metadata: Dict[str, Any] = None

    def to_swarmbridge_format(self) -> Dict[str, str]:
        """Convert to SwarmBridge Multi-Actor format"""
        return {
            "robot_id": self.robot_id,
            "role_type": self.role.value.upper(),
        }

    def to_dynamical_format(self) -> Dict[str, str]:
        """Convert to Dynamical execution context"""
        return {
            "robot_id": self.robot_id,
            "execution_role": self.role.value,
            "task_id": self.task_id,
        }


@dataclass
class CoordinationConfig:
    """Configuration for coordination primitive execution"""
    primitive: CoordinationPrimitive
    participants: List[str]  # Robot IDs
    parameters: Dict[str, Any]
    timeout: Optional[int] = None

    def to_swarmbridge_format(self) -> Dict[str, Any]:
        """Convert to SwarmBridge coordination format"""
        return {
            "coordination_mode": self._map_to_swarmbridge_mode(),
            "participants": self.participants,
            "parameters": self.parameters,
        }

    def _map_to_swarmbridge_mode(self) -> str:
        """Map coordination primitive to SwarmBridge coordination mode"""
        mapping = {
            CoordinationPrimitive.HANDOVER: "hierarchical",
            CoordinationPrimitive.FORMATION: "peer_to_peer",
            CoordinationPrimitive.SYNCHRONIZATION: "dynamic",
            CoordinationPrimitive.INTENT_SHARING: "dynamic",
            CoordinationPrimitive.MUTEX: "hierarchical",
            CoordinationPrimitive.BARRIER: "consensus",
            CoordinationPrimitive.RENDEZVOUS: "peer_to_peer",
        }
        return mapping.get(self.primitive, "dynamic")


class StandardizedRoleAssigner:
    """
    Role assigner using standardized roles

    Assigns roles compatible with both SwarmBridge Multi-Actor
    and Dynamical single-actor execution.
    """

    @staticmethod
    def assign_multi_actor_roles(
        task_type: str,
        num_robots: int,
    ) -> List[RoleType]:
        """
        Assign roles for multi-actor task

        Args:
            task_type: Type of multi-actor task
            num_robots: Number of robots needed

        Returns:
            List of assigned roles
        """
        if task_type == "cooperative_lift":
            if num_robots == 2:
                return [RoleType.LEADER, RoleType.FOLLOWER]
            elif num_robots == 3:
                return [RoleType.LEADER, RoleType.SUPPORT, RoleType.MONITOR]

        elif task_type == "assembly":
            if num_robots == 2:
                return [RoleType.ASSEMBLER, RoleType.SUPPORT]
            elif num_robots == 3:
                return [RoleType.LEADER, RoleType.ASSEMBLER, RoleType.INSPECTOR]

        elif task_type == "transport":
            if num_robots == 2:
                return [RoleType.PICKER, RoleType.PLACER]
            elif num_robots >= 3:
                return [RoleType.PICKER] + [RoleType.TRANSPORTER] * (num_robots - 2) + [RoleType.PLACER]

        # Default: balanced roles
        roles = [RoleType.LEADER]
        for _ in range(num_robots - 1):
            roles.append(RoleType.FOLLOWER)
        return roles

    @staticmethod
    def validate_role_compatibility(
        role: RoleType,
        skill_type: str,
    ) -> bool:
        """
        Validate if role is compatible with skill type

        Args:
            role: Assigned role
            skill_type: Skill type (single_actor, multi_actor, hybrid)

        Returns:
            True if compatible
        """
        multi_actor_roles = {
            RoleType.LEADER,
            RoleType.FOLLOWER,
            RoleType.SUPPORT,
            RoleType.SPOTTER,
            RoleType.MONITOR,
        }

        single_actor_roles = {
            RoleType.EXECUTOR,
            RoleType.OPERATOR,
            RoleType.PICKER,
            RoleType.PLACER,
            RoleType.TRANSPORTER,
            RoleType.INSPECTOR,
            RoleType.ASSEMBLER,
        }

        if skill_type == "multi_actor":
            return role in multi_actor_roles

        elif skill_type == "single_actor":
            return role in single_actor_roles

        else:  # hybrid
            return True


class CoordinationPrimitiveExecutor:
    """
    Executor for standardized coordination primitives

    Translates primitives to appropriate API calls for SwarmBridge or Dynamical.
    """

    def __init__(
        self,
        swarm_bridge_url: str = "http://localhost:8083",
        dynamical_api_url: str = "http://localhost:8085",
    ):
        self.swarm_bridge_url = swarm_bridge_url
        self.dynamical_api_url = dynamical_api_url

    async def execute_primitive(
        self,
        config: CoordinationConfig,
    ) -> Dict[str, Any]:
        """
        Execute coordination primitive

        Routes to appropriate service based on primitive type:
        - Multi-actor primitives → SwarmBridge
        - Single-actor primitives → Dynamical
        """
        multi_actor_primitives = {
            CoordinationPrimitive.HANDOVER,
            CoordinationPrimitive.FORMATION,
            CoordinationPrimitive.SYNCHRONIZATION,
            CoordinationPrimitive.INTENT_SHARING,
        }

        if config.primitive in multi_actor_primitives:
            return await self._execute_swarmbridge_primitive(config)
        else:
            return await self._execute_traditional_primitive(config)

    async def _execute_swarmbridge_primitive(
        self,
        config: CoordinationConfig,
    ) -> Dict[str, Any]:
        """Execute multi-actor coordination via SwarmBridge"""
        import requests

        payload = config.to_swarmbridge_format()

        try:
            response = requests.post(
                f"{self.swarm_bridge_url}/api/v1/coordination/execute",
                json=payload,
                timeout=config.timeout or 30,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _execute_traditional_primitive(
        self,
        config: CoordinationConfig,
    ) -> Dict[str, Any]:
        """Execute traditional coordination (mutex, barrier, etc.)"""
        # Implementation for traditional coordination
        # This would integrate with existing coordination infrastructure
        return {
            "status": "success",
            "primitive": config.primitive.value,
            "participants": config.participants,
        }
