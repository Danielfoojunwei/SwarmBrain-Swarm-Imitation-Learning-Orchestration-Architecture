# Multi-Actor Integration Guide

## Overview

This guide explains how SwarmBrain integrates with the [SwarmBridge Multi-Actor Swarm Imitation Learning Architecture](https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture) to enable cooperative humanoid robot coordination.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Docker Deployment](#docker-deployment)
8. [Advanced Topics](#advanced-topics)

---

## Architecture Overview

### Integration Layers

SwarmBrain now supports both **single-actor** and **multi-actor** swarm learning:

```
┌─────────────────────────────────────────────────────────────────┐
│                        SwarmBrain System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │               UNIFIED FL COORDINATOR                       │ │
│  │              (SwarmBridge Integration)                     │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │                                                            │ │
│  │  ┌──────────────────┐         ┌──────────────────┐       │ │
│  │  │ Single-Actor FL  │         │ Multi-Actor FL   │       │ │
│  │  │ (Flower + dynami-│         │ (OpenFL + CSA)   │       │ │
│  │  │  cal_2)          │         │                  │       │ │
│  │  └──────────────────┘         └──────────────────┘       │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌────────────────────────────┴──────────────────────────────┐ │
│  │            MULTI-ACTOR ORCHESTRATOR                       │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │ │
│  │  │ CSA Engine  │  │ Multi-Actor │  │ Formation   │      │ │
│  │  │ (Policy     │  │ Coordinator │  │ Control     │      │ │
│  │  │  Loading)   │  │ (Role Assign│  │             │      │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │ │
│  │                                                            │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │ │
│  │  │ Intent      │  │ Safety      │  │ BehaviorTree│      │ │
│  │  │ Communication│  │ Verifier    │  │ State       │      │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌────────────────────────────┴──────────────────────────────┐ │
│  │                  ROBOT CONTROLLERS                        │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │ │
│  │  │ Robot 1  │  │ Robot 2  │  │ Robot 3  │  │ Robot N  │ │ │
│  │  │ (Leader) │  │(Follower)│  │ (Support)│  │ (Monitor)│ │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. **CSA Engine** (`integration/multi_actor_adapter/csa_engine.py`)
Loads and executes Cooperative Skill Artifacts (CSA) from the Multi-Actor repository.

Features:
- CSA loading from tar.gz or directory
- Multi-actor observation processing
- Role-conditioned action computation
- Intent communication and prediction
- Safety verification (separation, velocity, intent conflicts)

#### 2. **Multi-Actor Coordinator** (`integration/multi_actor_adapter/multi_actor_coordinator.py`)
Manages multi-robot cooperative tasks with dynamic role assignment.

Features:
- Dynamic role assignment based on capabilities
- Formation control (line, circle, triangle, custom)
- Intent-based coordination
- Task lifecycle management
- Safety monitoring

#### 3. **SwarmBridge** (`integration/multi_actor_adapter/swarm_bridge.py`)
Unified federated learning coordinator bridging Flower (single-actor) and OpenFL (multi-actor).

Features:
- Dual-mode FL (single-actor + multi-actor)
- Unified privacy mechanisms (LDP, DP-SGD, HE, FHE, Secure Aggregation)
- Multiple aggregation strategies (mean, trimmed mean, median, Krum)
- Site management and round orchestration

#### 4. **Multi-Actor Orchestrator** (`orchestrator/task_planner/multi_actor_orchestrator.py`)
Extended mission orchestrator supporting mixed single/multi-actor missions.

Features:
- Mixed task graph (single + multi-actor tasks)
- Integrated role assignment
- Coordination primitive execution
- Extended mission status tracking

---

## Installation

### Prerequisites

- SwarmBrain base installation (see main [README.md](../../README.md))
- Python 3.9+
- ROS 2 Humble/Jazzy (for multi-actor execution)
- Docker & Docker Compose (for full deployment)

### Quick Setup

```bash
# Run the automated setup script
./scripts/setup_multi_actor.sh
```

### Manual Installation

1. **Initialize submodules:**
```bash
git submodule update --init --recursive external/multi_actor
```

2. **Install Multi-Actor dependencies:**
```bash
pip install -r external/multi_actor/requirements.txt
pip install -e external/multi_actor/
```

3. **Install additional dependencies:**
```bash
pip install pydantic python-onvif-zeep opencv-python mmpose lerobot
pip install openfl opacus crypten pyfhel py-trees py-trees-ros
```

4. **Create workspace directories:**
```bash
mkdir -p robot_control/skills/csa/{models,checkpoints,cache}
mkdir -p learning/swarm_workspace/{single_actor,multi_actor,hybrid}
```

5. **Copy configuration files:**
```bash
cp config/multi_actor/*.yaml config/
cp config/multi_actor/robot_capabilities.json config/
```

---

## Core Concepts

### Cooperative Skill Artifacts (CSA)

A CSA is a packaged multi-actor skill containing:

```
csa_v1.0.0.tar.gz
├── manifest.json              # Metadata
├── roles/
│   ├── leader_adapter.pt      # Role-specific policy adapters
│   ├── follower_adapter.pt
│   └── support_adapter.pt
├── coordination_encoder.pt     # Shared coordination latent encoder
├── phase_machine.xml          # BehaviorTree.CPP state machine
├── safety_envelope.json       # Constraints (velocity, force, workspace)
└── checksums.sha256           # Integrity verification
```

### Role Types

- **LEADER**: High strength/perception, coordinates the task
- **FOLLOWER**: Balanced capabilities, follows leader's commands
- **SUPPORT**: High strength, provides stability
- **SPOTTER**: High perception, monitors the environment
- **MONITOR**: Perception-only, observes without interaction
- **CUSTOM**: User-defined roles

### Actor Intents

- **GRASP**: Preparing to grasp object
- **MOVE**: General motion
- **WAIT**: Holding position
- **HANDOFF**: Transferring object
- **SUPPORT**: Providing support/stability
- **MONITOR**: Observing

### Coordination Modes

- **HIERARCHICAL**: Leader-follower structure
- **PEER_TO_PEER**: Equal collaboration
- **DYNAMIC**: Adaptive based on context
- **CONSENSUS**: Vote-based decisions

### Task Phases (BehaviorTree states)

- **APPROACH**: Move toward object/target
- **GRASP**: Grasp object
- **LIFT**: Lift object
- **TRANSFER**: Transfer between actors
- **PLACE**: Place object at destination
- **RETREAT**: Return to safe position
- **ABORT**: Emergency stop

---

## Configuration

### CSA Configuration (`config/multi_actor/csa_config.yaml`)

```yaml
workspace:
  base_dir: "./robot_control/skills/csa"

csas:
  - csa_id: "cooperative_lift_v1"
    path: "./robot_control/skills/csa/models/cooperative_lift_v1.tar.gz"
    num_actors: 2
    roles:
      leader: "LEADER"
      follower: "FOLLOWER"
    coordination_mode: "hierarchical"
    safety_envelope:
      min_separation: 0.5  # meters
      max_relative_velocity: 0.3  # m/s
```

### Robot Capabilities (`config/multi_actor/robot_capabilities.json`)

```json
{
  "robots": [
    {
      "robot_id": "robot_1",
      "strength": 0.9,
      "dexterity": 0.6,
      "perception": 0.7,
      "speed": 0.8,
      "reach": 1.2,
      "payload": 15.0
    }
  ]
}
```

### SwarmBridge Configuration (`config/multi_actor/swarm_bridge_config.yaml`)

```yaml
swarm_bridge:
  flower_server: "localhost:8080"
  openfl_coordinator: "localhost:8081"
  csa_registry: "http://localhost:8082"

privacy:
  default_mode: "dp_sgd"
  dp_sgd:
    epsilon: 1.0
    delta: 1e-5
    noise_multiplier: 1.1
```

---

## Usage Examples

### Example 1: Load a CSA

```python
from integration.multi_actor_adapter import CSAEngine

# Initialize CSA engine
csa_engine = CSAEngine(workspace_dir="./robot_control/skills/csa")

# Load CSA
config = csa_engine.load_csa(
    csa_path="./robot_control/skills/csa/models/cooperative_lift_v1.tar.gz",
    csa_id="cooperative_lift_v1"
)

print(f"Loaded CSA with {config.num_actors} actors")
print(f"Roles: {config.roles}")
print(f"Coordination mode: {config.coordination_mode}")
```

### Example 2: Execute Multi-Actor Skill

```python
from integration.multi_actor_adapter import (
    CSAEngine,
    MultiActorObservation,
    ActorIntent,
)
import numpy as np

# Initialize engine
csa_engine = CSAEngine(workspace_dir="./robot_control/skills/csa")
csa_engine.load_csa("path/to/csa.tar.gz", "cooperative_lift_v1")

# Create observations
observations = {
    "robot_1": MultiActorObservation(
        robot_id="robot_1",
        role="LEADER",
        position=np.array([0.0, 0.0, 1.0]),
        velocity=np.array([0.1, 0.0, 0.0]),
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
    ),
    "robot_2": MultiActorObservation(
        robot_id="robot_2",
        role="FOLLOWER",
        position=np.array([0.5, 0.0, 1.0]),
        velocity=np.array([0.1, 0.0, 0.0]),
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
    ),
}

# Current intents
current_intent = {
    "robot_1": ActorIntent.GRASP,
    "robot_2": ActorIntent.SUPPORT,
}

# Execute skill
actions = csa_engine.execute_multi_actor_skill(
    csa_id="cooperative_lift_v1",
    observations=observations,
    current_intent=current_intent,
)

# Process actions
for robot_id, action in actions.items():
    print(f"{robot_id}: action={action.action}, intent={action.intent}, confidence={action.confidence}")
```

### Example 3: Dynamic Role Assignment

```python
from integration.multi_actor_adapter import MultiActorCoordinator, CSAEngine
from integration.multi_actor_adapter.multi_actor_coordinator import RobotCapabilities
from integration.multi_actor_adapter import RoleType

# Initialize
csa_engine = CSAEngine(workspace_dir="./robot_control/skills/csa")
coordinator = MultiActorCoordinator(
    csa_engine=csa_engine,
    capabilities_file="config/multi_actor/robot_capabilities.json"
)

# Register robots with capabilities
coordinator.register_robot(RobotCapabilities(
    robot_id="robot_1",
    strength=0.9,
    dexterity=0.6,
    perception=0.7,
    speed=0.8,
    reach=1.2,
    payload=15.0,
))

# Create multi-actor task
from integration.multi_actor_adapter import MultiActorTask

task = MultiActorTask(
    task_id="lift_task_001",
    csa_id="cooperative_lift_v1",
    required_roles={
        RoleType.LEADER: 1,
        RoleType.FOLLOWER: 1,
    },
    assigned_robots={},
    formation=None,
    status="pending",
)

# Assign task (automatic role assignment based on capabilities)
success = coordinator.assign_multi_actor_task(task)

if success:
    print(f"Task assigned: {task.assigned_robots}")
```

### Example 4: Unified Federated Learning

```python
import asyncio
from integration.multi_actor_adapter import SwarmBridge, SwarmRoundConfig
from integration.multi_actor_adapter.swarm_bridge import (
    LearningMode,
    PrivacyMode,
    AggregationStrategy,
)

async def run_fl_round():
    # Initialize SwarmBridge
    bridge = SwarmBridge(
        workspace_dir="./learning/swarm_workspace",
        flower_server_address="localhost:8080",
        openfl_coordinator_address="localhost:8081",
        csa_registry_url="http://localhost:8082",
    )

    # Register sites
    await bridge.register_site(
        site_id="factory_1",
        site_type="factory",
        learning_mode=LearningMode.MULTI_ACTOR,
        privacy_preferences=[PrivacyMode.DP_SGD, PrivacyMode.SECURE_AGG],
    )

    # Configure round
    round_config = SwarmRoundConfig(
        round_id="round_001",
        learning_mode=LearningMode.MULTI_ACTOR,
        privacy_mode=PrivacyMode.DP_SGD,
        aggregation_strategy=AggregationStrategy.TRIMMED_MEAN,
        min_participants=2,
        max_participants=5,
        epsilon=1.0,
        delta=1e-5,
        csa_base_id="cooperative_lift_v1",
        num_actors=2,
    )

    # Start round
    result = await bridge.start_swarm_round(round_config)
    print(f"Round started: {result}")

# Run
asyncio.run(run_fl_round())
```

### Example 5: Multi-Actor Mission with SwarmBrain Orchestrator

```python
from orchestrator.task_planner.multi_actor_orchestrator import (
    MultiActorMissionOrchestrator,
    WorkOrder,
)
from integration.multi_actor_adapter import RoleType

# Initialize orchestrator
orchestrator = MultiActorMissionOrchestrator(
    csa_workspace="./robot_control/skills/csa",
    capabilities_file="config/multi_actor/robot_capabilities.json",
)

# Load CSA
orchestrator.load_csa(
    csa_path="./robot_control/skills/csa/models/cooperative_lift_v1.tar.gz",
    csa_id="cooperative_lift_v1",
)

# Register robots
orchestrator.register_robot_extended(
    robot_id="robot_1",
    capabilities=["lift", "grasp", "move"],
    robot_capabilities={"strength": 0.9, "dexterity": 0.6, "perception": 0.7},
)

# Create work order
work_order = WorkOrder(
    order_id="order_001",
    description="Cooperative lifting task",
    tasks=[
        {
            "id": "task_001",
            "task_type": "multi_actor",
            "skill": "cooperative_lift",
            "csa_id": "cooperative_lift_v1",
            "required_actors": 2,
            "required_roles": {
                "LEADER": 1,
                "FOLLOWER": 1,
            },
            "formation": {
                "type": "line",
                "separation": 0.8,
            },
        }
    ],
)

# Create mission
task_graph = orchestrator.create_multi_actor_mission(work_order)

# Assign tasks
assignments = await orchestrator.assign_tasks("order_001")
print(f"Assignments: {assignments}")

# Execute coordination steps
# (in a control loop)
while not task_completed:
    actions = await orchestrator.execute_multi_actor_step(
        task_id="order_001_task_001",
        observations=current_observations,
    )
    # Apply actions to robots...
```

---

## API Reference

### CSAEngine

```python
class CSAEngine:
    def __init__(self, workspace_dir: str, device: str = "cpu", safety_checks: bool = True)
    def load_csa(self, csa_path: str, csa_id: str) -> CSAConfig
    def execute_multi_actor_skill(
        self, csa_id: str,
        observations: Dict[str, MultiActorObservation],
        current_intent: Dict[str, ActorIntent]
    ) -> Dict[str, MultiActorAction]
    def get_coordination_metrics(self) -> Dict[str, Any]
```

### MultiActorCoordinator

```python
class MultiActorCoordinator:
    def __init__(self, csa_engine: CSAEngine, capabilities_file: Optional[str] = None)
    def register_robot(self, capabilities: RobotCapabilities) -> None
    def assign_multi_actor_task(self, task: MultiActorTask) -> bool
    def execute_coordination_step(self, task_id: str) -> Dict[str, MultiActorAction]
    def check_task_completion(self, task_id: str) -> bool
    def abort_task(self, task_id: str, reason: str = "") -> None
    def get_coordination_status(self, task_id: str) -> Dict[str, Any]
```

### SwarmBridge

```python
class SwarmBridge:
    def __init__(
        self, workspace_dir: str,
        flower_server_address: str = "localhost:8080",
        openfl_coordinator_address: str = "localhost:8081",
        csa_registry_url: str = "http://localhost:8082"
    )
    async def register_site(
        self, site_id: str, site_type: str,
        learning_mode: LearningMode,
        privacy_preferences: List[PrivacyMode]
    ) -> Dict[str, Any]
    async def start_swarm_round(self, round_config: SwarmRoundConfig) -> Dict[str, Any]
    async def aggregate_round(
        self, round_id: str,
        site_updates: Dict[str, Any]
    ) -> Dict[str, Any]
    def get_round_status(self, round_id: str) -> Dict[str, Any]
    def get_site_statistics(self) -> Dict[str, Any]
```

---

## Docker Deployment

### Services

The multi-actor integration adds the following services to `docker-compose.yml`:

- **openfl-coordinator** (port 8081): OpenFL swarm coordinator
- **csa-registry** (port 8082): CSA artifact registry (FastAPI + PostgreSQL)
- **multi-actor-aggregator** (port 8083): Multi-actor data aggregation service

### Docker Compose Usage

```bash
# Start all services including multi-actor
docker-compose up -d

# Start only multi-actor services
docker-compose --profile multi-actor up -d

# View logs
docker-compose logs -f openfl-coordinator csa-registry
```

---

## Advanced Topics

### Custom CSA Creation

See `external/multi_actor/docs/ADVANCED_MULTI_ACTOR.md` for:
- Training cooperative policies
- Packaging CSAs
- Custom role types
- Formation patterns

### Privacy-Preserving Training

SwarmBridge supports multiple privacy modes:
- **LDP**: Local Differential Privacy (edge-first)
- **DP-SGD**: Differential Privacy SGD (coordinator-based)
- **HE/FHE**: Homomorphic Encryption
- **Secure Aggregation**: SwarmBrain's dropout-resilient scheme

### BehaviorTree Integration

Multi-actor tasks use BehaviorTree.CPP for state management. See:
- `external/multi_actor/ros2_ws/src/swarm_skill_runtime/`
- [BehaviorTree.CPP documentation](https://www.behaviortree.dev/)

### Industrial Integration

Multi-actor tasks integrate with SwarmBrain's industrial data layer:
- MES work orders → Multi-actor missions
- SCADA equipment status → Robot availability for role assignment
- Formation constraints from factory layout

---

## Troubleshooting

### CSA Loading Errors

```python
# Check CSA integrity
import hashlib
with open("csa.tar.gz", "rb") as f:
    checksum = hashlib.sha256(f.read()).hexdigest()
print(f"Checksum: {checksum}")
```

### Role Assignment Failures

```python
# Check robot capabilities
coordinator.robot_capabilities  # Inspect registered capabilities

# Manually assign roles
task.assigned_robots = {
    "robot_1": RoleType.LEADER,
    "robot_2": RoleType.FOLLOWER,
}
```

### Safety Violations

```python
# Adjust safety envelope
csa_engine.min_separation = 0.3  # Reduce minimum separation
csa_engine.max_relative_velocity = 0.5  # Increase velocity threshold

# Disable safety checks (testing only)
csa_engine.safety_checks = False
```

---

## References

- [Multi-Actor Repository](https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture)
- [SwarmBrain Architecture](../architecture/COMPLETE_ARCHITECTURE.md)
- [Dynamical Integration](./dynamical_2_integration.md)
- [Industrial Integration](./industrial_integration.md)

---

**Next Steps:**
1. Try the [Quick Start](#installation) guide
2. Run the [Usage Examples](#usage-examples)
3. Explore [Advanced Topics](#advanced-topics)
4. Contribute to the [Multi-Actor repository](https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture)
