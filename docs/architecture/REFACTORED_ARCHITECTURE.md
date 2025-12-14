# SwarmBrain Refactored Architecture

## Overview

SwarmBrain has been refactored to serve as a **pure mission-orchestration and system-wide learning/aggregation layer**, integrating with external services rather than duplicating functionality.

---

## Architectural Principles

### **Separation of Concerns**

| Layer | Responsibility | Implementation |
|-------|---------------|----------------|
| **SwarmBrain** | Mission orchestration, fleet coordination, system-wide aggregation | This repository |
| **SwarmBridge** | Multi-actor cooperative learning, CSA management, OpenFL coordination | External service |
| **Dynamical** | Single-actor skill execution, robot control runtime | External service |

### **API-First Integration**

SwarmBrain no longer:
- ❌ Manages skills locally
- ❌ Executes skills directly on robots
- ❌ Runs embedded Flower FL servers
- ❌ Stores policy models

SwarmBrain now:
- ✅ Orchestrates missions via API calls
- ✅ Pulls skill metadata from unified registry
- ✅ Triggers FL rounds via SwarmBridge API
- ✅ Delegates skill execution to Dynamical runtime

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SwarmBrain                                │
│                   (Mission Orchestrator)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              ORCHESTRATION LAYER                           │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐      │ │
│  │  │   Mission   │  │     Task     │  │Coordination │      │ │
│  │  │   Planner   │─▶│   Scheduler  │─▶│  Executor   │      │ │
│  │  └─────────────┘  └──────────────┘  └─────────────┘      │ │
│  └────────────────────────────────────────────────────────────┘ │
│           │                   │                   │              │
│           ▼                   ▼                   ▼              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 REGISTRY CLIENTS                           │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │ │
│  │  │Skill Registry│  │  FL Service  │  │ Coordination │    │ │
│  │  │   Client     │  │   Client     │  │   Client     │    │ │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │ │
│  └─────────┼──────────────────┼──────────────────┼───────────┘ │
│            │                  │                  │              │
└────────────┼──────────────────┼──────────────────┼──────────────┘
             │                  │                  │
       API Calls          API Calls          API Calls
             │                  │                  │
             ▼                  ▼                  ▼
┌────────────────────┐ ┌────────────────┐ ┌───────────────────┐
│   CSA Registry     │ │  SwarmBridge   │ │   Dynamical API   │
│  (Multi-Actor)     │ │  FL Coordinator│ │ (Skill Execution) │
│                    │ │                │ │                   │
│ - Skill metadata   │ │ - FL rounds    │ │ - Robot control   │
│ - CSA artifacts    │ │ - Aggregation  │ │ - Skill runtime   │
│ - Versioning       │ │ - Privacy      │ │ - Edge devices    │
└────────────────────┘ └────────────────┘ └───────────────────┘
```

---

## Refactoring Changes

### 1. **Unified Skill Registry Integration**

**Before:**
```python
# Local skill management
orchestrator.register_skill("pick_item")
orchestrator.skill_registry.add("pick_item")
```

**After:**
```python
# Pull from unified registry
from orchestrator.registry import UnifiedSkillRegistryClient

registry = UnifiedSkillRegistryClient(
    csa_registry_url="http://localhost:8082",  # Multi-actor skills
    dynamical_api_url="http://localhost:8085",  # Single-actor skills
)

# List all available skills
skills = registry.list_skills()

# Get skill metadata
skill_def = registry.get_skill("cooperative_lift_v1")
print(f"Required capabilities: {skill_def.required_capabilities}")
print(f"Skill type: {skill_def.skill_type}")  # single_actor or multi_actor
```

**Files:**
- `orchestrator/registry/skill_registry_client.py` - Registry client
- `orchestrator/task_planner/mission_orchestrator.py` - Updated to use registry

---

### 2. **Federated Learning Service API**

**Before:**
```python
# Direct Flower FL integration
from learning.federated_client import SwarmBrainClient
client = SwarmBrainClient(...)
fl.server.start_server(...)
```

**After:**
```python
# API calls to SwarmBridge
from orchestrator.learning import FederatedLearningServiceClient, FLRoundConfig

fl_client = FederatedLearningServiceClient(
    swarm_bridge_url="http://localhost:8083"
)

# Start FL round
round_config = FLRoundConfig(
    round_id="round_001",
    learning_mode="multi_actor",
    privacy_mode="dp_sgd",
    aggregation_strategy="trimmed_mean",
    csa_base_id="cooperative_lift_v1",
)

result = fl_client.start_training_round(round_config)

# Monitor progress
status = fl_client.get_round_status("round_001")
print(f"Status: {status['status']}")
print(f"Participants: {status['participants']}")

# Wait for completion
final_status = fl_client.wait_for_round_completion("round_001")

# Trigger skill update on robots
fl_client.trigger_skill_update("cooperative_lift_v1")
```

**Files:**
- `orchestrator/learning/fl_service_client.py` - FL service client
- `learning/federated_client/` - **Deprecated** (moved to `robot_control/deprecated/`)

---

### 3. **Standardized Role Assignment**

**Before:**
```python
# Ad-hoc role strings
task.role = "worker"  # Not standardized
```

**After:**
```python
# Standardized roles compatible with SwarmBridge and Dynamical
from orchestrator.coordination import RoleType, RoleAssignment, StandardizedRoleAssigner

# Assign multi-actor roles
roles = StandardizedRoleAssigner.assign_multi_actor_roles(
    task_type="cooperative_lift",
    num_robots=2,
)
# Returns: [RoleType.LEADER, RoleType.FOLLOWER]

# Create role assignment
assignment = RoleAssignment(
    robot_id="robot_001",
    role=RoleType.LEADER,
    task_id="task_001",
)

# Convert to SwarmBridge format
swarmbridge_format = assignment.to_swarmbridge_format()
# {'robot_id': 'robot_001', 'role_type': 'LEADER'}

# Convert to Dynamical format
dynamical_format = assignment.to_dynamical_format()
# {'robot_id': 'robot_001', 'execution_role': 'leader', 'task_id': 'task_001'}
```

**Supported Roles:**
- **Multi-actor** (SwarmBridge): `LEADER`, `FOLLOWER`, `SUPPORT`, `SPOTTER`, `MONITOR`
- **Single-actor** (Dynamical): `EXECUTOR`, `OPERATOR`
- **Industrial**: `PICKER`, `PLACER`, `TRANSPORTER`, `INSPECTOR`, `ASSEMBLER`

**Files:**
- `orchestrator/coordination/standardized_roles.py` - Role definitions
- `orchestrator/coordination/primitives.py` - **Updated** to use standardized roles

---

### 4. **Removed Redundant Runtime Code**

**Removed:**
- `robot_control/ros2_nodes/robot_controller.py` - Moved to Dynamical runtime
- `robot_control/skills/` - Moved to unified registry
- Direct Flower FL server integration

**Migrated to:**
- `robot_control/deprecated/` - Documentation of deprecated code
- External Dynamical runtime for skill execution
- SwarmBridge API for FL coordination

**Migration Guide:** See `robot_control/deprecated/README.md`

---

### 5. **Enhanced Monitoring**

**Added unified metrics collection from:**
- SwarmBrain orchestrator
- SwarmBridge FL service
- CSA Registry
- Dynamical API

```python
from orchestrator.monitoring import UnifiedMetricsCollector, MetricsConfig

# Initialize metrics collector
config = MetricsConfig(
    swarm_bridge_url="http://localhost:8083",
    csa_registry_url="http://localhost:8082",
    dynamical_api_url="http://localhost:8085",
    collection_interval=30,  # seconds
)

collector = UnifiedMetricsCollector(config)

# Start collection
await collector.start_collection()

# Record orchestrator events
collector.record_mission_created(priority=1)
collector.record_task_assigned(skill_type="multi_actor")
collector.record_fl_round_started(
    learning_mode="multi_actor",
    privacy_mode="dp_sgd",
)
```

**Prometheus Metrics Exposed:**
- `swarmbrain_missions_total` - Total missions created
- `swarmbrain_tasks_assigned_total` - Tasks assigned to robots
- `swarmbrain_fl_rounds_total` - FL rounds initiated
- `swarmbrain_fl_participants` - Participants in FL round
- `swarmbrain_skills_registered` - Skills in registry
- `swarmbrain_edge_device_health` - Edge device health scores
- `swarmbrain_robot_utilization` - Robot utilization percentage

**Grafana Dashboards:**
- Mission progress (task status, critical path)
- FL training status (round progress, aggregation metrics)
- Edge device health (battery, latency, health scores)
- Skill registry status (versions, deployments)

**Files:**
- `orchestrator/monitoring/unified_metrics.py` - Metrics collection
- Prometheus metrics exposed on port `:9090`

---

## API Endpoints

### SwarmBrain Orchestrator

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/missions` | POST | Create new mission |
| `/api/v1/missions/{id}/status` | GET | Get mission status |
| `/api/v1/tasks/assign` | POST | Assign tasks to robots |
| `/api/v1/skills/list` | GET | List available skills (from registry) |
| `/api/v1/coordination/execute` | POST | Execute coordination primitive |
| `/metrics` | GET | Prometheus metrics |

### External Services (Integrated)

| Service | URL | Purpose |
|---------|-----|---------|
| **CSA Registry** | `http://localhost:8082` | Multi-actor skill metadata and artifacts |
| **SwarmBridge** | `http://localhost:8083` | Unified FL coordinator (Flower + OpenFL) |
| **Dynamical API** | `http://localhost:8085` | Single-actor skill execution and robot control |

---

## Configuration

### `config/orchestrator_config.yaml`

```yaml
# Unified registry configuration
skill_registry:
  csa_registry_url: "http://localhost:8082"
  dynamical_api_url: "http://localhost:8085"
  cache_ttl: 300  # seconds

# Federated learning configuration
federated_learning:
  swarm_bridge_url: "http://localhost:8083"
  default_privacy_mode: "dp_sgd"
  default_aggregation: "trimmed_mean"

# Coordination configuration
coordination:
  swarm_bridge_url: "http://localhost:8083"
  dynamical_api_url: "http://localhost:8085"
  enable_intent_sharing: true
  enable_formation_control: true

# Monitoring configuration
monitoring:
  collection_interval: 30  # seconds
  prometheus_port: 9090
  grafana_url: "http://localhost:3000"
```

---

## Migration Guide

### For Existing SwarmBrain Users

1. **Update skill references:**
   ```python
   # Before
   orchestrator.register_skill("pick_item")

   # After
   skills = orchestrator.list_available_skills()
   skill = orchestrator.get_skill_info("pick_item")
   ```

2. **Update FL integration:**
   ```python
   # Before
   from learning.federated_client import fl_client

   # After
   from orchestrator.learning import FederatedLearningServiceClient
   fl_client = FederatedLearningServiceClient()
   ```

3. **Update robot skill execution:**
   ```python
   # Before
   robot_controller.execute_skill(skill_name, observations)

   # After
   import requests
   requests.post(
       f"{dynamical_api_url}/api/v1/robots/{robot_id}/execute",
       json={"skill_id": skill_id, "observations": observations}
   )
   ```

---

## Benefits

### **Simplified Architecture**
- SwarmBrain focuses solely on orchestration
- No redundant skill/policy management
- Clear separation of concerns

### **Improved Scalability**
- Unified registry scales independently
- FL service handles 1000+ clients
- Centralized monitoring across all services

### **Better Maintainability**
- Single source of truth for skills (registry)
- Standardized APIs across services
- Reduced code duplication

### **Enhanced Observability**
- Unified metrics from all services
- Single Grafana dashboard for entire swarm
- Real-time FL progress monitoring

---

## Next Steps

1. **Deploy Services:**
   ```bash
   docker-compose --profile refactored up -d
   ```

2. **Verify Integration:**
   ```bash
   # Check skill registry
   curl http://localhost:8082/api/v1/csa/list

   # Check SwarmBridge
   curl http://localhost:8083/api/v1/sites/statistics

   # Check Dynamical API
   curl http://localhost:8085/api/v1/skills/list
   ```

3. **Monitor Metrics:**
   - Prometheus: `http://localhost:9090`
   - Grafana: `http://localhost:3000`

4. **Test Orchestration:**
   ```python
   from orchestrator.task_planner.mission_orchestrator import MissionOrchestrator

   orchestrator = MissionOrchestrator(
       csa_registry_url="http://localhost:8082",
       dynamical_api_url="http://localhost:8085",
   )

   # List skills
   skills = orchestrator.list_available_skills()

   # Create mission
   mission = orchestrator.create_mission(work_order)

   # Assign tasks
   assignments = await orchestrator.assign_tasks(mission.order_id)
   ```

---

## See Also

- [Multi-Actor Integration Guide](../guides/multi_actor_integration.md)
- [Industrial Integration Guide](../guides/industrial_integration.md)
- [Complete Architecture Documentation](./COMPLETE_ARCHITECTURE.md)
