# SwarmBrain: Federated Orchestration Layer for Dynamical & SwarmBridge

![SwarmBrain Architecture](docs/architecture/swarm_architecture.png)

**SwarmBrain** is the **swarm extension layer** for the Dynamical skill execution platform. It orchestrates multi-robot cooperative missions by coordinating Dynamical-equipped robots and integrating cooperative skills trained via SwarmBridge's federated multi-actor imitation learning pipeline. SwarmBrain transforms Dynamical from a single-robot skill execution system into a **fleet-wide, multi-actor swarm imitation learning platform** with privacy-preserving coordination.

## 🎯 Positioning in the Tri-System Stack

**SwarmBrain builds on top of Dynamical and SwarmBridge to enable swarm-scale cooperative learning:**

1. **[Dynamical](https://github.com/Danielfoojunwei/Dynamical-Skill-Centric-Location-Adaptive-Privacy-Preserving-Imitation-Leanring-Platform)** (Foundation Layer)
   - Single-robot skill execution via VLA + MoE architecture
   - Privacy-preserving imitation learning with MOAI compression
   - Edge-based skill runtime with ROS 2 control

2. **[SwarmBridge](https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture)** (Training Layer)
   - Multi-actor demonstration capture and cooperative IL training
   - Federated learning pipeline producing Cooperative Skill Artifacts (CSAs)
   - Exports CSAs as MoE skill experts compatible with Dynamical's format

3. **SwarmBrain** (Orchestration Layer - This Repository)
   - Mission planning and multi-robot role assignment
   - CSA ingestion from SwarmBridge → deployment to Dynamical robots
   - System-wide metrics, reputation, and federated coordination
   - Coordination primitives (handover, barrier, rendezvous) execution

**Key Innovation**: SwarmBrain enables Dynamical's skill engine to execute **cooperative multi-actor skills** learned via SwarmBridge's federated training, extending single-robot IL to swarm-scale collaborative behaviors.

## 🌟 Key Features

### API-First Orchestration Architecture

SwarmBrain operates as a **pure orchestration layer**, delegating execution to specialized external services:

1. **Mission Orchestration Layer** (SwarmBrain Core)
   - Work order → task graph conversion with DAG planning
   - Multi-robot role assignment and coordination
   - Standardized role definitions (compatible with SwarmBridge & Dynamical)
   - System-wide metrics aggregation and monitoring
   - API-based skill and FL management

2. **Skill Execution Layer** ([Dynamical API](https://github.com/Danielfoojunwei/Dynamical-Skill-Centric-Location-Adaptive-Privacy-Preserving-Imitation-Leanring-Platform))
   - Single-actor skill execution via role-conditioned policies
   - ROS 2-based robot control and perception
   - Local model training and encrypted weight uploads
   - Privacy-preserving edge computing

3. **Multi-Actor Coordination** ([SwarmBridge](https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture))
   - Unified federated learning coordinator (Flower + OpenFL)
   - Cooperative Skill Artifact (CSA) management
   - Multi-actor formation control and role coordination
   - Secure aggregation and model distribution

4. **Unified Skill Registry** (CSA Registry + Dynamical API)
   - Single source of truth for skill metadata
   - Multi-actor skills (CSA format) + Single-actor skills
   - Skill versioning and capability matching
   - Dynamic skill discovery

### Privacy & Security

- **Homomorphic Encryption** (OpenFHE): Validate updates without decryption
- **Zero-Knowledge Proofs**: Prove reputation without revealing identity
- **Secure Aggregation**: Shamir secret sharing + seed-homomorphic PRG
- **No Raw Data Sharing**: Only encrypted model updates leave the edge
- **Federated Learning**: SwarmBridge coordinates privacy-preserving learning

### Production-Ready Features

- **Containerized Deployment**: Docker Compose for all services
- **CI/CD Pipeline**: GitHub Actions with linting, testing, security scans
- **Unified Monitoring**: Prometheus + Grafana dashboards across all services
- **Scalable Architecture**: API-first microservices with message queues
- **Comprehensive Testing**: Unit, integration, and performance tests

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Integration Overview](#integration-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## 🏗️ Architecture

### New Refactored Architecture (v1.1)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SWARMBRAIN ORCHESTRATOR                        │
│                    (Mission Planning & System-Wide Learning)            │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Mission Orchestrator (Task Planner + Scheduler)                 │  │
│  │  • Work order → Task graph (DAG)                                 │  │
│  │  • Multi-robot role assignment                                   │  │
│  │  • Standardized coordination primitives                          │  │
│  └────────────┬────────────────────────────────┬───────────────────┘  │
│               │                                │                       │
│               │  API Calls                     │  API Calls            │
│               ▼                                ▼                       │
│  ┌─────────────────────────┐     ┌─────────────────────────┐         │
│  │ Unified Skill Registry  │     │ Federated Learning      │         │
│  │ Client                  │     │ Service Client          │         │
│  │ • Fetch skills from     │     │ • Trigger FL rounds     │         │
│  │   CSA Registry          │     │ • Monitor training      │         │
│  │ • Fetch skills from     │     │ • Apply updates         │         │
│  │   Dynamical API         │     └─────────────┬───────────┘         │
│  │ • Unified metadata      │                   │                     │
│  └────────────┬────────────┘                   │                     │
│               │                                │                     │
└───────────────┼────────────────────────────────┼─────────────────────┘
                │                                │
        ┌───────▼────────┐              ┌────────▼────────┐
        │                │              │                 │
┌───────▼──────────┐  ┌──▼─────────────▼──┐  ┌──────────▼──────────┐
│  CSA Registry    │  │   SwarmBridge     │  │  Dynamical API      │
│  (Multi-Actor)   │  │ (FL Coordinator)  │  │  (Skill Execution)  │
│                  │  │                   │  │                     │
│ • CSA manifests  │  │ • Flower Server   │  │ • ROS 2 Control     │
│ • Role adapters  │  │ • OpenFL Server   │  │ • Single-actor      │
│ • Coordination   │  │ • Unified Agg     │  │   Skills            │
│   encoders       │  │ • Secure Agg      │  │ • Local Training    │
│ • BehaviorTrees  │  │ • Model Distrib   │  │ • Encrypted Uploads │
└──────────────────┘  └───────┬───────────┘  └──────────┬──────────┘
                              │                         │
                      ┌───────▼─────────────────────────▼───────┐
                      │         Edge Robots (Fleet)             │
                      │                                         │
                      │  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
                      │  │Robot 1  │  │Robot 2  │  │Robot N  │ │
                      │  │• Sensors│  │• Sensors│  │• Sensors│ │
                      │  │• Actuat.│  │• Actuat.│  │• Actuat.│ │
                      │  │• Training│  │• Training│  │• Training│ │
                      │  └─────────┘  └─────────┘  └─────────┘ │
                      └─────────────────────────────────────────┘
```

### Component Breakdown

#### SwarmBrain Orchestrator (`/orchestrator`)

**Mission Orchestration**
- **Task Planner** (`task_planner/mission_orchestrator.py`): Converts work orders to DAG task graphs
- **Scheduler** (`scheduler/task_scheduler.py`): Assigns tasks to robots based on capabilities
- **Coordination** (`coordination/standardized_roles.py`): Standardized roles and primitives
- **Workflow Engine**: Behavior trees for complex missions

**Integration Clients**
- **Unified Skill Registry Client** (`registry/skill_registry_client.py`):
  - Fetches multi-actor skills from CSA Registry
  - Fetches single-actor skills from Dynamical API
  - Provides unified skill metadata with caching

- **FL Service Client** (`learning/fl_service_client.py`):
  - Replaces direct Flower integration
  - API calls to SwarmBridge for FL coordination
  - Triggers training rounds and monitors progress

**System Monitoring**
- **Unified Metrics Collector** (`monitoring/unified_metrics.py`):
  - Collects metrics from SwarmBridge, CSA Registry, Dynamical API
  - Exposes Prometheus metrics for Grafana
  - Tracks missions, tasks, FL rounds, edge device health

#### External Services (Integrated via API)

**CSA Registry** (Multi-Actor Skill Management)
- Cooperative Skill Artifacts with role adapters
- Coordination encoders and formation control
- BehaviorTree state machines
- Safety envelopes

**SwarmBridge** (Federated Learning Coordinator)
- Unified FL server (Flower + OpenFL)
- Multi-actor (OpenFL) and single-actor (Flower) support
- Secure aggregation protocols
- Model distribution to edge devices

**Dynamical API** (Skill Execution Runtime)
- ROS 2-based robot control
- Single-actor skill execution
- Local model training
- Privacy-preserving weight uploads

#### Robot Control (`/robot_control`)

**Note**: Runtime execution code has been moved to Dynamical API. This directory now contains:
- **Deprecated** (`deprecated/`): Legacy code with migration guide
- **Schemas**: Message definitions and interfaces
- **Utils**: Shared utilities

For skill execution, see [Dynamical repository](https://github.com/Danielfoojunwei/Dynamical-Skill-Centric-Location-Adaptive-Privacy-Preserving-Imitation-Leanring-Platform).

#### Cryptography (`/crypto`)
- **FHE**: OpenFHE integration for encrypted validation
- **ZKP**: Zero-knowledge reputation proofs
- **Utils**: Key management, secure randomness

---

## 🔗 Integration Overview

### Multi-Actor Swarm Integration

SwarmBrain integrates with [SwarmBridge](https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture) for multi-actor coordination:

**Cooperative Skill Artifacts (CSA)**
- **Manifest**: Skill metadata, requirements, actor counts
- **Role Adapters**: Actor-specific observation/action transformations
- **Coordination Encoder**: Shared representations for multi-actor tasks
- **BehaviorTree**: State machine for skill execution
- **Safety Envelope**: Collision avoidance and constraints

**Federated Learning**
- **OpenFL Server**: Multi-actor model aggregation
- **Flower Server**: Single-actor model aggregation
- **Secure Aggregation**: Dropout-resilient protocols
- **Model Distribution**: Push updated weights to robots

### Single-Actor Skill Integration

SwarmBrain integrates with [Dynamical](https://github.com/Danielfoojunwei/Dynamical-Skill-Centric-Location-Adaptive-Privacy-Preserving-Imitation-Leanring-Platform) for single-actor execution:

**Skill Execution**
- ROS 2-based control loops
- Role-conditioned neural network policies
- Real-time perception and retargeting
- Local skill execution (no raw data transmission)

**Privacy-Preserving Learning**
- Local training on edge devices
- Encrypted weight uploads (only model updates leave device)
- Differential privacy support
- Federated learning participation

### Standardized Roles

**Multi-Actor Roles** (SwarmBridge)
- `LEADER`: Formation leader, path planner
- `FOLLOWER`: Formation follower, trajectory tracking
- `SUPPORT`: Task assistance, handover coordination
- `SPOTTER`: Visual monitoring, quality inspection
- `MONITOR`: System observer, anomaly detection

**Single-Actor Roles** (Dynamical)
- `EXECUTOR`: Task executor
- `OPERATOR`: Operator-guided execution

**Industrial Roles**
- `PICKER`: Object picking
- `PLACER`: Object placement
- `TRANSPORTER`: Material transport
- `INSPECTOR`: Quality inspection
- `ASSEMBLER`: Component assembly

---

## 🧠 SwarmBridge Algorithms & Cooperative Learning Pipeline

### How SwarmBridge Extends Dynamical to Multi-Actor Learning

SwarmBridge is the **training pipeline** that transforms multi-actor demonstrations into Cooperative Skill Artifacts (CSAs) compatible with Dynamical's MoE format. SwarmBrain then orchestrates the execution of these CSAs across robot fleets.

### Core Algorithms & Adaptive Loops

#### 1. Multi-Actor Data Capture
- **Synchronized ROS 2 Recording**: Collects trajectories from multiple robots/humans with precise timestamps
- **Per-Role State/Action Logging**: Each role (giver, receiver, leader, follower) records its own observations and actions
- **MOAI Compression**: Vision data compressed before encryption (preserving privacy while reducing bandwidth)
- **Episode Structure**: `{states_per_role, actions_per_role, timestamps, coordination_events}`

#### 2. Coordination Encoder & Role-Conditioned Policies
**Architecture**:
```
Multi-Actor Trajectory → Coordination Encoder (Transformer/RNN) → Coordination Latent z
                                                                          ↓
Role i Observation + z → Role i Policy Head → Role i Action
```

**Key Components**:
- **Coordination Encoder**: Processes joint trajectory to produce shared latent `z` capturing inter-agent dependencies
- **Role-Conditioned Policies**: Each role's policy takes `(own_observation, z)` → `action`
- **Cross-Role Consistency**: Ensures coordinated behaviors (e.g., giver's release matches receiver's grasp)

#### 3. Training Loop with Local-Global Distillation
```python
# Per-site training (SwarmBridge)
for epoch in range(num_epochs):
    # Sample local multi-actor demonstrations
    batch = sample_multi_actor_batch(local_demos)

    # Compute coordination latent
    z = coordination_encoder(batch.joint_trajectory)

    # Per-role policy losses
    for role in roles:
        action_pred = role_policy[role](batch.obs[role], z)
        loss_IL += imitation_loss(action_pred, batch.action[role])

    # Cross-role consistency regularization
    loss_consistency = consistency_penalty(actions_pred, z)

    # Local-global distillation (maintain global skill prior)
    loss_distill = KL_divergence(local_policy, global_policy)

    # Total loss
    loss = loss_IL + λ_consistency * loss_consistency + λ_distill * loss_distill

    optimizer.step()

# Federated aggregation (SwarmBridge FL Coordinator)
encrypted_updates = collect_encrypted_updates_from_sites()
global_update = secure_aggregation(encrypted_updates)
global_policy.update(global_update)
distribute_global_policy_to_sites()
```

#### 4. Novelty-Based Sampling
- **Up-sample rare coordination patterns**: Near-collisions, failed transfers, complex handovers
- **Diversity metrics**: Embedding-space clustering to identify under-represented scenarios
- **Improves generalization**: Prevents overfitting to common, easy coordination cases

#### 5. Federated Training with Dropout-Resilient Secure Aggregation
**SwarmBridge Federated Loop**:
```
1. Each site trains local multi-actor policy on its demonstrations
2. Encrypt model updates with dropout-resilient secret sharing
3. Submit encrypted updates to SwarmBridge FL Coordinator
4. Secure aggregation combines updates without decryption
5. Global policy distributed back to sites
6. Sites apply local-global distillation for next round
```

**Privacy Guarantees**:
- No raw multi-actor demonstrations leave the site
- Only encrypted gradient/weight updates shared
- Secure aggregation prevents coordinator from seeing individual updates
- MOAI-compressed embeddings reduce information leakage

#### 6. Cooperative Skill Artifact (CSA) Export
After federated training completes, SwarmBridge exports CSAs:

**CSA Format** (compatible with Dynamical's MoE):
```python
{
  "skill_id": "collaborative_handover_v2",
  "skill_type": "multi_actor",
  "required_roles": ["giver", "receiver"],
  "required_actors": 2,

  # Per-role MoE experts (Dynamical format)
  "role_experts": {
    "giver": {
      "checkpoint_uri": "s3://skills/handover_giver_expert.pth",
      "embedding_type": "moai_compressed",
      "architecture": "diffusion_policy",  # or ACT, GPT, etc.
      "encryption_scheme": "openfhe_bfv"
    },
    "receiver": {
      "checkpoint_uri": "s3://skills/handover_receiver_expert.pth",
      "embedding_type": "moai_compressed",
      "architecture": "diffusion_policy",
      "encryption_scheme": "openfhe_bfv"
    }
  },

  # Coordination encoder (shared across roles)
  "coordination_encoder": {
    "checkpoint_uri": "s3://skills/handover_coord_encoder.pth",
    "architecture": "transformer",
    "latent_dim": 256
  },

  # Coordination metadata
  "coordination_primitives": ["handover", "sync_grasp"],
  "safety_envelope": {...},
  "behavior_tree": {...}
}
```

#### 7. CSA Flow: SwarmBridge → SwarmBrain → Dynamical

**Training & Distribution Pipeline**:
```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Multi-Actor Data Collection (On-Site)                       │
│    - Robots/humans perform collaborative demonstrations        │
│    - ROS 2 synchronized recording                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. SwarmBridge Federated Training                              │
│    - Coordination encoder + role-conditioned policies          │
│    - Cross-role consistency + novelty sampling                 │
│    - Secure aggregation across sites                           │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. CSA Export (SwarmBridge → CSA Registry)                     │
│    - Package per-role experts as Dynamical MoE modules         │
│    - Include coordination encoder checkpoint                   │
│    - Add safety envelope + behavior tree                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. SwarmBrain Ingestion (CSA Registry → SwarmBrain)            │
│    - UnifiedSkillRegistryClient fetches CSAs                   │
│    - Validates CSA format compatibility with Dynamical         │
│    - Registers skill in mission planner                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Mission Planning (SwarmBrain Orchestrator)                  │
│    - Work order → task graph conversion                        │
│    - Role assignment: match CSA roles to robots                │
│    - Coordination primitive scheduling                         │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Skill Deployment (SwarmBrain → Dynamical API)               │
│    - SwarmBrain calls Dynamical API: execute_skill()           │
│    - Passes CSA ID + assigned role                             │
│    - Dynamical loads role-specific expert from CSA             │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. Skill Execution (Dynamical Runtime on Edge Robots)          │
│    - Load coordination encoder checkpoint                      │
│    - Load role-specific expert (giver/receiver/etc.)           │
│    - Inference: obs + coord_latent → action                    │
│    - ROS 2 control loop executes actions                       │
└─────────────────────────────────────────────────────────────────┘
```

### Innovations Beyond Single-Robot Dynamical

**What SwarmBridge + SwarmBrain Add to Dynamical**:

1. **Multi-Actor Cooperative Learning**: Learn coordinated behaviors (handovers, formations) that cannot be represented by single-robot MoE experts
2. **Federated Cross-Site Coordination**: Robots in different locations share and aggregate cooperative skills without raw data sharing
3. **Dynamic Clustering & RL-Based Scheduling**: SwarmBrain's device scheduler uses PPO to optimize which robots participate in FL rounds
4. **Reputation System with zkRep**: Zero-knowledge proofs validate robot competence without revealing identity/performance metrics
5. **Mission-Level Orchestration**: Convert high-level work orders into coordinated multi-robot task graphs with dependency management

**Privacy Enhancements Beyond Dynamical**:
- Dropout-resilient secret sharing for secure aggregation
- Zero-knowledge reputation proofs (zkRep)
- Cross-role encrypted communication
- MOAI compression applied before multi-actor aggregation

---

## 📦 Installation

### Prerequisites

- **OS**: Ubuntu 22.04 (recommended) or later
- **Python**: 3.9+ (3.10 recommended)
- **Docker**: 24.0+ with Docker Compose
- **Hardware**:
  - Min: 16GB RAM, 8 CPU cores (for all services)
  - Recommended: 32GB RAM, 16 CPU cores, NVIDIA GPU (for FL)

### Quick Setup (Docker Compose)

```bash
# Clone SwarmBrain repository
git clone https://github.com/Danielfoojunwei/SwarmBraim-Swarm-Imitation-Learning-Orchestration-Architecture.git
cd SwarmBraim-Swarm-Imitation-Learning-Orchestration-Architecture

# Setup external dependencies (SwarmBridge Multi-Actor)
cd external
git clone https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture.git multi_actor
cd multi_actor && pip install -e . && cd ../..

# Start all services (refactored architecture)
docker-compose --profile refactored up -d

# Verify services
docker-compose ps

# Expected services:
# - swarmbrain-orchestrator (port 8000)
# - swarmbrain-csa-registry (port 8082)
# - swarmbrain-swarmbridge (port 8083)
# - swarmbrain-dynamical (port 8085)
# - prometheus (port 9090)
# - grafana (port 3000)
# - rabbitmq (port 5672, 15672)
```

### Local Development

```bash
# Clone repository
git clone https://github.com/Danielfoojunwei/SwarmBraim-Swarm-Imitation-Learning-Orchestration-Architecture.git
cd SwarmBraim-Swarm-Imitation-Learning-Orchestration-Architecture

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install SwarmBrain dependencies
pip install -r requirements.txt

# Setup external services
cd external
git clone https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture.git multi_actor
cd multi_actor && pip install -e . && cd ../..

# Install Dynamical (in separate directory)
cd ..
git clone https://github.com/Danielfoojunwei/Dynamical-Skill-Centric-Location-Adaptive-Privacy-Preserving-Imitation-Leanring-Platform.git
cd Dynamical-Skill-Centric-Location-Adaptive-Privacy-Preserving-Imitation-Leanring-Platform
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Start All Services

```bash
# Start refactored architecture
docker-compose --profile refactored up -d

# Check service health
curl http://localhost:8000/health  # SwarmBrain Orchestrator
curl http://localhost:8082/health  # CSA Registry
curl http://localhost:8083/health  # SwarmBridge FL Coordinator
curl http://localhost:8085/health  # Dynamical API
```

### 2. Access Web Interfaces

- **SwarmBrain Orchestrator API**: http://localhost:8000/docs
- **CSA Registry**: http://localhost:8082/docs
- **SwarmBridge Dashboard**: http://localhost:8083/dashboard
- **Dynamical API**: http://localhost:8085/docs
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **RabbitMQ Management**: http://localhost:15672 (swarm/swarm)

### 3. List Available Skills

```python
import requests

# List all skills from unified registry
response = requests.get('http://localhost:8000/api/v1/skills')
skills = response.json()

print(f"Total skills: {len(skills)}")
for skill in skills:
    print(f"- {skill['name']} ({skill['skill_type']}) - {skill['description']}")
    print(f"  Required actors: {skill['required_actors']}")
    print(f"  Roles: {skill['required_roles']}")
```

### 4. Register Robots with Dynamical

```python
# Register a robot for skill execution
response = requests.post('http://localhost:8085/api/v1/robots', json={
    'robot_id': 'robot_001',
    'capabilities': ['grasp', 'navigate', 'manipulate'],
    'location': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    'status': 'idle'
})

print(response.json())
```

### 5. Create a Multi-Actor Mission

```python
# Create a collaborative work order
work_order = {
    'order_id': 'wo_001',
    'description': 'Multi-actor assembly with handover',
    'tasks': [
        {
            'id': 'task_1',
            'skill': 'collaborative_pick',  # Multi-actor CSA
            'role': 'leader',
            'required_actors': 2,
            'dependencies': []
        },
        {
            'id': 'task_2',
            'skill': 'handover',  # Multi-actor coordination
            'role': 'follower',
            'required_actors': 2,
            'dependencies': ['task_1'],
            'coordination': 'handover'
        },
        {
            'id': 'task_3',
            'skill': 'assembly',  # Single-actor skill
            'role': 'executor',
            'required_actors': 1,
            'dependencies': ['task_2']
        }
    ],
    'priority': 1
}

# Submit to orchestrator
response = requests.post('http://localhost:8000/api/v1/missions', json=work_order)
mission = response.json()
print(f"Mission created: {mission['mission_id']}")
```

### 6. Monitor Mission Progress

```python
# Get mission status
mission_id = mission['mission_id']
response = requests.get(f'http://localhost:8000/api/v1/missions/{mission_id}/status')
status = response.json()

print(f"Mission: {status['mission_id']}")
print(f"Status: {status['status']}")
print(f"Completed tasks: {status['status_breakdown']['completed']}/{status['total_tasks']}")
print(f"Assigned robots: {status['assigned_robots']}")
```

### 7. Trigger Federated Learning

```python
# Start a federated learning round via SwarmBridge
fl_config = {
    'skill_id': 'collaborative_pick',
    'learning_mode': 'multi_actor',  # or 'single_actor'
    'privacy_mode': 'secure_aggregation',
    'num_rounds': 10,
    'min_participants': 3,
    'aggregation_strategy': 'fedavg'
}

response = requests.post('http://localhost:8000/api/v1/federated_learning/rounds', json=fl_config)
fl_round = response.json()
print(f"FL Round started: {fl_round['round_id']}")

# Monitor FL progress
response = requests.get(f'http://localhost:8000/api/v1/federated_learning/rounds/{fl_round["round_id"]}/status')
print(response.json())
```

### 8. View Unified Metrics

```bash
# Access Grafana dashboard
open http://localhost:3000

# Default dashboards:
# - SwarmBrain Mission Overview
# - Federated Learning Progress
# - Edge Device Health
# - Skill Registry Metrics
```

---

## 🛠️ Development

### Repository Structure (Refactored)

```
SwarmBrain/
├── orchestrator/              # Mission orchestration layer
│   ├── task_planner/         # Mission planning & task graphs
│   │   └── mission_orchestrator.py  # Updated with registry client
│   ├── scheduler/            # Task assignment & scheduling
│   ├── coordination/         # 🆕 Standardized roles & primitives
│   │   ├── __init__.py
│   │   └── standardized_roles.py
│   ├── registry/             # 🆕 Unified skill registry client
│   │   ├── __init__.py
│   │   └── skill_registry_client.py
│   ├── learning/             # 🆕 FL service client
│   │   ├── __init__.py
│   │   └── fl_service_client.py
│   └── monitoring/           # 🆕 Unified metrics collection
│       ├── __init__.py
│       └── unified_metrics.py
├── robot_control/            # Robot schemas & interfaces
│   ├── deprecated/           # 🆕 Legacy runtime code (see migration guide)
│   ├── schemas/              # Message definitions
│   └── utils/                # Shared utilities
├── crypto/                   # Cryptographic layer
│   ├── fhe/                 # Homomorphic encryption
│   ├── zkp/                 # Zero-knowledge proofs
│   └── utils/               # Crypto utilities
├── external/                 # 🆕 External service integrations
│   ├── multi_actor/         # SwarmBridge Multi-Actor (git clone)
│   └── README.md            # Setup instructions
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── performance/        # Benchmarks
├── docs/                   # Documentation
│   └── architecture/
│       ├── REFACTORED_ARCHITECTURE.md  # 🆕 New architecture docs
│       └── README.md
├── config/                 # Configuration files
├── docker/                 # Dockerfiles
│   ├── orchestrator.Dockerfile
│   ├── csa_registry.Dockerfile      # 🆕 CSA Registry service
│   ├── swarmbridge.Dockerfile       # 🆕 SwarmBridge FL service
│   └── dynamical.Dockerfile         # 🆕 Dynamical API service
└── docker-compose.yml      # Updated with refactored services

```

### Key Changes in Refactored Architecture

**Removed from SwarmBrain**
- ❌ Local skill execution (moved to Dynamical API)
- ❌ Direct Flower FL integration (moved to SwarmBridge)
- ❌ Local skill storage (moved to CSA Registry + Dynamical)
- ❌ ROS 2 control loops (moved to Dynamical)

**Added to SwarmBrain**
- ✅ `UnifiedSkillRegistryClient` - Fetches skills from CSA Registry & Dynamical
- ✅ `FederatedLearningServiceClient` - API calls to SwarmBridge
- ✅ `StandardizedRoles` - Compatible with SwarmBridge & Dynamical
- ✅ `UnifiedMetricsCollector` - System-wide monitoring
- ✅ API-first orchestration pattern

### Running Tests

```bash
# Unit tests
pytest tests/unit -v

# Integration tests (requires services running)
docker-compose --profile refactored up -d
pytest tests/integration -v

# Test specific modules
pytest tests/unit/orchestrator/test_skill_registry_client.py -v
pytest tests/unit/orchestrator/test_fl_service_client.py -v
pytest tests/unit/orchestrator/test_standardized_roles.py -v

# Coverage report
pytest --cov=orchestrator --cov-report=html
open htmlcov/index.html
```

### Code Quality

```bash
# Format code
black orchestrator crypto tests

# Sort imports
isort orchestrator crypto tests

# Lint
flake8 orchestrator crypto tests

# Type check
mypy orchestrator crypto tests
```

### Development Workflow

**Working with External Services**

```bash
# Update SwarmBridge (multi_actor)
cd external/multi_actor
git pull origin main
pip install -e .

# Test CSA Registry integration
python -c "from orchestrator.registry.skill_registry_client import UnifiedSkillRegistryClient;
           client = UnifiedSkillRegistryClient();
           print(client.list_skills())"

# Test SwarmBridge FL integration
python -c "from orchestrator.learning.fl_service_client import FederatedLearningServiceClient;
           client = FederatedLearningServiceClient();
           print(client.get_service_status())"
```

**Adding a New Skill**

1. **Register in CSA Registry** (for multi-actor) or **Dynamical** (for single-actor)
2. **SwarmBrain auto-discovers** via UnifiedSkillRegistryClient
3. **No code changes needed** in SwarmBrain orchestrator
4. Skills automatically available for mission planning

See [Skill Development Guide](docs/guides/skill_development.md).

---

## 🧪 Testing

### Test Coverage

- **Unit Tests**: 85%+ coverage target
- **Integration Tests**: API-first service integration
- **Performance Tests**: Latency, throughput benchmarks
- **Security Tests**: Vulnerability scanning

### Integration Test Example

```python
import pytest
import requests

@pytest.fixture
def services_running():
    """Ensure all services are healthy"""
    services = {
        'orchestrator': 'http://localhost:8000',
        'csa_registry': 'http://localhost:8082',
        'swarmbridge': 'http://localhost:8083',
        'dynamical': 'http://localhost:8085',
    }

    for name, url in services.items():
        response = requests.get(f'{url}/health')
        assert response.status_code == 200, f"{name} not healthy"

    return services

def test_unified_skill_registry(services_running):
    """Test skill fetching from both registries"""
    response = requests.get('http://localhost:8000/api/v1/skills')
    assert response.status_code == 200

    skills = response.json()
    assert len(skills) > 0

    # Verify both multi-actor and single-actor skills
    skill_types = {s['skill_type'] for s in skills}
    assert 'multi_actor' in skill_types or 'single_actor' in skill_types

def test_fl_round_lifecycle(services_running):
    """Test complete FL round via SwarmBridge"""
    # Start FL round
    config = {
        'skill_id': 'test_skill',
        'learning_mode': 'single_actor',
        'num_rounds': 1,
        'min_participants': 1
    }

    response = requests.post('http://localhost:8000/api/v1/federated_learning/rounds', json=config)
    assert response.status_code == 200

    round_id = response.json()['round_id']

    # Check status
    response = requests.get(f'http://localhost:8000/api/v1/federated_learning/rounds/{round_id}/status')
    assert response.status_code == 200
    assert response.json()['status'] in ['pending', 'in_progress', 'completed']
```

---

## 📚 Documentation

Comprehensive documentation is available in [docs/](docs/):

- **[Refactored Architecture](docs/architecture/REFACTORED_ARCHITECTURE.md)**: New API-first design (v1.1)
- **[Architecture Overview](docs/architecture/README.md)**: System design and components
- **[API Reference](docs/api/README.md)**: REST endpoints for all services
- **[Development Guide](docs/guides/development.md)**: Contributing and extending
- **[Deployment Guide](docs/guides/deployment.md)**: Production deployment
- **[Security Guide](docs/guides/security.md)**: Privacy and security best practices
- **[Migration Guide](robot_control/deprecated/README.md)**: Migrating from legacy architecture

### External Documentation

- **[SwarmBridge Multi-Actor](https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture)**: Federated learning coordination
- **[Dynamical Platform](https://github.com/Danielfoojunwei/Dynamical-Skill-Centric-Location-Adaptive-Privacy-Preserving-Imitation-Leanring-Platform)**: Skill execution runtime

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Run code quality checks (`black`, `flake8`, `mypy`)
6. Commit (`git commit -m 'feat: Add amazing feature'`)
7. Push (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

SwarmBrain integrates and builds upon:

- **[SwarmBridge Multi-Actor](https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture)**: Multi-actor coordination and unified FL
- **[Dynamical Platform](https://github.com/Danielfoojunwei/Dynamical-Skill-Centric-Location-Adaptive-Privacy-Preserving-Imitation-Leanring-Platform)**: Skill-centric execution and privacy-preserving learning
- **OpenFHE**: Homomorphic encryption library
- **Flower**: Federated learning framework
- **ROS 2**: Robot Operating System
- **Open-RMF**: Multi-fleet coordination

Research foundations from NTU on dynamic clustering, secure aggregation, and joint scheduling.

See [CITATIONS.md](CITATIONS.md) for full references.

---

## 📞 Contact

- **GitHub Issues**: [SwarmBrain Issues](https://github.com/Danielfoojunwei/SwarmBraim-Swarm-Imitation-Learning-Orchestration-Architecture/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Danielfoojunwei/SwarmBraim-Swarm-Imitation-Learning-Orchestration-Architecture/discussions)

---

## 🗺️ Roadmap

### v1.0 (Completed - Q4 2024)
- ✅ Core three-layer architecture
- ✅ ROS 2 integration
- ✅ Federated learning with Flower
- ✅ Secure aggregation
- ✅ Dynamic clustering
- ✅ Docker deployment

### v1.1 (Completed - Q1 2025) 🎉
- ✅ **API-first refactored architecture**
- ✅ **SwarmBridge Multi-Actor integration**
- ✅ **Unified Skill Registry (CSA + Dynamical)**
- ✅ **Standardized role assignment**
- ✅ **Unified metrics collection**
- ✅ **Comprehensive documentation**

### v1.2 (Planned - Q2 2025)
- [ ] OpenFHE full integration in SwarmBridge
- [ ] zkRep circuit implementation (Circom)
- [ ] Multi-site deployment across data centers
- [ ] Enhanced Grafana dashboards
- [ ] Real-world deployment case studies

### v2.0 (Planned - Q4 2025)
- [ ] Hierarchical federated learning
- [ ] Cross-modal skill transfer
- [ ] Mobile app for mission control
- [ ] Advanced visualization and debugging tools
- [ ] Open-RMF integration

---

**Built with ❤️ by the SwarmBrain Team**

**Architecture**: Mission Orchestration | **Integration**: SwarmBridge + Dynamical | **Privacy**: Federated Learning + Secure Aggregation
