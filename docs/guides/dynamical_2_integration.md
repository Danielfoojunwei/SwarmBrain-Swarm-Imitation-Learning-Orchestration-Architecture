># Dynamical.ai (dynamical_2) Integration Guide

## Overview

This guide explains how SwarmBrain integrates with the **dynamical_2** repository - the Dynamical.ai imitation learning pipeline for robot skill training.

**Integration Architecture**:
```
dynamical_2 (IL Training) ←→ SwarmBrain (Multi-Robot Orchestration)
                          ←→ Federated Learning (Cross-Site)
```

---

## What is dynamical_2?

The `dynamical_2` repository contains Dynamical.ai's core imitation learning system:

- **Perception**: MMPose, camera processing, human state estimation
- **Retargeting**: Human-to-robot motion mapping (OKAMI-style)
- **Data Collection**: Demonstration recording, chunking, encoding
- **Training**: Diffusion policies, ACT, Transformer-based IL
- **Encryption**: N2HE-LWE for privacy-preserving training

**SwarmBrain Integration** extends this with:
- Multi-robot orchestration
- Federated learning across sites
- Industrial data integration (SCADA/MES)
- Production-grade deployment

---

## Setup

### 1. Clone dynamical_2 as Submodule

```bash
cd /path/to/swarm-brain
./scripts/setup_dynamical_2.sh
```

This script:
- ✅ Clones dynamical_2 as git submodule
- ✅ Installs dependencies
- ✅ Creates model directories
- ✅ Generates default configuration
- ✅ Links legacy code

Manual setup:
```bash
# Add submodule
git submodule add https://github.com/Danielfoojunwei/dynamical_2.git external/dynamical_2
git submodule update --init --recursive

# Install dependencies
pip install -r external/dynamical_2/requirements.txt
pip install -r requirements.txt
```

### 2. Verify Installation

```python
from integration.dynamical_adapter.skill_engine import DynamicalSkillEngine

engine = DynamicalSkillEngine(skills_dir="robot_control/skills/dynamical/models")
print("✓ Skill engine initialized")
```

---

## Architecture

### Component Overview

```
┌────────────────────────────────────────────────────────────────┐
│                         dynamical_2                             │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐      │
│  │ Perception   │   │ Retargeting  │   │ Data Collect │      │
│  │ (MMPose)     │──▶│ (IK Solver)  │──▶│ (Recorder)   │      │
│  └──────────────┘   └──────────────┘   └──────┬───────┘      │
│                                                │               │
│                                         ┌──────▼───────┐      │
│                                         │  Training    │      │
│                                         │  Pipeline    │      │
│                                         └──────┬───────┘      │
└────────────────────────────────────────────────┼──────────────┘
                                                 │
                                          Trained Models
                                                 │
┌────────────────────────────────────────────────▼──────────────┐
│                    SwarmBrain Integration                      │
│                                                                │
│  ┌───────────────────────────────────────────────────────┐   │
│  │             Skill Engine Adapter                       │   │
│  │  • Load trained policies (diffusion, ACT, transformer)│   │
│  │  • Execute skills on robots                           │   │
│  │  • Manage skill library                               │   │
│  └─────────────────────┬─────────────────────────────────┘   │
│                        │                                      │
│  ┌─────────────────────▼─────────────────────────────────┐   │
│  │             Training Bridge                            │   │
│  │  • Local training with dynamical_2                     │   │
│  │  • Export for federated learning                       │   │
│  │  • Import FL-aggregated models                         │   │
│  └─────────────────────┬─────────────────────────────────┘   │
│                        │                                      │
│                        ▼                                      │
│         ┌──────────────────────────┐                          │
│         │  Robot Control Layer     │                          │
│         │  (ROS 2 + Orchestrator)  │                          │
│         └──────────────────────────┘                          │
└────────────────────────────────────────────────────────────────┘
```

### Key Modules

**1. Skill Engine** (`integration/dynamical_adapter/skill_engine.py`):
- Loads trained policies from dynamical_2
- Executes skills on robots
- Manages skill library
- Provides observation preprocessing

**2. Training Bridge** (`integration/dynamical_adapter/training_bridge.py`):
- Runs local training using dynamical_2 pipeline
- Exports models for federated learning
- Imports FL-aggregated models

**3. Robot Control Integration** (`robot_control/ros2_nodes/robot_controller.py`):
- Uses skill engine for task execution
- Integrates with orchestrator
- Publishes robot state

---

## Usage

### Train a New Skill

#### Collect Demonstrations

```bash
# Using dynamical_2's data collection
cd external/dynamical_2
python3 recorder.py --output demonstrations/pick_and_place --num-demos 50
```

#### Train Locally

```bash
# Using SwarmBrain training script
python scripts/train_skill.py \
    --skill pick_and_place \
    --data demonstrations/pick_and_place \
    --robot robot_001 \
    --epochs 100
```

Output:
```
Training completed successfully!
  Skill: pick_and_place
  Robot: robot_001
  Model saved to: workspace/training/robot_001/pick_and_place/pick_and_place_20250112_143022.pt
  Final loss: 0.0234
  Training time: 245.3s
```

#### Alternative: Direct dynamical_2 Training

```bash
cd external/dynamical_2
python3 il_pipeline_demo.py --config configs/pick_and_place.yaml
```

### Load Skill on Robot

```python
from integration.dynamical_adapter.skill_engine import (
    DynamicalSkillEngine,
    SkillConfig
)

# Initialize skill engine
engine = DynamicalSkillEngine(
    skills_dir="robot_control/skills/dynamical/models"
)

# Configure skill
skill_config = SkillConfig(
    skill_name="pick_and_place",
    model_path="workspace/training/robot_001/pick_and_place/pick_and_place_latest.pt",
    policy_type="diffusion",
    input_modalities=["rgb", "depth", "proprioception"],
    action_dim=7,
    observation_dim=1024
)

# Load skill
engine.load_skill(skill_config)

# Execute skill
observations = {
    'rgb': camera_image,          # (H, W, 3)
    'depth': depth_image,          # (H, W)
    'proprioception': joint_states # (n_joints,)
}

execution = engine.execute_skill("pick_and_place", observations)

# Apply action
robot.execute_action(execution.action)
```

### Federated Learning Integration

#### Export Model for FL

```bash
python scripts/train_skill.py \
    --skill pick_and_place \
    --data demonstrations/ \
    --robot robot_001 \
    --export-fl
```

#### Use with SwarmBrain FL

```python
from learning.federated_client.fl_client import SwarmBrainClient
from integration.dynamical_adapter.training_bridge import DynamicalTrainingBridge

# Initialize training bridge
bridge = DynamicalTrainingBridge(workspace_dir="workspace/training")

# Train locally
result = bridge.train_skill_locally(training_config)

# Export for FL
fl_export = bridge.export_for_federated_learning(
    model_path=result.model_path,
    robot_id="robot_001"
)

# FL client sends parameters to server
# (handled automatically by SwarmBrainClient)

# After FL aggregation, import updated model
aggregated_model_path = bridge.import_from_federated_learning(
    aggregated_params=fl_aggregated_params,
    parameter_names=param_names,
    output_path="models/pick_and_place_fl_round_5.pt"
)

# Reload skill with updated model
engine.load_skill(updated_skill_config)
```

---

## Configuration

### Skill Configuration (`config/dynamical/skills.yaml`)

```yaml
skills:
  - name: pick_and_place
    policy_type: diffusion
    model_path: robot_control/skills/dynamical/models/pick_and_place.pt
    input_modalities:
      - rgb
      - depth
      - proprioception
    action_dim: 7
    observation_dim: 1024
    enabled: true

  - name: assembly
    policy_type: act
    model_path: robot_control/skills/dynamical/models/assembly.pt
    input_modalities:
      - rgb
      - proprioception
    action_dim: 14  # Bimanual
    observation_dim: 512
    enabled: true

training:
  workspace_dir: /workspace/training
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  device: cuda

federated_learning:
  enabled: true
  server_address: fl-server:8080
  sync_interval: 3600  # seconds
```

### Policy Types

**1. Diffusion Policy**:
- Best for smooth, continuous motions
- Handles multi-modal action distributions
- Higher sample quality, slower inference

**2. ACT (Action Chunking Transformer)**:
- Predicts action sequences
- Good for long-horizon tasks
- Temporal consistency

**3. Transformer Policy**:
- Attention-based architecture
- Handles variable-length sequences
- Good for complex reasoning

---

## Docker Deployment

### Build with dynamical_2 Support

```bash
# Build robot controller with dynamical_2
docker build -f docker/Dockerfile.robot \
    --build-arg INCLUDE_DYNAMICAL=true \
    -t swarm-robot:dynamical .
```

### Docker Compose

```yaml
services:
  robot-with-dynamical:
    build:
      context: .
      dockerfile: docker/Dockerfile.robot
      args:
        INCLUDE_DYNAMICAL: "true"
    volumes:
      - ./external/dynamical_2:/workspace/dynamical_2
      - ./robot_control/skills/dynamical:/workspace/skills
    environment:
      - PYTHONPATH=/workspace/dynamical_2:$PYTHONPATH
```

---

## Advanced Topics

### Custom Policy Types

Add your own policy in `external/dynamical_2/models/`:

```python
# my_custom_policy.py
import torch.nn as nn

class MyCustomPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Your architecture

    def forward(self, obs):
        # Your forward pass
        return action
```

Register in skill engine:

```python
# In skill_engine.py
def _load_custom_policy(self, model_path, config):
    from models.my_custom_policy import MyCustomPolicy

    policy = MyCustomPolicy(config.observation_dim, config.action_dim)
    checkpoint = torch.load(model_path)
    policy.load_state_dict(checkpoint['model_state_dict'])

    return policy
```

### Multi-Modal Observations

```python
observations = {
    'rgb': rgb_image,               # (3, H, W)
    'depth': depth_image,           # (1, H, W)
    'proprioception': joint_states, # (n_joints,)
    'tactile': tactile_sensors,     # (n_sensors,)
    'force': force_torque,          # (6,)
}

execution = engine.execute_skill("complex_task", observations)
```

### Transfer Learning

```python
# Load pre-trained skill
engine.load_skill(pretrained_config)

# Fine-tune on new data
bridge = DynamicalTrainingBridge(workspace_dir)

fine_tune_config = TrainingConfig(
    robot_id="robot_002",
    skill_name="pick_and_place_variant",
    data_dir="demonstrations/variant",
    output_dir="models/fine_tuned",
    policy_type="diffusion",
    num_epochs=20  # Fewer epochs for fine-tuning
)

result = bridge.train_skill_locally(
    fine_tune_config,
    pretrained_model=pretrained_config.model_path
)
```

---

## Troubleshooting

### dynamical_2 Import Errors

```bash
# Ensure submodule is initialized
git submodule update --init --recursive

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Should include: /path/to/external/dynamical_2
```

### Model Loading Failures

```python
# Check model file exists
from pathlib import Path
model_path = Path("models/pick_and_place.pt")
print(f"Model exists: {model_path.exists()}")

# Check model format
import torch
checkpoint = torch.load(model_path, map_location='cpu')
print(f"Keys: {checkpoint.keys()}")
```

### CUDA Out of Memory

```bash
# Train on CPU
python scripts/train_skill.py --skill my_skill --data demos/ --device cpu

# Or reduce batch size
python scripts/train_skill.py --skill my_skill --data demos/ --batch-size 16
```

---

## Migration from Legacy Code

The `legacy_code/` directory contains the original Dynamical.ai implementation. Migration path:

```bash
# Link legacy code to dynamical_2
ln -s $(pwd)/legacy_code $(pwd)/external/dynamical_2/legacy_integration

# Use legacy modules in new system
from external.dynamical_2.legacy_integration import full_pipeline_demo
```

---

## References

- [dynamical_2 Repository](https://github.com/Danielfoojunwei/dynamical_2)
- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [ACT Paper](https://arxiv.org/abs/2304.13705)
- [SwarmBrain Federated Learning](federated_learning.md)

---

## Example Workflows

### Workflow 1: Single Robot Training

```bash
# 1. Collect demonstrations
python external/dynamical_2/recorder.py --output demos/skill1

# 2. Train skill
python scripts/train_skill.py --skill skill1 --data demos/skill1

# 3. Deploy to robot
python scripts/deploy_skill.py --robot robot_001 --skill skill1
```

### Workflow 2: Multi-Site Federated Learning

```bash
# Site A
python scripts/train_skill.py --skill shared_skill --robot site_a_robot --export-fl

# Site B
python scripts/train_skill.py --skill shared_skill --robot site_b_robot --export-fl

# FL Server aggregates

# Sites download aggregated model
python scripts/sync_fl_model.py --skill shared_skill --round 5
```

### Workflow 3: Industrial Deployment

```bash
# 1. Setup industrial integration
docker-compose --profile industrial up -d

# 2. MES work order arrives → Orchestrator creates mission

# 3. Robot loads appropriate skill based on task
# (Automatic via orchestrator industrial integration)

# 4. Skill executes using dynamical_2 policy

# 5. Results reported back to MES
```
