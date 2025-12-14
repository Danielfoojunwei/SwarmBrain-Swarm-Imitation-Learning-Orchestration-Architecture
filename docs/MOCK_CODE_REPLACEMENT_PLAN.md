# Mock Code Replacement Plan

## Overview

This document identifies all mock/placeholder code in SwarmBrain and provides a replacement strategy using open-source libraries aligned with the tri-system architecture (SwarmBrain + SwarmBridge + Dynamical).

---

## Mock Code Inventory

### 1. CSA Engine (`integration/multi_actor_adapter/csa_engine.py`)

**Location**: Lines 326-406
**Type**: Mock implementations for multi-actor coordination

**Issues**:
- `_encode_coordination_latent()` (L326-354): Returns placeholder zero tensors instead of actual hierarchical encoding
- `_compute_action()` (L378-393): Returns zero actions with placeholder confidence
- `_predict_next_intent()` (L395-405): Returns current intent without prediction logic
- Comment at L333: "This is a simplified version - in production, use the actual HierarchicalCoordinationEncoder from multi_actor repository"

**Replacement Strategy**:
```python
# Import from SwarmBridge Multi-Actor repository
from external.multi_actor.core.coordination import HierarchicalCoordinationEncoder
from external.multi_actor.core.intent_communication import IntentCommunicationModule
from external.multi_actor.models.role_adapters import RoleSpecificAdapter
```

**Dependencies**:
- ✅ SwarmBridge Multi-Actor repository (already in `external/multi_actor/`)
- Libraries: PyTorch, NumPy (already installed)

---

### 2. Training Bridge (`integration/dynamical_adapter/training_bridge.py`)

**Location**: Lines 194-250
**Type**: Mock training pipeline

**Issues**:
- `_create_mock_model()` (L194-206): Creates simple 2-layer MLP instead of actual IL policies
- `_run_training()` (L208-250): Uses random data instead of actual demonstrations
- Comment at L191: "dynamical_2 models not available, using mock model"
- Comment at L224: "In real implementation, use DataLoader from dynamical_2"

**Replacement Strategy**:
```python
# Import from Dynamical repository
from dynamical.models.diffusion_policy import DiffusionPolicy
from dynamical.models.act_policy import ACTPolicy
from dynamical.models.transformer_policy import TransformerPolicy
from dynamical.data.dataloader import DemonstrationDataLoader
from dynamical.training.trainer import ILTrainer
```

**Dependencies**:
- Dynamical repository (clone from GitHub)
- Libraries: diffusers, transformers, einops (add to requirements.txt)

---

### 3. Skill Engine (`integration/dynamical_adapter/skill_engine.py`)

**Location**: Lines 185-195
**Type**: Mock policy for inference

**Issues**:
- `_create_mock_policy()` (L185-195): Creates simple linear layer instead of actual trained policies
- Fallback to mock in L144-147 (diffusion), L164-165 (ACT), L182-183 (transformer)

**Replacement Strategy**:
```python
# Import from Dynamical repository (same as Training Bridge)
from dynamical.models.diffusion_policy import DiffusionPolicy
from dynamical.models.act_policy import ACTPolicy
from dynamical.models.transformer_policy import TransformerPolicy
from dynamical.inference.policy_wrapper import PolicyInferenceWrapper
```

**Dependencies**:
- Same as Training Bridge above

---

### 4. Industrial Data Aggregator (`industrial_data/streams/data_aggregator.py`)

**Location**: Lines 245-246
**Type**: TODO placeholders

**Issues**:
- `production_metrics`: Empty dict with TODO comment
- `equipment_status`: Empty dict with TODO comment

**Replacement Strategy**:
```python
# Add implementations using industrial libraries
import pandas as pd
from industrial_data.analytics.oee_calculator import OEECalculator
from industrial_data.analytics.equipment_monitor import EquipmentHealthMonitor
```

**New Implementation**:
- **OEECalculator**: Calculate Overall Equipment Effectiveness (Availability × Performance × Quality)
- **EquipmentHealthMonitor**: Track equipment health metrics (MTBF, MTTR, vibration, temperature)

**Dependencies**:
- pandas (already installed)
- scipy (for signal processing)
- scikit-learn (for anomaly detection)

---

### 5. zkRep Reputation System (`crypto/zkp/zkrep_reputation.py`)

**Location**: Lines 199-266, 354-378
**Type**: Mock zero-knowledge proofs

**Issues**:
- `generate_reputation_proof()` (L164-218): Returns mock proof structure with placeholder values
- `verify_reputation_proof()` (L220-266): Basic validation only, no cryptographic verification
- Comment at L197: "In production, this would call snarkjs via subprocess"
- Comment at L238: "In production, verify the actual cryptographic proof here using snarkjs or libsnark"
- TODO at L354-378: Example Circom circuit provided but not implemented

**Replacement Strategy**:

**Option 1: snarkjs + Circom** (Recommended)
```bash
# Install snarkjs and circom
npm install -g snarkjs
npm install -g circom

# Python wrapper
pip install py-snarkjs
```

```python
# Implementation using snarkjs
from crypto.zkp.snarkjs_wrapper import SnarkjsWrapper

class ZKRepSystem:
    def __init__(self):
        self.snarkjs = SnarkjsWrapper(
            circuit_path="circuits/reputation_tier.circom",
            protocol="groth16"
        )

    def generate_reputation_proof(self, robot_id, claimed_tier):
        # Generate witness
        witness = {
            "reputation_score": self.reputations[robot_id].score,
            "robot_id_hash": self._hash_robot_id(robot_id),
            "claimed_tier": claimed_tier.value
        }

        # Generate proof using snarkjs
        proof = self.snarkjs.generate_proof(witness)
        return proof

    def verify_reputation_proof(self, proof, claimed_tier, robot_commitment):
        # Verify using snarkjs
        public_signals = [claimed_tier, robot_commitment]
        return self.snarkjs.verify_proof(proof, public_signals)
```

**Option 2: arkworks (Rust-based)** (Alternative)
```bash
# More performant but requires Rust bindings
pip install arkworks-py
```

**Dependencies**:
- Node.js + npm (for snarkjs/circom)
- Python packages: `py-snarkjs` or `arkworks-py`

**Circom Circuit**: Implement `circuits/reputation_tier.circom` (example provided in TODO comment)

---

### 6. Device Scheduler (`learning/scheduling/device_scheduler.py`)

**Location**: Lines 446-452
**Type**: TODO for RL policy update

**Issues**:
- `update_policy()` method incomplete
- Comment at L446: "TODO: Implement full PPO or A3C update"
- Current implementation only computes reward, no gradient update

**Replacement Strategy**:

**Use stable-baselines3 for PPO**
```bash
pip install stable-baselines3
```

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
import gym

class FLSchedulingEnv(gym.Env):
    """Gym environment for FL device scheduling"""

    def __init__(self, max_devices=100, total_bandwidth=1000.0):
        super().__init__()

        # Observation space: device states
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(max_devices, 9),  # 9 features per device
            dtype=np.float32
        )

        # Action space: selection + bandwidth allocation
        self.action_space = spaces.Box(
            low=0, high=1,
            shape=(max_devices, 2),  # [selection_prob, bandwidth_frac]
            dtype=np.float32
        )

    def step(self, action):
        # Execute scheduling decision
        # Return: obs, reward, done, info
        pass

    def reset(self):
        # Reset environment
        pass

class DeviceScheduler:
    def __init__(self, ...):
        # Create environment
        env = DummyVecEnv([lambda: FLSchedulingEnv(...)])

        # Initialize PPO agent
        self.ppo_agent = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=2048,
            batch_size=64,
            learning_rate=3e-4,
            ent_coef=0.01,
        )

    def update_policy(self, states, decisions, actual_delays, actual_energies, accuracy_improvement):
        # Compute reward
        reward = ...

        # Update PPO agent
        self.ppo_agent.learn(total_timesteps=1000)
```

**Dependencies**:
- stable-baselines3
- gym
- PyTorch (already installed)

---

## Replacement Priority

### High Priority (Critical for production)
1. ✅ **zkRep System** - Security/privacy critical for FL
2. ✅ **CSA Engine** - Required for multi-actor coordination
3. ✅ **Device Scheduler** - Performance optimization for FL

### Medium Priority (Functionality complete but using mocks)
4. ✅ **Training Bridge** - Local training works but needs real IL models
5. ✅ **Skill Engine** - Skill execution works but needs real policies

### Low Priority (Minor TODOs)
6. ✅ **Industrial Data** - Placeholders don't block functionality

---

## Implementation Steps

### Step 1: Setup Dependencies (30 min)

```bash
# Install ZKP dependencies
npm install -g snarkjs circom
pip install py-snarkjs

# Install RL dependencies
pip install stable-baselines3 gym

# Install industrial analytics
pip install scipy scikit-learn

# Install IL model dependencies
pip install diffusers transformers einops

# Clone Dynamical repository
cd external
git clone https://github.com/Danielfoojunwei/Dynamical-Skill-Centric-Location-Adaptive-Privacy-Preserving-Imitation-Leanring-Platform.git dynamical
cd dynamical && pip install -e . && cd ../..
```

### Step 2: Replace zkRep System (2 hours)

- Implement Circom circuit: `circuits/reputation_tier.circom`
- Create snarkjs wrapper: `crypto/zkp/snarkjs_wrapper.py`
- Update `ZKRepSystem` class to use real proofs
- Generate proving/verification keys
- Write tests

### Step 3: Replace CSA Engine (2 hours)

- Import actual `HierarchicalCoordinationEncoder` from SwarmBridge
- Import `IntentCommunicationModule` from SwarmBridge
- Import `RoleSpecificAdapter` from SwarmBridge
- Update CSAEngine methods to use real implementations
- Write integration tests

### Step 4: Replace Training/Skill Engines (3 hours)

- Import Dynamical models (DiffusionPolicy, ACTPolicy, TransformerPolicy)
- Update `DynamicalTrainingBridge` to use real training pipeline
- Update `DynamicalSkillEngine` to load real policies
- Remove all mock model fallbacks
- Write integration tests

### Step 5: Replace Device Scheduler (2 hours)

- Create `FLSchedulingEnv` gym environment
- Replace LSTM network with PPO from stable-baselines3
- Implement full PPO update in `update_policy()`
- Write training script for scheduler
- Write tests

### Step 6: Add Industrial Data Analytics (1 hour)

- Implement `OEECalculator` class
- Implement `EquipmentHealthMonitor` class
- Update `get_aggregated_data()` to return real metrics
- Write tests

### Step 7: Integration Testing (2 hours)

- Test full workflow: Mission → CSA execution → FL training → Device scheduling
- Performance benchmarking
- Documentation updates

---

## Success Criteria

✅ All mock code removed
✅ Real open-source libraries integrated
✅ All tests passing
✅ No fallback to mock implementations
✅ Performance metrics meet requirements
✅ Documentation updated

---

## Open Source Libraries Used

### Core Tri-System
- ✅ **SwarmBridge**: Multi-actor coordination, FL coordination
- ✅ **Dynamical**: IL policies, training pipeline, skill execution

### Machine Learning
- ✅ **PyTorch**: Deep learning framework (already used)
- ✅ **stable-baselines3**: Reinforcement learning (PPO, A3C)
- ✅ **diffusers**: Diffusion models for IL
- ✅ **transformers**: Transformer models for IL

### Cryptography
- ✅ **snarkjs**: zk-SNARK proof generation/verification
- ✅ **circom**: Circuit compiler for ZK proofs

### Industrial/Analytics
- ✅ **pandas**: Data manipulation
- ✅ **scipy**: Signal processing, statistics
- ✅ **scikit-learn**: Anomaly detection, ML utilities

### Infrastructure
- ✅ **gym**: RL environment interface
- ✅ **numpy**: Numerical computing (already used)

---

## Estimated Timeline

- **Total Effort**: ~12 hours
- **Completion Target**: 1-2 days

## Risk Mitigation

- **Dependency conflicts**: Use virtual environment, pin versions
- **Breaking changes**: Maintain backward compatibility wrappers
- **Performance regression**: Benchmark before/after
- **Integration failures**: Comprehensive testing at each step

---

## Next Steps

1. Review and approve this replacement plan
2. Create feature branch: `feat/remove-all-mock-code`
3. Execute implementation steps 1-7
4. Code review and testing
5. Merge to main branch
