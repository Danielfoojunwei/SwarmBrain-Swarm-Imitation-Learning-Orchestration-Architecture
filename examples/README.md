# SwarmBrain Examples

This directory contains end-to-end examples demonstrating SwarmBrain's integration with Dynamical and SwarmBridge.

## Available Examples

### 1. End-to-End CSA Pipeline (`end_to_end_csa_pipeline.py`)

**Complete pipeline from SwarmBridge training to coordinated skill execution on Dynamical robots.**

**Pipeline Flow:**
```
SwarmBridge Training → CSA Export → Registry Upload →
SwarmBrain Import → Mission Planning → Dynamical Execution
```

**What it demonstrates:**
1. **SwarmBridge Federated Training**: Multi-actor imitation learning across sites
2. **CSA Export**: Converting trained models to Cooperative Skill Artifacts
3. **CSA Registration**: Importing CSAs and registering with Dynamical API
4. **Mission Planning**: Creating multi-robot coordinated missions
5. **Skill Execution**: Deploying and executing coordinated skills on robots

**Prerequisites:**
```bash
# Start all services
docker-compose --profile refactored up -d

# Verify services are healthy
curl http://localhost:8000/health  # SwarmBrain
curl http://localhost:8082/health  # CSA Registry
curl http://localhost:8083/health  # SwarmBridge
curl http://localhost:8085/health  # Dynamical API
```

**Run the example:**
```bash
# From SwarmBrain root directory
python examples/end_to_end_csa_pipeline.py
```

**Expected output:**
```
===================================================================================
SwarmBrain End-to-End CSA Pipeline Demo
===================================================================================

STEP 1: Training multi-actor skill via SwarmBridge
✅ Training completed successfully!

STEP 2: Exporting CSA from SwarmBridge and uploading to registry
✅ CSA exported and uploaded: collaborative_handover_v2

STEP 3: Importing CSA and registering with Dynamical
✅ CSA is compatible with Dynamical
✅ CSA registered with Dynamical: collaborative_handover_v2

STEP 4: Planning multi-actor mission
✅ Mission plan created: mission_handover_001

STEP 5: Executing mission on Dynamical robots
✅ Mission completed successfully!

🎉 End-to-end pipeline completed successfully!
===================================================================================
```

## Running Examples

### Option 1: With Live Services

Run examples against live SwarmBrain/SwarmBridge/Dynamical services:

```bash
# Start all services
docker-compose --profile refactored up -d

# Run example
python examples/end_to_end_csa_pipeline.py
```

### Option 2: With Mock Services (Development)

For development and testing without full infrastructure:

```python
# Set mock mode in environment
export SWARMBRAIN_MOCK_MODE=true
export SWARMBRIDGE_MOCK_MODE=true
export DYNAMICAL_MOCK_MODE=true

# Run example (will use mock responses)
python examples/end_to_end_csa_pipeline.py
```

## Key Concepts Demonstrated

### Cooperative Skill Artifacts (CSAs)

CSAs are the bridge between SwarmBridge's multi-actor training and Dynamical's skill execution:

```python
{
  "skill_id": "collaborative_handover_v2",
  "required_roles": ["giver", "receiver"],
  "required_actors": 2,

  "role_experts": {
    "giver": {
      "checkpoint_uri": "s3://skills/handover_giver.pth",
      "embedding_type": "moai_compressed",
      "architecture": "diffusion_policy",
      "encryption_scheme": "openfhe_bfv"
    },
    "receiver": { /* ... */ }
  },

  "coordination_encoder": {
    "checkpoint_uri": "s3://skills/handover_encoder.pth",
    "architecture": "transformer",
    "latent_dim": 256
  }
}
```

### Tri-System Integration

**Dynamical** (Foundation):
- Single-robot skill execution
- VLA + MoE architecture
- Privacy-preserving IL

**SwarmBridge** (Training):
- Multi-actor demonstration capture
- Federated cooperative learning
- CSA export in Dynamical format

**SwarmBrain** (Orchestration):
- Mission planning and coordination
- CSA import and deployment
- Fleet-wide skill orchestration

## Troubleshooting

### Services not responding
```bash
# Check service health
docker-compose ps
docker-compose logs swarmbrain-orchestrator
docker-compose logs swarmbrain-swarmbridge
docker-compose logs swarmbrain-dynamical
```

### CSA not found in registry
```bash
# List available CSAs
curl http://localhost:8082/api/v1/skills | jq '.[] | select(.skill_type=="multi_actor")'
```

### Robots not available
```bash
# Check robot status
curl http://localhost:8085/api/v1/robots | jq
```

### Training timeout
```bash
# Reduce training rounds for faster testing
# Edit FLRoundConfig in example:
FLRoundConfig(
    num_rounds=1,  # Reduce for quick testing
    timeout_seconds=600,  # Shorter timeout
)
```

## Next Steps

After running the examples:

1. **Modify CSA parameters**: Try different skill types, roles, coordination primitives
2. **Add more robots**: Scale to 3+ robot coordination
3. **Custom missions**: Create your own multi-actor task graphs
4. **Monitor with Grafana**: View metrics at http://localhost:3000

## Additional Resources

- [SwarmBrain Documentation](../docs/README.md)
- [SwarmBridge Repository](https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture)
- [Dynamical Repository](https://github.com/Danielfoojunwei/Dynamical-Skill-Centric-Location-Adaptive-Privacy-Preserving-Imitation-Leanring-Platform)
- [CSA Schema Reference](../orchestrator/schemas/csa_schema.py)
