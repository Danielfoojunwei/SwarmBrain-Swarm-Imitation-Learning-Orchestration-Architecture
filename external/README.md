# External Dependencies

This directory contains external repositories integrated with SwarmBrain.

## Multi-Actor Repository

The Multi-Actor Swarm Imitation Learning repository is cloned here for integration.

**Repository**: https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture.git

### Setup

To set up the multi-actor integration, run:

```bash
./scripts/setup_multi_actor.sh
```

This will:
1. Clone the multi_actor repository to `external/multi_actor`
2. Install dependencies
3. Configure the integration

### Manual Setup

Alternatively, clone manually:

```bash
git clone https://github.com/Danielfoojunwei/SwarmBridge-Multi-Actor-Swarm-Imitation-Learning-Architecture.git external/multi_actor
```

**Note**: The `external/multi_actor` directory is git-ignored to avoid submodule complexity.
The integration code in `integration/multi_actor_adapter/` references this repository.
