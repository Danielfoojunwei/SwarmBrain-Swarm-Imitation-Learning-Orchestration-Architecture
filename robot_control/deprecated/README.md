# Deprecated Robot Control Code

This directory contains robot control code that has been deprecated in favor of
API-based skill execution through the Dynamical platform.

## Refactoring Changes

### What Was Removed

1. **Local Skill Execution** - Robots no longer execute skills locally. Instead,
   they receive skill execution requests from the Dynamical API.

2. **Direct ROS 2 Control** - Low-level ROS 2 control loops have been moved to
   the Dynamical runtime. SwarmBrain now orchestrates robots via API calls.

3. **Embedded Policies** - Policy models are no longer stored locally. They are
   managed by the unified skill registry (CSA Registry for multi-actor,
   Dynamical API for single-actor).

### New Architecture

```
SwarmBrain (Orchestrator)
    ↓ API calls
Dynamical Runtime (Skill Execution)
    ↓ ROS 2 DDS
Robot Controllers (Low-level Control)
```

### Migration Guide

If you need to execute skills on robots:

1. **Use Dynamical API** instead of local execution:
   ```python
   # Old way (deprecated)
   robot_controller.execute_skill(skill_name, observations)

   # New way
   import requests
   requests.post(
       f"{dynamical_api_url}/api/v1/robots/{robot_id}/execute",
       json={"skill_id": skill_id, "observations": observations}
   )
   ```

2. **Use Skill Registry** instead of local skill management:
   ```python
   # Old way (deprecated)
   orchestrator.register_skill(skill_name)

   # New way
   from orchestrator.registry import UnifiedSkillRegistryClient
   registry = UnifiedSkillRegistryClient()
   skills = registry.list_skills()
   ```

3. **Use FL Service Client** instead of Flower:
   ```python
   # Old way (deprecated)
   from learning.federated_client import fl_client

   # New way
   from orchestrator.learning import FederatedLearningServiceClient
   fl_client = FederatedLearningServiceClient()
   fl_client.start_training_round(round_config)
   ```

## Files Moved Here

- `ros2_nodes/robot_controller.py` - Moved to Dynamical runtime
- `skills/` - Moved to unified skill registry
- Direct Flower FL integration - Replaced with FL service client

## See Also

- [Unified Skill Registry](../../orchestrator/registry/)
- [FL Service Client](../../orchestrator/learning/)
- [Multi-Actor Integration](../../docs/guides/multi_actor_integration.md)
