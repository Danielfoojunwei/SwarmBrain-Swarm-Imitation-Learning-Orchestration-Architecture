"""
End-to-End CSA Pipeline Example

This example demonstrates the complete pipeline from SwarmBridge training
to CSA deployment and execution on Dynamical-equipped robots.

Pipeline Flow:
1. SwarmBridge: Train multi-actor skill via federated learning
2. SwarmBridge → CSA Registry: Export and upload CSA
3. SwarmBrain: Import CSA and register with Dynamical
4. SwarmBrain: Plan mission with multi-actor coordination
5. Dynamical: Execute coordinated skill on robot fleet

Prerequisites:
- All services running (SwarmBrain, SwarmBridge, CSA Registry, Dynamical)
- Multi-actor demonstrations collected (ROS 2 bags)
- At least 2 robots registered with Dynamical
"""

import logging
import time
from typing import Dict, Any

# SwarmBrain components
from orchestrator.learning.fl_service_client import FederatedLearningServiceClient, FLRoundConfig
from orchestrator.integration.csa_importer import CSAImporter
from orchestrator.integration.dynamical_executor import (
    DynamicalExecutor,
    SkillExecutionRequest,
)
from orchestrator.schemas.csa_schema import CooperativeSkillArtifact

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def step_1_train_multi_actor_skill_via_swarmbridge() -> str:
    """
    Step 1: Train a multi-actor skill using SwarmBridge federated learning

    This simulates a multi-site federated learning session where multiple
    robots collect demonstrations and train a cooperative skill (e.g., handover).

    Returns:
        training_job_id: SwarmBridge training job ID
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Training multi-actor skill via SwarmBridge")
    logger.info("=" * 80)

    fl_client = FederatedLearningServiceClient(swarmbridge_url="http://localhost:8083")

    # Configure federated learning round
    round_config = FLRoundConfig(
        round_id="handover_training_v1",
        learning_mode="multi_actor",  # Multi-actor IL
        privacy_mode="secure_aggregation",
        aggregation_strategy="fedavg",
        min_participants=2,
        max_participants=5,
        timeout_seconds=3600,
        csa_base_id="collaborative_handover",
        num_actors=2,
    )

    logger.info(f"Starting FL training round: {round_config.round_id}")
    logger.info(f"  Learning mode: {round_config.learning_mode}")
    logger.info(f"  Privacy mode: {round_config.privacy_mode}")
    logger.info(f"  Required participants: {round_config.min_participants}")

    # Start training round
    round_result = fl_client.start_training_round(round_config)
    logger.info(f"✅ Training round started: {round_result}")

    # Wait for training to complete
    logger.info("Waiting for training to complete (this may take a while)...")
    final_status = fl_client.wait_for_round_completion(
        round_id=round_config.round_id,
        poll_interval=10,
        max_wait=3600,
    )

    if final_status.get("status") == "completed":
        logger.info("✅ Training completed successfully!")
        logger.info(f"   Aggregated metrics: {final_status.get('aggregated_metrics')}")

        # Extract training job ID for CSA export
        training_job_id = final_status.get("training_job_id", round_config.round_id)
        return training_job_id
    else:
        logger.error(f"❌ Training failed with status: {final_status.get('status')}")
        raise RuntimeError("Training did not complete successfully")


def step_2_export_and_upload_csa(training_job_id: str) -> str:
    """
    Step 2: Export CSA from SwarmBridge and upload to CSA Registry

    After training completes, SwarmBridge exports the trained multi-actor
    skill as a Cooperative Skill Artifact (CSA) in Dynamical-compatible format.

    Args:
        training_job_id: SwarmBridge training job ID

    Returns:
        skill_id: Exported CSA skill ID
    """
    logger.info("=" * 80)
    logger.info("STEP 2: Exporting CSA from SwarmBridge and uploading to registry")
    logger.info("=" * 80)

    fl_client = FederatedLearningServiceClient(swarmbridge_url="http://localhost:8083")

    # Complete training-to-deployment pipeline
    logger.info(f"Exporting CSA from training job: {training_job_id}")
    skill_id = fl_client.complete_training_to_deployment_pipeline(
        training_job_id=training_job_id,
        csa_registry_url="http://localhost:8082",
    )

    if skill_id:
        logger.info(f"✅ CSA exported and uploaded: {skill_id}")
        return skill_id
    else:
        logger.error("❌ Failed to export CSA")
        raise RuntimeError("CSA export failed")


def step_3_import_csa_and_register_with_dynamical(skill_id: str) -> CooperativeSkillArtifact:
    """
    Step 3: Import CSA and register with Dynamical API

    SwarmBrain fetches the CSA from the registry, validates compatibility
    with Dynamical's MoE format, and registers it for skill execution.

    Args:
        skill_id: CSA skill ID

    Returns:
        csa: CooperativeSkillArtifact object
    """
    logger.info("=" * 80)
    logger.info("STEP 3: Importing CSA and registering with Dynamical")
    logger.info("=" * 80)

    csa_importer = CSAImporter(
        csa_registry_url="http://localhost:8082",
        swarmbridge_url="http://localhost:8083",
        dynamical_api_url="http://localhost:8085",
    )

    # Fetch CSA from registry
    logger.info(f"Fetching CSA from registry: {skill_id}")
    csa = csa_importer.fetch_csa_from_registry(skill_id)

    if not csa:
        logger.error(f"❌ CSA not found in registry: {skill_id}")
        raise RuntimeError("CSA not found")

    logger.info(f"✅ Fetched CSA: {csa.skill_name} (v{csa.version})")
    logger.info(f"   Required roles: {csa.required_roles}")
    logger.info(f"   Required actors: {csa.required_actors}")
    logger.info(f"   Coordination primitives: {[p.value for p in csa.coordination_primitives]}")

    # Validate Dynamical compatibility
    logger.info("Validating Dynamical compatibility...")
    csa_importer.validate_dynamical_compatibility(csa)
    logger.info("✅ CSA is compatible with Dynamical")

    # Register with Dynamical API
    logger.info("Registering CSA with Dynamical...")
    success = csa_importer.register_csa_with_dynamical(csa)

    if success:
        logger.info(f"✅ CSA registered with Dynamical: {skill_id}")
        return csa
    else:
        logger.error("❌ Failed to register CSA with Dynamical")
        raise RuntimeError("CSA registration failed")


def step_4_plan_multi_actor_mission(csa: CooperativeSkillArtifact) -> Dict[str, Any]:
    """
    Step 4: Plan a multi-actor mission using the CSA

    SwarmBrain's mission planner creates a work order with coordinated
    multi-robot tasks using the registered CSA.

    Args:
        csa: CooperativeSkillArtifact

    Returns:
        mission_plan: Mission execution plan
    """
    logger.info("=" * 80)
    logger.info("STEP 4: Planning multi-actor mission")
    logger.info("=" * 80)

    # Simulate mission planning (in production, this would call MissionOrchestrator)
    logger.info(f"Planning mission with CSA: {csa.skill_id}")
    logger.info(f"   Skill: {csa.skill_name}")
    logger.info(f"   Roles needed: {csa.required_roles}")

    # Assign robots to roles
    robot_role_assignments = {
        "robot_001": "giver",    # Robot 1 will be the giver
        "robot_002": "receiver",  # Robot 2 will be the receiver
    }

    logger.info(f"   Robot assignments: {robot_role_assignments}")

    # Create mission plan
    mission_plan = {
        "mission_id": "mission_handover_001",
        "csa_skill_id": csa.skill_id,
        "robot_role_assignments": robot_role_assignments,
        "coordination_group_id": "handover_group_001",
        "parameters": {
            "object_id": "box_001",
            "handover_location": {"x": 1.0, "y": 0.5, "z": 0.3},
            "approach_speed": 0.2,
        },
        "timeout": 60.0,
    }

    logger.info("✅ Mission plan created:")
    logger.info(f"   Mission ID: {mission_plan['mission_id']}")
    logger.info(f"   Coordination group: {mission_plan['coordination_group_id']}")

    return mission_plan


def step_5_execute_mission_on_dynamical_robots(mission_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 5: Execute coordinated skill on Dynamical-equipped robots

    SwarmBrain deploys the mission to Dynamical API, which orchestrates
    the coordinated skill execution across the robot fleet.

    Args:
        mission_plan: Mission execution plan

    Returns:
        execution_results: Results from all robots
    """
    logger.info("=" * 80)
    logger.info("STEP 5: Executing mission on Dynamical robots")
    logger.info("=" * 80)

    executor = DynamicalExecutor(dynamical_api_url="http://localhost:8085")

    # Execute multi-actor skill
    logger.info(f"Deploying mission: {mission_plan['mission_id']}")
    logger.info(f"   CSA: {mission_plan['csa_skill_id']}")
    logger.info(f"   Robots: {list(mission_plan['robot_role_assignments'].keys())}")

    results = executor.execute_multi_actor_skill(
        csa_skill_id=mission_plan['csa_skill_id'],
        robot_role_assignments=mission_plan['robot_role_assignments'],
        coordination_group_id=mission_plan['coordination_group_id'],
        parameters=mission_plan['parameters'],
        timeout=mission_plan['timeout'],
    )

    # Report results
    logger.info("=" * 80)
    logger.info("Execution Results:")
    logger.info("=" * 80)

    all_success = True
    for robot_id, result in results.items():
        role = mission_plan['robot_role_assignments'][robot_id]
        logger.info(f"Robot {robot_id} (role={role}):")
        logger.info(f"  Status: {result.status.value}")
        logger.info(f"  Success: {result.success}")
        logger.info(f"  Duration: {result.duration:.2f}s" if result.duration else "  Duration: N/A")

        if result.error_message:
            logger.error(f"  Error: {result.error_message}")
            all_success = False

    if all_success:
        logger.info("✅ Mission completed successfully!")
    else:
        logger.error("⚠️ Mission completed with errors")

    return results


def main():
    """
    Run the complete end-to-end pipeline
    """
    logger.info("=" * 80)
    logger.info("SwarmBrain End-to-End CSA Pipeline Demo")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This demo shows the complete flow:")
    logger.info("1. SwarmBridge federated training")
    logger.info("2. CSA export and registry upload")
    logger.info("3. SwarmBrain CSA import and Dynamical registration")
    logger.info("4. Multi-actor mission planning")
    logger.info("5. Coordinated skill execution on robots")
    logger.info("")
    logger.info("=" * 80)

    try:
        # Step 1: Train via SwarmBridge
        training_job_id = step_1_train_multi_actor_skill_via_swarmbridge()

        # Step 2: Export CSA
        skill_id = step_2_export_and_upload_csa(training_job_id)

        # Step 3: Import and register CSA
        csa = step_3_import_csa_and_register_with_dynamical(skill_id)

        # Step 4: Plan mission
        mission_plan = step_4_plan_multi_actor_mission(csa)

        # Step 5: Execute mission
        results = step_5_execute_mission_on_dynamical_robots(mission_plan)

        logger.info("=" * 80)
        logger.info("🎉 End-to-end pipeline completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Trained CSA: {csa.skill_id}")
        logger.info(f"Mission: {mission_plan['mission_id']}")
        logger.info(f"Execution results: {len(results)} robots")

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ Pipeline failed: {e}")
        logger.error("=" * 80)
        raise


if __name__ == "__main__":
    main()
