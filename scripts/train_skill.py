#!/usr/bin/env python3
"""
Train Skill Script

Trains a robot skill using the dynamical_2 imitation learning pipeline
and prepares it for SwarmBrain deployment.

Usage:
    python scripts/train_skill.py --skill pick_and_place --data demonstrations/
    python scripts/train_skill.py --skill assembly --robot robot_001 --epochs 200
"""

import argparse
import sys
from pathlib import Path
import yaml
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integration.dynamical_adapter.skill_engine import SkillConfig
from integration.dynamical_adapter.training_bridge import (
    DynamicalTrainingBridge,
    TrainingConfig
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_skill_config(skill_name: str, config_file: str = "config/dynamical/skills.yaml") -> dict:
    """Load skill configuration from YAML."""
    config_path = PROJECT_ROOT / config_file

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Find skill in config
    for skill in config.get('skills', []):
        if skill['name'] == skill_name:
            return skill

    logger.error(f"Skill '{skill_name}' not found in configuration")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Train a robot skill')

    parser.add_argument(
        '--skill',
        type=str,
        required=True,
        help='Name of skill to train (e.g., pick_and_place, assembly)'
    )

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to demonstration data directory'
    )

    parser.add_argument(
        '--robot',
        type=str,
        default='robot_local',
        help='Robot ID for this training run'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='robot_control/skills/dynamical/models',
        help='Output directory for trained model'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to train on'
    )

    parser.add_argument(
        '--export-fl',
        action='store_true',
        help='Export model for federated learning'
    )

    args = parser.parse_args()

    # Load skill configuration
    logger.info(f"Loading configuration for skill: {args.skill}")
    skill_config = load_skill_config(args.skill)

    # Create training configuration
    training_config = TrainingConfig(
        robot_id=args.robot,
        skill_name=args.skill,
        data_dir=args.data,
        output_dir=args.output,
        policy_type=skill_config['policy_type'],
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )

    # Initialize training bridge
    workspace_dir = PROJECT_ROOT / "workspace" / "training"
    bridge = DynamicalTrainingBridge(workspace_dir=str(workspace_dir))

    # Train skill
    logger.info(f"Starting training for '{args.skill}' on robot {args.robot}")
    logger.info(f"  Policy type: {skill_config['policy_type']}")
    logger.info(f"  Data directory: {args.data}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Device: {args.device}")

    try:
        result = bridge.train_skill_locally(training_config)

        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)
        logger.info(f"  Skill: {result.skill_name}")
        logger.info(f"  Robot: {result.robot_id}")
        logger.info(f"  Model saved to: {result.model_path}")
        logger.info(f"  Final loss: {result.final_loss:.4f}")
        logger.info(f"  Training time: {result.training_time:.1f}s")
        logger.info("=" * 60)

        # Export for federated learning if requested
        if args.export_fl:
            logger.info("Exporting model for federated learning...")
            fl_export = bridge.export_for_federated_learning(
                model_path=result.model_path,
                robot_id=args.robot
            )

            export_path = workspace_dir / "fl_exports" / f"{args.skill}_{args.robot}.npz"
            export_path.parent.mkdir(parents=True, exist_ok=True)

            import numpy as np
            np.savez(export_path, **fl_export)

            logger.info(f"FL export saved to: {export_path}")

        # Print training summary
        summary = bridge.get_training_summary()
        logger.info("\nTraining Summary:")
        logger.info(f"  Total trainings: {summary['total_trainings']}")
        logger.info(f"  Skills trained: {', '.join(summary['skills_trained'])}")
        logger.info(f"  Average training time: {summary['avg_training_time']:.1f}s")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
