"""
Dynamical.ai Skill Engine Adapter

Integrates the Dynamical.ai imitation learning pipeline with SwarmBrain's
robot control layer. Allows robots to execute skills trained via the
dynamical_2 system.

Architecture:
    dynamical_2 (IL Pipeline) → Trained Policies → Skill Engine → Robot Execution
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import logging
from dataclasses import dataclass
from datetime import datetime

# Add dynamical_2 to Python path
DYNAMICAL_2_PATH = Path(__file__).parent.parent.parent / "external" / "dynamical_2"
if DYNAMICAL_2_PATH.exists():
    sys.path.insert(0, str(DYNAMICAL_2_PATH))


@dataclass
class SkillConfig:
    """Configuration for a trained skill."""
    skill_name: str
    model_path: str
    policy_type: str  # "diffusion", "act", "transformer"
    input_modalities: List[str]  # ["rgb", "depth", "proprioception"]
    action_dim: int
    observation_dim: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    confidence_threshold: float = 0.8


@dataclass
class SkillExecution:
    """Result of skill execution."""
    skill_name: str
    action: np.ndarray
    confidence: float
    timestamp: datetime
    observations: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


class DynamicalSkillEngine:
    """
    Skill engine that loads and executes policies trained with dynamical_2.

    Manages:
    - Policy loading from checkpoints
    - Observation preprocessing
    - Action inference
    - Skill execution monitoring
    """

    def __init__(
        self,
        skills_dir: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize skill engine.

        Args:
            skills_dir: Directory containing trained skill models
            device: Device to run inference on (cuda/cpu)
        """
        self.skills_dir = Path(skills_dir)
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Loaded skills
        self.skills: Dict[str, Any] = {}
        self.skill_configs: Dict[str, SkillConfig] = {}

        # Execution history
        self.execution_history: List[SkillExecution] = []

        self.logger.info(f"Initialized Dynamical Skill Engine on {device}")

    def load_skill(self, skill_config: SkillConfig):
        """
        Load a trained skill from checkpoint.

        Args:
            skill_config: Configuration for the skill to load
        """
        try:
            model_path = Path(skill_config.model_path)

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            # Load model based on policy type
            if skill_config.policy_type == "diffusion":
                policy = self._load_diffusion_policy(model_path, skill_config)
            elif skill_config.policy_type == "act":
                policy = self._load_act_policy(model_path, skill_config)
            elif skill_config.policy_type == "transformer":
                policy = self._load_transformer_policy(model_path, skill_config)
            else:
                raise ValueError(f"Unknown policy type: {skill_config.policy_type}")

            policy = policy.to(self.device)
            policy.eval()

            self.skills[skill_config.skill_name] = policy
            self.skill_configs[skill_config.skill_name] = skill_config

            self.logger.info(
                f"Loaded skill '{skill_config.skill_name}' "
                f"({skill_config.policy_type}) from {model_path}"
            )

        except Exception as e:
            self.logger.error(f"Error loading skill {skill_config.skill_name}: {e}")
            raise

    def _load_diffusion_policy(self, model_path: Path, config: SkillConfig) -> torch.nn.Module:
        """Load diffusion policy model."""
        try:
            # Import from dynamical_2
            from models.diffusion_policy import DiffusionPolicy

            policy = DiffusionPolicy(
                observation_dim=config.observation_dim,
                action_dim=config.action_dim,
                # Add other config parameters
            )

            checkpoint = torch.load(model_path, map_location=self.device)
            policy.load_state_dict(checkpoint['model_state_dict'])

            return policy

        except ImportError:
            self.logger.warning(
                "dynamical_2 models not found, using mock policy for development"
            )
            return self._create_mock_policy(config)

    def _load_act_policy(self, model_path: Path, config: SkillConfig) -> torch.nn.Module:
        """Load ACT (Action Chunking Transformer) policy model."""
        try:
            from models.act_policy import ACTPolicy

            policy = ACTPolicy(
                observation_dim=config.observation_dim,
                action_dim=config.action_dim,
            )

            checkpoint = torch.load(model_path, map_location=self.device)
            policy.load_state_dict(checkpoint['model_state_dict'])

            return policy

        except ImportError:
            return self._create_mock_policy(config)

    def _load_transformer_policy(self, model_path: Path, config: SkillConfig) -> torch.nn.Module:
        """Load transformer-based policy model."""
        try:
            from models.transformer_policy import TransformerPolicy

            policy = TransformerPolicy(
                observation_dim=config.observation_dim,
                action_dim=config.action_dim,
            )

            checkpoint = torch.load(model_path, map_location=self.device)
            policy.load_state_dict(checkpoint['model_state_dict'])

            return policy

        except ImportError:
            return self._create_mock_policy(config)

    def _create_mock_policy(self, config: SkillConfig) -> torch.nn.Module:
        """Create a mock policy for testing when dynamical_2 is not available."""
        class MockPolicy(torch.nn.Module):
            def __init__(self, obs_dim, act_dim):
                super().__init__()
                self.fc = torch.nn.Linear(obs_dim, act_dim)

            def forward(self, obs):
                return self.fc(obs)

        return MockPolicy(config.observation_dim, config.action_dim)

    def execute_skill(
        self,
        skill_name: str,
        observations: Dict[str, np.ndarray],
        context: Optional[Dict[str, Any]] = None
    ) -> SkillExecution:
        """
        Execute a skill given current observations.

        Args:
            skill_name: Name of skill to execute
            observations: Dictionary of observation modalities
            context: Optional context (e.g., object locations, goals)

        Returns:
            SkillExecution with action and metadata
        """
        if skill_name not in self.skills:
            raise ValueError(f"Skill '{skill_name}' not loaded")

        config = self.skill_configs[skill_name]
        policy = self.skills[skill_name]

        try:
            # Preprocess observations
            obs_tensor = self._preprocess_observations(observations, config)

            # Run inference
            with torch.no_grad():
                action_tensor = policy(obs_tensor)

            # Postprocess action
            action = self._postprocess_action(action_tensor, config)

            # Compute confidence (if policy supports it)
            confidence = self._compute_confidence(action_tensor, config)

            # Create execution result
            execution = SkillExecution(
                skill_name=skill_name,
                action=action,
                confidence=confidence,
                timestamp=datetime.utcnow(),
                observations=observations,
                metadata={
                    'policy_type': config.policy_type,
                    'device': str(self.device),
                    'context': context
                }
            )

            self.execution_history.append(execution)

            return execution

        except Exception as e:
            self.logger.error(f"Error executing skill {skill_name}: {e}")
            raise

    def _preprocess_observations(
        self,
        observations: Dict[str, np.ndarray],
        config: SkillConfig
    ) -> torch.Tensor:
        """Preprocess observations for policy input."""
        # Stack required modalities
        obs_list = []

        for modality in config.input_modalities:
            if modality not in observations:
                raise ValueError(f"Missing required modality: {modality}")

            obs = observations[modality]

            # Convert to tensor
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).float()

            obs_list.append(obs)

        # Concatenate
        obs_tensor = torch.cat(obs_list, dim=-1)

        # Add batch dimension if needed
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        return obs_tensor.to(self.device)

    def _postprocess_action(
        self,
        action_tensor: torch.Tensor,
        config: SkillConfig
    ) -> np.ndarray:
        """Postprocess action from policy output."""
        action = action_tensor.cpu().numpy()

        # Remove batch dimension
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]

        # Clip to valid range
        action = np.clip(action, -1.0, 1.0)

        return action

    def _compute_confidence(
        self,
        action_tensor: torch.Tensor,
        config: SkillConfig
    ) -> float:
        """Compute confidence score for the action."""
        # Simple heuristic: use action magnitude variance
        # More sophisticated methods could use ensemble, dropout, etc.

        action_norm = torch.norm(action_tensor)
        confidence = min(1.0, action_norm.item() / 10.0)

        return confidence

    def get_available_skills(self) -> List[str]:
        """Get list of loaded skills."""
        return list(self.skills.keys())

    def get_skill_info(self, skill_name: str) -> Dict[str, Any]:
        """Get information about a loaded skill."""
        if skill_name not in self.skills:
            raise ValueError(f"Skill '{skill_name}' not loaded")

        config = self.skill_configs[skill_name]

        return {
            'skill_name': config.skill_name,
            'policy_type': config.policy_type,
            'input_modalities': config.input_modalities,
            'action_dim': config.action_dim,
            'observation_dim': config.observation_dim,
            'device': config.device,
            'model_path': config.model_path
        }

    def unload_skill(self, skill_name: str):
        """Unload a skill to free memory."""
        if skill_name in self.skills:
            del self.skills[skill_name]
            del self.skill_configs[skill_name]
            self.logger.info(f"Unloaded skill: {skill_name}")

    def clear_history(self):
        """Clear execution history."""
        self.execution_history.clear()

    def get_execution_stats(self, skill_name: Optional[str] = None) -> Dict[str, Any]:
        """Get execution statistics."""
        if skill_name:
            executions = [e for e in self.execution_history if e.skill_name == skill_name]
        else:
            executions = self.execution_history

        if not executions:
            return {'count': 0}

        confidences = [e.confidence for e in executions]

        return {
            'count': len(executions),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'recent_executions': len([e for e in executions if (datetime.utcnow() - e.timestamp).seconds < 60])
        }


# Example skill configurations
EXAMPLE_SKILLS = [
    SkillConfig(
        skill_name="pick_and_place",
        model_path="models/pick_and_place_diffusion.pt",
        policy_type="diffusion",
        input_modalities=["rgb", "depth", "proprioception"],
        action_dim=7,  # 6 DoF arm + gripper
        observation_dim=1024
    ),
    SkillConfig(
        skill_name="assembly",
        model_path="models/assembly_act.pt",
        policy_type="act",
        input_modalities=["rgb", "proprioception"],
        action_dim=14,  # Bimanual robot
        observation_dim=512
    ),
    SkillConfig(
        skill_name="inspection",
        model_path="models/inspection_transformer.pt",
        policy_type="transformer",
        input_modalities=["rgb", "depth"],
        action_dim=7,
        observation_dim=2048
    ),
]
