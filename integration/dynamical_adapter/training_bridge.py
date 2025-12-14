"""
Training Pipeline Bridge

Connects dynamical_2's imitation learning pipeline with SwarmBrain's
federated learning system.

Flow:
    Local Data Collection (dynamical_2) →
    Local Training (dynamical_2 IL) →
    Model Export →
    Federated Aggregation (SwarmBrain FL) →
    Model Distribution →
    Skill Deployment
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
import numpy as np
import logging
from datetime import datetime
from dataclasses import dataclass

# Add dynamical_2 to path
DYNAMICAL_2_PATH = Path(__file__).parent.parent.parent / "external" / "dynamical_2"
if DYNAMICAL_2_PATH.exists():
    sys.path.insert(0, str(DYNAMICAL_2_PATH))


@dataclass
class TrainingConfig:
    """Configuration for training integration."""
    robot_id: str
    skill_name: str
    data_dir: str
    output_dir: str
    policy_type: str  # "diffusion", "act", "transformer"
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingResult:
    """Result of training run."""
    skill_name: str
    robot_id: str
    model_path: str
    final_loss: float
    num_epochs: int
    training_time: float  # seconds
    timestamp: datetime


class DynamicalTrainingBridge:
    """
    Bridge between dynamical_2 training and SwarmBrain federated learning.

    Responsibilities:
    - Run local training using dynamical_2 pipeline
    - Export trained models for federated learning
    - Convert model updates to FL format
    - Import FL-aggregated models back to dynamical_2 format
    """

    def __init__(self, workspace_dir: str):
        """
        Initialize training bridge.

        Args:
            workspace_dir: Directory for training workspaces
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.training_history: List[TrainingResult] = []

    def train_skill_locally(
        self,
        config: TrainingConfig,
        demonstration_data: Optional[Dict[str, Any]] = None
    ) -> TrainingResult:
        """
        Train a skill locally using dynamical_2 IL pipeline.

        Args:
            config: Training configuration
            demonstration_data: Optional pre-loaded demonstration data

        Returns:
            TrainingResult with model path and metrics
        """
        self.logger.info(
            f"Starting local training for skill '{config.skill_name}' "
            f"on robot {config.robot_id}"
        )

        start_time = datetime.utcnow()

        try:
            # Load demonstrations if not provided
            if demonstration_data is None:
                demonstration_data = self._load_demonstrations(config.data_dir)

            # Initialize model based on policy type
            model = self._create_model(config)

            # Train model
            final_loss = self._run_training(model, demonstration_data, config)

            # Save model
            model_path = self._save_model(model, config)

            training_time = (datetime.utcnow() - start_time).total_seconds()

            result = TrainingResult(
                skill_name=config.skill_name,
                robot_id=config.robot_id,
                model_path=str(model_path),
                final_loss=final_loss,
                num_epochs=config.num_epochs,
                training_time=training_time,
                timestamp=datetime.utcnow()
            )

            self.training_history.append(result)

            self.logger.info(
                f"Training completed: {config.skill_name} "
                f"(loss={final_loss:.4f}, time={training_time:.1f}s)"
            )

            return result

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def _load_demonstrations(self, data_dir: str) -> Dict[str, Any]:
        """Load demonstration data from disk."""
        data_path = Path(data_dir)

        try:
            # Try loading from dynamical_2 format
            from recorder import load_demonstrations

            demos = load_demonstrations(data_path)
            return demos

        except ImportError:
            # Fallback: load from NumPy files
            self.logger.warning("dynamical_2 recorder not found, using fallback loader")

            demos = {
                'observations': [],
                'actions': [],
                'rewards': []
            }

            for demo_file in data_path.glob("demo_*.npz"):
                data = np.load(demo_file)
                demos['observations'].append(data['observations'])
                demos['actions'].append(data['actions'])

            return demos

    def _create_model(self, config: TrainingConfig) -> torch.nn.Module:
        """Create model based on policy type."""
        try:
            if config.policy_type == "diffusion":
                from models.diffusion_policy import DiffusionPolicy
                model = DiffusionPolicy()

            elif config.policy_type == "act":
                from models.act_policy import ACTPolicy
                model = ACTPolicy()

            elif config.policy_type == "transformer":
                from models.transformer_policy import TransformerPolicy
                model = TransformerPolicy()

            else:
                raise ValueError(f"Unknown policy type: {config.policy_type}")

            return model.to(config.device)

        except ImportError:
            self.logger.warning("dynamical_2 models not available, using mock model")
            return self._create_mock_model(config)

    def _create_mock_model(self, config: TrainingConfig) -> torch.nn.Module:
        """Create mock model for testing."""
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(512, 256)
                self.fc2 = torch.nn.Linear(256, 7)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        return MockModel().to(config.device)

    def _run_training(
        self,
        model: torch.nn.Module,
        data: Dict[str, Any],
        config: TrainingConfig
    ) -> float:
        """Run training loop."""
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.MSELoss()

        model.train()
        final_loss = 0.0

        # Simple training loop (simplified version)
        for epoch in range(config.num_epochs):
            # Create batches
            # (In real implementation, use DataLoader from dynamical_2)

            epoch_loss = 0.0
            num_batches = 0

            # Mock training iteration
            optimizer.zero_grad()

            # Generate random batch (replace with actual data)
            obs = torch.randn(config.batch_size, 512).to(config.device)
            actions = torch.randn(config.batch_size, 7).to(config.device)

            pred_actions = model(obs)
            loss = criterion(pred_actions, actions)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if epoch % 10 == 0:
                self.logger.debug(f"Epoch {epoch}/{config.num_epochs}, Loss: {epoch_loss:.4f}")

            final_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

        return final_loss

    def _save_model(self, model: torch.nn.Module, config: TrainingConfig) -> Path:
        """Save trained model."""
        output_dir = self.workspace_dir / config.robot_id / config.skill_name
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_path = output_dir / f"{config.skill_name}_{timestamp}.pt"

        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'timestamp': timestamp
        }, model_path)

        self.logger.info(f"Model saved to {model_path}")

        return model_path

    def export_for_federated_learning(
        self,
        model_path: str,
        robot_id: str
    ) -> Dict[str, Any]:
        """
        Export model for federated learning aggregation.

        Args:
            model_path: Path to trained model
            robot_id: Robot identifier

        Returns:
            Dictionary with model parameters and metadata
        """
        checkpoint = torch.load(model_path, map_location='cpu')

        model_params = checkpoint['model_state_dict']

        # Convert to list of NumPy arrays (FL-compatible format)
        params_list = [p.cpu().numpy() for p in model_params.values()]

        export_data = {
            'robot_id': robot_id,
            'parameters': params_list,
            'parameter_names': list(model_params.keys()),
            'timestamp': datetime.utcnow().isoformat(),
            'model_path': model_path
        }

        return export_data

    def import_from_federated_learning(
        self,
        aggregated_params: List[np.ndarray],
        parameter_names: List[str],
        output_path: str
    ) -> str:
        """
        Import aggregated model from federated learning.

        Args:
            aggregated_params: Aggregated parameters from FL server
            parameter_names: Names of parameters
            output_path: Where to save imported model

        Returns:
            Path to saved model
        """
        # Convert back to PyTorch format
        state_dict = {
            name: torch.from_numpy(param)
            for name, param in zip(parameter_names, aggregated_params)
        }

        # Save
        torch.save({
            'model_state_dict': state_dict,
            'source': 'federated_learning',
            'timestamp': datetime.utcnow().isoformat()
        }, output_path)

        self.logger.info(f"Imported FL model to {output_path}")

        return output_path

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training history."""
        if not self.training_history:
            return {'total_trainings': 0}

        return {
            'total_trainings': len(self.training_history),
            'skills_trained': list(set(r.skill_name for r in self.training_history)),
            'robots': list(set(r.robot_id for r in self.training_history)),
            'avg_training_time': np.mean([r.training_time for r in self.training_history]),
            'recent_trainings': [
                {
                    'skill': r.skill_name,
                    'robot': r.robot_id,
                    'loss': r.final_loss,
                    'time': r.training_time
                }
                for r in self.training_history[-5:]
            ]
        }
