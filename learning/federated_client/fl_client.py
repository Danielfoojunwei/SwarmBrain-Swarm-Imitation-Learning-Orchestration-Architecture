"""
Federated Learning Client for SwarmBrain

Uses Flower framework for federated learning across robot fleets.
Supports privacy-preserving aggregation and dynamic clustering.
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging

try:
    import flwr as fl
    from flwr.common import (
        Code,
        EvaluateIns,
        EvaluateRes,
        FitIns,
        FitRes,
        GetParametersIns,
        GetParametersRes,
        Parameters,
        Status,
    )
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False
    logging.warning("Flower not installed. Install with: pip install flwr")


@dataclass
class ClientConfig:
    """Configuration for federated learning client."""
    client_id: str
    cluster_id: Optional[str] = None
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    enable_secure_aggregation: bool = True
    enable_reputation: bool = True
    bandwidth_limit_mbps: Optional[float] = None


class SwarmBrainClient(fl.client.NumPyClient if FLWR_AVAILABLE else object):
    """
    Federated learning client for SwarmBrain.

    Each robot/site runs this client to participate in federated learning
    without sharing raw teleoperation data.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: ClientConfig
    ):
        """
        Initialize federated learning client.

        Args:
            model: PyTorch model (skill policy network)
            train_loader: DataLoader for local training data
            val_loader: DataLoader for validation data
            config: Client configuration
        """
        if not FLWR_AVAILABLE:
            raise ImportError("Flower is required. Install with: pip install flwr")

        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        self.criterion = nn.MSELoss()

        self.logger.info(
            f"Initialized FL client {config.client_id} "
            f"(cluster: {config.cluster_id}, device: {config.device})"
        )

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """
        Get model parameters as NumPy arrays.

        Args:
            config: Configuration from server

        Returns:
            List of model parameters as NumPy arrays
        """
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Set model parameters from NumPy arrays.

        Args:
            parameters: Model parameters as NumPy arrays
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {
            k: torch.tensor(v) for k, v in params_dict
        }
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """
        Train model on local data.

        Args:
            parameters: Global model parameters from server
            config: Training configuration from server

        Returns:
            Tuple of (updated parameters, number of examples, metrics)
        """
        self.set_parameters(parameters)

        # Train for specified number of local epochs
        num_epochs = int(config.get("local_epochs", self.config.local_epochs))

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            total_loss += epoch_loss
            self.logger.debug(
                f"Client {self.config.client_id} - "
                f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(self.train_loader):.4f}"
            )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Get updated parameters
        updated_parameters = self.get_parameters({})

        # Number of training examples
        num_examples = len(self.train_loader.dataset)

        metrics = {
            "train_loss": avg_loss,
            "num_examples": num_examples,
            "cluster_id": self.config.cluster_id or "default"
        }

        self.logger.info(
            f"Client {self.config.client_id} completed training: "
            f"loss={avg_loss:.4f}, examples={num_examples}"
        )

        return updated_parameters, num_examples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str]
    ) -> Tuple[float, int, Dict[str, float]]:
        """
        Evaluate model on local validation data.

        Args:
            parameters: Model parameters from server
            config: Evaluation configuration

        Returns:
            Tuple of (loss, number of examples, metrics)
        """
        self.set_parameters(parameters)

        self.model.eval()
        total_loss = 0.0
        num_examples = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * len(inputs)
                num_examples += len(inputs)

        avg_loss = total_loss / num_examples if num_examples > 0 else 0.0

        metrics = {
            "val_loss": avg_loss,
            "num_examples": num_examples
        }

        self.logger.info(
            f"Client {self.config.client_id} evaluation: "
            f"loss={avg_loss:.4f}, examples={num_examples}"
        )

        return avg_loss, num_examples, metrics


def create_client_fn(
    model_fn: Callable[[], nn.Module],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: ClientConfig
) -> Callable[[str], SwarmBrainClient]:
    """
    Create a client factory function for Flower.

    Args:
        model_fn: Function that creates a new model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Client configuration

    Returns:
        Client factory function
    """
    def client_fn(cid: str) -> SwarmBrainClient:
        """Create a new client instance."""
        config.client_id = cid
        model = model_fn()
        return SwarmBrainClient(model, train_loader, val_loader, config)

    return client_fn


def start_client(
    server_address: str,
    client: SwarmBrainClient,
    root_certificates: Optional[bytes] = None
):
    """
    Start federated learning client and connect to server.

    Args:
        server_address: Address of FL server (e.g., "localhost:8080")
        client: SwarmBrain FL client instance
        root_certificates: Optional TLS certificates for secure connection
    """
    if not FLWR_AVAILABLE:
        raise ImportError("Flower is required. Install with: pip install flwr")

    logging.info(f"Starting FL client, connecting to {server_address}")

    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),
        root_certificates=root_certificates
    )
