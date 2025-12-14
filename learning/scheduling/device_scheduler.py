"""
Joint Device Scheduling and Bandwidth Allocation

Uses deep reinforcement learning with LSTM to decide which robots participate
in each federated learning round and how much bandwidth they receive.

Minimizes delay and energy consumption while maintaining learning quality.

Reference:
"Joint Device Scheduling and Resource Allocation for Federated Learning"
NTU research on efficient FL resource allocation
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


@dataclass
class DeviceState:
    """State of a device (robot) for scheduling decisions."""
    device_id: str
    battery_level: float  # 0.0 to 1.0
    cpu_usage: float  # 0.0 to 1.0
    memory_available: float  # GB
    network_bandwidth: float  # Mbps
    channel_quality: float  # 0.0 to 1.0 (SINR normalized)
    data_size: int  # Number of training samples
    model_staleness: int  # Rounds since last participation
    task_priority: float  # 0.0 to 1.0


@dataclass
class SchedulingDecision:
    """Scheduling decision for a device."""
    device_id: str
    selected: bool
    bandwidth_allocation: float  # Mbps
    priority: float  # Scheduling priority
    expected_delay: float  # Estimated completion time (seconds)
    expected_energy: float  # Estimated energy consumption (Joules)


class LSTMSchedulerNetwork(nn.Module):
    """
    LSTM-based network for device scheduling and bandwidth allocation.

    Takes device states and historical data as input, outputs scheduling
    decisions and bandwidth allocations.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        max_devices: int = 100
    ):
        """
        Initialize scheduler network.

        Args:
            state_dim: Dimension of device state vector
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            max_devices: Maximum number of devices to schedule
        """
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_devices = max_devices

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # Selection head (binary decision: participate or not)
        self.selection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output probability of selection
        )

        # Bandwidth allocation head
        self.bandwidth_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Output positive bandwidth value
        )

        # Value head for RL (estimated return)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        device_states: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.

        Args:
            device_states: Tensor of shape (batch, seq_len, num_devices, state_dim)
            hidden: Optional hidden state for LSTM

        Returns:
            Tuple of (selection_probs, bandwidth_allocs, values, new_hidden)
        """
        batch_size, seq_len, num_devices, _ = device_states.shape

        # Reshape for LSTM: (batch * num_devices, seq_len, state_dim)
        states_reshaped = device_states.view(batch_size * num_devices, seq_len, self.state_dim)

        # LSTM forward
        lstm_out, new_hidden = self.lstm(states_reshaped, hidden)

        # Take last timestep output
        last_output = lstm_out[:, -1, :]  # (batch * num_devices, hidden_dim)

        # Generate predictions
        selection_probs = self.selection_head(last_output)  # (batch * num_devices, 1)
        bandwidth_allocs = self.bandwidth_head(last_output)  # (batch * num_devices, 1)
        values = self.value_head(last_output)  # (batch * num_devices, 1)

        # Reshape back to (batch, num_devices, 1)
        selection_probs = selection_probs.view(batch_size, num_devices, 1)
        bandwidth_allocs = bandwidth_allocs.view(batch_size, num_devices, 1)
        values = values.view(batch_size, num_devices, 1)

        return selection_probs, bandwidth_allocs, values, new_hidden


class DeviceScheduler:
    """
    Joint device scheduler and bandwidth allocator for federated learning.

    Uses deep RL with LSTM to make scheduling decisions that minimize
    delay and energy consumption.
    """

    def __init__(
        self,
        state_dim: int = 9,  # Number of features in DeviceState
        total_bandwidth: float = 1000.0,  # Total available bandwidth (Mbps)
        max_devices_per_round: int = 20,
        min_devices_per_round: int = 5,
        energy_weight: float = 0.3,  # Weight for energy in reward
        delay_weight: float = 0.5,  # Weight for delay in reward
        accuracy_weight: float = 0.2,  # Weight for accuracy in reward
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize device scheduler.

        Args:
            state_dim: Dimension of device state
            total_bandwidth: Total bandwidth available for allocation
            max_devices_per_round: Maximum devices to select per round
            min_devices_per_round: Minimum devices to select per round
            energy_weight: Weight for energy consumption in reward
            delay_weight: Weight for delay in reward
            accuracy_weight: Weight for accuracy improvement in reward
            device: Device to run model on (cpu/cuda)
        """
        self.state_dim = state_dim
        self.total_bandwidth = total_bandwidth
        self.max_devices = max_devices_per_round
        self.min_devices = min_devices_per_round

        self.energy_weight = energy_weight
        self.delay_weight = delay_weight
        self.accuracy_weight = accuracy_weight

        self.device = device
        self.logger = logging.getLogger(__name__)

        # Initialize LSTM scheduler network
        self.network = LSTMSchedulerNetwork(
            state_dim=state_dim,
            hidden_dim=128,
            num_layers=2,
            max_devices=max_devices_per_round
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=1e-4
        )

        # History for LSTM
        self.state_history: List[List[DeviceState]] = []
        self.max_history_length = 10

        self.logger.info(
            f"Initialized device scheduler with {max_devices_per_round} max devices, "
            f"{total_bandwidth} Mbps total bandwidth"
        )

    def schedule_devices(
        self,
        device_states: List[DeviceState],
        greedy: bool = False
    ) -> List[SchedulingDecision]:
        """
        Schedule devices for the next FL round.

        Args:
            device_states: Current states of all devices
            greedy: If True, use greedy selection (no exploration)

        Returns:
            List of scheduling decisions
        """
        # Add to history
        self.state_history.append(device_states)
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)

        # Convert states to tensor
        state_tensor = self._states_to_tensor(self.state_history)

        # Run network forward
        with torch.no_grad():
            selection_probs, bandwidth_allocs, _, _ = self.network(state_tensor)

        # Convert to numpy
        selection_probs = selection_probs.squeeze().cpu().numpy()
        bandwidth_allocs = bandwidth_allocs.squeeze().cpu().numpy()

        # Make scheduling decisions
        decisions = []

        # Select devices based on probabilities
        if greedy:
            # Select top-k devices by probability
            selected_indices = np.argsort(selection_probs)[-self.max_devices:]
            selected = np.zeros(len(device_states), dtype=bool)
            selected[selected_indices] = True
        else:
            # Sample based on probabilities
            selected = np.random.rand(len(device_states)) < selection_probs

        # Ensure we have at least min_devices
        num_selected = np.sum(selected)
        if num_selected < self.min_devices:
            # Select additional devices
            unselected = np.where(~selected)[0]
            additional = np.random.choice(
                unselected,
                size=min(self.min_devices - num_selected, len(unselected)),
                replace=False
            )
            selected[additional] = True

        # Normalize bandwidth allocations for selected devices
        selected_bandwidth = bandwidth_allocs[selected]
        total_requested = np.sum(selected_bandwidth)

        if total_requested > 0:
            normalized_bandwidth = (
                selected_bandwidth / total_requested * self.total_bandwidth
            )
        else:
            # Equal allocation
            normalized_bandwidth = np.ones(np.sum(selected)) * (
                self.total_bandwidth / np.sum(selected)
            )

        # Create scheduling decisions
        bandwidth_idx = 0
        for idx, (device_state, is_selected) in enumerate(zip(device_states, selected)):
            if is_selected:
                bw = normalized_bandwidth[bandwidth_idx]
                bandwidth_idx += 1

                # Estimate delay and energy
                delay = self._estimate_delay(device_state, bw)
                energy = self._estimate_energy(device_state, bw)

                decision = SchedulingDecision(
                    device_id=device_state.device_id,
                    selected=True,
                    bandwidth_allocation=float(bw),
                    priority=float(selection_probs[idx]),
                    expected_delay=delay,
                    expected_energy=energy
                )
            else:
                decision = SchedulingDecision(
                    device_id=device_state.device_id,
                    selected=False,
                    bandwidth_allocation=0.0,
                    priority=float(selection_probs[idx]),
                    expected_delay=0.0,
                    expected_energy=0.0
                )

            decisions.append(decision)

        num_selected = sum(1 for d in decisions if d.selected)
        total_bw = sum(d.bandwidth_allocation for d in decisions)

        self.logger.info(
            f"Scheduled {num_selected} devices with total bandwidth {total_bw:.2f} Mbps"
        )

        return decisions

    def _states_to_tensor(
        self,
        state_history: List[List[DeviceState]]
    ) -> torch.Tensor:
        """Convert state history to tensor for network input."""
        # Extract features from each device state
        seq_data = []

        for states in state_history:
            timestep_data = []
            for state in states:
                features = [
                    state.battery_level,
                    state.cpu_usage,
                    state.memory_available / 100.0,  # Normalize
                    state.network_bandwidth / 1000.0,  # Normalize to Gbps
                    state.channel_quality,
                    np.log10(state.data_size + 1) / 6.0,  # Log scale, normalize
                    state.model_staleness / 10.0,  # Normalize
                    state.task_priority,
                    1.0  # Bias term
                ]
                timestep_data.append(features)
            seq_data.append(timestep_data)

        # Convert to tensor: (1, seq_len, num_devices, state_dim)
        tensor = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0)
        return tensor.to(self.device)

    def _estimate_delay(self, state: DeviceState, bandwidth: float) -> float:
        """
        Estimate communication delay for a device.

        Args:
            state: Device state
            bandwidth: Allocated bandwidth (Mbps)

        Returns:
            Estimated delay in seconds
        """
        # Simple model: delay = data_size / bandwidth + processing_time
        model_size_mb = 50.0  # Assume 50 MB model
        upload_delay = (model_size_mb * 8) / bandwidth  # Convert MB to Mbits

        # Processing time depends on CPU usage and data size
        processing_delay = state.data_size / 1000.0 * (1 + state.cpu_usage)

        total_delay = upload_delay + processing_delay

        return total_delay

    def _estimate_energy(self, state: DeviceState, bandwidth: float) -> float:
        """
        Estimate energy consumption for a device.

        Args:
            state: Device state
            bandwidth: Allocated bandwidth (Mbps)

        Returns:
            Estimated energy in Joules
        """
        # Simple energy model
        # E = P_compute * T_compute + P_transmit * T_transmit

        # Power consumption (Watts)
        compute_power = 50.0 * (1 + state.cpu_usage)  # Higher CPU usage = more power
        transmit_power = 5.0 + bandwidth / 100.0  # Power increases with bandwidth

        # Time
        delay = self._estimate_delay(state, bandwidth)
        compute_time = state.data_size / 1000.0 * (1 + state.cpu_usage)
        transmit_time = delay - compute_time

        # Energy
        compute_energy = compute_power * compute_time
        transmit_energy = transmit_power * transmit_time

        total_energy = compute_energy + transmit_energy

        return total_energy

    def update_policy(
        self,
        states: List[DeviceState],
        decisions: List[SchedulingDecision],
        actual_delays: List[float],
        actual_energies: List[float],
        accuracy_improvement: float
    ):
        """
        Update scheduling policy based on observed outcomes.

        Args:
            states: Device states used for scheduling
            decisions: Scheduling decisions made
            actual_delays: Actual observed delays
            actual_energies: Actual observed energies
            accuracy_improvement: Improvement in model accuracy
        """
        # Compute reward
        avg_delay = np.mean([d for d, dec in zip(actual_delays, decisions) if dec.selected])
        avg_energy = np.mean([e for e, dec in zip(actual_energies, decisions) if dec.selected])

        # Normalize and combine into reward (higher is better)
        reward = (
            self.accuracy_weight * accuracy_improvement
            - self.delay_weight * (avg_delay / 100.0)  # Normalize delay
            - self.energy_weight * (avg_energy / 1000.0)  # Normalize energy
        )

        self.logger.debug(
            f"Policy update: reward={reward:.4f}, "
            f"delay={avg_delay:.2f}s, energy={avg_energy:.2f}J, "
            f"accuracy_improvement={accuracy_improvement:.4f}"
        )

        # TODO: Implement full PPO or A3C update
        # For now, this is a placeholder
        # Full implementation would involve:
        # 1. Computing advantage estimates
        # 2. PPO clipped objective
        # 3. Gradient descent on policy and value losses

    def save_model(self, path: str):
        """Save scheduler model to disk."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        self.logger.info(f"Saved scheduler model to {path}")

    def load_model(self, path: str):
        """Load scheduler model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Loaded scheduler model from {path}")
