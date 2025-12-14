"""
Gymnasium Environment for FL Device Scheduling

Trains PPO agent to select devices and allocate bandwidth for federated learning.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple


class FLSchedulingEnv(gym.Env):
    """
    Gymnasium environment for federated learning device scheduling.

    Observation: Device states (battery, CPU, bandwidth, etc.)
    Action: Device selection probabilities + bandwidth allocation
    Reward: Accuracy improvement - delay penalty - energy penalty
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_devices: int = 100,
        total_bandwidth: float = 1000.0,
        energy_weight: float = 0.3,
        delay_weight: float = 0.5,
        accuracy_weight: float = 0.2,
    ):
        super().__init__()

        self.max_devices = max_devices
        self.total_bandwidth = total_bandwidth
        self.energy_weight = energy_weight
        self.delay_weight = delay_weight
        self.accuracy_weight = accuracy_weight

        # Observation space: device states
        # Each device has 9 features: battery, CPU, memory, bandwidth, channel quality,
        # data size (log scale), staleness, priority, bias
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(max_devices, 9),
            dtype=np.float32
        )

        # Action space: selection probability + bandwidth fraction for each device
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(max_devices, 2),
            dtype=np.float32
        )

        self.current_step = 0
        self.device_states = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize random device states
        self.device_states = self.observation_space.sample()
        self.current_step = 0

        return self.device_states, {}

    def step(self, action):
        """
        Execute scheduling action.

        Args:
            action: (max_devices, 2) array with [selection_prob, bandwidth_frac]

        Returns:
            observation, reward, terminated, truncated, info
        """
        selection_probs = action[:, 0]
        bandwidth_fracs = action[:, 1]

        # Select devices based on probabilities
        selected = np.random.rand(self.max_devices) < selection_probs
        num_selected = np.sum(selected)

        # Ensure minimum 5 devices selected
        if num_selected < 5:
            unselected = np.where(~selected)[0]
            additional = np.random.choice(unselected, size=min(5 - num_selected, len(unselected)), replace=False)
            selected[additional] = True
            num_selected = np.sum(selected)

        # Normalize bandwidth allocations
        selected_bandwidth = bandwidth_fracs[selected]
        total_requested = np.sum(selected_bandwidth)
        if total_requested > 0:
            normalized_bandwidth = selected_bandwidth / total_requested * self.total_bandwidth
        else:
            normalized_bandwidth = np.ones(num_selected) * (self.total_bandwidth / num_selected)

        # Compute metrics
        delays = self._estimate_delays(self.device_states[selected], normalized_bandwidth)
        energies = self._estimate_energies(self.device_states[selected], normalized_bandwidth)
        accuracy_improvement = self._estimate_accuracy_improvement(num_selected)

        # Compute reward
        avg_delay = np.mean(delays)
        avg_energy = np.mean(energies)

        reward = (
            self.accuracy_weight * accuracy_improvement
            - self.delay_weight * (avg_delay / 100.0)
            - self.energy_weight * (avg_energy / 1000.0)
        )

        # Update device states (simulate battery drain, staleness increase, etc.)
        self.device_states = self._update_device_states(self.device_states, selected)

        self.current_step += 1
        terminated = self.current_step >= 100
        truncated = False

        info = {
            'num_selected': num_selected,
            'avg_delay': avg_delay,
            'avg_energy': avg_energy,
            'accuracy_improvement': accuracy_improvement
        }

        return self.device_states, reward, terminated, truncated, info

    def _estimate_delays(self, states, bandwidths):
        """Estimate communication delays"""
        model_size_mb = 50.0
        upload_delays = (model_size_mb * 8) / bandwidths
        processing_delays = states[:, 1] * 10.0  # CPU usage factor
        return upload_delays + processing_delays

    def _estimate_energies(self, states, bandwidths):
        """Estimate energy consumption"""
        compute_power = 50.0 * (1 + states[:, 1])
        transmit_power = 5.0 + bandwidths / 100.0
        delays = self._estimate_delays(states, bandwidths)
        return (compute_power + transmit_power) * delays

    def _estimate_accuracy_improvement(self, num_participants):
        """Estimate accuracy improvement based on participants"""
        return min(1.0, num_participants / 20.0) * 0.05

    def _update_device_states(self, states, selected):
        """Update device states after round"""
        new_states = states.copy()
        # Drain battery for selected devices
        new_states[selected, 0] *= 0.95
        # Increase staleness for non-selected
        new_states[~selected, 6] = np.minimum(1.0, new_states[~selected, 6] + 0.1)
        # Reset staleness for selected
        new_states[selected, 6] = 0.0
        return new_states
