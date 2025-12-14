"""
Dropout-Resilient Secure Aggregation

Implements secure aggregation that tolerates robot disconnections during
federated learning rounds. Based on seed-homomorphic pseudorandom generators
and Shamir secret sharing.

Reference:
"Dropout-Resilient Secure Aggregation for Federated Learning"
NTU research on robust secure aggregation
"""

from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass
import hashlib
import secrets
from collections import defaultdict
import logging


@dataclass
class SecretShare:
    """A Shamir secret share."""
    participant_id: str
    share_value: bytes
    threshold: int
    total_participants: int


class SeedHomomorphicPRG:
    """
    Seed-homomorphic pseudorandom generator for secure aggregation.

    Allows combining encrypted values without decryption.
    """

    def __init__(self, seed: Optional[bytes] = None):
        """
        Initialize PRG with optional seed.

        Args:
            seed: Random seed (generated if not provided)
        """
        self.seed = seed or secrets.token_bytes(32)
        self.logger = logging.getLogger(__name__)

    def generate(self, length: int) -> np.ndarray:
        """
        Generate pseudorandom array.

        Args:
            length: Number of random values to generate

        Returns:
            NumPy array of random values
        """
        # Use seed to initialize RNG
        rng = np.random.RandomState(
            int.from_bytes(self.seed[:4], byteorder='big')
        )
        return rng.randn(length)

    def combine_seeds(self, other_seed: bytes) -> bytes:
        """
        Combine two seeds homomorphically (XOR operation).

        Args:
            other_seed: Seed to combine with

        Returns:
            Combined seed
        """
        return bytes(a ^ b for a, b in zip(self.seed, other_seed))


class ShamirSecretSharing:
    """
    Shamir's Secret Sharing for threshold cryptography.

    Allows recovery of secret even when some participants drop out.
    """

    def __init__(self, threshold: int, total_participants: int, prime: int = 2**127 - 1):
        """
        Initialize Shamir Secret Sharing.

        Args:
            threshold: Minimum number of shares needed to reconstruct
            total_participants: Total number of participants
            prime: Large prime number for modular arithmetic
        """
        self.threshold = threshold
        self.total_participants = total_participants
        self.prime = prime
        self.logger = logging.getLogger(__name__)

        if threshold > total_participants:
            raise ValueError(
                f"Threshold {threshold} cannot exceed total participants {total_participants}"
            )

    def share_secret(self, secret: int) -> List[Tuple[int, int]]:
        """
        Split secret into shares.

        Args:
            secret: Secret value to share

        Returns:
            List of (participant_id, share_value) tuples
        """
        # Generate random polynomial coefficients
        coeffs = [secret] + [
            secrets.randbelow(self.prime)
            for _ in range(self.threshold - 1)
        ]

        # Evaluate polynomial at different points
        shares = []
        for i in range(1, self.total_participants + 1):
            share_value = self._eval_polynomial(coeffs, i) % self.prime
            shares.append((i, share_value))

        return shares

    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> int:
        """
        Reconstruct secret from shares using Lagrange interpolation.

        Args:
            shares: List of (participant_id, share_value) tuples

        Returns:
            Reconstructed secret

        Raises:
            ValueError: If insufficient shares provided
        """
        if len(shares) < self.threshold:
            raise ValueError(
                f"Need at least {self.threshold} shares, got {len(shares)}"
            )

        # Use Lagrange interpolation at x=0
        secret = 0
        for i, (x_i, y_i) in enumerate(shares[:self.threshold]):
            numerator = 1
            denominator = 1

            for j, (x_j, _) in enumerate(shares[:self.threshold]):
                if i != j:
                    numerator = (numerator * (-x_j)) % self.prime
                    denominator = (denominator * (x_i - x_j)) % self.prime

            # Compute modular inverse
            lagrange_coeff = (numerator * pow(denominator, -1, self.prime)) % self.prime
            secret = (secret + y_i * lagrange_coeff) % self.prime

        return secret

    def _eval_polynomial(self, coeffs: List[int], x: int) -> int:
        """Evaluate polynomial at point x using Horner's method."""
        result = 0
        for coeff in reversed(coeffs):
            result = (result * x + coeff) % self.prime
        return result


class DropoutResilientAggregator:
    """
    Secure aggregator that tolerates participant dropouts.

    Combines seed-homomorphic PRG with Shamir secret sharing to enable
    secure aggregation even when robots disconnect mid-round.
    """

    def __init__(
        self,
        participant_ids: List[str],
        threshold_ratio: float = 0.5,
        seed_length: int = 32
    ):
        """
        Initialize dropout-resilient aggregator.

        Args:
            participant_ids: IDs of all participants
            threshold_ratio: Minimum ratio of participants needed (0.0-1.0)
            seed_length: Length of random seeds in bytes
        """
        self.participant_ids = participant_ids
        self.num_participants = len(participant_ids)
        self.threshold = max(1, int(threshold_ratio * self.num_participants))
        self.seed_length = seed_length

        self.logger = logging.getLogger(__name__)

        # Secret sharing scheme
        self.sss = ShamirSecretSharing(
            threshold=self.threshold,
            total_participants=self.num_participants
        )

        # Participant states
        self.participant_seeds: Dict[str, bytes] = {}
        self.participant_shares: Dict[str, List[Tuple[int, int]]] = {}
        self.masked_updates: Dict[str, np.ndarray] = {}

        self.logger.info(
            f"Initialized aggregator with {self.num_participants} participants, "
            f"threshold={self.threshold}"
        )

    def setup_round(self) -> Dict[str, Dict[str, any]]:
        """
        Set up a new aggregation round.

        Each participant generates seeds and shares them with others.

        Returns:
            Dictionary mapping participant IDs to their setup data
        """
        setup_data = {}

        for pid in self.participant_ids:
            # Generate random seed for this participant
            seed = secrets.token_bytes(self.seed_length)
            self.participant_seeds[pid] = seed

            # Create secret shares of the seed
            seed_int = int.from_bytes(seed, byteorder='big')
            shares = self.sss.share_secret(seed_int)
            self.participant_shares[pid] = shares

            setup_data[pid] = {
                'participant_id': pid,
                'shares_for_others': shares
            }

        return setup_data

    def add_masked_update(
        self,
        participant_id: str,
        model_update: np.ndarray,
        mask_seed: bytes
    ):
        """
        Add a masked model update from a participant.

        Args:
            participant_id: ID of participant
            model_update: Model update (gradient or parameter diff)
            mask_seed: Seed for generating privacy mask
        """
        # Generate mask from seed
        prg = SeedHomomorphicPRG(mask_seed)
        mask = prg.generate(len(model_update))

        # Apply mask to update
        masked_update = model_update + mask

        self.masked_updates[participant_id] = masked_update

        self.logger.debug(
            f"Added masked update from {participant_id}, "
            f"shape={model_update.shape}"
        )

    def aggregate(
        self,
        active_participants: Optional[Set[str]] = None
    ) -> np.ndarray:
        """
        Aggregate masked updates, handling dropouts.

        Args:
            active_participants: Set of participants who completed the round
                                (None = all participants)

        Returns:
            Aggregated model update

        Raises:
            ValueError: If too many participants dropped out
        """
        if active_participants is None:
            active_participants = set(self.participant_ids)

        num_active = len(active_participants)

        if num_active < self.threshold:
            raise ValueError(
                f"Too many dropouts: {num_active}/{self.num_participants} active, "
                f"need at least {self.threshold}"
            )

        self.logger.info(
            f"Aggregating with {num_active}/{self.num_participants} participants"
        )

        # Sum all masked updates
        aggregated = None
        for pid in active_participants:
            if pid in self.masked_updates:
                update = self.masked_updates[pid]
                if aggregated is None:
                    aggregated = np.zeros_like(update)
                aggregated += update

        if aggregated is None:
            raise ValueError("No updates to aggregate")

        # Remove masks by reconstructing seeds from shares
        for pid in active_participants:
            # Get shares from active participants
            shares = []
            for active_pid in active_participants:
                if pid in self.participant_shares.get(active_pid, {}):
                    participant_shares = self.participant_shares[active_pid]
                    # Find share for this participant
                    for idx, (share_id, share_val) in enumerate(participant_shares):
                        if self.participant_ids[share_id - 1] == pid:
                            shares.append((share_id, share_val))
                            break

            # Reconstruct seed if we have enough shares
            if len(shares) >= self.threshold:
                seed_int = self.sss.reconstruct_secret(shares)
                seed = seed_int.to_bytes(self.seed_length, byteorder='big')

                # Generate and subtract mask
                prg = SeedHomomorphicPRG(seed)
                mask = prg.generate(len(aggregated))
                aggregated -= mask

        # Average by number of participants
        aggregated /= num_active

        self.logger.info("Aggregation completed successfully")

        return aggregated

    def reset_round(self):
        """Reset state for a new round."""
        self.participant_seeds.clear()
        self.participant_shares.clear()
        self.masked_updates.clear()
        self.logger.debug("Round state reset")
