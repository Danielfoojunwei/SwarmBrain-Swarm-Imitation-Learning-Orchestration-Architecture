"""
zkRep-Style Reputation System

Zero-knowledge proof based reputation system for weighting robot contributions
in federated learning without revealing identities or performance details.

Based on zkRep architecture from privacy-preserving reputation systems.

## Architecture

1. Robots prove their reputation level without revealing exact score
2. Proofs are verified by aggregation server
3. Contributions are weighted based on verified reputation tier
4. No robot identities or performance metrics are leaked

## ZKP Framework Selection

Based on "Survey of Open-Source ZKP Frameworks" (2024):
- **zk-SNARKs**: Succinct proofs, fast verification
- **Circom/snarkjs**: Mature, JavaScript/WASM compatible
- **libsnark**: C++ library, production-ready
- **Noir**: Rust-based, user-friendly syntax

For SwarmBrain, we recommend **Circom + snarkjs** for:
- Easy integration with Python (via subprocess)
- Strong community support
- Flexible circuit design
- Browser compatibility (future web dashboard)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import subprocess
import logging
from pathlib import Path

from crypto.zkp.snarkjs_wrapper import SnarkjsWrapper


class ReputationTier(Enum):
    """Reputation tiers for robots."""
    NOVICE = 1      # Low reputation (new or poor performance)
    INTERMEDIATE = 2  # Medium reputation
    EXPERT = 3      # High reputation (consistent good performance)
    MASTER = 4      # Very high reputation (exceptional performance)


@dataclass
class ReputationScore:
    """Reputation score for a robot."""
    robot_id: str
    score: float  # 0.0 to 100.0
    tier: ReputationTier
    num_contributions: int
    average_accuracy: float
    last_updated: str  # ISO timestamp


class ZKRepSystem:
    """
    Zero-knowledge reputation system for federated learning.

    Robots can prove their reputation tier without revealing exact scores.
    """

    def __init__(
        self,
        circuit_path: Optional[str] = None,
        proving_key_path: Optional[str] = None,
        verification_key_path: Optional[str] = None,
        use_real_proofs: bool = True
    ):
        """
        Initialize zkRep system.

        Args:
            circuit_path: Path to compiled Circom circuit
            proving_key_path: Path to proving key (deprecated, auto-generated)
            verification_key_path: Path to verification key (deprecated, auto-generated)
            use_real_proofs: Whether to use real ZK proofs (True) or mock proofs for testing (False)
        """
        self.circuit_path = circuit_path or "circuits/reputation_tier.circom"
        self.use_real_proofs = use_real_proofs

        self.logger = logging.getLogger(__name__)

        # Initialize snarkjs wrapper if using real proofs
        if self.use_real_proofs:
            try:
                self.snarkjs = SnarkjsWrapper(
                    circuit_path=self.circuit_path,
                    protocol="groth16"
                )

                # Setup circuit if not already done
                if not self.snarkjs.is_setup_complete():
                    self.logger.info("Setting up zkRep circuit (one-time setup)...")
                    self.snarkjs.compile_circuit()
                    self.snarkjs.generate_keys()
                    self.logger.info("Circuit setup complete")

            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize snarkjs: {e}. "
                    "Falling back to mock proofs. Install snarkjs/circom: "
                    "npm install -g snarkjs circom"
                )
                self.use_real_proofs = False
                self.snarkjs = None
        else:
            self.snarkjs = None

        # Reputation database (in production, use secure storage)
        self.reputations: Dict[str, ReputationScore] = {}

        # Salt storage for commitments (in production, use secure storage)
        self.robot_salts: Dict[str, int] = {}

        # Tier thresholds
        self.tier_thresholds = {
            ReputationTier.NOVICE: 0.0,
            ReputationTier.INTERMEDIATE: 25.0,
            ReputationTier.EXPERT: 60.0,
            ReputationTier.MASTER: 85.0
        }

        # Contribution weights by tier
        self.tier_weights = {
            ReputationTier.NOVICE: 0.5,
            ReputationTier.INTERMEDIATE: 1.0,
            ReputationTier.EXPERT: 1.5,
            ReputationTier.MASTER: 2.0
        }

        self.logger.info("Initialized zkRep reputation system")

    def update_reputation(
        self,
        robot_id: str,
        accuracy: float,
        loss: float,
        contribution_quality: float
    ):
        """
        Update robot's reputation based on contribution quality.

        Args:
            robot_id: Robot identifier
            accuracy: Model accuracy achieved
            loss: Model loss achieved
            contribution_quality: Quality score (0.0 to 1.0)
        """
        if robot_id not in self.reputations:
            # Initialize new robot
            self.reputations[robot_id] = ReputationScore(
                robot_id=robot_id,
                score=50.0,  # Start at intermediate level
                tier=ReputationTier.INTERMEDIATE,
                num_contributions=0,
                average_accuracy=0.0,
                last_updated=self._get_timestamp()
            )

        reputation = self.reputations[robot_id]

        # Update metrics
        reputation.num_contributions += 1
        reputation.average_accuracy = (
            (reputation.average_accuracy * (reputation.num_contributions - 1) + accuracy)
            / reputation.num_contributions
        )

        # Compute score update based on contribution quality
        # Higher quality = higher score increase
        score_delta = contribution_quality * 10.0 - 5.0  # Range: -5 to +5

        # Update score with decay to prevent unbounded growth
        decay_factor = 0.95
        reputation.score = (reputation.score * decay_factor + score_delta)

        # Clamp to [0, 100]
        reputation.score = max(0.0, min(100.0, reputation.score))

        # Update tier
        reputation.tier = self._score_to_tier(reputation.score)

        reputation.last_updated = self._get_timestamp()

        self.logger.info(
            f"Updated reputation for {robot_id}: "
            f"score={reputation.score:.2f}, tier={reputation.tier.name}"
        )

    def generate_reputation_proof(
        self,
        robot_id: str,
        claimed_tier: ReputationTier
    ) -> Optional[Dict[str, any]]:
        """
        Generate zero-knowledge proof of reputation tier.

        The proof demonstrates: "I have reputation >= claimed_tier"
        without revealing the exact reputation score.

        Args:
            robot_id: Robot identifier
            claimed_tier: Reputation tier to prove

        Returns:
            ZK proof object (or None if proof generation fails)
        """
        if robot_id not in self.reputations:
            self.logger.error(f"Robot {robot_id} not found in reputation database")
            return None

        reputation = self.reputations[robot_id]

        # Check if claim is valid
        if reputation.tier.value < claimed_tier.value:
            self.logger.error(
                f"Robot {robot_id} cannot prove tier {claimed_tier.name}, "
                f"actual tier is {reputation.tier.name}"
            )
            return None

        # Generate or retrieve salt for commitment
        if robot_id not in self.robot_salts:
            import random
            self.robot_salts[robot_id] = random.randint(1, 2**64)

        salt = self.robot_salts[robot_id]

        # Use real ZK proofs if enabled
        if self.use_real_proofs and self.snarkjs:
            try:
                # Convert robot_id to numeric hash for circuit input
                robot_id_numeric = int(hashlib.sha256(robot_id.encode()).hexdigest()[:16], 16)

                # Prepare witness (private + public inputs)
                witness = {
                    "reputation_score": int(reputation.score),
                    "robot_id": robot_id_numeric,
                    "salt": salt,
                    "claimed_tier": claimed_tier.value
                }

                # Generate proof using snarkjs
                proof = self.snarkjs.generate_proof(witness)

                self.logger.info(
                    f"Generated real ZK proof for {robot_id} claiming tier {claimed_tier.name}"
                )

                return proof

            except Exception as e:
                self.logger.error(f"Failed to generate ZK proof: {e}")
                return None

        else:
            # Fallback to mock proof (for testing without snarkjs)
            robot_commitment = self._hash_robot_id(robot_id)

            proof = {
                'proof': {
                    'pi_a': ['0x...', '0x...', '0x...'],  # G1 point
                    'pi_b': [['0x...', '0x...'], ['0x...', '0x...'], ['0x...', '0x...']],  # G2 point
                    'pi_c': ['0x...', '0x...', '0x...'],  # G1 point
                },
                'publicSignals': [
                    str(claimed_tier.value),  # Public: claimed tier
                    robot_commitment,  # Public: commitment to robot ID (simplified)
                ],
                'protocol': 'groth16',
                'curve': 'bn128'
            }

            self.logger.warning(
                f"Generated MOCK ZK proof for {robot_id} claiming tier {claimed_tier.name}. "
                "Install snarkjs/circom for real proofs."
            )

            return proof

    def verify_reputation_proof(
        self,
        proof: Dict[str, any],
        claimed_tier: ReputationTier,
        robot_commitment: Optional[str] = None
    ) -> bool:
        """
        Verify a reputation proof.

        Args:
            proof: ZK proof to verify
            claimed_tier: Claimed reputation tier
            robot_commitment: Commitment to robot ID (optional, extracted from proof if not provided)

        Returns:
            True if proof is valid, False otherwise
        """
        if not proof or ('proof' not in proof and 'publicSignals' not in proof):
            self.logger.error("Invalid proof structure")
            return False

        # Use real verification if enabled
        if self.use_real_proofs and self.snarkjs:
            try:
                # Verify proof using snarkjs
                is_valid = self.snarkjs.verify_proof(proof)

                if is_valid:
                    self.logger.info(
                        f"Verified real ZK proof for tier {claimed_tier.name}"
                    )
                else:
                    self.logger.warning(
                        f"ZK proof verification failed for tier {claimed_tier.name}"
                    )

                return is_valid

            except Exception as e:
                self.logger.error(f"Failed to verify ZK proof: {e}")
                return False

        else:
            # Fallback to mock verification (basic validation)
            if proof.get('protocol') != 'groth16':
                self.logger.warning(f"Unsupported proof protocol: {proof.get('protocol')}")
                return False

            # Verify public inputs match
            public_signals = proof.get('publicSignals', proof.get('public_inputs', []))
            if len(public_signals) != 2:
                return False

            if public_signals[0] != str(claimed_tier.value):
                return False

            # If robot_commitment provided, verify it matches
            if robot_commitment and public_signals[1] != robot_commitment:
                return False

            # Mock verification: accept all proofs with valid structure
            self.logger.warning(
                f"MOCK verification for tier {claimed_tier.name}. "
                "Install snarkjs/circom for real verification."
            )

            return True

    def get_contribution_weight(
        self,
        robot_id: str,
        proof: Optional[Dict[str, any]] = None
    ) -> float:
        """
        Get contribution weight for a robot.

        If proof is provided and valid, use claimed tier for weighting.
        Otherwise, use stored reputation.

        Args:
            robot_id: Robot identifier
            proof: Optional ZK proof of higher reputation

        Returns:
            Contribution weight multiplier
        """
        if robot_id not in self.reputations:
            # Unknown robot, assign minimum weight
            return self.tier_weights[ReputationTier.NOVICE]

        reputation = self.reputations[robot_id]

        # If proof provided, verify and potentially upgrade weight
        if proof:
            # Extract claimed tier from public signals
            public_signals = proof.get('publicSignals', proof.get('public_inputs', []))
            if public_signals:
                claimed_tier = ReputationTier(int(public_signals[0]))
                robot_commitment = self._hash_robot_id(robot_id)

                if self.verify_reputation_proof(proof, claimed_tier, robot_commitment):
                    # Use claimed tier weight
                    return self.tier_weights[claimed_tier]

        # Use stored tier weight
        return self.tier_weights[reputation.tier]

    def _score_to_tier(self, score: float) -> ReputationTier:
        """Convert reputation score to tier."""
        if score >= self.tier_thresholds[ReputationTier.MASTER]:
            return ReputationTier.MASTER
        elif score >= self.tier_thresholds[ReputationTier.EXPERT]:
            return ReputationTier.EXPERT
        elif score >= self.tier_thresholds[ReputationTier.INTERMEDIATE]:
            return ReputationTier.INTERMEDIATE
        else:
            return ReputationTier.NOVICE

    def _hash_robot_id(self, robot_id: str) -> str:
        """Create commitment to robot ID (hash)."""
        return hashlib.sha256(robot_id.encode()).hexdigest()

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + 'Z'

    def get_reputation_stats(self) -> Dict[str, any]:
        """Get statistics about reputation distribution."""
        if not self.reputations:
            return {
                'total_robots': 0,
                'tier_distribution': {},
                'average_score': 0.0
            }

        tier_counts = {tier: 0 for tier in ReputationTier}
        total_score = 0.0

        for reputation in self.reputations.values():
            tier_counts[reputation.tier] += 1
            total_score += reputation.score

        return {
            'total_robots': len(self.reputations),
            'tier_distribution': {
                tier.name: count
                for tier, count in tier_counts.items()
            },
            'average_score': total_score / len(self.reputations),
            'tier_weights': {
                tier.name: weight
                for tier, weight in self.tier_weights.items()
            }
        }


# Circom circuit for reputation proofs has been implemented!
# See: circuits/reputation_tier.circom
# Wrapper: crypto/zkp/snarkjs_wrapper.py
#
# The circuit proves: reputation_score >= tier_threshold
# Without revealing the exact score (zero-knowledge proof)
#
# To use real ZK proofs:
#   1. Install snarkjs and circom: npm install -g snarkjs circom
#   2. Initialize zkRep with use_real_proofs=True (default)
#   3. Circuit will auto-compile and generate keys on first use
#
# For testing without snarkjs:
#   - Initialize with use_real_proofs=False
#   - Falls back to mock proofs (structure validation only)
