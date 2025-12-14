"""
Cooperative Skill Artifact (CSA) Schema

This module defines the standardized schema for Cooperative Skill Artifacts (CSAs)
produced by SwarmBridge's federated multi-actor imitation learning pipeline.

CSAs are compatible with Dynamical's MoE (Mixture of Experts) format, allowing
SwarmBrain to deploy multi-actor skills to Dynamical-equipped robots.

Format Compatibility:
- Per-role experts packaged as Dynamical MoE modules
- MOAI-compressed embeddings for privacy-preserving inference
- Encryption schemes compatible with Dynamical (OpenFHE BFV/CKKS)
- Checkpoint URIs pointing to S3/MinIO/local storage

CSA Flow:
SwarmBridge (training) → CSA Registry → SwarmBrain (orchestration) → Dynamical (execution)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from enum import Enum


class SkillType(str, Enum):
    """Skill type classification"""
    SINGLE_ACTOR = "single_actor"  # Single-robot skill (Dynamical native)
    MULTI_ACTOR = "multi_actor"    # Multi-robot cooperative skill (CSA)


class PolicyArchitecture(str, Enum):
    """Supported policy architectures for skill experts"""
    DIFFUSION_POLICY = "diffusion_policy"  # Diffusion-based IL
    ACT = "act"                            # Action Chunking Transformer
    GPT = "gpt"                            # GPT-based autoregressive
    TRANSFORMER = "transformer"            # Generic transformer
    LSTM = "lstm"                          # LSTM-based
    MLP = "mlp"                            # Multi-layer perceptron


class EmbeddingType(str, Enum):
    """Embedding compression types"""
    MOAI_COMPRESSED = "moai_compressed"    # MOAI compression (Dynamical native)
    RAW = "raw"                            # Uncompressed embeddings
    PCA = "pca"                            # PCA dimensionality reduction
    AUTOENCODER = "autoencoder"            # Learned compression


class EncryptionScheme(str, Enum):
    """Encryption schemes for model weights"""
    OPENFHE_BFV = "openfhe_bfv"           # OpenFHE BFV (integer arithmetic)
    OPENFHE_CKKS = "openfhe_ckks"         # OpenFHE CKKS (approximate arithmetic)
    SEAL_BFV = "seal_bfv"                 # Microsoft SEAL BFV
    NONE = "none"                         # No encryption (testing only)


class CoordinationPrimitive(str, Enum):
    """Standardized coordination primitives"""
    HANDOVER = "handover"                 # Object handover between robots
    FORMATION = "formation"               # Formation maintenance
    BARRIER = "barrier"                   # Synchronization barrier
    RENDEZVOUS = "rendezvous"            # Spatial rendezvous
    SYNC_GRASP = "sync_grasp"            # Synchronized grasping
    LEADER_FOLLOWER = "leader_follower"  # Leader-follower coordination
    COLLISION_AVOIDANCE = "collision_avoidance"


@dataclass
class RoleExpertConfig:
    """Configuration for a single role's expert model (Dynamical MoE compatible)"""
    checkpoint_uri: str                   # S3/MinIO/local path to model checkpoint
    embedding_type: EmbeddingType         # Embedding compression type
    architecture: PolicyArchitecture      # Policy architecture type
    encryption_scheme: EncryptionScheme   # Encryption scheme for weights

    # Model metadata
    input_dim: Optional[int] = None       # Observation dimension
    output_dim: Optional[int] = None      # Action dimension
    hidden_dims: Optional[List[int]] = None  # Hidden layer dimensions

    # Training metadata
    trained_episodes: Optional[int] = None
    validation_success_rate: Optional[float] = None

    # Dynamical compatibility
    dynamical_version: str = ">=1.0.0"    # Minimum Dynamical version required
    moai_compression_ratio: Optional[float] = None  # If using MOAI


@dataclass
class CoordinationEncoderConfig:
    """Configuration for the coordination encoder (shared across roles)"""
    checkpoint_uri: str                   # Path to encoder checkpoint
    architecture: PolicyArchitecture      # Encoder architecture (usually transformer)
    latent_dim: int                       # Coordination latent dimension

    # Architecture details
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None       # For transformer
    hidden_dim: Optional[int] = None

    # Input configuration
    max_sequence_length: Optional[int] = None  # Max trajectory length
    input_modalities: List[str] = field(default_factory=lambda: ["state", "action"])


@dataclass
class SafetyEnvelope:
    """Safety constraints for multi-actor skill execution"""
    min_inter_robot_distance: float = 0.3  # Minimum distance between robots (meters)
    max_velocity: float = 1.0              # Maximum velocity (m/s)
    max_acceleration: float = 2.0          # Maximum acceleration (m/s²)
    max_jerk: float = 5.0                  # Maximum jerk (m/s³)

    # Workspace bounds
    workspace_bounds: Optional[Dict[str, List[float]]] = None  # {"x": [min, max], "y": [...], "z": [...]}

    # Collision avoidance
    collision_check_frequency: float = 10.0  # Hz
    emergency_stop_deceleration: float = 5.0  # m/s²


@dataclass
class BehaviorTreeConfig:
    """Behavior tree configuration for skill state machine"""
    tree_structure: Dict[str, Any]        # BehaviorTree.CPP XML or JSON
    success_conditions: List[str]         # Conditions for skill success
    failure_conditions: List[str]         # Conditions for skill failure
    timeout: float = 30.0                 # Maximum execution time (seconds)


@dataclass
class CooperativeSkillArtifact:
    """
    Cooperative Skill Artifact (CSA) - Full specification

    CSAs are produced by SwarmBridge's federated multi-actor IL training and consumed
    by SwarmBrain for orchestration and Dynamical for execution.

    Format Guarantees:
    - All role experts are Dynamical MoE-compatible
    - Coordination encoder can be loaded by Dynamical's runtime
    - Encryption schemes supported by Dynamical's crypto layer
    - URIs accessible by edge robots (S3 pre-signed URLs or MinIO)
    """

    # --- Basic Metadata ---
    skill_id: str                         # Unique skill identifier
    skill_name: str                       # Human-readable name
    description: str                      # Skill description
    version: str = "1.0.0"               # Semantic versioning

    # --- Skill Type ---
    skill_type: SkillType = SkillType.MULTI_ACTOR

    # --- Actor Requirements ---
    required_roles: List[str]             # Required roles (e.g., ["giver", "receiver"])
    required_actors: int                  # Number of actors needed
    optional_roles: List[str] = field(default_factory=list)  # Optional support roles

    # --- Role Experts (Dynamical MoE Compatible) ---
    role_experts: Dict[str, RoleExpertConfig]  # Per-role expert configurations

    # --- Coordination Encoder ---
    coordination_encoder: CoordinationEncoderConfig

    # --- Coordination Metadata ---
    coordination_primitives: List[CoordinationPrimitive]  # Used primitives

    # --- Safety & Execution ---
    safety_envelope: SafetyEnvelope = field(default_factory=SafetyEnvelope)
    behavior_tree: Optional[BehaviorTreeConfig] = None

    # --- Training Metadata ---
    training_sites: Optional[int] = None           # Number of federated sites
    total_demonstrations: Optional[int] = None     # Total multi-actor demos
    federated_rounds: Optional[int] = None         # FL rounds completed
    final_validation_accuracy: Optional[float] = None

    # --- Compatibility ---
    swarmbridge_version: str = ">=1.0.0"          # SwarmBridge version used for training
    dynamical_version: str = ">=1.0.0"            # Minimum Dynamical version
    swarmbrain_version: str = ">=1.1.0"           # Minimum SwarmBrain version

    # --- Timestamps ---
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # --- Additional Metadata ---
    tags: List[str] = field(default_factory=list)  # Searchable tags
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate CSA integrity and compatibility"""
        # Check role experts match required roles
        if set(self.role_experts.keys()) != set(self.required_roles):
            raise ValueError(
                f"Role experts {set(self.role_experts.keys())} must match required roles {set(self.required_roles)}"
            )

        # Check required actors matches roles
        if len(self.required_roles) != self.required_actors:
            raise ValueError(
                f"Required actors ({self.required_actors}) must match number of required roles ({len(self.required_roles)})"
            )

        # Check all role experts have valid URIs
        for role, expert in self.role_experts.items():
            if not expert.checkpoint_uri:
                raise ValueError(f"Role '{role}' expert missing checkpoint_uri")

        # Check coordination encoder has valid URI
        if not self.coordination_encoder.checkpoint_uri:
            raise ValueError("Coordination encoder missing checkpoint_uri")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "description": self.description,
            "version": self.version,
            "skill_type": self.skill_type.value,
            "required_roles": self.required_roles,
            "required_actors": self.required_actors,
            "optional_roles": self.optional_roles,
            "role_experts": {
                role: {
                    "checkpoint_uri": expert.checkpoint_uri,
                    "embedding_type": expert.embedding_type.value,
                    "architecture": expert.architecture.value,
                    "encryption_scheme": expert.encryption_scheme.value,
                    "input_dim": expert.input_dim,
                    "output_dim": expert.output_dim,
                    "hidden_dims": expert.hidden_dims,
                    "trained_episodes": expert.trained_episodes,
                    "validation_success_rate": expert.validation_success_rate,
                    "dynamical_version": expert.dynamical_version,
                    "moai_compression_ratio": expert.moai_compression_ratio,
                }
                for role, expert in self.role_experts.items()
            },
            "coordination_encoder": {
                "checkpoint_uri": self.coordination_encoder.checkpoint_uri,
                "architecture": self.coordination_encoder.architecture.value,
                "latent_dim": self.coordination_encoder.latent_dim,
                "num_layers": self.coordination_encoder.num_layers,
                "num_heads": self.coordination_encoder.num_heads,
                "hidden_dim": self.coordination_encoder.hidden_dim,
                "max_sequence_length": self.coordination_encoder.max_sequence_length,
                "input_modalities": self.coordination_encoder.input_modalities,
            },
            "coordination_primitives": [p.value for p in self.coordination_primitives],
            "safety_envelope": {
                "min_inter_robot_distance": self.safety_envelope.min_inter_robot_distance,
                "max_velocity": self.safety_envelope.max_velocity,
                "max_acceleration": self.safety_envelope.max_acceleration,
                "max_jerk": self.safety_envelope.max_jerk,
                "workspace_bounds": self.safety_envelope.workspace_bounds,
                "collision_check_frequency": self.safety_envelope.collision_check_frequency,
                "emergency_stop_deceleration": self.safety_envelope.emergency_stop_deceleration,
            },
            "behavior_tree": {
                "tree_structure": self.behavior_tree.tree_structure,
                "success_conditions": self.behavior_tree.success_conditions,
                "failure_conditions": self.behavior_tree.failure_conditions,
                "timeout": self.behavior_tree.timeout,
            } if self.behavior_tree else None,
            "training_sites": self.training_sites,
            "total_demonstrations": self.total_demonstrations,
            "federated_rounds": self.federated_rounds,
            "final_validation_accuracy": self.final_validation_accuracy,
            "swarmbridge_version": self.swarmbridge_version,
            "dynamical_version": self.dynamical_version,
            "swarmbrain_version": self.swarmbrain_version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "custom_metadata": self.custom_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CooperativeSkillArtifact":
        """Create CSA from dictionary (JSON deserialization)"""
        # Convert enums
        skill_type = SkillType(data.get("skill_type", "multi_actor"))

        # Parse role experts
        role_experts = {
            role: RoleExpertConfig(
                checkpoint_uri=expert["checkpoint_uri"],
                embedding_type=EmbeddingType(expert["embedding_type"]),
                architecture=PolicyArchitecture(expert["architecture"]),
                encryption_scheme=EncryptionScheme(expert["encryption_scheme"]),
                input_dim=expert.get("input_dim"),
                output_dim=expert.get("output_dim"),
                hidden_dims=expert.get("hidden_dims"),
                trained_episodes=expert.get("trained_episodes"),
                validation_success_rate=expert.get("validation_success_rate"),
                dynamical_version=expert.get("dynamical_version", ">=1.0.0"),
                moai_compression_ratio=expert.get("moai_compression_ratio"),
            )
            for role, expert in data["role_experts"].items()
        }

        # Parse coordination encoder
        encoder_data = data["coordination_encoder"]
        coordination_encoder = CoordinationEncoderConfig(
            checkpoint_uri=encoder_data["checkpoint_uri"],
            architecture=PolicyArchitecture(encoder_data["architecture"]),
            latent_dim=encoder_data["latent_dim"],
            num_layers=encoder_data.get("num_layers"),
            num_heads=encoder_data.get("num_heads"),
            hidden_dim=encoder_data.get("hidden_dim"),
            max_sequence_length=encoder_data.get("max_sequence_length"),
            input_modalities=encoder_data.get("input_modalities", ["state", "action"]),
        )

        # Parse coordination primitives
        coordination_primitives = [
            CoordinationPrimitive(p) for p in data.get("coordination_primitives", [])
        ]

        # Parse safety envelope
        safety_data = data.get("safety_envelope", {})
        safety_envelope = SafetyEnvelope(
            min_inter_robot_distance=safety_data.get("min_inter_robot_distance", 0.3),
            max_velocity=safety_data.get("max_velocity", 1.0),
            max_acceleration=safety_data.get("max_acceleration", 2.0),
            max_jerk=safety_data.get("max_jerk", 5.0),
            workspace_bounds=safety_data.get("workspace_bounds"),
            collision_check_frequency=safety_data.get("collision_check_frequency", 10.0),
            emergency_stop_deceleration=safety_data.get("emergency_stop_deceleration", 5.0),
        )

        # Parse behavior tree
        behavior_tree = None
        if data.get("behavior_tree"):
            bt_data = data["behavior_tree"]
            behavior_tree = BehaviorTreeConfig(
                tree_structure=bt_data["tree_structure"],
                success_conditions=bt_data["success_conditions"],
                failure_conditions=bt_data["failure_conditions"],
                timeout=bt_data.get("timeout", 30.0),
            )

        # Parse timestamps
        created_at = datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow()
        updated_at = datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow()

        return cls(
            skill_id=data["skill_id"],
            skill_name=data["skill_name"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
            skill_type=skill_type,
            required_roles=data["required_roles"],
            required_actors=data["required_actors"],
            optional_roles=data.get("optional_roles", []),
            role_experts=role_experts,
            coordination_encoder=coordination_encoder,
            coordination_primitives=coordination_primitives,
            safety_envelope=safety_envelope,
            behavior_tree=behavior_tree,
            training_sites=data.get("training_sites"),
            total_demonstrations=data.get("total_demonstrations"),
            federated_rounds=data.get("federated_rounds"),
            final_validation_accuracy=data.get("final_validation_accuracy"),
            swarmbridge_version=data.get("swarmbridge_version", ">=1.0.0"),
            dynamical_version=data.get("dynamical_version", ">=1.0.0"),
            swarmbrain_version=data.get("swarmbrain_version", ">=1.1.0"),
            created_at=created_at,
            updated_at=updated_at,
            tags=data.get("tags", []),
            custom_metadata=data.get("custom_metadata", {}),
        )


# Example CSA factory functions

def create_handover_csa_example() -> CooperativeSkillArtifact:
    """Create an example handover CSA"""
    return CooperativeSkillArtifact(
        skill_id="collaborative_handover_v2",
        skill_name="Collaborative Object Handover",
        description="Two-robot object handover with synchronized grasp-release coordination",
        version="2.0.1",
        skill_type=SkillType.MULTI_ACTOR,
        required_roles=["giver", "receiver"],
        required_actors=2,
        role_experts={
            "giver": RoleExpertConfig(
                checkpoint_uri="s3://swarmbridge-skills/handover_v2/giver_expert.pth",
                embedding_type=EmbeddingType.MOAI_COMPRESSED,
                architecture=PolicyArchitecture.DIFFUSION_POLICY,
                encryption_scheme=EncryptionScheme.OPENFHE_BFV,
                input_dim=512,
                output_dim=7,
                hidden_dims=[256, 128],
                trained_episodes=5000,
                validation_success_rate=0.94,
                moai_compression_ratio=0.25,
            ),
            "receiver": RoleExpertConfig(
                checkpoint_uri="s3://swarmbridge-skills/handover_v2/receiver_expert.pth",
                embedding_type=EmbeddingType.MOAI_COMPRESSED,
                architecture=PolicyArchitecture.DIFFUSION_POLICY,
                encryption_scheme=EncryptionScheme.OPENFHE_BFV,
                input_dim=512,
                output_dim=7,
                hidden_dims=[256, 128],
                trained_episodes=5000,
                validation_success_rate=0.92,
                moai_compression_ratio=0.25,
            ),
        },
        coordination_encoder=CoordinationEncoderConfig(
            checkpoint_uri="s3://swarmbridge-skills/handover_v2/coord_encoder.pth",
            architecture=PolicyArchitecture.TRANSFORMER,
            latent_dim=256,
            num_layers=4,
            num_heads=8,
            hidden_dim=512,
            max_sequence_length=100,
        ),
        coordination_primitives=[
            CoordinationPrimitive.HANDOVER,
            CoordinationPrimitive.SYNC_GRASP,
        ],
        safety_envelope=SafetyEnvelope(
            min_inter_robot_distance=0.1,  # Close proximity for handover
            max_velocity=0.5,
            max_acceleration=1.0,
        ),
        training_sites=3,
        total_demonstrations=5000,
        federated_rounds=50,
        final_validation_accuracy=0.93,
        tags=["handover", "manipulation", "collaborative", "industrial"],
    )
