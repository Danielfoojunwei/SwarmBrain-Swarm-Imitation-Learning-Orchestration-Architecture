"""
Research Direction 1: Heterogeneous Cross-Embodiment Coordination

Academic Gap:
- Current multi-agent systems assume homogeneous agents (same morphology, sensors, actuators)
- Real industrial settings have heterogeneous robot fleets (arms, mobile bases, quadrupeds, drones)
- No existing work on federated learning of coordination skills across different embodiments

Novel Contribution:
- Morphology-invariant coordination encoder
- Cross-embodiment skill transfer via disentangled representations
- Federated aggregation that preserves cross-embodiment generalization

Publication Venues: CoRL, RSS, ICRA, NeurIPS (if strong theory)
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class RobotMorphology:
    """Robot morphology specification"""
    robot_type: str  # "manipulator", "mobile_base", "quadruped", "drone"
    dof: int  # Degrees of freedom
    action_space_dim: int
    observation_space_dim: int

    # Capability embeddings
    manipulation_capability: float  # 0-1
    mobility_capability: float      # 0-1
    sensing_capability: float       # 0-1

    # Physical constraints
    max_reach: Optional[float] = None
    max_payload: Optional[float] = None
    max_velocity: Optional[float] = None


class MorphologyInvariantCoordinationEncoder(nn.Module):
    """
    Cross-embodiment coordination encoder that learns morphology-invariant
    coordination representations.

    Key Innovation:
    - Disentangles coordination intent from morphology-specific execution
    - Enables coordination transfer across different robot types
    - Uses adversarial training to enforce morphology invariance

    Architecture:
    obs_1, ..., obs_N → Morphology Encoders → Coordination Encoder → z_coord
                                            ↓
                                    Morphology Discriminator (adversarial)
    """

    def __init__(
        self,
        morphology_types: List[str],
        coordination_latent_dim: int = 256,
        morphology_embedding_dim: int = 64,
    ):
        super().__init__()

        # Per-morphology observation encoders
        self.morphology_encoders = nn.ModuleDict({
            morph_type: nn.Sequential(
                nn.Linear(512, 256),  # Assume max obs dim
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, morphology_embedding_dim),
            )
            for morph_type in morphology_types
        })

        # Coordination encoder (morphology-invariant)
        self.coordination_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=morphology_embedding_dim,
                nhead=8,
                dim_feedforward=256,
            ),
            num_layers=4,
        )

        # Project to coordination latent
        self.coord_projection = nn.Linear(morphology_embedding_dim, coordination_latent_dim)

        # Morphology discriminator (for adversarial training)
        self.morphology_discriminator = nn.Sequential(
            nn.Linear(coordination_latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(morphology_types)),
        )

    def forward(
        self,
        observations: Dict[str, torch.Tensor],  # {agent_id: obs}
        morphologies: Dict[str, str],           # {agent_id: morphology_type}
        return_discriminator_logits: bool = False,
    ):
        """
        Encode multi-agent observations into morphology-invariant coordination latent.

        Args:
            observations: Per-agent observations
            morphologies: Per-agent morphology types
            return_discriminator_logits: If True, return discriminator predictions

        Returns:
            z_coord: Coordination latent (morphology-invariant)
            discriminator_logits: (optional) Morphology predictions (for adversarial loss)
        """
        # Encode each agent's observation with its morphology-specific encoder
        agent_embeddings = []
        for agent_id, obs in observations.items():
            morph_type = morphologies[agent_id]
            embedding = self.morphology_encoders[morph_type](obs)
            agent_embeddings.append(embedding)

        # Stack embeddings (batch, num_agents, embedding_dim)
        agent_embeddings = torch.stack(agent_embeddings, dim=1)

        # Coordination encoder (attention over agents)
        coord_features = self.coordination_encoder(agent_embeddings.transpose(0, 1))

        # Pool over agents (mean pooling)
        coord_pooled = coord_features.mean(dim=0)

        # Project to coordination latent
        z_coord = self.coord_projection(coord_pooled)

        if return_discriminator_logits:
            # Discriminator tries to predict morphology composition (for adversarial loss)
            discriminator_logits = self.morphology_discriminator(z_coord)
            return z_coord, discriminator_logits

        return z_coord


class CrossEmbodimentRolePolicy(nn.Module):
    """
    Role-conditioned policy that adapts to robot morphology.

    Key Innovation:
    - Takes morphology-invariant coordination latent + morphology embedding
    - Outputs morphology-specific actions
    - Enables same role to be executed by different robot types

    Example: "grasper" role can be filled by:
    - Robotic arm with gripper
    - Mobile manipulator
    - Humanoid hand
    """

    def __init__(
        self,
        role_name: str,
        coordination_latent_dim: int = 256,
        morphology_embedding_dim: int = 64,
        max_action_dim: int = 32,  # Max DoF across all morphologies
    ):
        super().__init__()
        self.role_name = role_name

        # Morphology encoder (encodes robot capabilities)
        self.morphology_encoder = nn.Sequential(
            nn.Linear(10, morphology_embedding_dim),  # [type, dof, capabilities...]
            nn.ReLU(),
        )

        # Policy network (coordination + morphology → action)
        self.policy = nn.Sequential(
            nn.Linear(coordination_latent_dim + morphology_embedding_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, max_action_dim),  # Outputs for all possible DoFs
        )

    def forward(
        self,
        z_coord: torch.Tensor,
        morphology_features: torch.Tensor,
        action_mask: torch.Tensor,  # Mask for valid action dims
    ):
        """
        Generate morphology-specific action given coordination latent.

        Args:
            z_coord: Coordination latent (morphology-invariant)
            morphology_features: Robot morphology encoding
            action_mask: Binary mask for valid action dimensions

        Returns:
            action: Morphology-specific action
        """
        # Encode morphology
        morph_embedding = self.morphology_encoder(morphology_features)

        # Concatenate coordination + morphology
        combined = torch.cat([z_coord, morph_embedding], dim=-1)

        # Generate action
        action_logits = self.policy(combined)

        # Mask invalid action dimensions (padding for smaller DoF robots)
        action = action_logits * action_mask

        return action


class CrossEmbodimentFederatedAggregation:
    """
    Federated aggregation strategy for cross-embodiment coordination.

    Key Innovation:
    - Aggregates updates from heterogeneous robot fleets
    - Preserves morphology-invariant coordination while allowing
      morphology-specific adaptations
    - Uses adversarial loss to enforce invariance

    Research Question:
    How to aggregate gradients from robots with different action/observation spaces
    while maintaining coordination quality?

    Solution:
    - Separate aggregation for coordination encoder (shared) and role policies (morphology-specific)
    - Weight updates by morphology diversity (encourage cross-embodiment learning)
    """

    def __init__(self, morphology_types: List[str]):
        self.morphology_types = morphology_types

    def aggregate_coordination_encoders(
        self,
        local_updates: Dict[str, Dict],  # {site_id: {param_name: tensor}}
        site_morphologies: Dict[str, List[str]],  # {site_id: [morphology_types]}
    ) -> Dict:
        """
        Aggregate coordination encoder updates with morphology diversity weighting.

        Innovation: Sites with more diverse morphologies get higher weight to
        encourage cross-embodiment generalization.
        """
        # Calculate morphology diversity scores
        diversity_scores = {}
        for site_id, morphs in site_morphologies.items():
            # Diversity = number of unique morphologies
            diversity_scores[site_id] = len(set(morphs)) / len(self.morphology_types)

        # Normalize weights
        total_diversity = sum(diversity_scores.values())
        weights = {
            site_id: score / total_diversity
            for site_id, score in diversity_scores.items()
        }

        # Weighted aggregation
        aggregated_params = {}
        param_names = list(local_updates[list(local_updates.keys())[0]].keys())

        for param_name in param_names:
            weighted_sum = sum(
                weights[site_id] * local_updates[site_id][param_name]
                for site_id in local_updates.keys()
            )
            aggregated_params[param_name] = weighted_sum

        return aggregated_params

    def aggregate_role_policies(
        self,
        local_updates: Dict[str, Dict],  # {site_id: updates}
        role_morphology_pairs: Dict[str, List[tuple]],  # {site_id: [(role, morphology)]}
    ) -> Dict:
        """
        Aggregate role policy updates per (role, morphology) pair.

        Innovation: Separate aggregation streams for each role-morphology combination
        while sharing coordination understanding.
        """
        # Group updates by (role, morphology)
        role_morph_updates = {}

        for site_id, updates in local_updates.items():
            for role, morphology in role_morphology_pairs[site_id]:
                key = (role, morphology)
                if key not in role_morph_updates:
                    role_morph_updates[key] = []
                role_morph_updates[key].append(updates)

        # Aggregate each (role, morphology) stream
        aggregated = {}
        for (role, morphology), update_list in role_morph_updates.items():
            # FedAvg for this specific (role, morphology)
            aggregated[(role, morphology)] = self._fedavg(update_list)

        return aggregated

    def _fedavg(self, updates: List[Dict]) -> Dict:
        """Standard FedAvg aggregation"""
        if not updates:
            return {}

        param_names = list(updates[0].keys())
        aggregated = {}

        for param_name in param_names:
            aggregated[param_name] = sum(u[param_name] for u in updates) / len(updates)

        return aggregated


# Example training loop
def train_cross_embodiment_coordination(
    coordination_encoder: MorphologyInvariantCoordinationEncoder,
    role_policies: Dict[str, CrossEmbodimentRolePolicy],
    demonstrations: List[Dict],  # Multi-actor demos with morphology info
    adversarial_weight: float = 0.1,
):
    """
    Training loop with adversarial morphology invariance.

    Loss = L_IL (imitation) + L_consistency (cross-role) + λ_adv * L_adversarial

    L_adversarial = -H(discriminator(z_coord)) → maximize entropy
                  = minimize discriminator's ability to predict morphology
    """
    optimizer_encoder = torch.optim.Adam(coordination_encoder.parameters(), lr=1e-4)
    optimizer_policies = torch.optim.Adam(
        [p for policy in role_policies.values() for p in policy.parameters()],
        lr=1e-4
    )
    optimizer_discriminator = torch.optim.Adam(
        coordination_encoder.morphology_discriminator.parameters(),
        lr=1e-4
    )

    for epoch in range(num_epochs):
        for batch in demonstrations:
            # Forward pass
            observations = batch['observations']  # {agent_id: obs}
            morphologies = batch['morphologies']  # {agent_id: morphology_type}
            actions_gt = batch['actions']         # {agent_id: action}

            # Encode coordination (morphology-invariant)
            z_coord, discriminator_logits = coordination_encoder(
                observations, morphologies, return_discriminator_logits=True
            )

            # Generate actions per role
            actions_pred = {}
            for agent_id, role in batch['roles'].items():
                morph_features = batch['morphology_features'][agent_id]
                action_mask = batch['action_masks'][agent_id]

                actions_pred[agent_id] = role_policies[role](
                    z_coord, morph_features, action_mask
                )

            # Loss 1: Imitation loss
            loss_il = sum(
                F.mse_loss(actions_pred[aid], actions_gt[aid])
                for aid in actions_pred.keys()
            )

            # Loss 2: Cross-role consistency (coordinated actions should be consistent)
            loss_consistency = compute_consistency_loss(actions_pred, batch['coordination_constraints'])

            # Loss 3: Adversarial morphology invariance
            # Discriminator tries to predict morphology composition
            # Encoder tries to fool discriminator (maximize entropy)
            morphology_labels = encode_morphology_composition(morphologies)

            # Train discriminator (minimize classification loss)
            loss_discriminator = F.cross_entropy(discriminator_logits, morphology_labels)
            optimizer_discriminator.zero_grad()
            loss_discriminator.backward(retain_graph=True)
            optimizer_discriminator.step()

            # Train encoder to fool discriminator (maximize entropy)
            discriminator_probs = F.softmax(discriminator_logits, dim=-1)
            loss_adversarial = -torch.sum(discriminator_probs * torch.log(discriminator_probs + 1e-8))

            # Total loss
            loss_total = loss_il + 0.5 * loss_consistency - adversarial_weight * loss_adversarial

            optimizer_encoder.zero_grad()
            optimizer_policies.zero_grad()
            loss_total.backward()
            optimizer_encoder.step()
            optimizer_policies.step()


"""
Research Contributions:

1. Novel Architecture:
   - First morphology-invariant coordination encoder
   - Adversarial training for cross-embodiment invariance
   - Disentangled coordination latent

2. Federated Learning Innovation:
   - Diversity-weighted aggregation (sites with more morphology diversity get higher weight)
   - Separate aggregation streams for coordination (shared) and execution (morphology-specific)

3. Theoretical Analysis:
   - Prove that morphology-invariant latent preserves coordination quality
   - Bound on cross-embodiment transfer error
   - Sample complexity analysis for heterogeneous fleets

4. Experimental Validation:
   - Test on heterogeneous robot fleet (e.g., Franka Panda + UR5 + mobile manipulator)
   - Show zero-shot transfer to new morphologies
   - Ablation: with/without adversarial loss, diversity weighting

Publication Potential:
- CoRL (Conference on Robot Learning) - Top venue for robot learning
- RSS (Robotics: Science and Systems) - Top robotics venue
- ICRA (International Conference on Robotics and Automation)
- If strong theory: NeurIPS, ICLR

Unique Selling Points:
- First work on cross-embodiment federated coordination
- Combines adversarial training + federated learning + multi-agent coordination
- Practical industrial application (heterogeneous warehouse robots, construction sites)
"""
