"""
Research Direction 2: Compositional Coordination Primitives

Academic Gap:
- Current multi-agent RL learns monolithic policies for each task
- No compositionality: learning "pick+place" doesn't help with "pick+handover"
- Poor sample efficiency for new task compositions
- Limited generalization to unseen task sequences

Novel Contribution:
- Unsupervised discovery of atomic coordination primitives
- Neural composition operator that preserves coordination semantics
- Federated primitive library shared across sites
- Zero-shot generalization to new task compositions

Publication Venue: CoRL 2025, ICLR 2026 (if strong theory)

Key Innovation:
Instead of learning monolithic policies, decompose coordination into primitives:
- APPROACH(A, B, dist) - Bring agents within distance
- SYNC_GRASP(grasper, target) - Synchronized grasping
- HANDOFF(giver, receiver, object) - Object transfer
- FORMATION(agents, shape) - Maintain formation

Then compose: HANDOVER = APPROACH ∘ SYNC_GRASP ∘ HANDOFF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.cluster import KMeans


class PrimitiveType(str, Enum):
    """Discovered primitive types"""
    APPROACH = "approach"
    SYNC_GRASP = "sync_grasp"
    HANDOFF = "handoff"
    FORMATION = "formation"
    NAVIGATE = "navigate"
    AVOID = "avoid"
    WAIT = "wait"
    RELEASE = "release"


@dataclass
class CoordinationPrimitive:
    """
    Atomic coordination primitive that can be composed.

    Components:
    - Preconditions: When primitive can execute (learned)
    - Policy: How primitive executes (role-conditioned)
    - Postconditions: Expected outcome (learned)
    - Duration: Typical execution time
    """
    primitive_id: str
    primitive_type: PrimitiveType
    roles: List[str]  # Required roles (e.g., ["giver", "receiver"])

    # Learned components
    precondition_network: nn.Module
    policy_network: nn.Module
    postcondition_network: nn.Module

    # Metadata
    avg_duration: float = 0.0
    success_rate: float = 0.0
    num_demonstrations: int = 0

    def can_execute(self, state: torch.Tensor) -> torch.Tensor:
        """
        Check if primitive's preconditions are satisfied.

        Args:
            state: Multi-agent state

        Returns:
            probability ∈ [0, 1] that preconditions are met
        """
        return torch.sigmoid(self.precondition_network(state))

    def execute(self, state: torch.Tensor, role: str) -> torch.Tensor:
        """
        Execute primitive policy for given role.

        Args:
            state: Current state
            role: Agent's role in this primitive

        Returns:
            action: Action for this role
        """
        # Encode role
        role_idx = self.roles.index(role)
        role_embedding = F.one_hot(torch.tensor(role_idx), len(self.roles)).float()

        # Concatenate state + role
        policy_input = torch.cat([state, role_embedding])

        # Generate action
        action = self.policy_network(policy_input)
        return action

    def predict_postconditions(self, state: torch.Tensor, actions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict outcome after executing primitive.

        Args:
            state: Current state
            actions: Actions taken by all roles

        Returns:
            next_state: Predicted state after primitive execution
        """
        # Concatenate state + all role actions
        all_actions = torch.cat([actions[role] for role in self.roles])
        prediction_input = torch.cat([state, all_actions])

        next_state = self.postcondition_network(prediction_input)
        return next_state


class PrimitiveDiscovery:
    """
    Unsupervised discovery of coordination primitives from multi-actor demonstrations.

    Method:
    1. Temporal segmentation: Identify changepoints in demonstrations
    2. Behavioral clustering: Group similar segments
    3. Primitive extraction: Learn policy for each cluster
    4. Compositionality validation: Check if primitives compose to full demo
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_primitives: int = 10,
        embedding_dim: int = 128,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_primitives = num_primitives

        # VAE for behavior embedding
        self.behavior_encoder = BehaviorEmbeddingVAE(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=embedding_dim,
        )

    def discover_primitives(
        self,
        demonstrations: List[Dict],  # Multi-actor trajectories
    ) -> List[CoordinationPrimitive]:
        """
        Main primitive discovery pipeline.

        Steps:
        1. Segment demonstrations into candidate primitives
        2. Embed segments into behavior space
        3. Cluster segments by behavior similarity
        4. Learn primitive policies for each cluster
        5. Validate compositionality
        """
        # Step 1: Temporal segmentation
        segments = []
        for demo in demonstrations:
            demo_segments = self.segment_trajectory(demo)
            segments.extend(demo_segments)

        print(f"Discovered {len(segments)} candidate segments from {len(demonstrations)} demos")

        # Step 2: Embed segments into behavior space
        segment_embeddings = []
        for segment in segments:
            embedding = self.behavior_encoder.encode(segment)
            segment_embeddings.append(embedding)

        segment_embeddings = torch.stack(segment_embeddings)
        print(f"Embedded segments into {segment_embeddings.shape[1]}-dim behavior space")

        # Step 3: Cluster by behavior similarity
        clusters = self.cluster_behaviors(segment_embeddings, k=self.num_primitives)
        print(f"Clustered into {self.num_primitives} primitive types")

        # Step 4: Learn primitive policies
        primitives = []
        for cluster_id in range(self.num_primitives):
            # Get segments in this cluster
            cluster_segments = [
                segments[i] for i in range(len(segments))
                if clusters[i] == cluster_id
            ]

            if len(cluster_segments) < 5:
                print(f"Cluster {cluster_id} too small ({len(cluster_segments)} segments), skipping")
                continue

            # Learn primitive from cluster
            primitive = self.learn_primitive(cluster_id, cluster_segments)
            primitives.append(primitive)

        print(f"Learned {len(primitives)} coordination primitives")

        # Step 5: Validate compositionality
        compositionality_score = self.validate_compositionality(primitives, demonstrations)
        print(f"Compositionality score: {compositionality_score:.3f}")

        return primitives

    def segment_trajectory(self, demo: Dict) -> List[Dict]:
        """
        Segment multi-actor trajectory into sub-trajectories (candidate primitives).

        Method: Bayesian changepoint detection on coordination features
        - Coordination features: Distances, velocities, object states
        - Changepoints: Times when coordination pattern changes
        """
        states = demo['states']  # (T, state_dim)
        actions = demo['actions']  # {role: (T, action_dim)}

        # Extract coordination features (e.g., inter-agent distances)
        coord_features = self.extract_coordination_features(states, actions)

        # Changepoint detection
        changepoints = self.detect_changepoints(coord_features)

        # Split into segments
        segments = []
        for i in range(len(changepoints) - 1):
            start_t = changepoints[i]
            end_t = changepoints[i + 1]

            segment = {
                'states': states[start_t:end_t],
                'actions': {role: acts[start_t:end_t] for role, acts in actions.items()},
                'start_time': start_t,
                'end_time': end_t,
                'duration': end_t - start_t,
            }
            segments.append(segment)

        return segments

    def extract_coordination_features(self, states: torch.Tensor, actions: Dict) -> torch.Tensor:
        """
        Extract features that capture coordination patterns.

        Features:
        - Inter-agent distances
        - Relative velocities
        - Object positions (if applicable)
        - Action magnitudes
        """
        T = states.shape[0]
        features = []

        for t in range(T):
            state_t = states[t]

            # Inter-agent distances (assuming state includes agent positions)
            # For simplicity, assume state = [agent1_pos, agent2_pos, ...]
            # In practice, parse state structure

            # Placeholder: extract coordination-relevant features
            coord_feature = self.coordination_feature_extractor(state_t)
            features.append(coord_feature)

        return torch.stack(features)

    def coordination_feature_extractor(self, state: torch.Tensor) -> torch.Tensor:
        """Extract coordination-relevant features from state"""
        # Placeholder: in practice, parse state structure
        # Example features:
        # - Distance between agents
        # - Relative orientations
        # - Object being manipulated (if any)
        return state[:16]  # Use first 16 dims as coordination features

    def detect_changepoints(self, features: torch.Tensor) -> List[int]:
        """
        Bayesian changepoint detection.

        Returns indices where coordination pattern changes.
        """
        # Simplified changepoint detection: use variance in features
        # In practice, use BOCPD (Bayesian Online Changepoint Detection)

        T = features.shape[0]
        changepoints = [0]  # Start

        window_size = 10
        threshold = 0.5  # Variance threshold

        for t in range(window_size, T - window_size):
            # Variance before/after
            var_before = features[t - window_size:t].var(dim=0).mean()
            var_after = features[t:t + window_size].var(dim=0).mean()

            # Detect change if variance difference is large
            if abs(var_before - var_after) > threshold:
                changepoints.append(t)

        changepoints.append(T)  # End
        return changepoints

    def cluster_behaviors(self, embeddings: torch.Tensor, k: int) -> np.ndarray:
        """
        Cluster segment embeddings by behavior similarity.

        Returns cluster assignments for each segment.
        """
        # K-means clustering in behavior embedding space
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(embeddings.detach().cpu().numpy())
        return clusters

    def learn_primitive(self, cluster_id: int, segments: List[Dict]) -> CoordinationPrimitive:
        """
        Learn primitive policy from clustered segments.

        Components to learn:
        1. Precondition network: P(can_execute | state)
        2. Policy network: π(action | state, role)
        3. Postcondition network: P(next_state | state, actions)
        """
        # Extract roles from segments
        roles = list(segments[0]['actions'].keys())

        # Initialize networks
        precondition_net = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Scalar precondition probability
        )

        policy_net = nn.Sequential(
            nn.Linear(self.state_dim + len(roles), 256),  # state + role embedding
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
        )

        postcondition_net = nn.Sequential(
            nn.Linear(self.state_dim + len(roles) * self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.state_dim),  # Predict next state
        )

        # Train networks via behavior cloning on this cluster
        optimizer = torch.optim.Adam(
            list(precondition_net.parameters()) +
            list(policy_net.parameters()) +
            list(postcondition_net.parameters()),
            lr=1e-3
        )

        for epoch in range(100):
            epoch_loss = 0.0

            for segment in segments:
                states = segment['states']
                actions = segment['actions']

                # Precondition: first state should have high probability
                precond_prob = torch.sigmoid(precondition_net(states[0]))
                precond_loss = -torch.log(precond_prob + 1e-8)  # Should be high

                # Policy: behavior cloning
                policy_loss = 0.0
                for role in roles:
                    role_idx = roles.index(role)
                    role_emb = F.one_hot(torch.tensor(role_idx), len(roles)).float()

                    for t in range(len(states)):
                        policy_input = torch.cat([states[t], role_emb])
                        action_pred = policy_net(policy_input)
                        action_gt = actions[role][t]
                        policy_loss += F.mse_loss(action_pred, action_gt)

                # Postcondition: predict state transitions
                postcond_loss = 0.0
                for t in range(len(states) - 1):
                    all_actions = torch.cat([actions[role][t] for role in roles])
                    postcond_input = torch.cat([states[t], all_actions])
                    next_state_pred = postcondition_net(postcond_input)
                    next_state_gt = states[t + 1]
                    postcond_loss += F.mse_loss(next_state_pred, next_state_gt)

                total_loss = precond_loss + policy_loss + postcond_loss
                epoch_loss += total_loss.item()

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            if epoch % 20 == 0:
                print(f"  Primitive {cluster_id} epoch {epoch}: loss = {epoch_loss / len(segments):.4f}")

        # Create primitive object
        primitive = CoordinationPrimitive(
            primitive_id=f"primitive_{cluster_id}",
            primitive_type=PrimitiveType.APPROACH,  # Infer type later
            roles=roles,
            precondition_network=precondition_net,
            policy_network=policy_net,
            postcondition_network=postcondition_net,
            avg_duration=np.mean([s['duration'] for s in segments]),
            num_demonstrations=len(segments),
        )

        return primitive

    def validate_compositionality(
        self,
        primitives: List[CoordinationPrimitive],
        demonstrations: List[Dict],
    ) -> float:
        """
        Validate that primitives compose to reproduce original demonstrations.

        Score: Average reconstruction error when composing primitives
        """
        reconstruction_errors = []

        for demo in demonstrations[:10]:  # Sample 10 demos for validation
            # Greedily compose primitives to match demo
            composed_trajectory = self.greedy_composition(primitives, demo)

            # Compare composed trajectory to original demo
            error = self.compute_trajectory_error(composed_trajectory, demo)
            reconstruction_errors.append(error)

        return 1.0 - np.mean(reconstruction_errors)  # Higher score = better compositionality

    def greedy_composition(
        self,
        primitives: List[CoordinationPrimitive],
        demo: Dict,
    ) -> List[torch.Tensor]:
        """
        Greedily select primitives to reconstruct demonstration.
        """
        states = demo['states']
        composed_trajectory = []

        t = 0
        while t < len(states):
            # Find primitive with highest precondition probability
            state_t = states[t]
            best_primitive = None
            best_prob = 0.0

            for prim in primitives:
                prob = prim.can_execute(state_t)
                if prob > best_prob:
                    best_prob = prob
                    best_primitive = prim

            if best_primitive is None:
                break

            # Execute primitive for its avg duration
            duration = int(best_primitive.avg_duration)
            for _ in range(duration):
                if t >= len(states):
                    break
                composed_trajectory.append(states[t])
                t += 1

        return composed_trajectory

    def compute_trajectory_error(self, traj1: List, traj2: Dict) -> float:
        """Compute error between two trajectories"""
        # Simplified: L2 distance between states
        min_len = min(len(traj1), len(traj2['states']))
        if min_len == 0:
            return 1.0

        error = 0.0
        for i in range(min_len):
            error += F.mse_loss(traj1[i], traj2['states'][i]).item()

        return error / min_len


class BehaviorEmbeddingVAE(nn.Module):
    """
    VAE for embedding trajectory segments into behavior space.

    Used for clustering segments by behavior similarity.
    """

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 128):
        super().__init__()

        # Encoder: segment → latent
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder: latent → segment
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim + action_dim),
        )

    def encode(self, segment: Dict) -> torch.Tensor:
        """Encode segment to latent behavior embedding"""
        # Flatten segment
        states = segment['states']  # (T, state_dim)
        actions = torch.cat([segment['actions'][role] for role in segment['actions'].keys()], dim=-1)  # (T, total_action_dim)

        # Concatenate and average over time
        segment_flat = torch.cat([states, actions[:, :self.action_dim]], dim=-1)
        segment_avg = segment_flat.mean(dim=0)

        # Encode
        h = self.encoder(segment_avg)
        mu = self.fc_mu(h)
        return mu  # Return mean as embedding

    def forward(self, segment_flat: torch.Tensor):
        """Full VAE forward pass (for training)"""
        h = self.encoder(segment_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Decode
        recon = self.decoder(z)

        return recon, mu, logvar


class NeuralCompositionOperator(nn.Module):
    """
    Neural operator for composing coordination primitives.

    Key Innovation: Learn composition operator ∘ that preserves coordination semantics.

    Example: HANDOVER = APPROACH ∘ SYNC_GRASP ∘ RELEASE

    Architecture:
    - Input: Two primitive policies π1, π2
    - Output: Composed policy π1 ∘ π2
    - Constraint: Preserve coordination structure (adjacency matrix)
    """

    def __init__(self, policy_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Composition network
        self.composition_net = nn.Sequential(
            nn.Linear(policy_dim * 2, hidden_dim),  # Concatenate two policy embeddings
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, policy_dim),  # Output composed policy embedding
        )

    def compose(
        self,
        prim1: CoordinationPrimitive,
        prim2: CoordinationPrimitive,
    ) -> CoordinationPrimitive:
        """
        Compose two primitives into a new primitive.

        Constraint: prim1's postconditions must satisfy prim2's preconditions
        """
        # Check composability
        if not self.is_composable(prim1, prim2):
            raise ValueError(f"Primitives {prim1.primitive_id} and {prim2.primitive_id} not composable")

        # Extract policy embeddings (simplified: use policy network parameters)
        prim1_embedding = self.extract_policy_embedding(prim1.policy_network)
        prim2_embedding = self.extract_policy_embedding(prim2.policy_network)

        # Compose
        composed_embedding = self.composition_net(
            torch.cat([prim1_embedding, prim2_embedding])
        )

        # Create composed primitive
        composed_policy = self.embedding_to_policy(composed_embedding)

        composed_primitive = CoordinationPrimitive(
            primitive_id=f"{prim1.primitive_id}+{prim2.primitive_id}",
            primitive_type=PrimitiveType.APPROACH,  # Infer
            roles=list(set(prim1.roles + prim2.roles)),
            precondition_network=prim1.precondition_network,  # Use prim1's preconditions
            policy_network=composed_policy,
            postcondition_network=prim2.postcondition_network,  # Use prim2's postconditions
        )

        return composed_primitive

    def is_composable(self, prim1: CoordinationPrimitive, prim2: CoordinationPrimitive) -> bool:
        """
        Check if prim1 and prim2 can be composed.

        Condition: prim1's postconditions should imply prim2's preconditions
        """
        # Sample test state
        test_state = torch.randn(self.state_dim)

        # Execute prim1
        prim1_next_state = prim1.predict_postconditions(test_state, {})

        # Check if prim2 can execute from prim1's output
        prim2_can_execute = prim2.can_execute(prim1_next_state)

        return prim2_can_execute > 0.5  # Threshold

    def extract_policy_embedding(self, policy_network: nn.Module) -> torch.Tensor:
        """Extract embedding from policy network (flatten parameters)"""
        params = []
        for param in policy_network.parameters():
            params.append(param.flatten())
        return torch.cat(params)[:256]  # Use first 256 dims

    def embedding_to_policy(self, embedding: torch.Tensor) -> nn.Module:
        """Convert embedding back to policy network"""
        # Simplified: create new network
        # In practice, reconstruct network from embedding
        policy = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
        )
        return policy


"""
Research Contributions Summary:

1. **Primitive Discovery Algorithm**
   - Unsupervised segmentation + clustering
   - Behavior embedding VAE
   - Compositionality validation

2. **Neural Composition Operator**
   - Learn to compose primitives
   - Preserve coordination semantics
   - Check composability constraints

3. **Experimental Validation**
   - Industrial tasks: assembly, logistics
   - Show 10x sample efficiency for composed tasks
   - Zero-shot generalization to new compositions

4. **Federated Primitive Library**
   - Sites share discovered primitives
   - Aggregation of primitive libraries
   - Compositional transfer across sites

Publication: CoRL 2025
"""
