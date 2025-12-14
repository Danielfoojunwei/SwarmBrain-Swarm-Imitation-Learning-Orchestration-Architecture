"""
End-to-End IL Pipeline Demo

This script demonstrates the complete imitation learning pipeline on the edge:

    HumanState → Retargeter → Episode → Chunks → Encoder → Embeddings

After this, embeddings would be encrypted (N2HE) and streamed to MOAI cloud.
That step is not shown here but follows trivially.

The demo:
1. Generates synthetic human demonstrations
2. Retargets them to a mock robot
3. Records episodes
4. Chunks episodes into fixed-horizon sequences
5. Encodes chunks using the transformer encoder
6. Reports statistics and timing

Run:
    cd /home/claude/dynamical_edge
    python examples/il_pipeline_demo.py
"""

import numpy as np
import time
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.human_state import Human3DState, DexterHandState, HumanState, EnvObject, fuse_to_human_state
from core.recorder import RobotObs, RobotAction, DemoStep, Episode
from core.retargeting import Retargeter, RetargetConfig, GripperMapping

# Import IL modules
from il.chunking import (
    DemoChunk,
    chunk_episode,
    chunk_episode_lazy,
    flatten_obs,
    flatten_act,
    collate_chunks,
    normalize_chunks,
)
from il.encoder import (
    EncoderConfig,
    ChunkEncoder,
    create_encoder,
    encode_chunks_gpu,
    encode_single_chunk,
)

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available, encoder tests will be skipped")


# ---------------------------------------------------------------------------
# Mock IK Solver (for testing without a real robot)
# ---------------------------------------------------------------------------

class MockIKSolver:
    """
    Simple mock IK solver that returns plausible joint configurations.
    
    For testing the pipeline without a real robot URDF.
    """
    
    def __init__(self, n_joints: int = 7):
        self.n_joints = n_joints
        self._joint_limits_lower = np.full(n_joints, -2.0)
        self._joint_limits_upper = np.full(n_joints, 2.0)
    
    def solve(
        self,
        q_init: np.ndarray,
        T_ee_target: np.ndarray,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> Tuple[np.ndarray, bool, float]:
        """Return a solution that's close to the initial config."""
        # Simulate IK by adding small perturbation to initial config
        q_solution = q_init + np.random.randn(self.n_joints) * 0.1
        q_solution = np.clip(q_solution, self._joint_limits_lower, self._joint_limits_upper)
        
        # Simulate occasional IK failure
        success = np.random.random() > 0.02  # 2% failure rate
        error = np.random.random() * 0.01  # Small random error
        
        return q_solution, success, error
    
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Return a mock EE pose based on joint config."""
        T = np.eye(4)
        # Mock: EE position varies with first 3 joints
        T[0, 3] = 0.3 + 0.1 * np.sin(q[0])
        T[1, 3] = 0.1 * np.sin(q[1])
        T[2, 3] = 0.5 + 0.1 * np.cos(q[2])
        return T
    
    @property
    def joint_limits_lower(self) -> np.ndarray:
        return self._joint_limits_lower
    
    @property
    def joint_limits_upper(self) -> np.ndarray:
        return self._joint_limits_upper


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------

def generate_synthetic_trajectory(
    n_steps: int = 100,
    task: str = 'reach'
) -> List[HumanState]:
    """
    Generate a synthetic human demonstration trajectory.
    
    Creates a sequence of HumanState objects simulating a human
    reaching toward and grasping an object.
    """
    states = []
    
    # Object position (fixed)
    object_pos = np.array([0.5, 0.1, 0.3])
    object_pose = np.eye(4)
    object_pose[:3, 3] = object_pos
    
    target_object = EnvObject(
        object_id='target_cup',
        class_id=1,
        class_name='cup',
        pose_world=object_pose,
    )
    
    # Starting hand position
    start_pos = np.array([0.3, 0.3, 0.4])
    
    for i in range(n_steps):
        t = i / n_steps
        timestamp = time.time() + i * 0.05  # 20 Hz
        
        # Smooth trajectory (minimum jerk profile)
        s = 10 * t**3 - 15 * t**4 + 6 * t**5  # Smooth 0→1
        
        # Hand position: interpolate from start to object
        hand_pos = start_pos + s * (object_pos - start_pos)
        
        # Grasp closure: starts opening, closes near object
        grasp_phase = max(0, (t - 0.6) / 0.3)  # Grasp in last 30%
        
        # Create body keypoints (17 joints)
        keypoints = np.zeros((17, 3))
        
        # Set wrist positions (indices 13 and 16 for left/right)
        keypoints[16] = hand_pos  # right wrist
        keypoints[13] = hand_pos + np.array([-0.4, 0, 0])  # left wrist (offset)
        
        # Set other body parts (rough approximation)
        keypoints[0] = np.array([0, 0, 1.0])  # pelvis
        keypoints[14] = hand_pos + np.array([0.1, -0.2, 0.2])  # right shoulder
        keypoints[15] = (keypoints[14] + hand_pos) / 2  # right elbow
        
        body_state = Human3DState(
            timestamp=timestamp,
            keypoints_3d=keypoints,
            keypoint_confidence=np.ones(17) * 0.9,
        )
        
        # Create hand state with grasp
        finger_angles = np.ones((5, 3)) * grasp_phase * 1.2  # Close as grasp_phase increases
        
        hand_state = DexterHandState(
            timestamp=timestamp,
            side='right',
            finger_angles=finger_angles,
            finger_abduction=np.zeros(4),
            wrist_quat_local=np.array([0, 0, 0, 1]),  # Identity
        )
        
        # Fuse into HumanState
        human_state = fuse_to_human_state(
            body_state,
            hand_right=hand_state,
            hand_left=None,
            objects=[target_object],
            primary_object_id='target_cup',
        )
        
        states.append(human_state)
    
    return states


# ---------------------------------------------------------------------------
# Main Demo Functions
# ---------------------------------------------------------------------------

def demo_retargeting_and_recording():
    """
    Demonstrate retargeting human states to robot actions and recording episodes.
    """
    print("\n" + "="*70)
    print("PHASE 1: Retargeting and Recording")
    print("="*70)
    
    # Create mock IK solver and retargeter
    ik_solver = MockIKSolver(n_joints=7)
    config = RetargetConfig(
        max_joint_velocity=2.0,
        gripper_mapping=GripperMapping(strategy='weighted'),
    )
    retargeter = Retargeter(ik_solver, config)
    
    # Generate synthetic demonstration
    print("\nGenerating synthetic human demonstration...")
    human_states = generate_synthetic_trajectory(n_steps=100)
    print(f"  Generated {len(human_states)} timesteps")
    
    # Create episode
    episode = Episode(
        episode_id='demo_ep_001',
        task_id='reach_and_grasp',
        env_id='mock_environment',
        robot_type='mock_7dof',
    )
    
    # Initialize robot state
    q_current = np.zeros(7)
    prev_q = q_current.copy()
    
    print("\nRetargeting human demonstration to robot...")
    start_time = time.time()
    
    for i, human_state in enumerate(human_states):
        # Create current robot observation
        robot_obs = RobotObs(
            timestamp=human_state.timestamp,
            joint_positions=q_current,
            joint_velocities=np.zeros(7),
            gripper_position=0.0,
        )
        
        # Retarget human state to robot action
        step = retargeter.human_to_robot_action(
            human=human_state,
            robot_obs=robot_obs,
            prev_q=prev_q,
            episode_id=episode.episode_id,
            t_index=i,
            task_id=episode.task_id,
            env_id=episode.env_id,
        )
        
        # Add step to episode
        episode.append_step(step)
        
        # Update state for next iteration
        q_current = step.action.joint_position_target.copy()
        prev_q = q_current.copy()
    
    elapsed = time.time() - start_time
    
    print(f"  Retargeting complete in {elapsed:.3f}s ({len(episode)/elapsed:.1f} steps/sec)")
    print(f"  Episode length: {len(episode)} steps")
    print(f"  Valid steps: {episode.n_valid_steps}")
    print(f"  IK success rate: {retargeter.ik_success_rate*100:.1f}%")
    print(f"  Duration: {episode.duration:.2f}s")
    
    # Save episode
    output_path = Path('/home/claude/dynamical_edge/demo_il_episode.npz')
    episode.save(output_path)
    print(f"\n  Saved episode to {output_path}")
    
    return episode


def demo_chunking(episode: Episode):
    """
    Demonstrate chunking an episode into fixed-horizon sequences.
    """
    print("\n" + "="*70)
    print("PHASE 2: Episode Chunking")
    print("="*70)
    
    H = 20  # Chunk horizon
    overlap = 5  # Overlap between chunks
    
    print(f"\nChunking with H={H}, overlap={overlap}...")
    start_time = time.time()
    
    chunks = chunk_episode(
        episode_id=episode.episode_id,
        steps=episode.steps,
        H=H,
        overlap=overlap,
        min_valid_ratio=0.7,
    )
    
    elapsed = time.time() - start_time
    
    print(f"  Generated {len(chunks)} chunks in {elapsed*1000:.1f}ms")
    
    if chunks:
        sample_chunk = chunks[0]
        print(f"\nChunk dimensions:")
        print(f"  obs_seq shape: {sample_chunk.obs_seq.shape}")
        print(f"  act_seq shape: {sample_chunk.act_seq.shape}")
        print(f"  d_obs: {sample_chunk.d_obs}")
        print(f"  d_act: {sample_chunk.d_act}")
        
        # Show chunk quality distribution
        qualities = [c.mean_quality for c in chunks]
        print(f"\nChunk quality statistics:")
        print(f"  Mean quality: {np.mean(qualities):.3f}")
        print(f"  Min quality: {np.min(qualities):.3f}")
        print(f"  Max quality: {np.max(qualities):.3f}")
    
    # Also test lazy chunking
    print("\nTesting lazy chunking (streaming mode)...")
    lazy_count = sum(1 for _ in chunk_episode_lazy(
        episode.episode_id, episode.steps, H, overlap
    ))
    print(f"  Lazy chunking yielded {lazy_count} chunks")
    
    return chunks


def demo_encoding(chunks: List[DemoChunk]):
    """
    Demonstrate encoding chunks with the transformer encoder.
    """
    print("\n" + "="*70)
    print("PHASE 3: Transformer Encoding")
    print("="*70)
    
    if not TORCH_AVAILABLE:
        print("\n  Skipping encoder demo (PyTorch not available)")
        return None
    
    if not chunks:
        print("\n  No chunks to encode")
        return None
    
    # Get dimensions from first chunk
    d_obs = chunks[0].d_obs
    d_act = chunks[0].d_act
    
    print(f"\nCreating encoder (d_obs={d_obs}, d_act={d_act})...")
    
    # Create encoder
    encoder = create_encoder(
        d_obs=d_obs,
        d_act=d_act,
        d_embed=64,
        model_size='small',
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    
    # Encode all chunks
    print(f"\nEncoding {len(chunks)} chunks...")
    start_time = time.time()
    
    embeddings = encode_chunks_gpu(
        encoder=encoder,
        chunks=chunks,
        device=device,
        batch_size=16,
    )
    
    elapsed = time.time() - start_time
    
    print(f"  Encoding complete in {elapsed*1000:.1f}ms ({len(chunks)/elapsed:.1f} chunks/sec)")
    print(f"  Embedding shape: {embeddings.shape}")
    
    # Embedding statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {np.mean(embeddings):.4f}")
    print(f"  Std: {np.std(embeddings):.4f}")
    print(f"  Min: {np.min(embeddings):.4f}")
    print(f"  Max: {np.max(embeddings):.4f}")
    
    # Test single chunk encoding (streaming mode)
    print("\nTesting single-chunk encoding (streaming mode)...")
    start_time = time.time()
    single_embed = encode_single_chunk(encoder, chunks[0], device)
    single_elapsed = time.time() - start_time
    print(f"  Single chunk encoded in {single_elapsed*1000:.2f}ms")
    
    return embeddings


def demo_normalization(chunks: List[DemoChunk]):
    """
    Demonstrate chunk normalization for training.
    """
    print("\n" + "="*70)
    print("PHASE 4: Data Normalization")
    print("="*70)
    
    if not chunks:
        print("\n  No chunks to normalize")
        return
    
    print(f"\nNormalizing {len(chunks)} chunks...")
    
    normalized_chunks, stats = normalize_chunks(chunks, compute_stats=True)
    
    print(f"\nNormalization statistics:")
    print(f"  obs_mean shape: {stats['obs_mean'].shape}")
    print(f"  obs_std shape: {stats['obs_std'].shape}")
    print(f"  act_mean shape: {stats['act_mean'].shape}")
    print(f"  act_std shape: {stats['act_std'].shape}")
    
    # Verify normalization
    all_obs = np.concatenate([c.obs_seq for c in normalized_chunks], axis=0)
    all_act = np.concatenate([c.act_seq for c in normalized_chunks], axis=0)
    
    print(f"\nNormalized data statistics:")
    print(f"  obs mean: {np.mean(all_obs):.4f} (should be ~0)")
    print(f"  obs std: {np.std(all_obs):.4f} (should be ~1)")
    print(f"  act mean: {np.mean(all_act):.4f} (should be ~0)")
    print(f"  act std: {np.std(all_act):.4f} (should be ~1)")


def demo_collation(chunks: List[DemoChunk]):
    """
    Demonstrate batch collation for PyTorch dataloaders.
    """
    print("\n" + "="*70)
    print("PHASE 5: Batch Collation")
    print("="*70)
    
    if not chunks:
        print("\n  No chunks to collate")
        return
    
    # Take a subset for batching
    batch_chunks = chunks[:8]
    
    print(f"\nCollating batch of {len(batch_chunks)} chunks...")
    
    obs_batch, act_batch, metadata = collate_chunks(batch_chunks)
    
    print(f"\nCollated batch shapes:")
    print(f"  obs_batch: {obs_batch.shape}")
    print(f"  act_batch: {act_batch.shape}")
    print(f"  valid_masks: {metadata['valid_masks'].shape}")
    
    print(f"\nMetadata:")
    print(f"  chunk_ids: {metadata['chunk_ids'][:3]}...")
    print(f"  task_ids: {set(metadata['task_ids'])}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the complete IL pipeline demo."""
    print("\n" + "="*70)
    print("DYNAMICAL.AI EDGE IL PIPELINE DEMO")
    print("="*70)
    print("\nThis demo shows the complete IL pipeline on the edge:")
    print("  1. Human demonstration → Retargeting → Robot actions")
    print("  2. Robot episode → Chunking → Fixed-horizon sequences")
    print("  3. Chunks → Transformer encoder → Embeddings")
    print("  4. (Next step: N2HE encryption → Cloud upload)")
    
    # Phase 1: Retargeting and recording
    episode = demo_retargeting_and_recording()
    
    # Phase 2: Chunking
    chunks = demo_chunking(episode)
    
    # Phase 3: Encoding
    embeddings = demo_encoding(chunks)
    
    # Phase 4: Normalization
    demo_normalization(chunks)
    
    # Phase 5: Batch collation
    demo_collation(chunks)
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"\n  Episode: {len(episode)} steps, {episode.duration:.2f}s")
    print(f"  Chunks: {len(chunks)} (H=20, overlap=5)")
    if embeddings is not None:
        print(f"  Embeddings: {embeddings.shape}")
    
    print(f"\n  Next steps for production:")
    print(f"    1. Connect real cameras and glove")
    print(f"    2. Load real robot URDF into IK solver")
    print(f"    3. Integrate N2HE encryption")
    print(f"    4. Set up MOAI cloud streaming")
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)


if __name__ == '__main__':
    main()
