"""
Episode Chunking for ACT-style Imitation Learning

ACT (Action Chunking with Transformers) and similar IL methods operate on 
fixed-horizon "chunks" of demonstrations. This module handles the conversion
from variable-length Episodes to fixed-length DemoChunks.

Why Chunking?

1. Transformers need fixed sequence lengths for efficient batching
2. Action chunking improves temporal consistency - predicting H future 
   actions at once reduces compounding errors
3. Overlapping chunks provide data augmentation and multiple views of
   the same transition

Chunking Strategy:

Given an episode of length T and chunk horizon H with overlap O:
- Generate chunks starting at indices [0, H-O, 2*(H-O), ...]
- Each chunk contains H consecutive (obs, action) pairs
- If T < H, the episode is too short and should be filtered out
- Partial chunks at the end can be zero-padded or discarded

Memory Considerations:

On the edge device (Jetson Orin), memory is limited. Chunks should be
generated on-demand during streaming to cloud, not all materialized at once.
This module supports both eager (all-at-once) and lazy (generator) modes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Iterator, Optional, Tuple
import uuid

# Import from sibling modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.recorder import Episode, DemoStep, RobotObs, RobotAction


@dataclass
class DemoChunk:
    """
    A fixed-horizon chunk of a demonstration episode.
    
    This is the unit of data that gets:
    1. Encoded by the transformer encoder on the edge
    2. Encrypted with N2HE
    3. Streamed to the MOAI cloud
    
    The chunk contains flattened observation and action sequences,
    ready for neural network consumption.
    """
    chunk_id: str                               # Unique identifier for this chunk
    episode_id: str                             # Source episode
    task_id: str                                # Task being demonstrated
    env_id: str                                 # Physical environment/site
    
    # Sequence data as numpy arrays
    obs_seq: np.ndarray                         # [H, d_obs] observation sequence
    act_seq: np.ndarray                         # [H, d_act] action sequence
    
    # Temporal metadata
    start_step: int                             # Starting step index in episode
    horizon: int                                # Chunk length H
    timestamps: np.ndarray                      # [H] timestamps for each step
    
    # Quality metrics (from retargeting)
    quality_scores: np.ndarray                  # [H] per-step quality scores
    valid_mask: np.ndarray                      # [H] boolean mask of valid steps
    
    # Optional: robot state for first step (useful for execution)
    initial_joint_positions: Optional[np.ndarray] = None
    
    @property
    def d_obs(self) -> int:
        """Observation dimension."""
        return self.obs_seq.shape[1]
    
    @property
    def d_act(self) -> int:
        """Action dimension."""
        return self.act_seq.shape[1]
    
    @property
    def mean_quality(self) -> float:
        """Average quality score for valid steps."""
        if not np.any(self.valid_mask):
            return 0.0
        return float(np.mean(self.quality_scores[self.valid_mask]))
    
    @property
    def valid_ratio(self) -> float:
        """Fraction of valid steps in the chunk."""
        return float(np.mean(self.valid_mask))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'chunk_id': self.chunk_id,
            'episode_id': self.episode_id,
            'task_id': self.task_id,
            'env_id': self.env_id,
            'obs_seq': self.obs_seq.tolist(),
            'act_seq': self.act_seq.tolist(),
            'start_step': self.start_step,
            'horizon': self.horizon,
            'timestamps': self.timestamps.tolist(),
            'quality_scores': self.quality_scores.tolist(),
            'valid_mask': self.valid_mask.tolist(),
        }


def flatten_obs(step: DemoStep) -> np.ndarray:
    """
    Flatten a DemoStep's observation to a vector.
    
    Default layout: [joint_positions, joint_velocities, gripper_position]
    
    This function defines the canonical observation format for the IL pipeline.
    Modify this if your robot has additional sensors (e.g., force/torque).
    """
    return step.obs.to_vector(include_gripper=True)


def flatten_act(step: DemoStep) -> np.ndarray:
    """
    Flatten a DemoStep's action to a vector.
    
    Default layout: [joint_position_targets, gripper_target]
    
    This function defines the canonical action format for the IL pipeline.
    """
    return step.action.to_vector(include_gripper=True)


def chunk_episode(
    episode_id: str,
    steps: List[DemoStep],
    H: int,
    overlap: int = 0,
    min_valid_ratio: float = 0.8,
    pad_mode: str = 'zero'
) -> List[DemoChunk]:
    """
    Chunk an episode into fixed-horizon sequences.
    
    This is the main chunking function. It takes an episode's steps and
    produces a list of DemoChunks, each containing H timesteps of 
    observation-action pairs.
    
    Args:
        episode_id: Unique episode identifier
        steps: List of DemoStep objects (should all be from same episode)
        H: Chunk horizon (sequence length)
        overlap: Number of steps shared between consecutive chunks.
                 Setting overlap=H//2 gives 50% overlap for data augmentation.
        min_valid_ratio: Minimum fraction of valid steps required per chunk.
                        Chunks with lower valid ratios are discarded.
        pad_mode: How to handle chunks that extend past episode end.
                  'zero': Pad with zeros
                  'repeat': Repeat the last valid step
                  'discard': Don't generate partial chunks
    
    Returns:
        List of DemoChunk objects
    
    Example:
        If episode has T=100 steps, H=20, overlap=5:
        - Chunk 0: steps [0:20]
        - Chunk 1: steps [15:35] (overlap of 5)
        - Chunk 2: steps [30:50]
        - ... and so on
        
        This yields ceil((T - H) / (H - overlap)) + 1 = 6 chunks
    """
    T = len(steps)
    
    # Handle edge cases
    if T == 0:
        return []
    
    if T < H and pad_mode == 'discard':
        # Episode too short, no chunks possible
        return []
    
    # Validate steps have consistent metadata
    task_id = steps[0].task_id
    env_id = steps[0].env_id
    
    # Determine observation and action dimensions from first valid step
    first_valid = next((s for s in steps if s.is_valid), steps[0])
    d_obs = len(flatten_obs(first_valid))
    d_act = len(flatten_act(first_valid))
    
    # Calculate chunk start indices
    stride = H - overlap
    if stride <= 0:
        raise ValueError(f"Overlap ({overlap}) must be less than horizon ({H})")
    
    chunks = []
    start_idx = 0
    
    while start_idx < T:
        end_idx = min(start_idx + H, T)
        chunk_len = end_idx - start_idx
        
        # Initialize arrays for this chunk
        obs_seq = np.zeros((H, d_obs), dtype=np.float32)
        act_seq = np.zeros((H, d_act), dtype=np.float32)
        timestamps = np.zeros(H, dtype=np.float64)
        quality_scores = np.zeros(H, dtype=np.float32)
        valid_mask = np.zeros(H, dtype=bool)
        
        # Fill in data from steps
        for i, step_idx in enumerate(range(start_idx, end_idx)):
            step = steps[step_idx]
            
            obs_seq[i] = flatten_obs(step)
            act_seq[i] = flatten_act(step)
            timestamps[i] = step.obs.timestamp
            quality_scores[i] = step.quality_score
            valid_mask[i] = step.is_valid
        
        # Handle padding for partial chunks
        if chunk_len < H:
            if pad_mode == 'zero':
                # Already initialized to zeros, just update valid_mask
                # (already False for these indices)
                pass
            
            elif pad_mode == 'repeat':
                # Repeat the last valid observation and action
                last_idx = chunk_len - 1
                for i in range(chunk_len, H):
                    obs_seq[i] = obs_seq[last_idx]
                    act_seq[i] = act_seq[last_idx]
                    timestamps[i] = timestamps[last_idx]
                    # Keep quality_scores and valid_mask at 0/False for padding
            
            elif pad_mode == 'discard':
                # Skip this partial chunk
                break
        
        # Check if chunk meets quality threshold
        if np.mean(valid_mask) >= min_valid_ratio:
            # Get initial joint positions for potential execution
            initial_joints = None
            if steps[start_idx].is_valid:
                initial_joints = steps[start_idx].obs.joint_positions.copy()
            
            chunk = DemoChunk(
                chunk_id=f"{episode_id}_chunk_{start_idx:05d}",
                episode_id=episode_id,
                task_id=task_id,
                env_id=env_id,
                obs_seq=obs_seq,
                act_seq=act_seq,
                start_step=start_idx,
                horizon=H,
                timestamps=timestamps,
                quality_scores=quality_scores,
                valid_mask=valid_mask,
                initial_joint_positions=initial_joints,
            )
            chunks.append(chunk)
        
        start_idx += stride
        
        # If we've processed all data and padding is disabled, stop
        if end_idx >= T and pad_mode != 'discard':
            break
    
    return chunks


def chunk_episode_lazy(
    episode_id: str,
    steps: List[DemoStep],
    H: int,
    overlap: int = 0,
    min_valid_ratio: float = 0.8,
) -> Iterator[DemoChunk]:
    """
    Lazily chunk an episode, yielding one chunk at a time.
    
    This is memory-efficient for streaming to cloud. Instead of materializing
    all chunks in memory, we generate them on-demand.
    
    Same parameters as chunk_episode() but returns an iterator.
    
    Usage:
        for chunk in chunk_episode_lazy(episode_id, steps, H=20):
            embedding = encoder.encode(chunk)
            encrypted = encrypt(embedding)
            stream_to_cloud(encrypted)
    """
    T = len(steps)
    
    if T == 0:
        return
    
    task_id = steps[0].task_id
    env_id = steps[0].env_id
    
    # Get dimensions
    first_valid = next((s for s in steps if s.is_valid), steps[0])
    d_obs = len(flatten_obs(first_valid))
    d_act = len(flatten_act(first_valid))
    
    stride = H - overlap
    start_idx = 0
    
    while start_idx < T:
        end_idx = min(start_idx + H, T)
        chunk_len = end_idx - start_idx
        
        # Skip partial chunks
        if chunk_len < H:
            break
        
        # Build chunk data
        obs_seq = np.zeros((H, d_obs), dtype=np.float32)
        act_seq = np.zeros((H, d_act), dtype=np.float32)
        timestamps = np.zeros(H, dtype=np.float64)
        quality_scores = np.zeros(H, dtype=np.float32)
        valid_mask = np.zeros(H, dtype=bool)
        
        for i, step_idx in enumerate(range(start_idx, end_idx)):
            step = steps[step_idx]
            obs_seq[i] = flatten_obs(step)
            act_seq[i] = flatten_act(step)
            timestamps[i] = step.obs.timestamp
            quality_scores[i] = step.quality_score
            valid_mask[i] = step.is_valid
        
        if np.mean(valid_mask) >= min_valid_ratio:
            initial_joints = None
            if steps[start_idx].is_valid:
                initial_joints = steps[start_idx].obs.joint_positions.copy()
            
            yield DemoChunk(
                chunk_id=f"{episode_id}_chunk_{start_idx:05d}",
                episode_id=episode_id,
                task_id=task_id,
                env_id=env_id,
                obs_seq=obs_seq,
                act_seq=act_seq,
                start_step=start_idx,
                horizon=H,
                timestamps=timestamps,
                quality_scores=quality_scores,
                valid_mask=valid_mask,
                initial_joint_positions=initial_joints,
            )
        
        start_idx += stride


def chunk_episode_from_arrays(
    episode_id: str,
    task_id: str,
    env_id: str,
    obs_array: np.ndarray,
    act_array: np.ndarray,
    timestamps: np.ndarray,
    H: int,
    overlap: int = 0,
) -> List[DemoChunk]:
    """
    Chunk from pre-flattened arrays (e.g., loaded from saved episode).
    
    This is useful when loading episodes that were saved in array format
    and you want to chunk them without reconstructing DemoStep objects.
    
    Args:
        episode_id: Episode identifier
        task_id: Task identifier
        env_id: Environment identifier
        obs_array: [T, d_obs] observation array
        act_array: [T, d_act] action array
        timestamps: [T] timestamp array
        H: Chunk horizon
        overlap: Overlap between consecutive chunks
        
    Returns:
        List of DemoChunk objects
    """
    T = obs_array.shape[0]
    d_obs = obs_array.shape[1]
    d_act = act_array.shape[1]
    
    if T < H:
        return []
    
    stride = H - overlap
    chunks = []
    start_idx = 0
    
    while start_idx + H <= T:
        end_idx = start_idx + H
        
        chunk = DemoChunk(
            chunk_id=f"{episode_id}_chunk_{start_idx:05d}",
            episode_id=episode_id,
            task_id=task_id,
            env_id=env_id,
            obs_seq=obs_array[start_idx:end_idx].copy(),
            act_seq=act_array[start_idx:end_idx].copy(),
            start_step=start_idx,
            horizon=H,
            timestamps=timestamps[start_idx:end_idx].copy(),
            quality_scores=np.ones(H, dtype=np.float32),  # Assume all valid
            valid_mask=np.ones(H, dtype=bool),
        )
        chunks.append(chunk)
        
        start_idx += stride
    
    return chunks


# ---------------------------------------------------------------------------
# Batch Utilities for Dataloader Integration
# ---------------------------------------------------------------------------

def collate_chunks(chunks: List[DemoChunk]) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Collate a list of chunks into batched arrays for neural network training.
    
    This function is designed to be compatible with PyTorch dataloaders.
    
    Args:
        chunks: List of DemoChunk objects (should all have same H, d_obs, d_act)
        
    Returns:
        obs_batch: [B, H, d_obs] batched observations
        act_batch: [B, H, d_act] batched actions
        metadata: Dictionary with chunk_ids, valid_masks, etc.
    """
    B = len(chunks)
    H = chunks[0].horizon
    d_obs = chunks[0].d_obs
    d_act = chunks[0].d_act
    
    obs_batch = np.zeros((B, H, d_obs), dtype=np.float32)
    act_batch = np.zeros((B, H, d_act), dtype=np.float32)
    valid_masks = np.zeros((B, H), dtype=bool)
    
    chunk_ids = []
    episode_ids = []
    task_ids = []
    
    for i, chunk in enumerate(chunks):
        obs_batch[i] = chunk.obs_seq
        act_batch[i] = chunk.act_seq
        valid_masks[i] = chunk.valid_mask
        chunk_ids.append(chunk.chunk_id)
        episode_ids.append(chunk.episode_id)
        task_ids.append(chunk.task_id)
    
    metadata = {
        'chunk_ids': chunk_ids,
        'episode_ids': episode_ids,
        'task_ids': task_ids,
        'valid_masks': valid_masks,
    }
    
    return obs_batch, act_batch, metadata


def normalize_chunks(
    chunks: List[DemoChunk],
    obs_mean: Optional[np.ndarray] = None,
    obs_std: Optional[np.ndarray] = None,
    act_mean: Optional[np.ndarray] = None,
    act_std: Optional[np.ndarray] = None,
    compute_stats: bool = False,
) -> Tuple[List[DemoChunk], dict]:
    """
    Normalize observation and action sequences in chunks.
    
    Normalization is crucial for stable training. Options:
    1. Provide pre-computed statistics (for inference/deployment)
    2. Compute statistics from the provided chunks (for training)
    
    Args:
        chunks: List of DemoChunk objects
        obs_mean, obs_std: Pre-computed observation statistics
        act_mean, act_std: Pre-computed action statistics
        compute_stats: If True, compute statistics from chunks
        
    Returns:
        normalized_chunks: New list of chunks with normalized data
        stats: Dictionary with computed statistics
    """
    if compute_stats:
        # Collect all data for statistics computation
        all_obs = np.concatenate([c.obs_seq for c in chunks], axis=0)
        all_act = np.concatenate([c.act_seq for c in chunks], axis=0)
        
        obs_mean = np.mean(all_obs, axis=0)
        obs_std = np.std(all_obs, axis=0) + 1e-6  # Avoid division by zero
        act_mean = np.mean(all_act, axis=0)
        act_std = np.std(all_act, axis=0) + 1e-6
    
    # Create normalized copies
    normalized_chunks = []
    
    for chunk in chunks:
        norm_obs = (chunk.obs_seq - obs_mean) / obs_std
        norm_act = (chunk.act_seq - act_mean) / act_std
        
        norm_chunk = DemoChunk(
            chunk_id=chunk.chunk_id,
            episode_id=chunk.episode_id,
            task_id=chunk.task_id,
            env_id=chunk.env_id,
            obs_seq=norm_obs.astype(np.float32),
            act_seq=norm_act.astype(np.float32),
            start_step=chunk.start_step,
            horizon=chunk.horizon,
            timestamps=chunk.timestamps,
            quality_scores=chunk.quality_scores,
            valid_mask=chunk.valid_mask,
            initial_joint_positions=chunk.initial_joint_positions,
        )
        normalized_chunks.append(norm_chunk)
    
    stats = {
        'obs_mean': obs_mean,
        'obs_std': obs_std,
        'act_mean': act_mean,
        'act_std': act_std,
    }
    
    return normalized_chunks, stats
