"""
Transformer Encoder for Demonstration Chunk Embedding

This module implements the encoder component of the ACT-style IL pipeline.
The encoder runs on the edge device (Jetson Orin) and compresses demonstration
chunks into compact embeddings that can be:

1. Encrypted with N2HE (Homomorphic Encryption)
2. Streamed to the MOAI cloud
3. Used for policy training without exposing raw trajectories

Architecture Overview:

    Input:  [B, H, d_obs]  Observation sequence
            [B, H, d_act]  Action sequence
            
    ↓ Concatenate and project
    
    Projected: [B, H, d_model]
    
    ↓ Add positional encoding
    
    ↓ Transformer encoder layers (self-attention)
    
    Contextualized: [B, H, d_model]
    
    ↓ Pooling (mean/CLS token)
    
    Output: [B, d_embed]  Chunk embedding
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# ---------------------------------------------------------------------------
# Configuration (always available)
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    """Configuration for the chunk encoder."""
    
    # Input dimensions (set based on robot)
    d_obs: int = 15                     # Observation dimension (7 joints * 2 + gripper)
    d_act: int = 8                      # Action dimension (7 joints + gripper)
    
    # Model architecture
    d_model: int = 128                  # Transformer hidden dimension
    n_heads: int = 4                    # Number of attention heads
    n_layers: int = 2                   # Number of transformer layers
    d_ff: int = 256                     # Feed-forward intermediate dimension
    dropout: float = 0.1                # Dropout probability
    
    # Output
    d_embed: int = 64                   # Final embedding dimension
    
    # Sequence
    max_horizon: int = 50               # Maximum chunk length
    
    # Pooling strategy
    pooling: str = 'mean'               # 'mean', 'cls', 'last'
    
    @property
    def d_input(self) -> int:
        """Total input dimension per timestep."""
        return self.d_obs + self.d_act


# ---------------------------------------------------------------------------
# PyTorch-dependent implementations
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    
    class SinusoidalPositionalEncoding(nn.Module):
        """
        Sinusoidal positional encoding from "Attention Is All You Need".
        
        This gives the transformer information about position in the sequence
        without requiring learned parameters. Good for generalization to
        sequence lengths not seen during training.
        """
        
        def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            # Create positional encoding matrix
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            
            # Compute the div_term for sinusoidal encoding
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * 
                (-np.log(10000.0) / d_model)
            )
            
            # Apply sin to even indices, cos to odd indices
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            # Add batch dimension and register as buffer (not a parameter)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            self.register_buffer('pe', pe)
        
        def forward(self, x):
            """Add positional encoding to input tensor [B, L, d_model]."""
            seq_len = x.size(1)
            x = x + self.pe[:, :seq_len, :]
            return self.dropout(x)


    class LearnedPositionalEncoding(nn.Module):
        """Learned positional encoding - better for fixed-length sequences."""
        
        def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            self.pe = nn.Embedding(max_len, d_model)
        
        def forward(self, x):
            """Add learned positional encoding to input tensor [B, L, d_model]."""
            B, L, _ = x.shape
            positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
            x = x + self.pe(positions)
            return self.dropout(x)


    class TransformerEncoderLayer(nn.Module):
        """Single transformer encoder layer with pre-norm architecture."""
        
        def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
            super().__init__()
            
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x, mask=None, key_padding_mask=None):
            """Forward pass: self-attention + feed-forward with residuals."""
            # Self-attention with pre-norm
            norm_x = self.norm1(x)
            attn_out, _ = self.self_attn(
                norm_x, norm_x, norm_x,
                attn_mask=mask,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
            x = x + self.dropout(attn_out)
            
            # Feed-forward with pre-norm
            norm_x = self.norm2(x)
            ff_out = self.ff(norm_x)
            x = x + ff_out
            
            return x


    class ChunkEncoder(nn.Module):
        """
        Transformer encoder for demonstration chunk embedding.
        
        Takes a sequence of (observation, action) pairs and produces
        a fixed-size embedding vector that captures the demonstration's behavior.
        """
        
        def __init__(self, config: EncoderConfig):
            super().__init__()
            self.config = config
            
            # Input projection
            self.input_proj = nn.Linear(config.d_input, config.d_model)
            
            # Optional CLS token
            if config.pooling == 'cls':
                self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
            else:
                self.cls_token = None
            
            # Positional encoding
            self.pos_enc = SinusoidalPositionalEncoding(
                config.d_model,
                max_len=config.max_horizon + 1,
                dropout=config.dropout
            )
            
            # Transformer layers
            self.layers = nn.ModuleList([
                TransformerEncoderLayer(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout
                )
                for _ in range(config.n_layers)
            ])
            
            # Final normalization
            self.final_norm = nn.LayerNorm(config.d_model)
            
            # Output projection
            self.output_proj = nn.Sequential(
                nn.Linear(config.d_model, config.d_embed),
                nn.LayerNorm(config.d_embed),
            )
            
            self._init_weights()
        
        def _init_weights(self):
            """Initialize weights for stable training."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
        
        def forward(self, obs_seq, act_seq, padding_mask=None):
            """
            Encode observation-action sequences to embeddings.
            
            Args:
                obs_seq: [B, H, d_obs] observation sequence
                act_seq: [B, H, d_act] action sequence
                padding_mask: [B, H] boolean mask where True = padding (ignore)
                
            Returns:
                embeddings: [B, d_embed] chunk embeddings
            """
            B, H, _ = obs_seq.shape
            
            # Concatenate and project
            x = torch.cat([obs_seq, act_seq], dim=-1)
            x = self.input_proj(x)
            
            # Add CLS token if using CLS pooling
            if self.config.pooling == 'cls' and self.cls_token is not None:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)
                if padding_mask is not None:
                    cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
                    padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
            
            # Positional encoding and transformer layers
            x = self.pos_enc(x)
            for layer in self.layers:
                x = layer(x, key_padding_mask=padding_mask)
            x = self.final_norm(x)
            
            # Pooling
            if self.config.pooling == 'cls':
                pooled = x[:, 0, :]
            elif self.config.pooling == 'mean':
                if padding_mask is not None:
                    mask = ~padding_mask.unsqueeze(-1)
                    x_masked = x * mask.float()
                    pooled = x_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    pooled = x.mean(dim=1)
            elif self.config.pooling == 'last':
                pooled = x[:, -1, :]
            else:
                raise ValueError(f"Unknown pooling strategy: {self.config.pooling}")
            
            return self.output_proj(pooled)
        
        def get_sequence_embeddings(self, obs_seq, act_seq, padding_mask=None):
            """Get per-timestep embeddings (before pooling)."""
            x = torch.cat([obs_seq, act_seq], dim=-1)
            x = self.input_proj(x)
            x = self.pos_enc(x)
            for layer in self.layers:
                x = layer(x, key_padding_mask=padding_mask)
            return self.final_norm(x)


    def encode_chunks_gpu(encoder, chunks, device='cuda', batch_size=32):
        """Encode a list of chunks using the GPU, returning numpy embeddings."""
        encoder = encoder.to(device)
        encoder.eval()
        
        all_embeddings = []
        n_chunks = len(chunks)
        
        with torch.no_grad():
            for i in range(0, n_chunks, batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                obs_batch = np.stack([c.obs_seq for c in batch_chunks])
                act_batch = np.stack([c.act_seq for c in batch_chunks])
                mask_batch = np.stack([~c.valid_mask for c in batch_chunks])
                
                obs_tensor = torch.from_numpy(obs_batch).float().to(device)
                act_tensor = torch.from_numpy(act_batch).float().to(device)
                mask_tensor = torch.from_numpy(mask_batch).bool().to(device)
                
                embeddings = encoder(obs_tensor, act_tensor, padding_mask=mask_tensor)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)


    def encode_single_chunk(encoder, chunk, device='cuda'):
        """Encode a single chunk (for streaming)."""
        encoder = encoder.to(device)
        encoder.eval()
        
        with torch.no_grad():
            obs = torch.from_numpy(chunk.obs_seq).float().unsqueeze(0).to(device)
            act = torch.from_numpy(chunk.act_seq).float().unsqueeze(0).to(device)
            mask = torch.from_numpy(~chunk.valid_mask).bool().unsqueeze(0).to(device)
            embedding = encoder(obs, act, padding_mask=mask)
        
        return embedding.cpu().numpy().squeeze(0)


    def export_to_onnx(encoder, output_path, batch_size=1, horizon=20):
        """Export encoder to ONNX format for TensorRT compilation."""
        encoder.eval()
        config = encoder.config
        
        dummy_obs = torch.randn(batch_size, horizon, config.d_obs)
        dummy_act = torch.randn(batch_size, horizon, config.d_act)
        
        torch.onnx.export(
            encoder,
            (dummy_obs, dummy_act),
            output_path,
            input_names=['obs_seq', 'act_seq'],
            output_names=['embedding'],
            dynamic_axes={
                'obs_seq': {0: 'batch_size', 1: 'horizon'},
                'act_seq': {0: 'batch_size', 1: 'horizon'},
                'embedding': {0: 'batch_size'},
            },
            opset_version=13,
        )
        logger.info(f"Exported encoder to {output_path}")


    def create_encoder(d_obs, d_act, d_embed=64, model_size='small'):
        """Create an encoder with sensible defaults for the given size."""
        size_configs = {
            'tiny': dict(d_model=64, n_heads=2, n_layers=1, d_ff=128),
            'small': dict(d_model=128, n_heads=4, n_layers=2, d_ff=256),
            'medium': dict(d_model=256, n_heads=8, n_layers=4, d_ff=512),
            'large': dict(d_model=512, n_heads=8, n_layers=6, d_ff=1024),
        }
        
        if model_size not in size_configs:
            raise ValueError(f"Unknown model size: {model_size}")
        
        config = EncoderConfig(
            d_obs=d_obs,
            d_act=d_act,
            d_embed=d_embed,
            **size_configs[model_size]
        )
        return ChunkEncoder(config)


    def load_encoder(checkpoint_path, device='cpu'):
        """Load a trained encoder from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = EncoderConfig(**checkpoint['config'])
        encoder = ChunkEncoder(config)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        encoder.to(device)
        return encoder, config


    def save_encoder(encoder, output_path):
        """Save encoder checkpoint."""
        checkpoint = {
            'config': encoder.config.__dict__,
            'model_state_dict': encoder.state_dict(),
        }
        torch.save(checkpoint, output_path)
        logger.info(f"Saved encoder to {output_path}")

else:
    # Stub implementations when PyTorch is not available
    
    class SinusoidalPositionalEncoding:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for SinusoidalPositionalEncoding")
    
    class LearnedPositionalEncoding:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for LearnedPositionalEncoding")
    
    class TransformerEncoderLayer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for TransformerEncoderLayer")
    
    class ChunkEncoder:
        def __init__(self, config):
            raise ImportError("PyTorch required for ChunkEncoder")
    
    def encode_chunks_gpu(*args, **kwargs):
        raise ImportError("PyTorch required for encode_chunks_gpu")
    
    def encode_single_chunk(*args, **kwargs):
        raise ImportError("PyTorch required for encode_single_chunk")
    
    def export_to_onnx(*args, **kwargs):
        raise ImportError("PyTorch required for export_to_onnx")
    
    def create_encoder(*args, **kwargs):
        raise ImportError("PyTorch required for create_encoder")
    
    def load_encoder(*args, **kwargs):
        raise ImportError("PyTorch required for load_encoder")
    
    def save_encoder(*args, **kwargs):
        raise ImportError("PyTorch required for save_encoder")
