"""PyTorch Neural Architecture for XMCA-Net Dense Combiner and Deep Cause-Attribution Head.

Implements:
1. DenseMultiBranchEncoder: Multi-stream diagnostic encoder (Vision, Retrieval, Utilization, Interaction).
2. XMCANetwork: Deep dense residual network with missingness-aware routing for hallucination detection.
3. XMCAAttributionNetwork: Deep multi-stage cause-attribution classifier with diagnostic skip connections.
4. TemperatureScaler: Post-hoc probability calibration via temperature optimization on validation logits.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseMultiBranchEncoder(nn.Module):
    """Encodes diagnostic features into multi-stream representations with missingness gating."""

    def __init__(self, hidden_stream_dim: int = 64, interaction_dim: int = 32):
        super().__init__()
        # Stream 1: Vision (5 features: sp_score, visual_support, visual_contradiction, coverage, uncertainty)
        self.vision_branch = nn.Sequential(
            nn.Linear(5, hidden_stream_dim),
            nn.LayerNorm(hidden_stream_dim),
            nn.GELU(),
            nn.Linear(hidden_stream_dim, hidden_stream_dim),
        )

        # Stream 2: Retrieval (6 features: sr_score, top_rel, mean_rel, margin, cov, m_r)
        self.retrieval_branch = nn.Sequential(
            nn.Linear(6, hidden_stream_dim),
            nn.LayerNorm(hidden_stream_dim),
            nn.GELU(),
            nn.Linear(hidden_stream_dim, hidden_stream_dim),
        )

        # Stream 3: Utilization (6 features: su_score, entail, contra, omission, unsupported, m_u)
        self.utilization_branch = nn.Sequential(
            nn.Linear(6, hidden_stream_dim),
            nn.LayerNorm(hidden_stream_dim),
            nn.GELU(),
            nn.Linear(hidden_stream_dim, hidden_stream_dim),
        )

        # Stream 4: Interactions (3 features: total_claims, sp_x_sr, sp_x_su)
        self.interaction_branch = nn.Sequential(
            nn.Linear(3, interaction_dim),
            nn.LayerNorm(interaction_dim),
            nn.GELU(),
            nn.Linear(interaction_dim, interaction_dim),
        )

        # Missingness-conditioned routing gate (m_r, m_u)
        self.rag_condition_gate = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, hidden_stream_dim * 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Partition 20-dim input into 4 diagnostic streams and fuse."""
        v_feat = x[:, 0:5]
        r_feat = x[:, 5:11]
        u_feat = x[:, 11:17]
        i_feat = x[:, 17:20]
        masks = x[:, [10, 16]]  # m_r and m_u

        h_v = self.vision_branch(v_feat)
        h_r = self.retrieval_branch(r_feat)
        h_u = self.utilization_branch(u_feat)
        h_i = self.interaction_branch(i_feat)

        # Condition RAG streams: if m_r=1 or m_u=1, attenuate RAG features so vision decisions stay unpolluted
        gate = self.rag_condition_gate(masks)
        gate_r = gate[:, :h_r.shape[-1]]
        gate_u = gate[:, h_r.shape[-1]:]

        h_r_gated = h_r * (1.0 - masks[:, 0:1] * 0.85) * gate_r
        h_u_gated = h_u * (1.0 - masks[:, 1:2] * 0.85) * gate_u

        # Concatenate all stream representations (64 + 64 + 64 + 32 = 224 dims)
        return torch.cat([h_v, h_r_gated, h_u_gated, h_i], dim=-1)


class DenseResidualBlock(nn.Module):
    """Pre-LayerNorm residual block with GELU activations and dropout."""

    def __init__(self, dim: int, dropout: float = 0.12):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear1(self.act(self.norm1(x)))
        out = self.drop(out)
        out = self.linear2(self.norm2(out))
        out = self.drop(out)
        return residual + out


# Alias for backward compatibility
ResidualBlock = DenseResidualBlock


class XMCANetwork(nn.Module):
    """Dense multi-stream deep residual neural network for hallucination detection."""

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.12,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Multi-branch multi-stream encoder
        self.encoder = DenseMultiBranchEncoder(hidden_stream_dim=64, interaction_dim=32)
        encoded_dim = 64 + 64 + 64 + 32  # 224

        # Dense projection into unified diagnostic latent space
        self.fusion_proj = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Deep residual diagnostic blocks
        self.blocks = nn.ModuleList([
            DenseResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])

        # Final representation norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Multi-layer detection classification head
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def extract_features_and_masks(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract missingness indicators [m_r, m_u] located at indices 10 and 16."""
        m_r = x[:, 10:11]
        m_u = x[:, 16:17]
        masks = torch.cat([m_r, m_u], dim=-1)
        return x, masks

    def forward_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass up to the penultimate diagnostic embedding."""
        encoded = self.encoder(x)
        h = self.fusion_proj(encoded)
        for block in self.blocks:
            h = block(h)
        return self.final_norm(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes raw detection logits [batch_size]."""
        rep = self.forward_representation(x)
        logits = self.detection_head(rep)
        return logits.squeeze(-1)


class XMCAAttributionNetwork(nn.Module):
    """Deep cause-attribution classifier with raw diagnostic skip connections."""

    def __init__(self, hidden_dim: int = 128, raw_feat_dim: int = 20, num_causes: int = 3):
        super().__init__()
        input_dim = hidden_dim + raw_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.10),
            DenseResidualBlock(96, dropout=0.10),
            nn.Linear(96, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Linear(48, num_causes),
        )

    def forward(self, rep: torch.Tensor, raw_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Outputs 3-class logits for [Perception, Retrieval, Utilization]."""
        if raw_x is not None:
            combined = torch.cat([rep, raw_x], dim=-1)
        else:
            # Fallback zero-pad if raw features not passed
            pad = torch.zeros(rep.size(0), 20, device=rep.device, dtype=rep.dtype)
            combined = torch.cat([rep, pad], dim=-1)
        return self.classifier(combined)


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling to calibrate neural model probabilities."""

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(torch.clamp(torch.exp(self.log_temperature), min=0.1, max=10.0).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(torch.exp(self.log_temperature), min=0.1, max=10.0)
        return logits / temp

    def calibrate(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50):
        """Optimize scalar temperature on validation logits to minimize NLL."""
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.01, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
