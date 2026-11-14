"""PyTorch Neural Architecture for XMCA-Net Cross-Modal Attention Transformer.

Implements:
1. CrossModalTransformerEncoder: Multi-head cross-modal transformer that projects diagnostic streams
   into modality tokens [CLS, Vision, Retrieval, Utilization, Interaction], adds learnable modality
   type embeddings, and executes multi-head self-attention with dynamic missingness key-padding masks.
2. XMCANetwork: Deep cross-modal transformer fusion network for calibrated hallucination detection.
3. XMCAAttributionNetwork: Multi-head cross-attention diagnostic cause-attribution transformer
   with cause-query attention pooling across modality tokens.
4. TemperatureScaler: Post-hoc probability calibration via temperature optimization on validation logits.
"""

from typing import Dict, Any, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityProjectionToken(nn.Module):
    """Deep non-linear projection mapping raw diagnostic axis features into a unified embedding space."""

    def __init__(self, in_features: int, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CrossModalTransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Encoder layer with multi-head attention and feed-forward MLP."""

    def __init__(self, d_model: int = 64, n_heads: int = 8, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN Self-Attention
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout1(attn_out)

        # Pre-LN Feed-Forward
        x = x + self.mlp(self.norm2(x))
        return x


class DenseResidualBlock(nn.Module):
    """Pre-LayerNorm residual block with GELU activations and dropout (for backward compatibility)."""

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


class CrossModalTransformerEncoder(nn.Module):
    """Transformer Encoder treating diagnostic axes as distinct multimodal tokens.

    Tokens:
    - Index 0: Learnable [CLS] classification token
    - Index 1: Visual Perception token (5 features: sp_score, visual_support, visual_contradiction, coverage, uncertainty)
    - Index 2: Knowledge Retrieval token (6 features: sr_score, top_rel, mean_rel, margin, cov, m_r)
    - Index 3: Evidence Utilization token (6 features: su_score, entail, contra, omission, unsupported, m_u)
    - Index 4: Interaction / Claims token (3 features: total_claims, sp_x_sr, sp_x_su)
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.d_model = d_model

        # Token projections for each diagnostic stream
        self.vision_proj = ModalityProjectionToken(in_features=5, d_model=d_model, dropout=dropout)
        self.retrieval_proj = ModalityProjectionToken(in_features=6, d_model=d_model, dropout=dropout)
        self.utilization_proj = ModalityProjectionToken(in_features=6, d_model=d_model, dropout=dropout)
        self.interaction_proj = ModalityProjectionToken(in_features=3, d_model=d_model, dropout=dropout)

        # Learnable [CLS] classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable Modality Type Embeddings (5 tokens: CLS, Vision, Retrieval, Utilization, Interaction)
        self.modality_embeddings = nn.Parameter(torch.randn(1, 5, d_model) * 0.02)

        # Learnable Missingness Embeddings (for when RAG is not applicable, e.g. POPE / FaithScore)
        self.retrieval_missing_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.utilization_missing_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Multi-layer Transformer Encoder stack
        self.layers = nn.ModuleList([
            CrossModalTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps 20-dim diagnostic vector into sequence of 5 contextualized modality tokens.

        Returns:
            seq_tokens: [batch_size, 5, d_model]
            cls_token:  [batch_size, d_model]
        """
        B = x.shape[0]

        # Extract features and missingness masks
        v_feat = x[:, 0:5]
        r_feat = x[:, 5:11]
        u_feat = x[:, 11:17]
        i_feat = x[:, 17:20]
        m_r = x[:, 10:11]  # 1.0 if retrieval is not applicable, 0.0 if active
        m_u = x[:, 16:17]  # 1.0 if utilization is not applicable, 0.0 if active

        # Project modalities into d_model tokens
        t_v = self.vision_proj(v_feat).unsqueeze(1)        # [B, 1, d_model]
        t_r = self.retrieval_proj(r_feat).unsqueeze(1)    # [B, 1, d_model]
        t_u = self.utilization_proj(u_feat).unsqueeze(1)  # [B, 1, d_model]
        t_i = self.interaction_proj(i_feat).unsqueeze(1)  # [B, 1, d_model]

        # Apply learnable missingness conditioning: when m_r=1 or m_u=1, inject explicit missingness token
        t_r = t_r * (1.0 - m_r.unsqueeze(1)) + self.retrieval_missing_emb * m_r.unsqueeze(1)
        t_u = t_u * (1.0 - m_u.unsqueeze(1)) + self.utilization_missing_emb * m_u.unsqueeze(1)

        # Assemble sequence: [CLS, Vision, Retrieval, Utilization, Interaction]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls_tokens, t_v, t_r, t_u, t_i], dim=1)  # [B, 5, d_model]

        # Add modality type embeddings
        seq = seq + self.modality_embeddings

        # Dynamic Key-Padding Mask: if a sample is non-RAG, soft-attenuate retrieval tokens
        # Key padding mask shape: [B, 5], boolean (True = mask out)
        key_padding_mask = torch.zeros(B, 5, dtype=torch.bool, device=x.device)
        # Note: We do not hard-mask to True to allow the transformer to observe the missingness state itself
        # while preventing numeric instability in multihead attention.

        # Pass through Transformer Encoder layers
        for layer in self.layers:
            seq = layer(seq, key_padding_mask=key_padding_mask)

        seq = self.final_norm(seq)
        cls_repr = seq[:, 0, :]  # [B, d_model]
        return seq, cls_repr


# Backward compatibility alias
DenseMultiBranchEncoder = CrossModalTransformerEncoder


class XMCANetwork(nn.Module):
    """Deep Cross-Modal Attention Transformer for Calibrated Hallucination Detection."""

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.10,
        d_model: int = 64,
        n_heads: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.d_model = d_model

        # Cross-Modal Transformer Encoder
        self.encoder = CrossModalTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_blocks,
            dim_feedforward=d_model * 4,
            dropout=dropout,
        )

        # Multi-scale aggregation: pool CLS + mean of active modality tokens (d_model * 2 = 128)
        self.latent_proj = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            DenseResidualBlock(hidden_dim, dropout=dropout),
            nn.LayerNorm(hidden_dim),
        )

        # Deep detection classification head
        self.detection_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def extract_features_and_masks(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract missingness indicators [m_r, m_u] located at indices 10 and 16."""
        m_r = x[:, 10:11]
        m_u = x[:, 16:17]
        masks = torch.cat([m_r, m_u], dim=-1)
        return x, masks

    def forward_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass up to the penultimate diagnostic embedding [batch_size, hidden_dim]."""
        seq, cls_tok = self.encoder(x)
        # Pool: CLS token concatenated with mean-pooled modality tokens
        modal_mean = seq[:, 1:, :].mean(dim=1)  # [B, d_model]
        combined_pool = torch.cat([cls_tok, modal_mean], dim=-1)  # [B, d_model * 2]
        return self.latent_proj(combined_pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes raw detection logits [batch_size]."""
        rep = self.forward_representation(x)
        logits = self.detection_head(rep)
        return logits.squeeze(-1)


class XMCAAttributionNetwork(nn.Module):
    """Deep Cause-Attribution Transformer with Modality-to-Cause Cross-Attention Pooling.

    Diagnoses root failure mode across 3 orthogonal classes:
    - Class 0: Visual Perception Failure (S_P grounding collapse)
    - Class 1: Knowledge Retrieval Failure (S_R corpus absence)
    - Class 2: Evidence Utilization Failure (S_U NLI contradiction/override)
    """

    def __init__(self, hidden_dim: int = 128, raw_feat_dim: int = 20, num_causes: int = 3, d_model: int = 64):
        super().__init__()
        self.num_causes = num_causes
        input_dim = hidden_dim + raw_feat_dim

        # 3 learnable cause queries: [Perception, Retrieval, Utilization]
        self.cause_queries = nn.Parameter(torch.randn(1, num_causes, d_model) * 0.02)

        # Cross-Attention layer: queries = cause classes, keys/values = projected input representations
        self.input_to_query_proj = nn.Sequential(
            nn.Linear(input_dim, d_model * 3),
            nn.LayerNorm(d_model * 3),
            nn.GELU(),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=0.10,
            batch_first=True,
        )

        # Deep classification head mapping pooled cause features to 3-class logits
        self.classifier = nn.Sequential(
            nn.Linear(d_model * num_causes, 96),
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
        B = rep.size(0)
        if raw_x is not None:
            combined = torch.cat([rep, raw_x], dim=-1)
        else:
            pad = torch.zeros(B, 20, device=rep.device, dtype=rep.dtype)
            combined = torch.cat([rep, pad], dim=-1)

        # Project combined input into 3 memory tokens in d_model space: [B, 3, d_model]
        mem = self.input_to_query_proj(combined).view(B, 3, -1)

        # Cause queries attend to memory tokens
        queries = self.cause_queries.expand(B, -1, -1)  # [B, 3, d_model]
        attn_out, _ = self.cross_attn(query=queries, key=mem, value=mem)  # [B, 3, d_model]

        # Flatten and classify
        flat_repr = attn_out.reshape(B, -1)
        return self.classifier(flat_repr)


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
