"""PyTorch Neural Architecture for XMCA-Net Combiner and Cause-Attribution Head.

Implements:
1. XMCANetwork: Deep residual MLP with missingness-aware gating for hallucination detection.
2. TemperatureScaler: Post-hoc probability calibration via temperature optimization.
3. XMCAAttributionNetwork: Neural multi-class cause-attribution classifier.
"""

import math
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MissingnessAwareGating(nn.Module):
    """Gating mechanism that conditions representations on RAG applicability masks."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden representation [batch_size, hidden_dim]
            masks: Missingness indicators [m_r, m_u] [batch_size, 2]
        """
        gate_weights = self.gate(masks)
        return h * (1.0 + gate_weights)


class ResidualBlock(nn.Module):
    """Pre-LayerNorm residual block with GELU activations and dropout."""

    def __init__(self, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear1(self.act(self.norm1(x)))
        out = self.drop(out)
        out = self.linear2(self.norm2(out))
        out = self.drop(out)
        return residual + out


class XMCANetwork(nn.Module):
    """Deep residual neural network for cross-modal consistency hallucination detection."""

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Missingness conditioning gate (m_r, m_u)
        self.missingness_gate = MissingnessAwareGating(hidden_dim)

        # Residual diagnostic fusion blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])

        # Final representation norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Detection classification head
        self.detection_head = nn.Linear(hidden_dim, 1)

    def extract_features_and_masks(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract missingness indicators [m_r, m_u] located at indices 10 and 16."""
        m_r = x[:, 10:11]
        m_u = x[:, 16:17]
        masks = torch.cat([m_r, m_u], dim=-1)
        return x, masks

    def forward_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass up to the penultimate diagnostic embedding."""
        _, masks = self.extract_features_and_masks(x)
        h = self.input_proj(x)
        h = self.missingness_gate(h, masks)
        for block in self.blocks:
            h = block(h)
        return self.final_norm(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes raw detection logits [batch_size, 1]."""
        rep = self.forward_representation(x)
        logits = self.detection_head(rep)
        return logits.squeeze(-1)


class XMCAAttributionNetwork(nn.Module):
    """Neural cause-attribution classifier diagnosing primary failure stage."""

    def __init__(self, hidden_dim: int = 64, num_causes: int = 3):
        super().__init__()
        self.attribution_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, num_causes),
        )

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        """Outputs 3-class logits for [Perception, Retrieval, Utilization]."""
        return self.attribution_head(rep)


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling to calibrate neural model probabilities."""

    def __init__(self):
        super().__init__()
        # Initialize log_temperature to 0.0 so temperature starts at 1.0
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.exp(self.log_temperature)
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

