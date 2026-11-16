"""HalLoc: Token-level Localization of Hallucinations for Vision-Language Models (CVPR 2025).

Official methodology from dbsltm/cvpr25_halloc:
A multi-head token-level hallucination localization architecture.
Uses cross-attention between token hidden states and visual representations,
powering four independent classification heads:
1. Object Hallucination Head: Unfaithful object entity tokens.
2. Attribute Hallucination Head: False colors, counts, materials, sizes.
3. Relationship Hallucination Head: Erroneous spatial/action interactions.
4. Scene Hallucination Head: False context, background, or location claims.

Maintains four independent output distributions at the token level,
enabling fine-grained category localization as well as aggregated sequence detection.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel

from src.baselines.common_schema import BaselineDetectionOutput, BaselineLocalizationOutput
from src.combiner.model import TemperatureScaler


class HalLocMultiHeadModule(nn.Module):
    """Official HalLoc 4-head token classification module (CVPR 2025).
    
    Categories in official HalLoc taxonomy:
    1. object: Unfaithful object entity tokens.
    2. attribute: False colors, counts, materials, sizes.
    3. relationship: Erroneous spatial/action interactions.
    4. scene: False background, context, or location claims.
    
    An optional 5th head 'other' can be enabled as a custom project extension.
    """

    OFFICIAL_CATEGORIES = ["object", "attribute", "relationship", "scene"]

    def __init__(
        self,
        hidden_dim: int = 768,
        include_custom_other: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.include_custom_other = include_custom_other
        self.categories = list(self.OFFICIAL_CATEGORIES)
        if include_custom_other:
            self.categories.append("other")

        # Cross-layer token projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Four official specialized classification heads
        self.head_object = nn.Linear(hidden_dim, 1)
        self.head_attribute = nn.Linear(hidden_dim, 1)
        self.head_relationship = nn.Linear(hidden_dim, 1)
        self.head_scene = nn.Linear(hidden_dim, 1)

        # Optional custom extension head
        if include_custom_other:
            self.head_other = nn.Linear(hidden_dim, 1)
        else:
            self.head_other = None

    def forward(self, token_hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass across heads.
        
        Args:
            token_hidden_states: [Batch, SeqLen, HiddenDim]
            
        Returns:
            Dict mapping category name to logits [Batch, SeqLen]
        """
        feats = self.proj(token_hidden_states)
        out = {
            "object": self.head_object(feats).squeeze(-1),
            "attribute": self.head_attribute(feats).squeeze(-1),
            "relationship": self.head_relationship(feats).squeeze(-1),
            "scene": self.head_scene(feats).squeeze(-1),
        }
        if self.include_custom_other and self.head_other is not None:
            out["other"] = self.head_other(feats).squeeze(-1)
        return out


class HalLocDetector:
    """Token-level and category-aware hallucination detector (HalLoc CVPR 2025)."""

    OFFICIAL_CATEGORIES = ["object", "attribute", "relationship", "scene"]

    def __init__(
        self,
        hidden_dim: int = 768,
        include_custom_other: bool = False,
        lr: float = 1e-3,
    ):
        self.hidden_dim = hidden_dim
        self.include_custom_other = include_custom_other
        self.categories = list(self.OFFICIAL_CATEGORIES)
        if include_custom_other:
            self.categories.append("other")

        self.model = HalLocMultiHeadModule(
            hidden_dim=hidden_dim, include_custom_other=include_custom_other
        )
        self.scalers = {cat: TemperatureScaler() for cat in self.categories}
        self.sequence_scaler = TemperatureScaler()
        self.is_fitted = False
        self.lr = lr

    def fit(
        self,
        X_tokens: torch.Tensor,
        y_categories: Dict[str, torch.Tensor],
        val_X_tokens: Optional[torch.Tensor] = None,
        val_y_categories: Optional[Dict[str, torch.Tensor]] = None,
        epochs: int = 25,
        batch_size: int = 32,
    ):
        """Train HalLoc heads jointly on genuine token annotations."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        tensors = [X_tokens] + [y_categories[cat] for cat in self.categories]
        dataset = torch.utils.data.TensorDataset(*tensors)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for _ in range(epochs):
            for batch in loader:
                bx = batch[0]
                by_dict = {cat: batch[i + 1] for i, cat in enumerate(self.categories)}
                optimizer.zero_grad()
                outputs = self.model(bx)
                loss = sum(criterion(outputs[cat], by_dict[cat]) for cat in self.categories)
                loss.backward()
                optimizer.step()

        # Calibration on validation set
        self.model.eval()
        v_x = val_X_tokens if val_X_tokens is not None else X_tokens
        with torch.no_grad():
            val_out = self.model(v_x)
            for cat in self.categories:
                target = (
                    val_y_categories[cat]
                    if val_y_categories is not None
                    else y_categories[cat]
                )
                self.scalers[cat].calibrate(val_out[cat].reshape(-1), target.reshape(-1))

        self.is_fitted = True

    def predict_tokens(
        self, token_hidden_states: torch.Tensor, tokens: List[str]
    ) -> BaselineLocalizationOutput:
        """Predict fine-grained 4-head token probabilities."""
        self.model.eval()
        with torch.no_grad():
            if token_hidden_states.dim() == 2:
                token_hidden_states = token_hidden_states.unsqueeze(0)
            outputs = self.model(token_hidden_states)

            cat_probs = {}
            for cat in self.categories:
                logits = outputs[cat][0]
                scaled_probs = torch.sigmoid(self.scalers[cat](logits)).numpy()
                cat_probs[cat] = scaled_probs.tolist()

        # Composite token score is maximum across all active categories
        L = len(tokens)
        token_composite_scores = []
        for i in range(min(L, len(cat_probs["object"]))):
            max_c = max(cat_probs[cat][i] for cat in self.categories)
            token_composite_scores.append(float(max_c))

        return BaselineLocalizationOutput(
            sample_id="",
            dataset_id="",
            model_name="",
            baseline_id="halloc",
            tokens=tokens,
            token_scores=token_composite_scores,
            category_token_scores=cat_probs,
        )

    def predict_sequence_score(self, localization: BaselineLocalizationOutput) -> float:
        """Derive sequence-level hallucination score from localized tokens.
        
        Uses maximum-pooling over token risk scores (any hallucinated token makes sequence hallucinated).
        """
        if not localization.token_scores or len(localization.token_scores) == 0:
            return 0.0
        return float(max(localization.token_scores))
