"""HALP Feature Extraction Module for Vision-Language Models (EACL 2026 Adaptation).

Extracts the authentic 11 HALP probe feature targets during a single forward pass:
1. 1 Visual Feature (VF) probe target:
   - Mean-pooled vision encoder output prior to multimodal projection.
2. 5 Vision Token (VT) probe targets:
   - Hidden states at the final visual-token position across 5 decoder depth ratios:
     {1, L/4, L/2, 3L/4, L}.
3. 5 Query Token (QT) probe targets:
   - Hidden states at the final query-token position across the same 5 decoder depth ratios.

Critical HuggingFace Tuple Mapping:
- outputs.hidden_states has length L + 1.
- hidden_states[0] is the input embedding layer (NOT decoder layer 1).
- 1-based decoder layer l corresponds to hidden_states[l], which is the output of
  0-based decoder block model.layers[l - 1].
- Exact layer positions:
  * Pos 1 (Layer 1):      hidden_states[1]
  * Pos 2 (Layer L/4):    hidden_states[L // 4]
  * Pos 3 (Layer L/2):    hidden_states[L // 2]
  * Pos 4 (Layer 3L/4):  hidden_states[(3 * L) // 4]
  * Pos 5 (Layer L):      hidden_states[L] (or hidden_states[-1])
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from pydantic import BaseModel, Field


class HALPExtractedSample(BaseModel):
    """Pydantic container for a single sample's extracted 11 HALP feature vectors."""

    sample_id: str
    dataset_id: str
    model_name: str
    partition: str
    image_token_end_idx: int
    query_token_end_idx: int
    
    # Feature vectors (flattened 1D numpy lists)
    vf_pooled: List[float]
    vt_pos_1: List[float]
    vt_pos_L_div_4: List[float]
    vt_pos_L_div_2: List[float]
    vt_pos_3L_div_4: List[float]
    vt_pos_L: List[float]
    qt_pos_1: List[float]
    qt_pos_L_div_4: List[float]
    qt_pos_L_div_2: List[float]
    qt_pos_3L_div_4: List[float]
    qt_pos_L: List[float]
    
    has_nan: bool = False
    has_inf: bool = False


class HALPActivationExtractor:
    """Activation extractor for the 11 HALP probe targets."""

    DEPTH_RATIOS = ["pos_1", "pos_L_div_4", "pos_L_div_2", "pos_3L_div_4", "pos_L"]

    def __init__(self, model_family: str, num_layers: int, hidden_dim: int):
        self.model_family = model_family.lower()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.tuple_indices = self._compute_hidden_state_tuple_indices(num_layers)

    @classmethod
    def _compute_hidden_state_tuple_indices(cls, L: int) -> Dict[str, int]:
        """Compute the exact tuple index in outputs.hidden_states.
        
        Since hidden_states[0] is the embedding layer, decoder layer l (1-based)
        is situated at hidden_states[l].
        """
        return {
            "pos_1": 1,
            "pos_L_div_4": max(1, L // 4),
            "pos_L_div_2": max(1, L // 2),
            "pos_3L_div_4": max(1, (3 * L) // 4),
            "pos_L": L, # equivalent to hidden_states[-1]
        }

    def extract_from_forward(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        vf_pooled_tensor: torch.Tensor,
        vt_pos: int,
        qt_pos: int,
        sample_id: str = "",
        dataset_id: str = "",
        partition: str = "dev_train",
    ) -> HALPExtractedSample:
        """Extract all 11 HALP targets from model forward pass activations.
        
        Args:
            hidden_states: outputs.hidden_states tuple of length L + 1.
            vf_pooled_tensor: Pooled vision tower representation [1, D_vis].
            vt_pos: Sequence index of final visual token.
            qt_pos: Sequence index of final query token.
        """
        assert len(hidden_states) == self.num_layers + 1, (
            f"Expected {self.num_layers + 1} hidden states (embedding + {self.num_layers} layers), "
            f"got {len(hidden_states)}"
        )

        extracted_vectors = {}
        has_nan = False
        has_inf = False

        # 1. Visual Feature (VF) probe target
        vf_flat = vf_pooled_tensor.squeeze().detach().cpu()
        if torch.isnan(vf_flat).any():
            has_nan = True
        if torch.isinf(vf_flat).any():
            has_inf = True
        extracted_vectors["vf_pooled"] = vf_flat.numpy().tolist()

        # 2. Vision Token (VT) and Query Token (QT) probe targets across 5 depth positions
        for depth_key, tuple_idx in self.tuple_indices.items():
            layer_hidden = hidden_states[tuple_idx] # [Batch, SeqLen, HiddenDim]
            
            # Bound check on token positions
            seq_len = layer_hidden.shape[1]
            safe_vt = min(vt_pos, seq_len - 1)
            safe_qt = min(qt_pos, seq_len - 1)

            vt_vec = layer_hidden[0, safe_vt, :].detach().cpu()
            qt_vec = layer_hidden[0, safe_qt, :].detach().cpu()

            if torch.isnan(vt_vec).any() or torch.isnan(qt_vec).any():
                has_nan = True
            if torch.isinf(vt_vec).any() or torch.isinf(qt_vec).any():
                has_inf = True

            extracted_vectors[f"vt_{depth_key}"] = vt_vec.numpy().tolist()
            extracted_vectors[f"qt_{depth_key}"] = qt_vec.numpy().tolist()

        return HALPExtractedSample(
            sample_id=sample_id,
            dataset_id=dataset_id,
            model_name=self.model_family,
            partition=partition,
            image_token_end_idx=vt_pos,
            query_token_end_idx=qt_pos,
            vf_pooled=extracted_vectors["vf_pooled"],
            vt_pos_1=extracted_vectors["vt_pos_1"],
            vt_pos_L_div_4=extracted_vectors["vt_pos_L_div_4"],
            vt_pos_L_div_2=extracted_vectors["vt_pos_L_div_2"],
            vt_pos_3L_div_4=extracted_vectors["vt_pos_3L_div_4"],
            vt_pos_L=extracted_vectors["vt_pos_L"],
            qt_pos_1=extracted_vectors["qt_pos_1"],
            qt_pos_L_div_4=extracted_vectors["qt_pos_L_div_4"],
            qt_pos_L_div_2=extracted_vectors["qt_pos_L_div_2"],
            qt_pos_3L_div_4=extracted_vectors["qt_pos_3L_div_4"],
            qt_pos_L=extracted_vectors["qt_pos_L"],
            has_nan=has_nan,
            has_inf=has_inf,
        )

