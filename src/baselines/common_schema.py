"""Common schemas and interfaces for XMCA-Net baseline benchmark evaluation.

Defines standardized data contracts for:
1. Detection Baselines (HALP, VisER, HalLoc, REFIND, XMCA)
2. Localization Baselines (HalLoc, REFIND, VisER, XMCA)
3. Mitigation Runners (Base VLM, OPERA, Re-Align)
4. Sensitivity & Context Verification (REFIND, Re-Align, XMCA)

Strict zero-fabrication contract: preserves original outputs without forced approximations.
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np


class BaselineDetectionOutput(BaseModel):
    """Standard output schema for hallucination detection baselines."""

    sample_id: str = Field(..., description="Unique benchmark sample identifier")
    dataset_id: str = Field(..., description="Dataset: pope, faithscore, infoseek, a_okvqa")
    model_name: str = Field(..., description="Evaluated backbone VLM")
    baseline_id: str = Field(..., description="Baseline identifier: halp, viser, halloc, refind, xmca")
    
    # Raw baseline score (higher indicates higher probability / risk of hallucination)
    raw_score: float = Field(..., description="Raw continuous baseline score")
    
    # Calibrated probability (calibrated strictly on validation set)
    calibrated_risk: Optional[float] = Field(None, description="Validation-calibrated risk in [0, 1]")
    
    # Metadata specific to baseline
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaselineLocalizationOutput(BaseModel):
    """Standard output schema for token/span hallucination localization baselines."""

    sample_id: str
    dataset_id: str
    model_name: str
    baseline_id: str
    
    # Tokenized sequence
    tokens: List[str] = Field(..., description="Generated text tokens")
    
    # Token-level hallucination risk scores
    token_scores: List[float] = Field(..., description="Continuous hallucination score per token")
    
    # Category-specific token scores (for HalLoc 4 heads: object, attribute, relation, scene)
    category_token_scores: Optional[Dict[str, List[float]]] = Field(
        None, description="Head-specific probabilities per token"
    )
    
    # Binary hallucinated token flags (thresholded on validation set)
    is_hallucinated_token: Optional[List[bool]] = None


class BaselineMitigationOutput(BaseModel):
    """Standard output schema for generation mitigation runners (Base VLM, OPERA, Re-Align)."""

    sample_id: str
    dataset_id: str
    model_name: str
    mitigation_id: str = Field(..., description="base_vlm, opera, realign")
    
    prompt: str
    image_id: str
    generated_text: str = Field(..., description="Response generated under this decoding/alignment regime")
    
    # Latency and efficiency telemetry
    generation_time_ms: float = Field(..., description="Wall-clock generation time in ms")
    tokens_generated: int = Field(..., description="Number of newly generated tokens")
    latency_ms_per_token: float = Field(..., description="Per-token generation latency")
    
    # Ground truth answer & evaluation
    ground_truth_label: Optional[int] = Field(None, description="0=factual, 1=hallucinated")
    accuracy_score: Optional[float] = Field(None, description="Downstream task accuracy (e.g. POPE / VQA accuracy)")
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextSensitivityOutput(BaseModel):
    """Standard output schema for context sensitivity & retrieval awareness (REFIND, Re-Align, XMCA)."""

    sample_id: str
    dataset_id: str
    model_name: str
    
    # Context Sensitivity Ratio (CSR) per token
    token_csr_scores: List[float] = Field(..., description="Token-level CSR values")
    
    # Aggregated sequence context dependence
    mean_csr: float = Field(..., description="Mean context sensitivity across tokens")
    min_csr: float = Field(..., description="Minimum context sensitivity (most context-independent token)")
    
    # Evidence divergence
    retrieval_reliance_score: float = Field(..., description="Overall sensitivity to retrieved evidence")

