"""Re-Align: Aligning Vision-Language Models via Retrieval-Augmented Direct Preference Optimization (2025).

Official methodology from taco-group/Re-Align:
An alignment framework that mitigates object hallucinations in VLMs via image retrieval:
1. Dual Preference Dataset Construction:
   Pairs each input image I_0 with a retrieved similar image I_ret, creating preference pairs
   (y_w, y_l) where the chosen response y_w is factually grounded in I_0, and the rejected
   response y_l contains plausible hallucinations injected via entities from I_ret.
2. Retrieval-Augmented Direct Preference Optimization (rDPO):
   Optimizes standard DPO combined with an explicit visual preference objective:
   L_rDPO = L_DPO(I_0, q, y_w, y_l) + lambda_vis * L_vis_pref(I_0, I_ret, q, y_w, y_l)

Evaluation Runner:
Compares the base unaligned VLM against the Re-Align rDPO-aligned VLM checkpoint
on identical test prompts, measuring hallucination reduction and VQA accuracy.
"""

from typing import List, Dict, Any, Optional, Tuple
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from src.baselines.common_schema import BaselineMitigationOutput


class RDPOLoss(nn.Module):
    """Retrieval-augmented Direct Preference Optimization (rDPO) loss."""

    def __init__(self, beta: float = 0.1, lambda_vis: float = 0.5):
        super().__init__()
        self.beta = beta
        self.lambda_vis = lambda_vis

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        policy_vis_pref_logps: Optional[torch.Tensor] = None,
        ref_vis_pref_logps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute rDPO loss combining standard DPO and visual preference objective."""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        dpo_logits = self.beta * (pi_logratios - ref_logratios)
        dpo_loss = -F.logsigmoid(dpo_logits).mean()

        if policy_vis_pref_logps is not None and ref_vis_pref_logps is not None:
            vis_pi_ratios = policy_chosen_logps - policy_vis_pref_logps
            vis_ref_ratios = ref_chosen_logps - ref_vis_pref_logps
            vis_logits = self.beta * (vis_pi_ratios - vis_ref_ratios)
            vis_loss = -F.logsigmoid(vis_logits).mean()
            return dpo_loss + self.lambda_vis * vis_loss

        return dpo_loss


class ReAlignMitigationRunner:
    """Runner for evaluating Re-Align aligned VLM generation against Base VLM."""

    def __init__(self, aligned_model_path: Optional[str] = None):
        self.aligned_model_path = aligned_model_path

    def evaluate_generation(
        self,
        model: Any,
        processor: Any,
        prompt: str,
        image: Any,
        sample_id: str = "",
        dataset_id: str = "",
        model_name: str = "",
        max_new_tokens: int = 128,
    ) -> BaselineMitigationOutput:
        """Generate response from Re-Align aligned VLM and record latency and accuracy."""
        start_time = time.perf_counter()

        if hasattr(processor, "tokenizer"):
            inputs = processor(text=prompt, images=image, return_tensors="pt")
        else:
            inputs = processor(prompt, image, return_tensors="pt")

        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if hasattr(processor, "decode"):
            gen_text = processor.decode(outputs[0], skip_special_tokens=True)
        elif hasattr(processor, "tokenizer"):
            gen_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            gen_text = str(outputs[0])

        new_tokens = outputs.shape[1] - inputs.get("input_ids", outputs).shape[1]
        new_tokens = max(1, new_tokens)
        ms_per_token = elapsed_ms / new_tokens

        return BaselineMitigationOutput(
            sample_id=sample_id,
            dataset_id=dataset_id,
            model_name=model_name,
            mitigation_id="realign",
            prompt=prompt,
            image_id="",
            generated_text=gen_text,
            generation_time_ms=elapsed_ms,
            tokens_generated=new_tokens,
            latency_ms_per_token=ms_per_token,
            metadata={"aligned_checkpoint": self.aligned_model_path or "in-memory"},
        )

