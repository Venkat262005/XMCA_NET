# XMCA-Net: VLM Independence & Comparative Protocol

This document outlines the architectural independence and fair-comparison protocol for the three Vision-Language Model (VLM) families evaluated in XMCA-Net.

---

## 1. Selected VLM Families

XMCA-Net evaluates three structurally distinct open-source VLM families:

| Model Family | Specific Model ID | Pinned Git Revision | Visual Backbone | Language Backbone |
|---|---|---|---|---|
| **LLaVA** | `llava-hf/llava-1.5-7b-hf` | `a272c74` | CLIP ViT-L/14 @ 336px | Vicuna-7B (LLaMA-based) |
| **Qwen2-VL** | `Qwen/Qwen2-VL-7B-Instruct` | `847be15` | Native Dynamic ViT | Qwen2-7B |
| **InternVL** | `OpenGVLab/InternVL2-8B` | `9826f68` | InternViT-6B | InternLM2.5-7B / Yi |

---

## 2. Architectural Independence

To ensure findings reflect generalizable consistency dynamics rather than model-specific idiosyncrasies, the three families span distinct design paradigms:

1. **Visual Feature Representation**:
   - LLaVA uses standard static patch resolution from a pretrained CLIP encoder with a 2-layer MLP projection.
   - Qwen2-VL processes arbitrary aspect ratios dynamically with dynamic resolution tokens.
   - InternVL leverages a high-capacity custom 6B visual transformer (InternViT) with progressive alignment.

2. **Cross-Attention & Token Routing**:
   - Each model routes visual tokens into the language decoder using distinct attention topologies and token downsampling strategies.
   - Diagnostics ($S_P, S_R, S_U$) are extracted natively from each model's internal representation without altering the backbone weights.

---

## 3. Strict Comparative Constraints

1. **Commit Revision Pinning**: Every model is pinned to an explicit git commit SHA. Floating `latest` or unpinned repository branches are strictly prohibited by configuration validation (`src/utils/config.py`).
2. **Identical Generation Inputs**: Each model receives the identical matched prompt and preprocessed image.
3. **Deterministic Generation Parameters**:
   - `temperature = 0.0` (Greedy decoding)
   - `do_sample = false`
   - `max_new_tokens = 128`
4. **Isolated Artifact Storage**:
   - Generations are stored independently in `artifacts/generations/{model_key}/{dataset_name}_{tier}.jsonl`.
   - Manifest validation (`src/utils/manifest.py`) asserts that the exact same sample IDs exist across all three model outputs before any comparative metrics are computed.
5. **No Cross-Model Contamination**: Models are loaded and executed sequentially. At no point are outputs or intermediate states of one model exposed to another during inference.
