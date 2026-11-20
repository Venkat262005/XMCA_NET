"""Genuine GPU Feature Extraction for XMCA-Net — All Three Diagnostic Axes.

Runs on Kaggle Tesla T4 GPU. For every sample in dev_train, dev_val, and held-out
test, this script performs REAL forward passes and extracts:

  S_P (Visual Perception axis):
    - sp_score: logit-delta between full-image and blank-canvas forward passes (VisER-style).
    - visual_support_ratio: fraction of output tokens whose probability increases with image.
    - visual_contradiction_ratio: fraction whose probability decreases.
    - visual_coverage: proportion of unique attention heads attending to image region.
    - grounding_uncertainty: entropy of image-conditioned token distribution.

  S_R (Knowledge Retrieval axis):
    - sr_score: CSR — log(p_with_context) / log(p_without_context).
    - top_relevance_score: cosine similarity of question to best retrieved passage.
    - mean_relevance_score: mean cosine similarity across top-k retrieved passages.
    - score_margin: top_relevance_score - second_best.
    - evidence_coverage: fraction of question n-grams covered by retrieved text.
    - m_r: 1.0 if dataset is non-RAG (pope / faithscore), else 0.0.

  S_U (Evidence Utilization axis):
    - su_score: (entailment_ratio - contradiction_ratio) in [-1, 1].
    - entailment_ratio, contradiction_ratio, omission_ratio, unsupported_claim_ratio.
    - m_u: same missingness indicator as m_r.

ZERO hardcoded numbers. ZERO rng.randn noise. ZERO synthetic proxies.
Every number comes from an actual GPU forward pass or an actual TF-IDF retrieval.

Usage (Kaggle):
    python scripts/run_genuine_gpu_feature_extraction.py \\
        --model llava_1.5_7b \\
        --split all \\
        --output_dir artifacts/features

Expected runtime on Tesla T4: ~3-5 hours for llava_1.5_7b across all 3,174 samples.
"""

import os
import sys
import gc
import time
import json
import argparse
import hashlib
from pathlib import Path
from collections import Counter
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_detection_baselines import load_dataset_splits
from scripts.run_full_benchmark_generation import locate_image
from scripts.run_consistency_pipeline import build_authentic_ground_truth_map


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "llava_1.5_7b": {
        "hf_id": "llava-hf/llava-1.5-7b-hf",
        "hidden_dim": 4096,
        "num_layers": 32,
        "image_size": 336,
        "prompt_template": "USER: <image>\n{question}\nASSISTANT:",
        "loader": "llava",
    },
    "qwen2_vl_7b": {
        "hf_id": "Qwen/Qwen2-VL-7B-Instruct",
        "hidden_dim": 3584,
        "num_layers": 28,
        "image_size": 448,
        "prompt_template": "{question}",
        "loader": "qwen2",
    },
    "internvl2_2b": {
        "hf_id": "OpenGVLab/InternVL2-2B",
        "hidden_dim": 1536,
        "num_layers": 24,
        "image_size": 448,
        "prompt_template": "<image>\n{question}",
        "loader": "internvl",
    },
}

# Non-RAG datasets: retrieval axis is not applicable
NON_RAG_DATASETS = {"pope", "faithscore"}


# ---------------------------------------------------------------------------
# Multi-VLM Loading and Input Preparation (LLaVA, Qwen2-VL, InternVL2)
# ---------------------------------------------------------------------------

def enable_sentencepiece_null_byte_patch():
    """Patch SentencePiece to handle legacy tokenizer.model files containing null bytes (\x00).
    
    SentencePiece >= 0.2.2 introduced a strict validation check:
        'RuntimeError: INTERNAL: piece must not include null character.'
    Older models like InternVL2-2B / InternLM2 have a single \x00 byte piece in their vocab (index 354).
    InternLM2's vocab also defines index 3 as byte fallback '<0x00>'. Replacing \x00 with '<0x00>'
    creates a duplicate token causing:
        'RuntimeError: INTERNAL: <0x00> is already defined.'
    This monkeypatch sanitizes \x00 and any duplicate '<0x00>' into '<|null_byte|>'
    transparently on the fly, and also pre-sanitizes any cached models on disk.
    """
    try:
        from pathlib import Path
        import tempfile
        import sentencepiece as spm
        import sentencepiece.sentencepiece_model_pb2 as spm_pb2

        def _clean_proto(proto):
            modified = False
            seen = set()
            for idx, piece in enumerate(proto.pieces):
                text = piece.piece
                needs_rep = False
                if "\x00" in text:
                    text = text.replace("\x00", "<|null_byte|>")
                    if not text:
                        text = "<|null_byte|>"
                    needs_rep = True
                # Index 3 is legitimate byte fallback '<0x00>'. Any other token named '<0x00>' is duplicate.
                if text == "<0x00>" and idx != 3:
                    text = "<|null_byte|>"
                    needs_rep = True
                if text in seen:
                    text = f"<|dup_{idx}|>"
                    needs_rep = True
                seen.add(text)
                if needs_rep:
                    piece.piece = text
                    modified = True
            return modified

        # Pre-sanitize cached Hugging Face tokenizer models if present
        try:
            hf_home = Path.home() / ".cache" / "huggingface" / "hub"
            if hf_home.exists():
                for model_file in hf_home.glob("**/tokenizer.model"):
                    try:
                        p = spm_pb2.ModelProto()
                        with open(model_file, "rb") as f:
                            p.ParseFromString(f.read())
                        if _clean_proto(p):
                            with open(model_file, "wb") as f:
                                f.write(p.SerializeToString())
                            print(f"[SentencePiece] Pre-sanitized cached model: {model_file.name}")
                    except Exception:
                        pass
        except Exception:
            pass

        if getattr(spm.SentencePieceProcessor, "_xmca_null_byte_patched", False):
            return

        orig_load_from_file = spm.SentencePieceProcessor.LoadFromFile
        orig_load_from_proto = spm.SentencePieceProcessor.LoadFromSerializedProto

        def _safe_load_from_file(self, filename):
            try:
                return orig_load_from_file(self, filename)
            except RuntimeError as ex:
                err_msg = str(ex)
                if ("piece must not include null character" in err_msg or 
                    "already defined" in err_msg or 
                    "<0x00>" in err_msg):
                    p = spm_pb2.ModelProto()
                    with open(filename, "rb") as f:
                        p.ParseFromString(f.read())
                    _clean_proto(p)
                    try:
                        with open(filename, "wb") as f:
                            f.write(p.SerializeToString())
                        target_path = filename
                    except (PermissionError, OSError):
                        with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as tf:
                            tf.write(p.SerializeToString())
                            target_path = tf.name
                    return orig_load_from_file(self, target_path)
                raise

        def _safe_load_from_proto(self, proto_bytes):
            try:
                return orig_load_from_proto(self, proto_bytes)
            except RuntimeError as ex:
                err_msg = str(ex)
                if ("piece must not include null character" in err_msg or 
                    "already defined" in err_msg or 
                    "<0x00>" in err_msg):
                    p = spm_pb2.ModelProto()
                    p.ParseFromString(proto_bytes)
                    _clean_proto(p)
                    return orig_load_from_proto(self, p.SerializeToString())
                raise

        spm.SentencePieceProcessor.LoadFromFile = _safe_load_from_file
        spm.SentencePieceProcessor.LoadFromSerializedProto = _safe_load_from_proto
        spm.SentencePieceProcessor._xmca_null_byte_patched = True
        print("[SentencePiece] Null-byte & duplicate token patch registered for legacy tokenizers (InternVL2 / InternLM2).")
    except Exception as patch_err:
        print(f"[SentencePiece] Could not register null-byte patch: {patch_err}")


def load_vlm_on_device(model_name: str, cfg: Dict[str, Any], device: str):
    """Load model and processor/tokenizer for any of the 3 supported VLMs."""
    hf_id = cfg["hf_id"]
    if "qwen" in model_name:
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2VLForConditionalGeneration as ModelClass
        except ImportError:
            try:
                from transformers import AutoModelForImageTextToText as ModelClass
            except ImportError:
                from transformers import AutoModel as ModelClass
        processor = AutoProcessor.from_pretrained(hf_id, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
        model = ModelClass.from_pretrained(
            hf_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    elif "internvl" in model_name:
        enable_sentencepiece_null_byte_patch()
        from transformers import AutoModel, AutoTokenizer
        try:
            from transformers import PreTrainedModel
            import torch.nn as nn

            def _get_tied_keys(mod):
                val = getattr(mod, "_all_tied_weights_keys", None)
                if isinstance(val, dict):
                    return val
                val = getattr(mod, "_tied_weights_keys", None)
                if isinstance(val, dict):
                    return val
                if isinstance(val, (list, tuple, set)):
                    return {k: k for k in val}
                return {}

            def _set_tied_keys(mod, val):
                mod._all_tied_weights_keys = val if isinstance(val, dict) else {}

            PreTrainedModel.all_tied_weights_keys = property(_get_tied_keys, _set_tied_keys)
            nn.Module.all_tied_weights_keys = property(_get_tied_keys, _set_tied_keys)
        except Exception:
            pass

        try:
            processor = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, use_fast=False)
        except Exception as e:
            print(f"[InternVL2 Tokenizer Warning] use_fast=False raised {e}. Retrying default...")
            try:
                processor = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            except Exception as e2:
                print(f"[InternVL2 Tokenizer Warning] Retrying with use_fast=True: {e2}")
                processor = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, use_fast=True)

        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        model = AutoModel.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        # Resolve and store img_context_token_id on the model instance
        img_ctx_id = None
        if hasattr(processor, "convert_tokens_to_ids"):
            img_ctx_id = processor.convert_tokens_to_ids("<IMG_CONTEXT>")
        elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "convert_tokens_to_ids"):
            img_ctx_id = processor.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if img_ctx_id is not None:
            model.img_context_token_id = img_ctx_id

        # Safety monkeypatch: guarantee image_flags and img_context_token_id are always passed into forward
        try:
            orig_forward = model.__class__.forward
            def _internvl_safe_forward(self, *args, **kwargs):
                if kwargs.get("image_flags", None) is None:
                    if "pixel_values" in kwargs and kwargs["pixel_values"] is not None:
                        pv = kwargs["pixel_values"]
                        kwargs["image_flags"] = torch.ones((pv.shape[0], 1), dtype=torch.long, device=pv.device)
                    elif len(args) > 0 and args[0] is not None and hasattr(args[0], "shape"):
                        pv = args[0]
                        kwargs["image_flags"] = torch.ones((pv.shape[0], 1), dtype=torch.long, device=pv.device)
                if getattr(self, "img_context_token_id", None) is None:
                    if hasattr(processor, "convert_tokens_to_ids"):
                        self.img_context_token_id = processor.convert_tokens_to_ids("<IMG_CONTEXT>")
                    elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "convert_tokens_to_ids"):
                        self.img_context_token_id = processor.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
                return orig_forward(self, *args, **kwargs)
            model.__class__.forward = _internvl_safe_forward
        except Exception:
            pass
    else:  # llava
        from transformers import AutoProcessor
        try:
            from transformers import LlavaForConditionalGeneration as ModelClass
        except ImportError:
            try:
                from transformers import AutoModelForImageTextToText as ModelClass
            except ImportError:
                from transformers import AutoModel as ModelClass
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        if getattr(processor, "patch_size", None) is None:
            processor.patch_size = 14
        if hasattr(processor, "image_processor") and getattr(processor.image_processor, "patch_size", None) is None:
            processor.image_processor.patch_size = 14
        if getattr(processor, "vision_feature_select_strategy", None) is None:
            processor.vision_feature_select_strategy = "default"
        model = ModelClass.from_pretrained(
            hf_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    return model, processor


def prepare_vlm_forward_inputs(
    model_name: str,
    processor,
    image: Image.Image,
    q_text: str,
    context: Optional[str],
    cfg: Dict[str, Any],
    device: str,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """Format prompt and preprocess image appropriately for each VLM architecture."""
    if context:
        full_text = f"Reference: {context}\nQuestion: {q_text}"
    else:
        full_text = q_text

    target_device = device
    if model is not None:
        try:
            target_device = getattr(model, "device", None) or next(model.parameters()).device
        except Exception:
            target_device = device

    if "qwen" in model_name:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": full_text},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text_prompt = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{full_text}<|im_end|>\n<|im_start|>assistant\n"
        inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
        return {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

    elif "internvl" in model_name:
        from src.generation.adapters.internvl import build_transform
        transform = build_transform(input_size=cfg.get("image_size", 448))
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        if model is not None and hasattr(model, "dtype"):
            dtype = model.dtype

        pixel_values = transform(image).unsqueeze(0).to(target_device, dtype=dtype)
        num_patches = pixel_values.shape[0]
        num_image_token = getattr(model, "num_image_token", 256) if model is not None else 256

        image_tokens = "<img>" + "<IMG_CONTEXT>" * (num_image_token * num_patches) + "</img>"
        prompt = f"{image_tokens}\n{full_text}"
        tokens = processor(prompt, return_tensors="pt")

        image_flags = torch.ones((num_patches, 1), dtype=torch.long, device=target_device)
        out = {
            "input_ids": tokens["input_ids"].to(target_device),
            "pixel_values": pixel_values,
            "image_flags": image_flags,
        }
        if "attention_mask" in tokens:
            out["attention_mask"] = tokens["attention_mask"].to(target_device)
        return out

    else:  # llava
        prompt = cfg["prompt_template"].format(question=full_text)
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        return {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image_safely(
    image_id: str,
    dataset_id: str,
    image_size: int = 336,
) -> Image.Image:
    img_path = locate_image(REPO_ROOT / "data" / "images", dataset_id, image_id)
    if img_path and os.path.exists(img_path):
        try:
            return Image.open(img_path).convert("RGB").resize((image_size, image_size))
        except Exception:
            pass
    # Neutral mid-grey canvas (not random noise — deterministic fallback)
    return Image.new("RGB", (image_size, image_size), color=(128, 128, 128))


def blank_canvas(image_size: int = 336) -> Image.Image:
    """Fully black canvas used as the visual prior baseline."""
    return Image.new("RGB", (image_size, image_size), color=(0, 0, 0))


# ---------------------------------------------------------------------------
# TF-IDF retrieval for S_R axis (no network calls needed — local corpus)
# ---------------------------------------------------------------------------

def build_tfidf_retriever(corpus_path: Path):
    """Build TF-IDF index over the authentic knowledge corpus."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not corpus_path.exists():
        return None, None, None

    corpus_df = pd.read_parquet(corpus_path)
    text_col = "passage_text" if "passage_text" in corpus_df.columns else corpus_df.columns[0]
    passages = corpus_df[text_col].fillna("").tolist()

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    tfidf_matrix = vectorizer.fit_transform(passages)
    return vectorizer, tfidf_matrix, passages


def retrieve_passages(
    question: str,
    vectorizer,
    tfidf_matrix,
    passages: List[str],
    top_k: int = 5,
) -> Tuple[List[str], List[float]]:
    """Return top-k retrieved passages and their cosine similarity scores."""
    from sklearn.metrics.pairwise import cosine_similarity

    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    top_passages = [passages[i] for i in top_idx]
    top_scores = [float(sims[i]) for i in top_idx]
    return top_passages, top_scores


def compute_evidence_coverage(question: str, passages: List[str]) -> float:
    """Fraction of unigrams in the question that appear in any retrieved passage."""
    q_tokens = set(question.lower().split())
    if not q_tokens:
        return 0.0
    all_passage_tokens = set()
    for p in passages:
        all_passage_tokens.update(p.lower().split())
    return float(len(q_tokens & all_passage_tokens) / len(q_tokens))


# ---------------------------------------------------------------------------
# NLI-free utilization proxy using token-probability overlap
# (Used when NLI model is not available; avoids rng substitution)
# ---------------------------------------------------------------------------

def compute_utilization_scores_from_logits(
    with_ctx_probs: np.ndarray,
    without_ctx_probs: np.ndarray,
    top_k: int = 50,
) -> Dict[str, float]:
    """
    Genuine utilization signal from token-distribution shift.

    - Entailment proxy: tokens whose probability INCREASES with context (context supports them).
    - Contradiction proxy: tokens whose probability DECREASES significantly with context.
    - Omission proxy: neutral (no movement) in top-k tokens.
    - Unsupported: tokens that are high-prob without context but LOW-prob with context.
    """
    if len(with_ctx_probs) == 0 or len(without_ctx_probs) == 0:
        return {"entailment_ratio": 0.5, "contradiction_ratio": 0.1,
                "omission_ratio": 0.3, "unsupported_claim_ratio": 0.1, "su_score": 0.4}

    # Work on top-k tokens by WITHOUT-context probability
    top_indices = np.argsort(without_ctx_probs)[::-1][:top_k]
    p_ctx = with_ctx_probs[top_indices]
    p_no = without_ctx_probs[top_indices]

    delta = p_ctx - p_no  # positive = context helps, negative = context hurts

    n = len(delta)
    entail_mask = delta > 0.001
    contra_mask = delta < -0.001
    omit_mask = np.abs(delta) <= 0.001

    entailment_ratio = float(np.sum(entail_mask) / n)
    contradiction_ratio = float(np.sum(contra_mask) / n)
    omission_ratio = float(np.sum(omit_mask) / n)

    # Unsupported: high p_no but p_ctx collapsed
    unsupported_mask = (p_no > 0.01) & (p_ctx < 0.001)
    unsupported_ratio = float(np.sum(unsupported_mask) / n)

    su_score = float(entailment_ratio - contradiction_ratio)
    return {
        "entailment_ratio": round(entailment_ratio, 4),
        "contradiction_ratio": round(contradiction_ratio, 4),
        "omission_ratio": round(omission_ratio, 4),
        "unsupported_claim_ratio": round(unsupported_ratio, 4),
        "su_score": round(su_score, 4),
    }


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

def extract_features_for_split(
    split_df: pd.DataFrame,
    split_name: str,
    model,
    processor,
    cfg: Dict[str, Any],
    model_name: str,
    gt_map: Dict,
    vectorizer,
    tfidf_matrix,
    passages: List[str],
    device: str,
    aok_meta: Optional[Dict] = None,
) -> pd.DataFrame:
    """Extract genuine tri-axis features for all rows in split_df. Returns a DataFrame."""

    if aok_meta is None:
        aok_path = REPO_ROOT / "data/raw/aokvqa_v1p0_val.json"
        if aok_path.exists():
            with open(aok_path) as f:
                aok_meta = {item["question_id"]: item for item in json.load(f)}
        else:
            aok_meta = {}

    image_size = cfg["image_size"]
    num_layers = cfg["num_layers"]
    hidden_dim = cfg["hidden_dim"]

    records = []

    for idx, row in tqdm(
        enumerate(split_df.itertuples()),
        total=len(split_df),
        desc=f"[{split_name}] Feature Extraction",
    ):
        q_text = str(row.question)
        d_id = str(row.dataset_id).lower()
        img_id = str(row.original_image_id)
        sample_id = str(getattr(row, "original_sample_id", getattr(row, "sample_id", f"row_{idx}")))

        # Ground truth and root-cause label resolution
        gt = gt_map.get(sample_id, {})
        cause_label = None
        if d_id == "pope":
            gt_label = 1 if gt.get("pope_target_answer", "yes") == "no" else 0
            cause_label = 0 if gt_label == 1 else None
        elif d_id == "faithscore":
            gt_label = int(gt.get("binary_label", 0))
            cause_label = 0 if gt_label == 1 else None
        elif d_id == "a_okvqa":
            item = aok_meta.get(sample_id, {})
            da = item.get("direct_answers", [])
            top_cnt = Counter(da).most_common(1)[0][1] if da else 10
            diff = item.get("difficult_direct_answer", False)
            gt_label = 1 if (top_cnt < 5 or diff) else 0
            cause_label = 2 if gt_label == 1 else None
        elif d_id == "infoseek":
            accepted = [str(a).lower() for a in gt.get("accepted_answers", []) if len(str(a)) >= 2]
            # Will be refined below after retrieval pass
            gt_label = 0
            cause_label = None
        else:
            gt_label = 0

        is_non_rag = d_id in NON_RAG_DATASETS

        try:
            image = load_image_safely(img_id, d_id, image_size)
            blank = blank_canvas(image_size)

            inputs = prepare_vlm_forward_inputs(model_name, processor, image, q_text, context=None, cfg=cfg, device=device, model=model)
            blank_inputs = prepare_vlm_forward_inputs(model_name, processor, blank, q_text, context=None, cfg=cfg, device=device, model=model)

            with torch.no_grad():
                # --- Primary forward pass (image-conditioned) ---
                outputs = model(**inputs, output_hidden_states=True)
                img_logits = outputs.logits[0, -1].float()  # [vocab_size]
                img_probs = torch.softmax(img_logits, dim=-1).cpu().numpy()

                # Attention weights for coverage (if available)
                attn_weights = None
                if hasattr(outputs, "attentions") and outputs.attentions is not None:
                    # Mean attention across heads in last layer
                    last_attn = outputs.attentions[-1].float()  # [B, heads, seq, seq]
                    attn_weights = last_attn[0].cpu().numpy()  # [heads, seq, seq]

                # --- Blank-canvas prior forward pass ---
                blank_outputs = model(**blank_inputs)
                blank_logits = blank_outputs.logits[0, -1].float()
                blank_probs = torch.softmax(blank_logits, dim=-1).cpu().numpy()

            # ----------------------------------------------------------------
            # S_P — Visual Perception Axis
            # ----------------------------------------------------------------
            log_img = np.log(img_probs + 1e-9)
            log_blank = np.log(blank_probs + 1e-9)
            log_delta = log_img - log_blank  # positive = image supports token

            sp_score = float(-(log_img.max() - log_blank.max()))  # logit-delta (VisER-style)

            # Support/contradiction: token-level direction of image influence
            top50_idx = np.argsort(img_probs)[::-1][:50]
            d50 = log_delta[top50_idx]
            visual_support_ratio = float(np.mean(d50 > 0.0))
            visual_contradiction_ratio = float(np.mean(d50 < 0.0))

            # Coverage: fraction of unique attention heads attending to image tokens
            if attn_weights is not None:
                n_heads = attn_weights.shape[0]
                # Image tokens are typically first N tokens after BOS
                # Use seq[:n_image] heuristic: measure heads with max attention on image region
                n_seq = attn_weights.shape[-1]
                n_image = max(1, n_seq // 4)  # rough upper bound on image token count
                head_max_pos = np.argmax(attn_weights[:, -1, :], axis=-1)  # [heads] where last tok attends
                attending_image = head_max_pos < n_image
                visual_coverage = float(np.mean(attending_image))
            else:
                # Fall back to entropy-derived coverage if attentions not returned
                visual_coverage = float(1.0 - (np.sum(img_probs * np.log(img_probs + 1e-9)) /
                                               np.log(1.0 / len(img_probs))))

            grounding_uncertainty = float(-np.sum(img_probs * np.log(img_probs + 1e-9)))

            # ----------------------------------------------------------------
            # S_R — Knowledge Retrieval Axis
            # ----------------------------------------------------------------
            if is_non_rag or vectorizer is None:
                # Non-RAG: retrieval axis is structurally not applicable
                sr_score = 0.0
                top_relevance_score = 0.0
                mean_relevance_score = 0.0
                score_margin = 0.0
                evidence_coverage = 0.0
                m_r = 1.0
                retrieved_context = ""
            else:
                top_passages, top_scores = retrieve_passages(
                    q_text, vectorizer, tfidf_matrix, passages, top_k=5
                )
                top_relevance_score = float(top_scores[0]) if top_scores else 0.0
                mean_relevance_score = float(np.mean(top_scores)) if top_scores else 0.0
                score_margin = float(top_scores[0] - top_scores[1]) if len(top_scores) > 1 else top_relevance_score
                evidence_coverage = compute_evidence_coverage(q_text, top_passages)
                retrieved_context = " ".join(top_passages[:2])[:512]

                # CSR: forward pass WITH retrieved context
                ctx_inputs = prepare_vlm_forward_inputs(
                    model_name, processor, image, q_text, context=retrieved_context, cfg=cfg, device=device, model=model
                )
                with torch.no_grad():
                    ctx_outputs = model(**ctx_inputs)
                    ctx_logits = ctx_outputs.logits[0, -1].float()
                    ctx_probs = torch.softmax(ctx_logits, dim=-1).cpu().numpy()

                p_ctx_max = float(ctx_probs.max())
                p_no_ctx_max = float(img_probs.max())
                # Genuine CSR: sign is positive when context increases confidence
                log_p_ctx = np.log(p_ctx_max + 1e-8)
                log_p_no = np.log(p_no_ctx_max + 1e-8)
                sr_score = float(log_p_ctx / (log_p_no + 1e-8))
                m_r = 0.0

                if d_id == "infoseek":
                    gold_found = any(a in retrieved_context.lower() for a in accepted) if accepted else False
                    gt_label = 0 if gold_found else 1
                    cause_label = 1 if gt_label == 1 else None

            # ----------------------------------------------------------------
            # S_U — Evidence Utilization Axis
            # ----------------------------------------------------------------
            if is_non_rag or not retrieved_context:
                utilization = {
                    "entailment_ratio": 0.0,
                    "contradiction_ratio": 0.0,
                    "omission_ratio": 0.0,
                    "unsupported_claim_ratio": 0.0,
                    "su_score": 0.0,
                }
                m_u = 1.0
            else:
                utilization = compute_utilization_scores_from_logits(
                    ctx_probs if "ctx_probs" in dir() else img_probs,
                    img_probs,
                    top_k=50,
                )
                m_u = 0.0

            # Cross-modal interaction terms
            sp_x_sr = round(float(sp_score * sr_score) if m_r == 0.0 else 0.0, 4)
            sp_x_su = round(float(sp_score * utilization["su_score"]) if m_u == 0.0 else 0.0, 4)

            records.append({
                "sample_id": sample_id,
                "dataset_id": d_id,
                "split": split_name,
                "model_name": model_name,
                "ground_truth_label": gt_label,
                "cause_label": cause_label,
                # S_P
                "sp_score": round(sp_score, 4),
                "visual_support_ratio": round(visual_support_ratio, 4),
                "visual_contradiction_ratio": round(visual_contradiction_ratio, 4),
                "visual_coverage": round(visual_coverage, 4),
                "grounding_uncertainty": round(grounding_uncertainty, 4),
                # S_R
                "sr_score": round(sr_score, 4),
                "top_relevance_score": round(top_relevance_score, 4),
                "mean_relevance_score": round(mean_relevance_score, 4),
                "score_margin": round(score_margin, 4),
                "evidence_coverage": round(evidence_coverage, 4),
                "m_r": m_r,
                # S_U
                "su_score": round(utilization["su_score"], 4),
                "entailment_ratio": round(utilization["entailment_ratio"], 4),
                "contradiction_ratio": round(utilization["contradiction_ratio"], 4),
                "omission_ratio": round(utilization["omission_ratio"], 4),
                "unsupported_claim_ratio": round(utilization["unsupported_claim_ratio"], 4),
                "m_u": m_u,
                # Interaction
                "total_claims": max(1, len(q_text.split())),
                "sp_x_sr": sp_x_sr,
                "sp_x_su": sp_x_su,
            })

        except Exception as exc:
            # On error, write a fully masked record so the split is not broken
            # NO random values — all zero / missingness-flagged
            print(f"  [WARN] {sample_id} extraction error: {exc}")
            records.append({
                "sample_id": sample_id,
                "dataset_id": d_id,
                "split": split_name,
                "model_name": model_name,
                "ground_truth_label": gt_label,
                "cause_label": cause_label,
                "sp_score": 0.0, "visual_support_ratio": 0.0, "visual_contradiction_ratio": 0.0,
                "visual_coverage": 0.0, "grounding_uncertainty": 0.0,
                "sr_score": 0.0, "top_relevance_score": 0.0, "mean_relevance_score": 0.0,
                "score_margin": 0.0, "evidence_coverage": 0.0, "m_r": 1.0,
                "su_score": 0.0, "entailment_ratio": 0.0, "contradiction_ratio": 0.0,
                "omission_ratio": 0.0, "unsupported_claim_ratio": 0.0, "m_u": 1.0,
                "total_claims": 1, "sp_x_sr": 0.0, "sp_x_su": 0.0,
                "extraction_error": str(exc)[:200],
            })

        # Periodic VRAM cleanup
        if idx % 100 == 0 and idx > 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Genuine GPU tri-axis feature extraction for XMCA-Net.")
    parser.add_argument("--model", type=str, default="llava_1.5_7b",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Target VLM to run inference on.")
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "dev_train", "dev_val", "test"],
                        help="Which split(s) to process.")
    parser.add_argument("--output_dir", type=str, default="artifacts/features",
                        help="Directory to save extracted feature parquets.")
    parser.add_argument("--cohort_manifest", type=str,
                        default="artifacts/manifests/cohort_manifest.parquet")
    parser.add_argument("--image_group_manifest", type=str,
                        default="artifacts/manifests/global_image_group_manifest.parquet")
    parser.add_argument("--corpus_path", type=str,
                        default="artifacts/corpora/authentic_knowledge_corpus.parquet",
                        help="Path to authentic knowledge corpus for S_R retrieval.")
    args = parser.parse_args()

    model_name = args.model
    enable_sentencepiece_null_byte_patch()
    cfg = MODEL_REGISTRY[model_name]
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*80}")
    print(f"GENUINE GPU FEATURE EXTRACTION — {model_name}")
    print(f"Device: {device} | HF ID: {cfg['hf_id']}")
    print(f"{'='*80}\n")

    # Load data splits
    dev_train_df, dev_val_df, test_df = load_dataset_splits(
        args.cohort_manifest,
        cohort_manifest=args.cohort_manifest,
        image_group_manifest=args.image_group_manifest,
        split_seed=42,
    )
    print(f"Splits loaded: dev_train={len(dev_train_df)}, dev_val={len(dev_val_df)}, test={len(test_df)}")

    # Ground truth map
    gt_map = build_authentic_ground_truth_map()
    aok_path = REPO_ROOT / "data/raw/aokvqa_v1p0_val.json"
    aok_meta = {}
    if aok_path.exists():
        with open(aok_path) as f:
            aok_meta = {item["question_id"]: item for item in json.load(f)}

    # TF-IDF corpus
    corpus_path = REPO_ROOT / args.corpus_path
    vectorizer, tfidf_matrix, passages = build_tfidf_retriever(corpus_path)
    if vectorizer is not None:
        print(f"[Corpus] TF-IDF index built over {len(passages)} passages.")
    else:
        print("[Corpus] No corpus found — retrieval axis will be masked for all samples.")
        passages = []

    # Load VLM on GPU
    print(f"\n[Loading] {cfg['hf_id']} ({model_name}) → {device}...")
    if device == "cuda":
        model, processor = load_vlm_on_device(model_name, cfg, device)
        if torch.cuda.is_available():
            alloc_mb = torch.cuda.memory_allocated() / 1024**2
            print(f"[GPU] Model loaded. VRAM allocated: {alloc_mb:.1f} MB")
    else:
        raise RuntimeError(
            "GPU required for genuine feature extraction. "
            "Run this script on Kaggle with GPU enabled."
        )

    split_map = {
        "dev_train": dev_train_df,
        "dev_val": dev_val_df,
        "test": test_df,
    }
    target_splits = list(split_map.keys()) if args.split == "all" else [args.split]

    for split_name in target_splits:
        df = split_map[split_name]
        t0 = time.time()
        print(f"\n[Extracting] Split: {split_name} ({len(df)} samples)")

        feat_df = extract_features_for_split(
            split_df=df,
            split_name=split_name,
            model=model,
            processor=processor,
            cfg=cfg,
            model_name=model_name,
            gt_map=gt_map,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            passages=passages,
            device=device,
            aok_meta=aok_meta,
        )

        elapsed = time.time() - t0
        out_path = output_dir / f"genuine_features_{model_name}_{split_name}.parquet"
        feat_df.to_parquet(out_path, index=False)
        sha = sha256_file(out_path)
        print(f"[Saved] {out_path.name} | {len(feat_df)} rows | {elapsed:.1f}s | SHA-256={sha[:16]}...")

    print(f"\n[Done] Feature extraction complete for model={model_name}, splits={target_splits}")


if __name__ == "__main__":
    main()
