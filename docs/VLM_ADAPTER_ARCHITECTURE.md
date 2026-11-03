# XMCA-Net: VLM Adapter Architecture & Dual-T4 Protocol

This document outlines the architectural implementation of the Vision-Language Model (VLM) adapters in XMCA-Net, establishing a unified interface across three distinct open-source model families while strictly adhering to hardware constraints and research reproducibility standards.

---

## 1. Architectural Overview

XMCA-Net diagnoses perception, retrieval, and utilization consistency across three distinct VLM families:

| Model Family | Specific Model ID | Pinned Git Revision | Architecture Class | Precision | Primary Parameter Scale |
|:---|:---|:---|:---|:---|:---|
| **LLaVA** | `llava-hf/llava-1.5-7b-hf` | `a272c74` | `LlavaForConditionalGeneration` | FP16 | 7.06B |
| **Qwen2-VL** | `Qwen/Qwen2-VL-7B-Instruct` | `847be15` | `Qwen2VLForConditionalGeneration` | FP16 | 7.61B |
| **InternVL** | `OpenGVLab/InternVL2-2B` | `cc06eec` | `InternVLChatModel` (Custom) | FP16 | 2.21B (Scale test: 8B `9826f68`) |

---

## 2. Unified Interface Specification

All model families inherit from `BaseVLMAdapter` defined in [`src/generation/base.py`](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/src/generation/base.py), exposing a strictly standardized contract:

```python
class BaseVLMAdapter(ABC):
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def format_prompt(self, prompt: str) -> Any: ...
    def preprocess_image(self, image: Image.Image) -> Any: ...
    def generate(self, image: Union[Image.Image, str, Path], prompt: str, **kwargs) -> VLMGenerationOutput: ...
    def get_memory_allocated(self) -> Dict[str, float]: ...
```

### Standardized Generation Output Schema (`VLMGenerationOutput`)
Every generation output adheres strictly to the unified Pydantic schema:
- `model_key`: Canonical identifier (`llava`, `qwen2_vl`, `internvl`).
- `model_id`: Exact Hugging Face repository ID.
- `revision`: Pinned git commit SHA.
- `prompt`: Input prompt text.
- `generated_text`: Decoded textual response.
- `latency_ms`: Execution latency in milliseconds.
- `input_tokens` / `output_tokens`: Token accounting.
- `device_memory_gib`: Memory profile per accelerator.
- `metadata`: Model-specific execution metadata.

---

## 3. Model-Specific Implementations

### 3.1 LLaVA Adapter ([`src/generation/adapters/llava.py`](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/src/generation/adapters/llava.py))
- **Processor**: `AutoProcessor`
- **Vision Backbone**: Static CLIP-ViT-L/14 with 2-layer MLP projection.
- **Conversational Templating**: Standard Vicuna conversational format `USER: <image>\n{prompt}\nASSISTANT:`.
- **Decoding**: Greedy decoding (`temperature=0.0`, `do_sample=False`, `max_new_tokens=128`).

### 3.2 Qwen2-VL Adapter ([`src/generation/adapters/qwen2_vl.py`](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/src/generation/adapters/qwen2_vl.py))
- **Processor**: Native `AutoProcessor` with dynamic resolution visual tokens (`<|vision_start|><|image_pad|><|vision_end|>`).
- **Conversational Templating**: Formats role-based messages and invokes `processor.apply_chat_template`.
- **Dynamic Resizing**: Retains native aspect ratios without fixed letterboxing or cropping.

### 3.3 InternVL Adapter ([`src/generation/adapters/internvl.py`](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/src/generation/adapters/internvl.py))
- **Checkpoint Strategy**:
  - `OpenGVLab/InternVL2-2B` (rev `cc06eec`) used for adapter validation and fast smoke testing.
  - `OpenGVLab/InternVL2-8B` (rev `9826f68`) configured for scale testing.
- **Dynamic Image Preprocessing**: Implements `dynamic_preprocess` with 448x448 patch slicing (1 to 6 tiles) plus thumbnail, normalized using ImageNet statistics.
- **Multi-GPU Splitting (`build_internvl2_8b_device_map`)**:
  - `InternVL2-8B` total FP16 weight size is ~16 GB. Fits across dual NVIDIA T4s (15 GB each) using custom device mapping:
    - GPU 0: `vision_model` (InternViT-6B) + `mlp1` + `language_model.model.embed_tokens` + language layers 0–15.
    - GPU 1: language layers 16–31 + `norm` + `lm_head`.

---

## 4. Multi-GPU Memory Management on Dual NVIDIA T4

1. **Precision**: All models are strictly instantiated in `torch.float16`. Turing T4 GPUs do not support native hardware BF16.
2. **Quota per GPU**: Max memory allocation capped at `14GiB` per T4 to allow headroom for CUDA runtime kernels and KV caching.
3. **Sequential Execution & Cache Purging**:
   - Models are loaded and executed strictly sequentially.
   - When transitioning models, `unload()` deletes model, tokenizer, and processor references, calls Python `gc.collect()`, and executes `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()`.
   - VRAM drops to ~0 GiB before the next model is instantiated, preventing Out-Of-Memory (OOM) failures.

---

## 5. Safeguards and Strict Isolation

1. **Zero Inference on Local Mac**: Any invocation of `adapter.load()` outside the verified dual NVIDIA T4 environment raises an explicit `EnvironmentVerificationError`.
2. **InfoSeek Materialization Guard**: If an attempt is made to run image-conditioned inference on InfoSeek samples before OVEN images are downloaded via authentic Kaggle Secrets `HF_TOKEN`, the generator raises an immediate `RuntimeError`.
3. **Partition Isolation**: The 24 overlapping COCO images are barred from all training, calibration, early stopping, threshold selection, and model selection routines.

