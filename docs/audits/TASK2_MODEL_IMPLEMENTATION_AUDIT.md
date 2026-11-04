# XMCA-Net: Task 2.0.1 Model Implementation Forensic Audit

**Audit Target**: Model Adapters, Factory Registry, Generation Pipeline, and Hugging Face API Compliance  
**Auditor**: Antigravity Static Code Analysis Suite

---

## 1. Execution Traces per Architecture

### 1.1 LLaVA Implementation Audit
- **File**: [`src/generation/adapters/llava.py`](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/src/generation/adapters/llava.py)
- **Model Key**: `llava`
- **Configured Model**: `llava-hf/llava-1.5-7b-hf`
- **Revision Argument**: `revision=self.revision` passed directly to `AutoProcessor.from_pretrained` (line 34) and `LlavaForConditionalGeneration.from_pretrained` (line 40).
- **Quantization & Dtype**: `dtype = torch.float16 if self.dtype_str == "float16" else torch.float32` (line 26).
- **Device Placement**: `device_map=self.config.device_map` (`"auto"`), `max_memory={0: "14GiB", 1: "14GiB"}` (line 43).
- **Image Input Verification**: Lines 103-109 passes the authentic PIL image into `processor(text=formatted_prompt, images=pil_image, return_tensors="pt")`.
- **Text Slicing**: Line 118 calculates `input_token_count = inputs["input_ids"].shape[-1]`. Line 135 slices `output_ids[:, input_token_count:]`. The model prompt and visual prefix are completely excluded from the recorded `generated_answer`.
- **Finding**: Implementation strictly adheres to official Hugging Face `transformers` LLaVA conditional generation contracts.

### 1.2 Qwen2-VL Implementation Audit
- **File**: [`src/generation/adapters/qwen2_vl.py`](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/src/generation/adapters/qwen2_vl.py)
- **Model Key**: `qwen2_vl`
- **Configured Model**: `Qwen/Qwen2-VL-7B-Instruct`
- **Revision Argument**: Pinned commit SHA `eed13092ef92e448dd6875b2a00151bd3f7db0ac` passed to `from_pretrained` (lines 34, 39).
- **Processor Pipeline**: Lines 91-96 passes the conversational prompt and PIL image into `AutoProcessor`. Dynamic aspect ratio scaling is handled natively by the processor.
- **Generation Call**: Lines 115-118 calls `self.model.generate(**inputs, **gen_kwargs)` with greedy decoding (`temperature=0.0`, `do_sample=False`).
- **Finding**: Implementation complies with `Qwen2VLForConditionalGeneration` official specifications in Transformers 4.45+.

### 1.3 InternVL2 Implementation Audit
- **File**: [`src/generation/adapters/internvl.py`](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/src/generation/adapters/internvl.py)
- **Model Key**: `internvl`
- **Configured Model**: `OpenGVLab/InternVL2-2B` (Adapter validation) / `OpenGVLab/InternVL2-8B` (Scale experiment)
- **Image Preprocessing Method**: Lines 58-101 implements `dynamic_preprocess`:
  - `find_best_aspect_ratio`: Calculates target grid minimising distortion.
  - Patch generation: Crops image into blocks of `image_size=448` (1 to 6 tiles).
  - Thumbnail: Appends a 448x448 thumbnail image if multiple patches are created.
  - Normalization: Uses standard ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
- **Chat Interface**: Lines 228-233 calls official `self.model.chat(self.tokenizer, pixel_values, formatted_question, generation_config=generation_config)`.
- **Finding**: Adheres to OpenGVLab's official InternVL2 chat API and patch generation procedures.

---

## 2. Multi-GPU Device Map & Scale Experiment Inspection

- **File**: [`src/generation/adapters/internvl.py`](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/src/generation/adapters/internvl.py#L104-L128)
- **Function**: `build_internvl2_8b_device_map(num_gpus=2)`
- **Detailed Layer Allocation**:
  - `vision_model`: Assigned to GPU 0 (holds the 6B vision transformer).
  - `mlp1`: Assigned to GPU 0 (projector).
  - `language_model.model.embed_tokens`: Assigned to GPU 0.
  - Language model layers 0 to 15: Assigned to GPU 0.
  - Language model layers 16 to 31: Assigned to GPU 1.
  - `language_model.model.norm`: Assigned to GPU 1.
  - `language_model.lm_head` and `output`: Assigned to GPU 1.
- **Physical Feasibility**: This distribution balances the ~16 GB weights of InternVL2-8B across the two 15 GB NVIDIA T4 GPUs (~8 GB per GPU), leaving ample headroom for activations and KV cache.

---

## 3. Orchestration & Isolation Guards

- **File**: [`src/generation/generator.py`](file:///Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC/src/generation/generator.py#L67-L75)
- **Safeguard 1 (InfoSeek Gated Access)**:
  ```python
  if dataset == "infoseek":
      img_p = Path(image_path)
      if not img_p.exists() or img_p.stat().st_size == 0:
          raise RuntimeError(
              f"InfoSeek image-conditioned inference is BLOCKED: Image '{image_path}' "
              "is not materialized. Gated OVEN access must be resolved on Kaggle via HF_TOKEN."
          )
  ```
  Any attempt to execute inference without materialized authentic image raises an immediate hard exception.
- **Safeguard 2 (Sequential Memory Cleanliness)**:
  Lines 32-34 and 44-48 ensure `unload_active_adapter()` is invoked prior to loading a subsequent model, preventing multi-model co-existence in VRAM.

---

## 4. Model Implementation Audit Verdict

| Backbone | Official Class Used | Genuine Processor | Dynamic Vision Pipeline | Device Map Strategy | Status |
|:---|:---|:---|:---|:---|:---|
| **LLaVA 1.5 7B** | `LlavaForConditionalGeneration` | `AutoProcessor` | Static RGB + CLIP ViT-L/14 | `device_map="auto"`, max 14GiB/GPU | PASS |
| **Qwen2-VL 7B** | `Qwen2VLForConditionalGeneration` | `AutoProcessor` | Dynamic resolution tokens | `device_map="auto"`, max 14GiB/GPU | PASS |
| **InternVL2 2B** | `InternVLChatModel` (`AutoModel`) | `AutoTokenizer` | Dynamic 448x448 patches + thumbnail | `device_map="auto"` | PASS |
| **InternVL2 8B** | `InternVLChatModel` (`AutoModel`) | `AutoTokenizer` | Dynamic 448x448 patches + thumbnail | Custom layer map across GPU 0 & 1 | PASS |
