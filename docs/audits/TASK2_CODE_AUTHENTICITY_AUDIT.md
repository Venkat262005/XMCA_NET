# XMCA-Net: Task 2.0.1 Code Authenticity Forensic Audit

**Audit Date**: 2026-11-04  
**Audit Target**: Vision-Language Model Adapters, Factory Registry, and Execution Scripts  
**Repository Working Directory**: `/Users/ketharajupallivenkat/Desktop/XMCA_NET_SRC`  
**Audit Scope**: Read-only static code analysis, execution tracing, and anti-manipulation audit.

---

## 1. Executive Summary

A comprehensive forensic audit was conducted on the VLM adapter architecture and execution scripts before remote Kaggle GPU execution. The codebase was analyzed for:
- Simulated or hardcoded model answers.
- Fabricated or static GPU telemetry values.
- Mocks in execution pipelines.
- Contamination between dry-run fixtures and research artifacts.
- Correctness of Hugging Face multimodal pipelines.

**Forensic Finding**: The codebase represents a **genuine, authentic research implementation**. There are zero hardcoded model predictions in research directories, zero synthetic labels, and zero fabricated GPU telemetry measurements. All simulated data generated during local macOS contract testing has been quarantined in `tests/fixtures/task2_dry_run/`. Research generation paths (`artifacts/generations/task2_verified/`) remain strictly reserved for genuine Kaggle dual-T4 GPU outputs.

---

## 2. Forensic Trace per Backbone

### 2.1 LLaVA 1.5 7B (`src/generation/adapters/llava.py`)
- **Model Class**: `LlavaForConditionalGeneration` (line 24, line 38) with `AutoModelForVision2Seq` fallback (line 48).
- **Processor**: `AutoProcessor.from_pretrained(self.model_id, revision=self.revision)` (lines 32-35).
- **Image Resolution**: Calls `self._resolve_image(image)` verifying image existence on disk and converting to RGB (lines 103).
- **Prompt Templating**: Uses `processor.apply_chat_template` or standard Vicuna multimodal template `USER: <image>\n{prompt}\nASSISTANT:` (lines 60-77).
- **Forward & Decoding**:
  ```python
  inputs = self.processor(text=formatted_prompt, images=pil_image, return_tensors="pt") # line 106
  inputs = {k: v.to(target_device) for k, v in inputs.items()} # lines 113-116
  output_ids = self.model.generate(**inputs, **gen_kwargs) # lines 129-132
  generated_ids = output_ids[:, input_token_count:] # line 135
  decoded_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip() # line 137
  ```
- **Lineage**: The response string is decoded strictly from `generated_ids`. Prompt tokens are sliced out using `input_token_count`.
- **Verdict**: **PASS** (Static audit). Hardware inference: **NOT VERIFIABLE** on local macOS.

### 2.2 Qwen2-VL 7B (`src/generation/adapters/qwen2_vl.py`)
- **Model Class**: `Qwen2VLForConditionalGeneration` (line 24, line 37).
- **Processor**: `AutoProcessor.from_pretrained(self.model_id, revision=self.revision)` (lines 32-35).
- **Dynamic Resolution**: Compatible with native Qwen2-VL visual token representation (`<|vision_start|><|image_pad|><|vision_end|>`).
- **Prompt Templating**: Uses `processor.apply_chat_template` with conversation role mapping (lines 49-62).
- **Forward & Decoding**:
  ```python
  inputs = self.processor(text=[formatted_prompt], images=[pil_image], padding=True, return_tensors="pt") # lines 91-96
  output_ids = self.model.generate(**inputs, **gen_kwargs) # lines 115-118
  generated_ids = output_ids[:, input_token_count:] # line 121
  decoded_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip() # lines 123-125
  ```
- **Lineage**: Answers are decoded dynamically from the model's forward token ids.
- **Verdict**: **PASS** (Static audit). Hardware inference: **NOT VERIFIABLE** on local macOS.

### 2.3 InternVL2 2B / 8B (`src/generation/adapters/internvl.py`)
- **Model Class**: `AutoModel.from_pretrained(self.model_id, revision=self.revision, torch_dtype=dtype, trust_remote_code=True, ...)` (lines 161-169).
- **Tokenizer**: `AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, trust_remote_code=True, use_fast=False)` (lines 150-155).
- **Dynamic Patch Slicing**: Implements `dynamic_preprocess` (lines 58-101) tiling 1 to 6 patches of 448x448 based on aspect ratio plus thumbnail, normalized with ImageNet mean/std (lines 18-32).
- **Autoregressive Generation**:
  ```python
  response = self.model.chat(
      self.tokenizer,
      pixel_values,
      formatted_question,
      generation_config=generation_config,
  ) # lines 228-233
  ```
- **Multi-GPU Splitting**: `build_internvl2_8b_device_map(num_gpus=2)` (lines 104-128) assigns vision model, embeddings, and language layers 0-15 to GPU 0; language layers 16-31, norm, and lm_head to GPU 1.
- **Verdict**: **PASS** (Static audit). Hardware inference: **NOT VERIFIABLE** on local macOS.

---

## 3. Findings & Severity Classification Summary

| Component | Target File | Verification Item | Code Evidence | Severity | Audit Result |
|:---|:---|:---|:---|:---|:---|
| LLaVA Adapter | `src/generation/adapters/llava.py` | Official model class & tokenizer | Lines 24, 38 | PASS | Verified official Hugging Face class |
| LLaVA Adapter | `src/generation/adapters/llava.py` | Autoregressive generation & token slicing | Lines 129-137 | PASS | Decodes generated tokens only |
| Qwen2-VL Adapter | `src/generation/adapters/qwen2_vl.py` | Dynamic resolution processor | Lines 91-96 | PASS | Passes PIL image to native multimodal processor |
| InternVL Adapter | `src/generation/adapters/internvl.py` | Dynamic patch slicing | Lines 58-101 | PASS | 448x448 patches + thumbnail transform |
| InternVL Adapter | `src/generation/adapters/internvl.py` | Dual-GPU split mapping | Lines 104-128 | PASS | Layer distribution across GPU 0 & 1 |
| GPU Telemetry | `src/generation/telemetry.py` | Hardware driver calls | Lines 12-42 | PASS | Genuine `nvidia-smi` CLI query |
| GPU Telemetry | `src/generation/telemetry.py` | PyTorch memory accounting | Lines 76-94 | PASS | Live `allocated`, `reserved`, and peak calls |
| Dry-Run Separation | `scripts/kaggle_vlm_smoke_test.py` | Isolation from research dirs | Lines 308-314 | PASS | Dry-run writes to `tests/fixtures/task2_dry_run/` |
| Research Directory | `artifacts/generations/task2_verified/` | No simulated records | Empty (`.gitkeep` only) | PASS | Zero contamination |
