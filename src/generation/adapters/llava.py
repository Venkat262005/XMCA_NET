"""LLaVA model family adapter for XMCA-Net.

Supports LlavaForConditionalGeneration and LlavaOnevision architectures
with dual NVIDIA T4 multi-GPU device mapping and float16 precision.
"""

from pathlib import Path
import time
from typing import Dict, Any, Optional, Union
from PIL import Image

from src.generation.base import BaseVLMAdapter, VLMGenerationOutput
from src.utils.environment import enforce_strict_environment


class LLaVAAdapter(BaseVLMAdapter):
    """Adapter for LLaVA multimodal models."""

    def load(self) -> None:
        """Load LLaVA processor and model checkpoint onto available GPU(s)."""
        enforce_strict_environment(self.app_config)

        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            AutoModelForImageTextToText = None

        dtype = torch.float16 if self.dtype_str == "float16" else torch.float32

        max_memory = {0: self.config.max_memory_per_gpu, 1: self.config.max_memory_per_gpu}
        if torch.cuda.device_count() == 1:
            max_memory = {0: self.config.max_memory_per_gpu}

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=self.revision,
            patch_size=14,
        )
        if getattr(self.processor, "patch_size", None) is None:
            self.processor.patch_size = 14
        if hasattr(self.processor, "image_processor") and getattr(self.processor.image_processor, "patch_size", None) is None:
            self.processor.image_processor.patch_size = 14
        if getattr(self.processor, "vision_feature_select_strategy", None) is None:
            self.processor.vision_feature_select_strategy = "default"
        if getattr(self.processor, "num_additional_image_tokens", 0) == 0:
            self.processor.num_additional_image_tokens = 1

        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id,
                revision=self.revision,
                torch_dtype=dtype,
                device_map=self.config.device_map,
                max_memory=max_memory,
                low_cpu_mem_usage=True,
            )
        except Exception:
            if AutoModelForImageTextToText is not None:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    revision=self.revision,
                    torch_dtype=dtype,
                    device_map=self.config.device_map,
                    max_memory=max_memory,
                    low_cpu_mem_usage=True,
                )
            else:
                raise

        # Synchronize processor attributes with model configuration
        vision_cfg = getattr(self.model.config, "vision_config", None)
        patch_sz = getattr(vision_cfg, "patch_size", 14) if vision_cfg is not None else 14
        self.processor.patch_size = patch_sz
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.patch_size = patch_sz
        strat = getattr(self.model.config, "vision_feature_select_strategy", "default")
        self.processor.vision_feature_select_strategy = strat
        self.processor.num_additional_image_tokens = 1

        self.model.eval()
        self._is_loaded = True

    def format_prompt(self, prompt: str) -> str:
        """Format prompt with standard LLaVA conversational template."""
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            try:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                return self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            except Exception:
                pass
        # Default standard Vicuna prompt format for LLaVA-1.5
        return f"USER: <image>\n{prompt}\nASSISTANT:"

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Return validated RGB PIL Image."""
        return image.convert("RGB")

    def generate(
        self,
        image: Union[Image.Image, str, Path],
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ) -> VLMGenerationOutput:
        """Generate response given an image and prompt."""
        if not self.is_loaded or self.model is None or self.processor is None:
            raise RuntimeError(f"Model '{self.model_key}' must be loaded before calling generate().")

        import torch

        gen_params = self.config.generation_params.copy()
        max_tokens = max_new_tokens if max_new_tokens is not None else gen_params.get("max_new_tokens", 128)
        sample = do_sample if do_sample is not None else gen_params.get("do_sample", False)
        temp = temperature if temperature is not None else gen_params.get("temperature", 0.0)

        pil_image = self._resolve_image(image)
        formatted_prompt = self.format_prompt(prompt)

        # Guard against upstream transformers LlavaProcessor patch_size & strategy None bug
        if getattr(self.processor, "patch_size", None) is None:
            self.processor.patch_size = 14
        if hasattr(self.processor, "image_processor") and getattr(self.processor.image_processor, "patch_size", None) is None:
            self.processor.image_processor.patch_size = 14
        if getattr(self.processor, "vision_feature_select_strategy", None) is None:
            self.processor.vision_feature_select_strategy = "default"
        if getattr(self.processor, "num_additional_image_tokens", 0) == 0:
            self.processor.num_additional_image_tokens = 1

        inputs = self.processor(
            text=formatted_prompt,
            images=pil_image,
            return_tensors="pt",
        )

        # Dynamic token alignment guard: ensure count of <image> tokens in input_ids exactly matches expected features
        if "pixel_values" in inputs and "input_ids" in inputs:
            image_token_id = getattr(self.model.config, "image_token_id", None)
            if image_token_id is None:
                image_token_id = getattr(self.processor, "image_token_id", None)
            if image_token_id is None and hasattr(self.processor, "tokenizer"):
                image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")

            if image_token_id is not None:
                pv = inputs["pixel_values"]
                h, w = pv.shape[-2], pv.shape[-1]
                patch_sz = getattr(self.processor, "patch_size", 14) or 14
                expected_features = (h // patch_sz) * (w // patch_sz)

                input_ids = inputs["input_ids"]
                current_tokens = (input_ids == image_token_id).sum().item()
                if current_tokens != expected_features and current_tokens > 0:
                    diff = expected_features - current_tokens
                    new_input_ids = []
                    new_att_masks = []
                    has_att = "attention_mask" in inputs
                    for b in range(input_ids.shape[0]):
                        b_ids = input_ids[b]
                        b_att = inputs["attention_mask"][b] if has_att else torch.ones_like(b_ids)
                        mask = (b_ids == image_token_id)
                        if not mask.any():
                            new_input_ids.append(b_ids)
                            if has_att:
                                new_att_masks.append(b_att)
                            continue
                        indices = mask.nonzero(as_tuple=True)[0]
                        last_pos = indices[-1].item()
                        if diff > 0:
                            insert_tokens = torch.full((diff,), image_token_id, dtype=b_ids.dtype, device=b_ids.device)
                            insert_att = torch.ones((diff,), dtype=b_att.dtype, device=b_att.device)
                            fixed_ids = torch.cat([b_ids[:last_pos + 1], insert_tokens, b_ids[last_pos + 1:]])
                            fixed_att = torch.cat([b_att[:last_pos + 1], insert_att, b_att[last_pos + 1:]])
                        else:
                            remove_count = abs(diff)
                            remove_indices = set(indices[-remove_count:].tolist())
                            keep_mask = torch.tensor([i not in remove_indices for i in range(len(b_ids))], dtype=torch.bool, device=b_ids.device)
                            fixed_ids = b_ids[keep_mask]
                            fixed_att = b_att[keep_mask]
                        new_input_ids.append(fixed_ids)
                        if has_att:
                            new_att_masks.append(fixed_att)
                    inputs["input_ids"] = torch.stack(new_input_ids)
                    if has_att:
                        inputs["attention_mask"] = torch.stack(new_att_masks)

        target_device = self.model.device
        inputs = {
            k: v.to(target_device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        input_token_count = inputs["input_ids"].shape[-1]

        start_time = time.perf_counter()
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": sample,
            }
            if sample and temp > 0:
                gen_kwargs["temperature"] = temp

            output_ids = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        generated_ids = output_ids[:, input_token_count:]
        output_token_count = generated_ids.shape[-1]
        decoded_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        memory_alloc = self.get_memory_allocated()

        return VLMGenerationOutput(
            model_key=self.model_key,
            model_id=self.model_id,
            revision=self.revision,
            prompt=prompt,
            generated_text=decoded_text,
            finish_reason="stop",
            latency_ms=round(latency_ms, 2),
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            device_memory_gib=memory_alloc,
            metadata={"device_map": str(self.config.device_map)},
        )

