"""Qwen2-VL model family adapter for XMCA-Net.

Supports dynamic resolution visual tokens, native chat templates,
and dual NVIDIA T4 multi-GPU device mapping.
"""

from pathlib import Path
import time
from typing import Dict, Any, Optional, Union, List
from PIL import Image

from src.generation.base import BaseVLMAdapter, VLMGenerationOutput
from src.utils.environment import enforce_strict_environment


class Qwen2VLAdapter(BaseVLMAdapter):
    """Adapter for Qwen2-VL multimodal models."""

    def load(self) -> None:
        """Load Qwen2-VL processor and model checkpoint."""
        enforce_strict_environment(self.app_config)

        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        dtype = torch.float16 if self.dtype_str == "float16" else torch.float32

        max_memory = {0: self.config.max_memory_per_gpu, 1: self.config.max_memory_per_gpu}
        if torch.cuda.device_count() == 1:
            max_memory = {0: self.config.max_memory_per_gpu}

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=self.revision,
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=dtype,
            device_map=self.config.device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        self._is_loaded = True

    def format_prompt(self, prompt: str) -> str:
        """Construct Qwen2-VL chat conversation template."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Ensure standard RGB mode for dynamic resolution processing."""
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
        """Execute Qwen2-VL generation."""
        if not self.is_loaded or self.model is None or self.processor is None:
            raise RuntimeError(f"Model '{self.model_key}' must be loaded before calling generate().")

        import torch

        gen_params = self.config.generation_params.copy()
        max_tokens = max_new_tokens if max_new_tokens is not None else gen_params.get("max_new_tokens", 128)
        sample = do_sample if do_sample is not None else gen_params.get("do_sample", False)
        temp = temperature if temperature is not None else gen_params.get("temperature", 0.0)

        pil_image = self._resolve_image(image)
        formatted_prompt = self.format_prompt(prompt)

        inputs = self.processor(
            text=[formatted_prompt],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )

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
        decoded_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

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

