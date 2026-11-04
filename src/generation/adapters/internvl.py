"""InternVL model family adapter for XMCA-Net.

Supports OpenGVLab/InternVL2-2B and InternVL2-8B architectures with
dynamic patch preprocessing, custom dual NVIDIA T4 device mapping,
and standard chat interfaces.
"""

import math
from pathlib import Path
import time
from typing import Dict, Any, Optional, Union, List, Tuple
from PIL import Image

from src.generation.base import BaseVLMAdapter, VLMGenerationOutput
from src.utils.environment import enforce_strict_environment


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    """Build torchvision transform for InternVL patch normalization."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_best_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    orig_width: int,
    orig_height: int,
    image_size: int,
) -> Tuple[int, int]:
    """Find target grid aspect ratio minimizing scaling distortion."""
    best_ratio = (1, 1)
    best_diff = float("inf")
    area = orig_width * orig_height
    for r in target_ratios:
        target_aspect_ratio = r[0] / r[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_diff:
            best_diff = ratio_diff
            best_ratio = r
        elif ratio_diff == best_diff:
            if area > 0.5 * image_size * image_size * r[0] * r[1]:
                best_ratio = r
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    """Split high-resolution image into dynamic patch grid plus thumbnail."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = []
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_num and i * j >= min_num:
                    target_ratios.append((i, j))
    target_ratios = sorted(set(target_ratios), key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_best_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def build_internvl2_8b_device_map(num_gpus: int = 2) -> Dict[str, Union[int, str]]:
    """Build multi-GPU device map for InternVL2-8B on dual NVIDIA T4 GPUs.
    
    Vision transformer (InternViT) and initial embeddings reside on GPU 0.
    Language layers (InternLM 32 layers) are distributed across GPU 0 (0-15) and GPU 1 (16-31).
    Final norm and lm_head reside on GPU 1.
    """
    if num_gpus < 2:
        return {"": 0}

    device_map: Dict[str, Union[int, str]] = {
        "vision_model": 0,
        "mlp1": 0,
        "language_model.model.embed_tokens": 0,
    }
    num_layers = 32
    layers_per_gpu = math.ceil(num_layers / 2)
    for i in range(num_layers):
        gpu_id = 0 if i < layers_per_gpu else 1
        device_map[f"language_model.model.layers.{i}"] = gpu_id

    device_map["language_model.model.norm"] = 1
    device_map["language_model.output"] = 1
    device_map["language_model.lm_head"] = 1
    return device_map


class InternVLAdapter(BaseVLMAdapter):
    """Adapter for InternVL multimodal models (InternVL2-2B and InternVL2-8B)."""

    def load(self) -> None:
        """Load InternVL tokenizer and model checkpoint with multi-GPU mapping."""
        enforce_strict_environment(self.app_config)

        import torch
        from transformers import AutoTokenizer, AutoModel

        dtype = torch.float16 if self.dtype_str == "float16" else torch.float32

        # Device mapping strategy
        gpu_count = torch.cuda.device_count()
        if "8B" in self.model_id and gpu_count >= 2:
            device_map = build_internvl2_8b_device_map(num_gpus=gpu_count)
        else:
            device_map = self.config.device_map

        # Guard against upstream SentencePiece >= 0.2.2 regression that breaks InternLM2 tokenizers
        try:
            import sentencepiece
            sp_ver = getattr(sentencepiece, "__version__", "")
            if sp_ver.startswith("0.2.2") or (sp_ver and tuple(map(int, sp_ver.split(".")[:3])) >= (0, 2, 2)):
                import subprocess
                print(f"Warning: SentencePiece version {sp_ver} detected. InternVL requires sentencepiece < 0.2.2.")
                print("Automatically adjusting to sentencepiece<0.2.2...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece<0.2.2"], check=True)
                import importlib
                importlib.reload(sentencepiece)
        except Exception:
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=True,
            use_fast=False,
        )

        max_memory = {0: self.config.max_memory_per_gpu, 1: self.config.max_memory_per_gpu}
        if gpu_count == 1:
            max_memory = {0: self.config.max_memory_per_gpu}

        # Guard against transformers 4.49+ / 5.0+ missing all_tied_weights_keys on custom remote models
        from transformers import PreTrainedModel
        import torch.nn as nn

        def _get_tied_keys(mod):
            if hasattr(mod, "_all_tied_weights_keys"):
                return mod._all_tied_weights_keys
            return getattr(mod, "_tied_weights_keys", {})

        def _set_tied_keys(mod, val):
            mod._all_tied_weights_keys = val

        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = property(_get_tied_keys, _set_tied_keys)
        if not hasattr(nn.Module, "all_tied_weights_keys"):
            nn.Module.all_tied_weights_keys = property(_get_tied_keys, _set_tied_keys)

        self.model = AutoModel.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=dtype,
            device_map=device_map,
            max_memory=max_memory if device_map == "auto" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        self._is_loaded = True

    def format_prompt(self, prompt: str) -> str:
        """Construct question prompt with visual placeholder."""
        if "<image>" not in prompt:
            return f"<image>\n{prompt}"
        return prompt

    def preprocess_image(self, image: Image.Image) -> Any:
        """Run dynamic patch slicing and tensor transform for InternVL."""
        import torch
        transform = build_transform(input_size=448)
        patches = dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True)
        pixel_values = [transform(p) for p in patches]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def generate(
        self,
        image: Union[Image.Image, str, Path],
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ) -> VLMGenerationOutput:
        """Execute InternVL generation using official chat interface."""
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError(f"Model '{self.model_key}' must be loaded before calling generate().")

        import torch

        gen_params = self.config.generation_params.copy()
        max_tokens = max_new_tokens if max_new_tokens is not None else gen_params.get("max_new_tokens", 128)
        sample = do_sample if do_sample is not None else gen_params.get("do_sample", False)
        temp = temperature if temperature is not None else gen_params.get("temperature", 0.0)

        pil_image = self._resolve_image(image)
        pixel_values = self.preprocess_image(pil_image)

        # Move pixel_values to model device/dtype
        target_dtype = torch.float16 if self.dtype_str == "float16" else torch.float32
        device = next(self.model.parameters()).device
        pixel_values = pixel_values.to(dtype=target_dtype, device=device)

        formatted_question = self.format_prompt(prompt)

        generation_config = {
            "max_new_tokens": max_tokens,
            "do_sample": sample,
        }
        if sample and temp > 0:
            generation_config["temperature"] = temp

        start_time = time.perf_counter()
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                formatted_question,
                generation_config=generation_config,
            )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Rough token approximation from tokenizer
        output_token_count = len(self.tokenizer.encode(response))
        input_token_count = len(self.tokenizer.encode(formatted_question)) + pixel_values.shape[0] * 256

        memory_alloc = self.get_memory_allocated()

        return VLMGenerationOutput(
            model_key=self.model_key,
            model_id=self.model_id,
            revision=self.revision,
            prompt=prompt,
            generated_text=response.strip(),
            finish_reason="stop",
            latency_ms=round(latency_ms, 2),
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            device_memory_gib=memory_alloc,
            metadata={"patches_count": int(pixel_values.shape[0])},
        )

