import logging
import os
import time
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from backend.services.request_context import current_request_id
from backend.services.inference_lock import GPU_MODEL_SCHEDULER

logger = logging.getLogger(__name__)

BASE_MODEL_ID = os.environ.get("ADV_VLM_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
DEFAULT_CAPTION_PROMPT = "Describe the main visible content in this image in one concise sentence."
# Keep vision activations bounded on 6 GiB cards. VLM and SAM3 are placed
# exclusively by GPUModelScheduler so their full weights do not compete.
MIN_PIXELS = 256 * 256
MAX_PIXELS = 384 * 384

CAPTIONING_MODEL_DIR = Path(os.environ.get("CAPTIONING_MODEL_DIR", Path(__file__).resolve().parent / "captioning-model")).expanduser()

PROCESSOR_DIR = CAPTIONING_MODEL_DIR / "processor"
ADAPTER_DIR = CAPTIONING_MODEL_DIR / "model"

ImageInput = Union[str, Path, Image.Image]


class CaptioningBackend:
    def __init__(
        self,
        base_model_id: str = BASE_MODEL_ID,
        adapter_path: Path = ADAPTER_DIR,
        processor_path: Path = PROCESSOR_DIR,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 0.9,
        do_sample: bool = False,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        use_flash_attention: bool = True,
    ):
        init_started = time.monotonic()
        logger.info("Loading captioning backend | request_id=%s | base_model=%s | adapter_dir=%s",
                    current_request_id(), base_model_id, adapter_path)
        self.device = "cpu"
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_id = base_model_id
        self.adapter_path = Path(adapter_path)
        self.use_adapter = (
            (self.adapter_path / "adapter_model.safetensors").exists()
            or (self.adapter_path / "adapter_model.bin").exists()
        )
        if not self.use_adapter:
            logger.warning("No PEFT adapter weights found at %s; captioning uses the base model without fine-tuning", self.adapter_path)
        self.processor_path = Path(processor_path)
        # The local processor/adapter was prepared for the 8B base. A smaller
        # base uses its own matching tokenizer and vision processor.
        if base_model_id != BASE_MODEL_ID or "2B" in base_model_id:
            self.processor_path = Path(base_model_id)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if self.torch_device.type == "cuda" else torch.float32
        self.autocast_dtype = self.dtype if self.torch_device.type == "cuda" else torch.float32

        if not self.adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found at {self.adapter_path}")
        if self.use_adapter:
            import json
            adapter_config_path = self.adapter_path / "adapter_config.json"
            if not adapter_config_path.is_file():
                raise RuntimeError(f"Adapter weights exist but adapter_config.json is missing: {self.adapter_path}")
            adapter_config = json.loads(adapter_config_path.read_text())
            adapter_base = adapter_config.get("base_model_name_or_path")
            if adapter_base and adapter_base.rstrip("/") != self.base_model_id.rstrip("/"):
                raise RuntimeError(
                    f"Caption adapter targets {adapter_base}, but configured base model is {self.base_model_id}"
                )
            if not adapter_config.get("peft_type"):
                raise RuntimeError(f"Caption adapter configuration is incomplete: {adapter_config_path}")
        if "Qwen/" not in str(self.processor_path) and not self.processor_path.exists():
            raise FileNotFoundError(f"Processor not found at {self.processor_path}")

        processor_started = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(
            self.processor_path,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        logger.info("Caption processor initialized | request_id=%s | duration_s=%.3f",
                    current_request_id(), time.monotonic() - processor_started)

        model_kwargs = {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        }
        # Keep the 2B VLM on CPU between tasks. The scheduler moves it to the
        # GPU only while it owns the shared inference lock; SAM3 and VLM never
        # hold their full weights on the 6 GiB device simultaneously.

        try:
            base_model = AutoModelForImageTextToText.from_pretrained(
                self.base_model_id,
                **model_kwargs,
            )
        except Exception as exc:
            raise

        if self.use_adapter:
            self.model = PeftModel.from_pretrained(
                base_model,
                str(self.adapter_path),
                torch_dtype=self.dtype,
            )
        else:
            self.model = base_model

        self.model.eval()
        self._model_device = "cpu"
        GPU_MODEL_SCHEDULER.register_vlm(self)
        logger.info(
            "Caption model initialized | request_id=%s | duration_s=%.3f | model=%s | initial_device=cpu | gpu_allocated_mib=%s | gpu_reserved_mib=%s",
            current_request_id(),
            time.monotonic() - init_started,
            self.base_model_id,
            round(torch.cuda.memory_allocated(self.torch_device) / 1024**2) if self.torch_device.type == "cuda" else 0,
            round(torch.cuda.memory_reserved(self.torch_device) / 1024**2) if self.torch_device.type == "cuda" else 0,
        )

    def caption(self, image_input: ImageInput) -> str:
        image = self._ensure_image(image_input)
        return self._generate(DEFAULT_CAPTION_PROMPT, image)

    def move_to(self, device: str) -> None:
        if self._model_device == device:
            return
        started = time.monotonic()
        self.model.to(device)
        self._model_device = device
        self.device = device
        logger.info("VLM placement changed | request_id=%s | device=%s | duration_s=%.3f",
                    current_request_id(), device, time.monotonic() - started)

    def vqa(self, image_input: ImageInput, question: str) -> str:
        image = self._ensure_image(image_input)
        return self._generate(question, image)

    def generate(self, user_prompt: str, image_input: ImageInput, **overrides) -> str:
        image = self._ensure_image(image_input)
        return self._generate(user_prompt, image, **overrides)

    @staticmethod
    def _ensure_image(image_input: ImageInput) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image path not found: {path}")
            with Image.open(path) as img:
                return img.convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    def _generate(self, user_prompt: str, image: Image.Image, **overrides) -> str:
        request_started = time.monotonic()
        prompt = (user_prompt or DEFAULT_CAPTION_PROMPT).strip()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        preprocess_started = time.monotonic()
        inputs = self.processor(
            images=[image],
            text=[chat_prompt],
            return_tensors="pt",
        )
        # Accelerate dispatch hooks own placement for CPU/disk-offloaded blocks.
        # Inputs should begin at the model's first execution device, not force
        # the whole inference (including large vision tensors) onto CUDA.
        input_device = next(self.model.parameters()).device
        inputs = {
            k: v.to(input_device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }
        logger.info(
            "Caption preprocessing complete | request_id=%s | dimensions=%sx%s | duration_s=%.3f | input_device=%s",
            current_request_id(), image.width, image.height, time.monotonic() - preprocess_started, input_device,
        )

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            prompt_len = attention_mask.sum(dim=-1).to(torch.int32)
        else:
            prompt_len = torch.tensor(
                [inputs["input_ids"].shape[-1]],
                dtype=torch.int32,
                device=self.torch_device,
            )

        max_new_tokens = overrides.get("max_new_tokens", self.max_new_tokens)
        max_time = overrides.get("max_time", 50)
        temperature = overrides.get("temperature", self.temperature)
        top_p = overrides.get("top_p", self.top_p)
        do_sample = overrides.get("do_sample", self.do_sample)

        placement_s = GPU_MODEL_SCHEDULER.activate_vlm()
        logger.info("VLM placement ready | request_id=%s | placement_s=%.3f | target=cuda",
                    current_request_id(), placement_s)
        input_device = next(self.model.parameters()).device
        inputs = {k: v.to(input_device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        generation_started = time.monotonic()
        with torch.inference_mode():
            autocast_enabled = self.torch_device.type == "cuda"
            with torch.autocast(
                device_type=self.torch_device.type,
                dtype=self.autocast_dtype,
                enabled=autocast_enabled,
            ):
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **({"max_time": max_time} if max_time is not None else {}),
                    do_sample=do_sample,
                    **({"top_p": top_p, "temperature": temperature} if do_sample else {}),
                )
        if self.torch_device.type == "cuda":
            torch.cuda.synchronize(self.torch_device)
        generation_duration = time.monotonic() - generation_started
        if max_time is not None and generation_duration >= max_time:
            raise TimeoutError(
                f"Vision-language generation exceeded its {max_time:g}-second time limit; no partial caption was returned."
            )

        decode_started = time.monotonic()
        start = prompt_len[0].item()
        trimmed = generated_ids[0][start:]

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            text = tokenizer.decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        else:
            text = self.processor.batch_decode(
                trimmed.unsqueeze(0),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        logger.info(
            "Caption generation complete | request_id=%s | generation_s=%.3f | decode_s=%.3f | output_tokens=%d | gpu_allocated_mib=%s | gpu_reserved_mib=%s | total_model_s=%.3f",
            current_request_id(),
            generation_duration,
            time.monotonic() - decode_started,
            int(trimmed.shape[-1]),
            round(torch.cuda.memory_allocated(self.torch_device) / 1024**2) if self.torch_device.type == "cuda" else 0,
            round(torch.cuda.memory_reserved(self.torch_device) / 1024**2) if self.torch_device.type == "cuda" else 0,
            time.monotonic() - request_started,
        )
        return text.strip()
