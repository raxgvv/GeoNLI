import logging
import os
import re
import threading
from pathlib import Path
from typing import Literal, Optional

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from .device import get_best_device

logger = logging.getLogger(__name__)

TaskType = Literal["binary", "numeric", "semantic"]

VQA_MODEL_DIR = Path(os.environ.get("VQA_MODEL_DIR", Path(__file__).resolve().parent / "vqa-model")).expanduser()

BASE_PROCESSOR_ID = "Qwen/Qwen3-VL-8B-Instruct"
PROCESSOR_DIR = Path(os.environ.get("VQA_PROCESSOR_DIR", VQA_MODEL_DIR / "processor"))

_TEMPLATES = {
    "binary": {
        "system": "You are a visual question answering assistant. Answer concisely.",
        "user_suffix": "Answer with yes or no.",
    },
    "numeric": {
        "system": "You are a visual question answering assistant. Answer concisely.",
        "user_suffix": "Return only a number in digits, no units, rounded to one decimal place.",
    },
    "semantic": {
        "system": "You are a visual question answering assistant. Answer concisely.",
        "user_suffix": "",
    },
}


def _build_prompt(task_type: TaskType, question: str) -> str:
    template = _TEMPLATES.get(task_type, _TEMPLATES["semantic"])
    return (
        "<|im_start|>system\n"
        f"{template['system']}<|im_end|>\n"
        "<|im_start|>user\n"
        "Picture 1: <|vision_start|><|image_pad|><|vision_end|>\n"
        f"{template['user_suffix']} {question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _validate_output(text: str, task_type: TaskType) -> str:
    text = (text or "").strip()
    if task_type == "binary":
        lowered = text.lower()
        if re.search(r"\byes\b", lowered):
            return "Yes"
        if re.search(r"\bno\b", lowered):
            return "No"
        raise ValueError("VQA model did not return a valid yes/no answer")
    if task_type == "numeric":
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        if match:
            return match.group()
        raise ValueError("VQA model did not return a valid numeric answer")
    return text


class VQABackend:
    """Fine-tuned Qwen3-VL VQA backend."""

    def __init__(
        self,
        checkpoint_dir: Path = VQA_MODEL_DIR,
        device: Optional[str] = None,
        max_new_tokens: int = 128,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"VQA checkpoint not found at {self.checkpoint_dir}")

        resolved_device = device or get_best_device()
        self.torch_device = torch.device(resolved_device)
        self.dtype = torch.bfloat16 if self.torch_device.type == "cuda" else torch.float32
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

        logger.info("Loading VQA model from %s", self.checkpoint_dir)
        config = AutoConfig.from_pretrained(
            self.checkpoint_dir,
            trust_remote_code=True,
        )
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None and getattr(text_cfg, "rope_scaling", None) is None:
            rope_params = getattr(text_cfg, "rope_parameters", {}) or {}
            text_cfg.rope_scaling = {
                "mrope_interleaved": rope_params.get("mrope_interleaved", True),
                "mrope_section": rope_params.get("mrope_section", [24, 20, 20]),
                "rope_type": rope_params.get("rope_type", "default"),
            }
            logger.info("Injected default rope_scaling into VQA text config")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.checkpoint_dir,
            config=config,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        self.model.to(self.torch_device)
        self.model.eval()
        self.processor = self._load_processor()

    def answer(self, image: Image.Image, question: str, task_type: TaskType = "semantic") -> str:
        if not question:
            raise ValueError("VQA question must be provided.")

        prompt_text = _build_prompt(task_type, question)
        pil_image = self._ensure_image(image)

        inputs = self.processor(
            text=[prompt_text],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: (v.to(self.torch_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with self._lock, torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return _validate_output(output_text, task_type)

    @staticmethod
    def _ensure_image(image: Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError(f"Unsupported image type for VQA backend: {type(image)}")

    def _load_processor(self) -> AutoProcessor:
        if not (PROCESSOR_DIR / "tokenizer_config.json").exists():
            PROCESSOR_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading VQA processor (%s) to %s", BASE_PROCESSOR_ID, PROCESSOR_DIR)
            processor = AutoProcessor.from_pretrained(BASE_PROCESSOR_ID, trust_remote_code=True)
            processor.save_pretrained(PROCESSOR_DIR)
            return processor

        logger.info("Using cached VQA processor at %s", PROCESSOR_DIR)
        return AutoProcessor.from_pretrained(PROCESSOR_DIR, trust_remote_code=True)
