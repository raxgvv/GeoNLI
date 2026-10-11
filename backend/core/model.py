import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from models.captioning import CaptioningBackend
from models.vqa import VQABackend
from backend.services.sam3_runner import run_sam3_grounding_from_image
from backend.services.inference_lock import GPU_INFERENCE_LOCK, GPU_MODEL_SCHEDULER
from backend.services.request_context import current_request_id
from backend.services.query_router import (
    QueryRouter,
    RouteDecision,
    get_area,
    get_largest_side,
    get_perimeter,
)

logger = logging.getLogger(__name__)


def ensure_pil_image(image: Any) -> Image.Image:
    """Normalize incoming image inputs to an RGB Pillow image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, Path):
        return ensure_pil_image(str(image))

    if isinstance(image, str):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image path not found: {image}")
        with Image.open(path) as img:
            return img.convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image)}")


class AdvancedVLMAdapter:
    """Adapter that routes backend requests to the fine-tuned Qwen3-VL model."""

    def __init__(self, config):
        self.config = config
        self.temperature = getattr(config, 'ADV_VLM_TEMPERATURE', 0.0)
        self.max_new_tokens = getattr(config, 'ADV_VLM_MAX_NEW_TOKENS', 64)
        self.top_p = getattr(config, 'ADV_VLM_TOP_P', 0.9)
        default_do_sample = self.temperature > 0
        self.do_sample = getattr(config, 'ADV_VLM_DO_SAMPLE', default_do_sample)
        captioning_model_dir = getattr(config, 'CAPTIONING_MODEL_DIR', None)
        if captioning_model_dir:
            os.environ.setdefault('captioning_model_dir', str(captioning_model_dir))
            os.environ.setdefault('CAPTIONING_MODEL_DIR', str(captioning_model_dir))
        self._lock = threading.Lock()
        self._load_backend()

    def _load_backend(self) -> None:
        self.model = CaptioningBackend(
            base_model_id=getattr(self.config, 'ADV_VLM_MODEL_ID', 'Qwen/Qwen3-VL-2B-Instruct'),
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
        )
        logger.info("[AdvancedVLMAdapter] Loaded shared vision-language backend model=%s", self.model.base_model_id)

    def _prepare_generate_kwargs(self, temperature: Optional[float], max_new_tokens: Optional[int]) -> Dict[str, Any]:
        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
        }
        temp = self.temperature if temperature is None else temperature
        if temp and temp > 0:
            generate_kwargs.update({
                "temperature": temp,
                "do_sample": True,
            })
        else:
            generate_kwargs["temperature"] = max(temp, 0.0)
            generate_kwargs["do_sample"] = False
        generate_kwargs.setdefault("top_p", self.top_p)
        return generate_kwargs

    def generate(self, image: Image.Image, prompt: Optional[str], *, temperature: Optional[float] = None,
                 max_new_tokens: Optional[int] = None, max_time: Optional[float] = None) -> str:
        pil_image = ensure_pil_image(image)
        prompt_clean = (prompt or "").strip()
        if prompt is not None and not prompt_clean:
            raise ValueError("Prompt must be a non-empty string when provided for VLM inference.")
        text_input = prompt_clean or "Describe the remote sensing scene in detail."
        generate_kwargs = self._prepare_generate_kwargs(temperature, max_new_tokens)
        if max_time is not None:
            generate_kwargs["max_time"] = max_time

        waiting = time.monotonic()
        with GPU_INFERENCE_LOCK, self._lock:
            lock_wait_s = time.monotonic() - waiting
            inference_started = time.monotonic()
            placement_wait_s = GPU_MODEL_SCHEDULER.activate_vlm()
            logger.info("VLM inference started | request_id=%s | lock_wait_s=%.3f | placement_s=%.3f | model=%s",
                        current_request_id(), lock_wait_s, placement_wait_s, self.model.base_model_id)
            result = self.model.generate(text_input, pil_image, **generate_kwargs)
            logger.info("Qwen inference finished | request_id=%s | inference_s=%.3f",
                        current_request_id(), time.monotonic() - inference_started)
            return result


class BaseModel:
    def __init__(self, adapter: AdvancedVLMAdapter):
        self.adapter = adapter

    def predict(self, image, query=None):
        raise NotImplementedError


class CaptioningModel(BaseModel):
    DEFAULT_PROMPT = "Describe the main visible content in this satellite image in one concise sentence."

    def __init__(self, adapter: AdvancedVLMAdapter, max_generation_seconds: float = 480,
                 max_new_tokens: int = 64):
        super().__init__(adapter)
        self.max_generation_seconds = max_generation_seconds
        self.max_new_tokens = max_new_tokens

    def predict(self, image):
        prompt = self.DEFAULT_PROMPT
        answer = self.adapter.generate(
            image, prompt, max_new_tokens=self.max_new_tokens, max_time=self.max_generation_seconds
        )
        return {
            "answer": answer,
            "prompt": prompt,
        }


class GroundingModel(BaseModel):
    DEFAULT_PROMPT = "Identify the key objects in this scene and describe their locations."

    def predict(self, image, query):
        prompt = (query or self.DEFAULT_PROMPT).strip()
        pil_image = ensure_pil_image(image)
        answer = self.adapter.generate(pil_image, prompt)
        return {
            "answer": answer,
            "prompt": prompt,
            "detections": [],  # Qwen3-VL returns text-only grounding info.
            "bboxes": [],
        }


class VQAModel(BaseModel):
    def __init__(self, adapter: AdvancedVLMAdapter, vqa_backend: Optional[VQABackend] = None):
        super().__init__(adapter)
        self.vqa_backend = vqa_backend

    def predict(self, image, query, task_type: str = 'semantic'):
        if not query:
            raise ValueError("VQA queries must include an instruction.")
        prompt = query.strip()
        pil_image = ensure_pil_image(image)

        if self.vqa_backend:
            answer = self.vqa_backend.answer(pil_image, prompt, task_type=task_type or 'semantic')
        else:
            task_type = (task_type or 'semantic').lower()
            instruction = {
                'binary': 'Answer with Yes or No only. ',
                'numeric': 'Answer with a number only. ',
                'semantic': 'Answer concisely in one sentence. ',
            }.get(task_type, '')
            token_limit = {'binary': 8, 'numeric': 16, 'semantic': 64}.get(task_type, 64)
            answer = self.adapter.generate(
                pil_image, instruction + prompt, max_new_tokens=token_limit
            )
            if task_type == 'binary':
                match = re.search(r'\b(yes|no)\b', answer, re.IGNORECASE)
                if not match:
                    raise ValueError("The VLM did not return a valid yes/no answer")
                answer = match.group(1).capitalize()
            elif task_type == 'numeric':
                match = re.search(r'[-+]?\d*\.?\d+', answer)
                if not match:
                    raise ValueError("The VLM did not return a valid numeric answer")
                answer = match.group(0)

        return {
            "answer": answer,
            "prompt": prompt,
            "task_type": task_type,
        }


class ModelManager:
    def __init__(self, config):
        logger.info("Initializing Advanced VLM Model Manager...")
        self.config = config
        self.adapter = AdvancedVLMAdapter(config)

        self.vqa_backend = None
        vqa_dir = getattr(config, 'VQA_MODEL_DIR', None)
        if vqa_dir and self._vqa_checkpoint_complete(vqa_dir):
            try:
                self.vqa_backend = VQABackend(
                    checkpoint_dir=vqa_dir,
                    device=getattr(config, 'VQA_DEVICE', None),
                    max_new_tokens=getattr(config, 'VQA_MAX_NEW_TOKENS', 128),
                )
            except Exception as exc:
                logger.error("Failed to load dedicated VQA model (%s). Falling back to captioning adapter.", exc)
        elif vqa_dir:
            logger.warning("Dedicated VQA checkpoint is incomplete at %s; using the shared Qwen3-VL captioning model", vqa_dir)

        self.captioner = CaptioningModel(
            self.adapter,
            max_generation_seconds=getattr(config, 'CAPTION_GENERATION_TIMEOUT_S', 50),
        )
        self.grounder = GroundingModel(self.adapter)
        self.vqa = VQAModel(self.adapter, self.vqa_backend)
        # Routing is optional. Loading a second 7B model competes with Qwen and
        # SAM3 for the limited GPU memory and is unnecessary for the core flows.
        self.router = None

    @staticmethod
    def _vqa_checkpoint_complete(checkpoint_dir) -> bool:
        """Only load a sharded checkpoint if every indexed shard is present."""
        import json

        directory = Path(checkpoint_dir)
        index_path = directory / 'model.safetensors.index.json'
        if index_path.exists():
            try:
                weight_map = json.loads(index_path.read_text()).get('weight_map', {})
                shards = set(weight_map.values())
                return bool(shards) and all((directory / shard).is_file() for shard in shards)
            except (OSError, ValueError, TypeError):
                return False
        return any((directory / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin'))

    def _safe_predict(self, model_func, *args, **kwargs):
        """Wrapper to catch errors in threads without crashing the whole batch."""
        try:
            return model_func(*args, **kwargs)
        except Exception as exc:
            logger.exception("Prediction failed")
            return {"error": str(exc), "error_type": type(exc).__name__}

    def _vlm_only_answer(
        self,
        image: Image.Image,
        query: str,
        task_type: str,
        decision: Optional[RouteDecision] = None,
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self.vqa.predict(image, query, task_type)
        result.setdefault("bboxes", [])
        result.setdefault("filtered_instances", [])
        result["route"] = "VLM_ONLY"
        result["numeric_type"] = (decision.numeric_type if decision else "NONE")
        result.setdefault("numeric_value", None)
        if task_type == "binary":
            result["answer"] = self._canonical_binary_answer(result.get("answer", ""))
        elif task_type == "numeric":
            numeric_value = self._extract_first_float(result.get("answer", ""))
            if numeric_value is not None:
                result["numeric_value"] = numeric_value
                result["answer"] = numeric_value
        if fallback_reason:
            result["fallback_reason"] = fallback_reason
        return result

    def _canonical_binary_answer(self, text: str) -> str:
        lowered = (text or "").strip().lower()
        if "yes" in lowered and "no" not in lowered:
            return "Yes"
        if "no" in lowered and "yes" not in lowered:
            return "No"
        if lowered.startswith("yes"):
            return "Yes"
        if lowered.startswith("no"):
            return "No"
        raise ValueError("The VLM did not return a valid yes/no answer")

    def _extract_first_float(self, text: str) -> Optional[float]:
        if not text:
            return None
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _polygon_area(self, corners: List[List[float]]) -> float:
        area = 0.0
        for i in range(len(corners)):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % len(corners)]
            area += (x1 * y2) - (x2 * y1)
        return abs(area) / 2.0

    def _compute_numeric_value(self, instances: List[Dict[str, Any]], numeric_type: str) -> Optional[float]:
        if numeric_type == "COUNT":
            return float(len(instances))

        # Get spatial resolution for real-world measurements
        spatial_resolution = getattr(self, '_current_spatial_resolution', 1.0)
        
        values: List[float] = []
        for inst in instances:
            corners = inst.get("obb_corners")
            if not corners or len(corners) != 4:
                continue
            if numeric_type == "LENGTH":
                # Convert pixel length to meters using spatial resolution
                pixel_length = get_largest_side(corners)
                values.append(pixel_length * spatial_resolution)
            elif numeric_type == "PERIMETER":
                # Convert pixel perimeter to meters using spatial resolution
                pixel_perimeter = get_perimeter(corners)
                values.append(pixel_perimeter * spatial_resolution)
            elif numeric_type == "AREA":
                pixel_area = inst.get("area")
                if pixel_area is None:
                    pixel_area = self._polygon_area(corners)
                # Convert pixel area to square meters using spatial resolution squared
                real_area = pixel_area * (spatial_resolution ** 2)
                values.append(real_area)

        if not values:
            return None

        if numeric_type == "AREA":
            return float(sum(values))  # Total area in square meters
        if numeric_type == "PERIMETER":
            return float(sum(values))  # Total perimeter in meters
        if numeric_type == "LENGTH":
            return max(values)  # Maximum length in meters
        return None

    def _process_routed_query(self, image: Image.Image, instruction: str, task_type: str) -> Dict[str, Any]:
        query = (instruction or "").strip()
        if not query:
            raise ValueError("Attribute query must include an instruction.")

        if task_type not in {"binary", "numeric"}:
            return self._vlm_only_answer(image, query, task_type)

        if self.router is None:
            return self._vlm_only_answer(image, query, task_type)

        decision = self.router.route(query, task_type)
        if decision.route != "SAM_PLUS_VLM":
            return self._vlm_only_answer(image, query, task_type, decision)

        try:
            spatial_resolution = getattr(self, '_current_spatial_resolution', 1.0)
            sam_payload = run_sam3_grounding_from_image(image, query, spatial_resolution_m=spatial_resolution)
        except Exception as exc:
            logger.error("SAM grounding failed for routed query: %s", exc)
            return self._vlm_only_answer(
                image,
                query,
                task_type,
                decision,
                fallback_reason="sam_error",
            )
        filtered_instances = sam_payload.get("filtered_instances", [])

        if not filtered_instances:
            return self._vlm_only_answer(
                image,
                query,
                task_type,
                decision,
                fallback_reason="sam_empty",
            )

        result: Dict[str, Any] = {
            "answer": "",
            "prompt": query,
            "task_type": task_type,
            "route": "SAM_PLUS_VLM",
            "numeric_type": decision.numeric_type,
            "filtered_instances": filtered_instances,
            "bboxes": sam_payload.get("bboxes", []),
            "sam_payload": sam_payload,
        }

        if task_type == "binary":
            count = len(filtered_instances)
            result["answer"] = "Yes" if count > 0 else "No"
            return result

        numeric_value = self._compute_numeric_value(filtered_instances, decision.numeric_type)
        if numeric_value is None:
            return self._vlm_only_answer(
                image,
                query,
                task_type,
                decision,
                fallback_reason="deterministic_failure",
            )

        result["numeric_value"] = numeric_value
        result["answer"] = float(numeric_value)
        return result

    def process_all_queries(self, image: Image.Image, image_type: str, queries: dict, spatial_resolution_m: float = 1.0):
        """Runs all potential tasks sequentially using simple loops."""
        pil_image = ensure_pil_image(image)
        results: Dict[str, Any] = {}
        
        # Store spatial resolution for numeric computations
        self._current_spatial_resolution = spatial_resolution_m

        # Process caption query
        if 'caption_query' in queries:
            results['caption'] = self._safe_predict(self.captioner.predict, pil_image)

        # Process grounding query
        if 'grounding_query' in queries:
            instr = queries['grounding_query'].get('instruction', '')
            results['grounding'] = self._safe_predict(self.grounder.predict, pil_image, instr)

        # Process attribute queries
        if 'attribute_query' in queries:
            attr_q = queries['attribute_query']

            if 'binary' in attr_q:
                results['attr_binary'] = self._safe_predict(
                    self._process_routed_query,
                    pil_image,
                    attr_q['binary'].get('instruction', ''),
                    'binary',
                )

            if 'numeric' in attr_q:
                results['attr_numeric'] = self._safe_predict(
                    self._process_routed_query,
                    pil_image,
                    attr_q['numeric'].get('instruction', ''),
                    'numeric',
                )

            if 'semantic' in attr_q:
                results['attr_semantic'] = self._safe_predict(
                    self.vqa.predict,
                    pil_image,
                    attr_q['semantic'].get('instruction', ''),
                    'semantic',
                )

        return results


_manager_instance = None
_manager_lock = threading.Lock()


def get_model(config=None):
    global _manager_instance
    if config is None:
        from backend.config import Config
        config = Config
    if _manager_instance is None and config:
        with _manager_lock:
            if _manager_instance is None:
                with GPU_INFERENCE_LOCK:
                    _manager_instance = ModelManager(config)
    return _manager_instance
