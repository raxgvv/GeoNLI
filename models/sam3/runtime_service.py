"""Persistent SAM3 grounding service that keeps heavyweight models alive."""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from backend.services.request_context import current_request_id
from backend.services.inference_lock import GPU_MODEL_SCHEDULER

_SAM3_DIR = Path(__file__).resolve().parent
_ORIG_CWD = os.getcwd()
try:
    os.chdir(str(_SAM3_DIR))
    if str(_SAM3_DIR) not in sys.path:
        sys.path.insert(0, str(_SAM3_DIR))
    try:
        from . import unified_inference as sam3_pipeline
    except ImportError as exc:  # pragma: no cover - deployment guard
        raise RuntimeError("Could not import models.sam3.unified_inference") from exc
finally:
    os.chdir(_ORIG_CWD)

logger = logging.getLogger(__name__)


@dataclass
class Sam3Result:
    filtered_instances: List[Dict[str, Any]]
    raw_candidate_count: int
    image_size: Tuple[int, int]
    output_image: Path
    logs: List[str]


class Sam3GroundingService:
    """In-process SAM3 pipeline that loads models once per backend process."""

    def __init__(
        self,
        sam3_root: Path,
        *,
        score_threshold: float = 0.4,
        min_area: int = 100,
        nms_threshold: float = 0.6,
        clip_threshold: float = 0.24,
        disable_clip: bool = True,
        refiner_temperature: float = 0.1,
        qwen_model: str = "qwen3-vl:30b",
        vlm_agent: Optional[Any] = None,
    ) -> None:
        self.sam3_root = sam3_root
        configured_checkpoint = os.environ.get("SAM3_CHECKPOINT")
        self.ckpt_path = Path(configured_checkpoint).expanduser() if configured_checkpoint else sam3_root / "sam.pt"
        if not self.ckpt_path.is_file():
            try:
                from huggingface_hub import hf_hub_download

                cached = hf_hub_download(
                    repo_id="facebook/sam3", filename="sam3.pt", local_files_only=True
                )
                self.ckpt_path = Path(cached)
            except Exception as exc:
                raise FileNotFoundError(
                    "SAM3 checkpoint was not found at SAM3_CHECKPOINT, models/sam3/sam.pt, "
                    "or in the local facebook/sam3 Hugging Face cache"
                ) from exc
        self.score_threshold = float(os.environ.get("SAM3_SCORE_THRESHOLD", score_threshold))
        self.min_area = min_area
        self.nms_threshold = nms_threshold
        self.clip_threshold = clip_threshold
        self.disable_clip = disable_clip
        self.refiner_temperature = refiner_temperature
        # Load weights on CPU first. The shared scheduler places SAM3 on the
        # GPU only for a grounding turn, then returns it to CPU before VLM use.
        self.device = "cpu"
        self._lock = threading.Lock()
        self._logs: List[str] = []
        self.vlm_agent = vlm_agent

        init_started = time.monotonic()
        logger.info("[SAM3] Loading vision model once | request_id=%s | device=%s | checkpoint=%s",
                    current_request_id(), self.device, self.ckpt_path)
        self.model = sam3_pipeline.load_sam3_model(str(self.ckpt_path), self.device)
        # Preserve candidates here so we can inspect genuine raw SAM scores;
        # apply the documented 0.5 model confidence cutoff in get_prediction.
        self.processor = sam3_pipeline.Sam3Processor(
            self.model, device=self.device, confidence_threshold=0.0
        )
        GPU_MODEL_SCHEDULER.register_sam3(self)
        logger.info("[SAM3] Vision model initialized | request_id=%s | duration_s=%.3f",
                    current_request_id(), time.monotonic() - init_started)

        # The incoming prompt works directly with SAM3's text encoder. Avoid
        # allocating a second 7B language model just to remove location words.
        self.refiner = None

        self.clip_context = None
        if not disable_clip:
            self.clip_context = sam3_pipeline.load_clip_context(
                model_name="ViT-H-14",
                pretrained="laion2b_s32b_b79k",
                device=self.device,
                disable_clip=False,
            )

        if self.vlm_agent is not None:
            logger.info(
                "[SAM3] Using injected VLM agent for grounding decisions (%s)",
                type(self.vlm_agent).__name__,
            )

        self.output_image = self.sam3_root / "output.jpg"
        self.intermediate_path = self.sam3_root / "intermediate_output.json"
        self.vlm_json_path = self.sam3_root / "final_output.json"

    def move_to(self, device: str) -> None:
        if self.device == device:
            return
        with self._lock:
            if self.device == device:
                return
            started = time.monotonic()
            self.model.to(device)
            self.device = device
            self.processor = sam3_pipeline.Sam3Processor(
                self.model, device=device, confidence_threshold=0.0
            )
            logger.info("SAM3 placement changed | request_id=%s | device=%s | duration_s=%.3f",
                        current_request_id(), device, time.monotonic() - started)

    def ground(self, image_path: Path, query: str) -> Sam3Result:
        with self._lock:
            return self._ground_locked(image_path, query)

    def set_vlm_agent(self, agent: Any) -> None:
        with self._lock:
            self.vlm_agent = agent
            logger.info(
                "[SAM3] Replaced VLM agent with %s",
                type(agent).__name__ if agent else "None",
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        logger.info("[SAM3] %s", message)
        self._logs.append(message)

    def _ground_locked(self, image_path: Path, query: str) -> Sam3Result:
        request_started = time.monotonic()
        self._logs = []
        query = (query or "object").strip() or "object"
        refined = self.refiner.extract(query, temperature=self.refiner_temperature) if self.refiner else self._grounding_phrase(query)
        sam_prompt = refined or query
        target_prompt = refined or query
        self._log(f"Prompt normalized | original={query!r} | sam_prompt={sam_prompt!r}")

        results, masks_np, raw_candidate_count = sam3_pipeline.get_prediction(
            self.model,
            self.processor,
            str(image_path),
            sam_prompt,
            self.device,
            score_threshold=self.score_threshold,
            min_area=self.min_area,
        )
        logger.info(
            "SAM3 inference complete | prompt=%r | candidates_after_filter=%d | total_s=%.3f | gpu_allocated_mib=%s | gpu_reserved_mib=%s",
            sam_prompt, len(results), time.monotonic() - request_started,
            round(torch.cuda.memory_allocated() / 1024**2) if torch.cuda.is_available() else 0,
            round(torch.cuda.memory_reserved() / 1024**2) if torch.cuda.is_available() else 0,
        )

        if not results:
            self._write_json(self.intermediate_path, {"instances": [], "meta": {"target_prompt": target_prompt}})
            self._write_json(self.vlm_json_path, {"selected_instances": [], "best_guess": {"index": None}})
            if masks_np is not None:
                sam3_pipeline.visualize_result(str(image_path), [], masks_np, str(self.output_image))
            return Sam3Result(
                filtered_instances=[],
                raw_candidate_count=raw_candidate_count,
                image_size=self._image_size(image_path),
                output_image=self.output_image,
                logs=self._logs.copy(),
            )

        nms_results = sam3_pipeline.apply_nms(results, iou_threshold=self.nms_threshold)
        final_results = nms_results

        if not self.disable_clip:
            clip_scores = sam3_pipeline.score_instances_with_clip(
                image_path=str(image_path),
                instances=final_results,
                text_prompt=target_prompt,
                clip_context=self.clip_context,
                device=self.device,
                padding_ratio=0.15,
            )
            for inst, clip_score in zip(final_results, clip_scores):
                inst["clip_score"] = clip_score

        json_payload = {
            "instances": final_results,
            "meta": {
                "target_prompt": target_prompt,
                "clip_model": None if self.disable_clip else "ViT-H-14",
                "clip_pretrained": None if self.disable_clip else "laion2b_s32b_b79k",
                "clip_threshold": self.clip_threshold,
                "nms_threshold": self.nms_threshold,
                "refiner_used": self.refiner is not None,
            },
        }
        self._write_json(self.intermediate_path, json_payload)

        filtered_instances = final_results
        if final_results and self.vlm_agent is not None:
            try:
                vlm_prompt = sam3_pipeline.build_vlm_prompt(
                    image_path=str(self.output_image),
                    clip_threshold=self.clip_threshold,
                    target_prompt=target_prompt,
                    json_payload=json_payload,
                )
                sam3_pipeline.visualize_result(str(image_path), final_results, masks_np, str(self.output_image))
                vlm_response = self.vlm_agent.generate_json(vlm_prompt, str(self.output_image))
                if vlm_response:
                    self._write_json(self.vlm_json_path, vlm_response)
                    selected = []
                    for entry in vlm_response.get("selected_instances", []):
                        idx = entry.get("index")
                        if idx is None:
                            continue
                        for candidate in final_results:
                            if candidate.get("index") == idx:
                                selected.append(candidate)
                                break
                    if selected:
                        filtered_instances = selected
            except Exception as exc:  # pragma: no cover - defensive logging
                self._log(f"VLM reasoning failed: {exc}")

        sam3_pipeline.visualize_result(str(image_path), filtered_instances, masks_np, str(self.output_image))
        return Sam3Result(
            filtered_instances=filtered_instances,
            raw_candidate_count=raw_candidate_count,
            image_size=self._image_size(image_path),
            output_image=self.output_image,
            logs=self._logs.copy(),
        )

    @staticmethod
    def _grounding_phrase(query: str) -> str:
        """Reduce common UI commands/questions to their open-vocabulary target."""
        phrase = re.sub(r"[?!.,;:]+$", "", query.strip().lower())
        phrase = re.sub(
            r"^(?:where\s+(?:is|are)|locate|find|detect|identify|show(?:\s+me)?|ground|segment)\s+",
            "",
            phrase,
        )
        phrase = re.sub(r"^(?:(?:the|a|an|any|all)\s+)+", "", phrase)
        phrase = re.sub(r"\s+(?:in|on|at)\s+(?:this|the)\s+(?:image|scene)$", "", phrase)
        phrase = re.sub(r"\s+(?:visible|here)$", "", phrase).strip()
        # The detector is more reliable on the object class label than a plural
        # sentence form (e.g. `airplanes` -> `airplane`).
        words = phrase.split()
        if words:
            word = words[-1]
            if len(word) > 4 and word.endswith("ies"):
                words[-1] = word[:-3] + "y"
            elif len(word) > 4 and word.endswith(("sses", "shes", "ches", "xes", "zes")):
                words[-1] = word[:-2]
            elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
                words[-1] = word[:-1]
        return " ".join(words) or query

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2))

    def _image_size(self, image_path: Path) -> Tuple[int, int]:
        with Image.open(image_path) as img:
            return img.size
