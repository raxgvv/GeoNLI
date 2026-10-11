"""Shared helpers for running the persistent SAM3 grounding pipeline."""
from __future__ import annotations

import base64
import logging
import os
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import uuid4

from PIL import Image
import torch

from models.sam3.runtime_service import Sam3GroundingService
from backend.services.inference_lock import GPU_INFERENCE_LOCK, GPU_MODEL_SCHEDULER
from backend.services.request_context import current_request_id

logger = logging.getLogger(__name__)

SAM3_LOCK = threading.Lock()
SAM3_ROOT: Optional[Path] = None
SAM3_SCRIPT: Optional[Path] = None
SAM3_SERVICE: Optional[Sam3GroundingService] = None


def _candidate_sam3_roots() -> List[Path]:
    roots: List[Path] = []
    env_root = os.environ.get('SAM3_ROOT')
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.append(Path(__file__).resolve().parents[2] / 'models' / 'sam3')  # Bundled copy
    roots.append(Path('/root/sam3'))  # Dedicated SAM3 checkout
    return roots


def _resolve_sam3_paths() -> Tuple[Path, Path]:
    for root in _candidate_sam3_roots():
        script = root / 'unified_inference.py'
        if script.exists():
            return root.resolve(), script.resolve()
    raise FileNotFoundError(
        "Could not locate unified_inference.py. Set SAM3_ROOT or ensure '/root/sam3' exists."
    )


def _get_sam3_paths() -> Tuple[Path, Path]:
    global SAM3_ROOT, SAM3_SCRIPT
    if SAM3_ROOT is None or SAM3_SCRIPT is None or not SAM3_SCRIPT.exists():
        SAM3_ROOT, SAM3_SCRIPT = _resolve_sam3_paths()
    return SAM3_ROOT, SAM3_SCRIPT


def _get_sam3_service() -> Sam3GroundingService:
    global SAM3_SERVICE
    if SAM3_SERVICE is None:
        sam3_root, _ = _get_sam3_paths()
        SAM3_SERVICE = Sam3GroundingService(sam3_root)
    return SAM3_SERVICE


def _normalize_corners(corners: List[List[float]], image_size: Optional[Tuple[int, int]]) -> Optional[List[float]]:
    if not image_size:
        return None
    width, height = image_size
    if not width or not height:
        return None
    flat: List[float] = []
    for point in corners:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        x, y = point
        flat.append(max(0.0, min(1.0, float(x) / width)))
        flat.append(max(0.0, min(1.0, float(y) / height)))
    return flat


def _encode_output_image(output_image: Path) -> Optional[str]:
    if not output_image.exists():
        logger.warning("SAM3 did not produce %s; returning empty visualization", output_image)
        return None
    encoded_bytes = output_image.read_bytes()
    return f"data:image/jpeg;base64,{base64.b64encode(encoded_bytes).decode('ascii')}"


def run_sam3_grounding(image_bytes: bytes, query: str, source_filename: Optional[str] = None, spatial_resolution_m: float = 1.0) -> dict:
    """Runs the persistent SAM3 grounding pipeline and returns encoded output image."""
    request_started = time.monotonic()
    sam3_root, _ = _get_sam3_paths()

    cleaned_query = (query or "").strip() or "object"
    upload_dir = sam3_root / "user_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(source_filename or "").suffix.lower() or ".png"
    upload_path = upload_dir / f"user_upload_{uuid4().hex}{suffix}"
    upload_path.write_bytes(image_bytes)

    lock_started = time.monotonic()
    with GPU_INFERENCE_LOCK:
        gpu_wait = time.monotonic() - lock_started
        placement_started = time.monotonic()
        GPU_MODEL_SCHEDULER.prepare_sam3()
        placement_duration = time.monotonic() - placement_started
        initialization_started = time.monotonic()
        service = _get_sam3_service()
        initialization_duration = time.monotonic() - initialization_started
        with SAM3_LOCK:
            placement_duration += GPU_MODEL_SCHEDULER.activate_sam3(service)
            inference_started = time.monotonic()
            result = service.ground(upload_path, cleaned_query)
            inference_duration = time.monotonic() - inference_started

    serialization_started = time.monotonic()
    filtered_instances = result.filtered_instances
    image_size = result.image_size
    output_image = result.output_image
    grounded_image = _encode_output_image(output_image)

    answer_payload: List[List[float]] = []
    for inst in filtered_instances:
        corners = inst.get("obb_corners") if isinstance(inst, dict) else None
        if not corners:
            continue
        normalized = _normalize_corners(corners, image_size)
        if normalized:
            answer_payload.append(normalized)
    answer_text = f"Grounding completed for '{cleaned_query}'." if answer_payload else "No groundings found."

    logger.info(
        "Grounding request complete | request_id=%s | dimensions=%sx%s | model_init_s=%.3f | gpu_lock_wait_s=%.3f | placement_s=%.3f | inference_s=%.3f | serialization_s=%.3f | total_s=%.3f | raw_count=%d | final_count=%d | scores=%s | gpu_allocated_mib=%s | gpu_reserved_mib=%s",
        current_request_id(), image_size[0], image_size[1], initialization_duration, gpu_wait, placement_duration,
        inference_duration, time.monotonic() - serialization_started, time.monotonic() - request_started,
        result.raw_candidate_count, len(answer_payload),
        [round(float(i.get("score", 0.0)), 4) for i in filtered_instances if isinstance(i, dict)],
        round(torch.cuda.memory_allocated() / 1024**2) if torch.cuda.is_available() else 0,
        round(torch.cuda.memory_reserved() / 1024**2) if torch.cuda.is_available() else 0,
    )

    return {
        "answer": answer_text,
        "bboxes": answer_payload,
        "grounded_image": grounded_image,
        "filtered_instances": filtered_instances,
        "raw_detection_count": result.raw_candidate_count,
        "logs": result.logs,
        "output_path": str(output_image),
        "input_path": str(upload_path),
    }


def run_sam3_grounding_from_image(image: Image.Image, query: str, filename: Optional[str] = None, spatial_resolution_m: float = 1.0) -> dict:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return run_sam3_grounding(buffer.getvalue(), query, filename, spatial_resolution_m=spatial_resolution_m)


# SAM3 is initialized on the first grounding request, not during Flask import.
