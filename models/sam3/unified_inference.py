#!/usr/bin/env python3
"""Single-file inference pipeline for SAM3 + CLIP + Qwen VLM reasoning."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from backend.services.request_context import current_request_id

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import ollama  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ollama = None

try:
    import open_clip
except ImportError:
    open_clip = None

#  Ensure the current directory is in the path to import sam3
sys.path.append(os.getcwd())

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("Error: Could not import 'sam3'. Please run this script from the root of the sam3 repository.")
    sys.exit(1)


def load_sam3_model(ckpt_path, device):
    """
    Loads the SAM 3 model using the official builder.
    """
    print(f"Loading SAM 3 model...")
    model = None

    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"A valid SAM3 checkpoint is required; not found: {ckpt_path}")

    # Pass the actual upstream keyword. The old positional fallback treated
    # the checkpoint path as a BPE path, then built a default CUDA model,
    # temporarily allocating SAM3 alongside the VLM before moving it to CPU.
    model = build_sam3_image_model(
        checkpoint_path=ckpt_path,
        device=device,
        load_from_HF=False,
    )

    model.to(device)
    model.eval()
    return model


def mask_to_oriented_bbox(mask_np, debug=False):
    """
    Convert a mask array to an oriented bounding box (4 corner points).
    """
    # Squeeze singleton dims
    mask = np.squeeze(mask_np)

    # If mask has more than 2 dims -> collapse channels
    if mask.ndim == 3:
        if mask.shape[0] <= 4 and mask.shape[0] != mask.shape[1]:
            mask = np.moveaxis(mask, 0, -1)
        mask = np.any(mask > 0, axis=-1)

    # Now mask should be 2D
    if mask.ndim != 2:
        mask = mask.reshape(mask.shape[-2], mask.shape[-1])

    # Convert to uint8 single channel 0/255
    if mask.dtype == np.bool_:
        mask_uint8 = (mask.astype(np.uint8) * 255)
    elif np.issubdtype(mask.dtype, np.floating):
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    elif np.issubdtype(mask.dtype, np.integer):
        if mask.max() <= 1:
            mask_uint8 = (mask.astype(np.uint8) * 255)
        else:
            mask_uint8 = mask.astype(np.uint8)
            _, mask_uint8 = cv2.threshold(mask_uint8, 1, 255, cv2.THRESH_BINARY)
    else:
        mask_uint8 = (mask > 0).astype(np.uint8) * 255

    mask_uint8 = np.ascontiguousarray(mask_uint8)
    if mask_uint8.ndim != 2:
        mask_uint8 = cv2.cvtColor(mask_uint8, cv2.COLOR_BGR2GRAY)

    if mask_uint8.max() == 0:
        return None

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # choose largest contour (by area)
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return None

    # oriented bounding box
    rect = cv2.minAreaRect(largest)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect)        # 4x2 float
    box = np.round(box).astype(np.int32)  # convert to int
    return box


def reorder_corners_clockwise(corners):
    pts = np.array(corners, dtype=float)
    cx, cy = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(angles)  # CCW
    pts_sorted = pts[order]
    idx_tl = np.lexsort((pts_sorted[:, 0], pts_sorted[:, 1]))[0]
    pts_reordered = np.roll(pts_sorted, -idx_tl, axis=0)
    return pts_reordered.tolist()


def generate_color_palette(n):
    """
    Generate a color palette with n distinct colors.
    Returns list of BGR tuples (for OpenCV).
    """
    # Predefined color palette with distinct colors
    base_colors = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (255, 165, 0),    # Orange
        (0, 128, 255),    # Orange-red
        (128, 128, 0),    # Teal
        (255, 192, 203),  # Pink
        (0, 255, 128),    # Spring green
        (128, 0, 0),      # Maroon
        (0, 128, 128),    # Olive
        (192, 192, 192),  # Silver
    ]
    
    if n <= len(base_colors):
        return base_colors[:n]
    
    # If we need more colors, generate them by varying hue
    colors = list(base_colors)
    for i in range(len(base_colors), n):
        hue = int(180 * i / n)  # OpenCV uses 0-180 for hue
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color)))
    
    return colors


def apply_nms(results, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression (NMS) to the results.
    Since results contain Oriented Bounding Boxes (OBBs), we approximate
    them as Axis-Aligned Bounding Boxes (AABBs) for the NMS calculation.
    """
    if not results:
        return []

    # 1. Convert OBB corners to AABB [x1, y1, x2, y2]
    # x1=min_x, y1=min_y, x2=max_x, y2=max_y
    boxes = []
    scores = []
    for r in results:
        pts = np.array(r['obb_corners'])
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        boxes.append([x1, y1, x2, y2])
        scores.append(r['score'])

    boxes = np.array(boxes)
    scores = np.array(scores)

    # 2. Sort by score descending
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(results[i])
        
        # If this was the last one, break
        if order.size == 1:
            break

        # Calculate IoU of the kept box with the rest
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # Area of the boxes
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])

        union = area_i + area_rest - inter
        
        # Avoid division by zero
        iou = inter / (union + 1e-6)

        # Keep only boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        
        # Because we sliced order[1:], the indices are shifted by 1
        order = order[inds + 1]

    print(f"NMS: Reduced {len(results)} detections to {len(keep)} (Threshold: {iou_threshold})")
    return keep


def get_prediction(model, processor, image_path, prompt, device,
                   score_threshold=0.5, min_area=100, debug=False):
    """
    Runs SAM3 inference and returns raw candidate results.
    """
    from PIL import Image
    decode_started = time.monotonic()
    image = Image.open(image_path).convert("RGB")
    decode_duration = time.monotonic() - decode_started
    image_started = time.monotonic()
    inference_state = processor.set_image(image)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    image_duration = time.monotonic() - image_started
    text_started = time.monotonic()
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    text_duration = time.monotonic() - text_started
    logger.info(
        "SAM3 stages | request_id=%s | dimensions=%sx%s | image_decode_s=%.3f | image_encoder_s=%.3f | text_and_grounding_s=%.3f",
        current_request_id(), image.width, image.height, decode_duration, image_duration, text_duration,
    )

    masks = None
    scores = None

    # Handle various output formats
    if isinstance(output, dict):
        if "masks" in output: masks = output["masks"]
        elif "pred_masks" in output: masks = output["pred_masks"]
        elif "masks_pred" in output: masks = output["masks_pred"]
        
        if "scores" in output: scores = output["scores"]
        elif "pred_scores" in output: scores = output["pred_scores"]
    else:
        for k in ("masks", "pred_masks", "masks_pred"):
            if hasattr(output, k):
                masks = getattr(output, k); break
        for k in ("scores", "pred_scores"):
            if hasattr(output, k):
                scores = getattr(output, k); break

    masks_np = None
    if masks is not None:
        masks_np = masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.array(masks)

    scores_np = None
    if scores is not None:
        scores_np = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.array(scores)
        raw_scores = np.asarray(scores_np).reshape(-1)
        logger.info(
            "SAM3 raw candidates | request_id=%s | prompt=%r | raw_masks=%d | top_scores=%s | score_range=%s | count_at_threshold=%d | post_filter_threshold=%.3f",
            current_request_id(),
            prompt,
            int(masks_np.shape[0]) if masks_np is not None and masks_np.ndim else 0,
            [round(float(score), 4) for score in sorted(raw_scores, reverse=True)[:10]],
            (round(float(raw_scores.min()), 4), round(float(raw_scores.max()), 4)) if raw_scores.size else None,
            int(np.count_nonzero(raw_scores >= score_threshold)),
            score_threshold,
        )

    raw_candidate_count = int(masks_np.shape[0]) if masks_np is not None and masks_np.ndim else 0

    if masks_np is None:
        return [], None, 0

    postprocess_started = time.monotonic()
    # Ensure masks shape is (N, H, W)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    elif masks_np.ndim == 3:
        pass # Assume (N, H, W)
    else:
        masks_np = np.any(masks_np, axis=tuple(range(1, masks_np.ndim - 2))).astype(np.float32)
        if masks_np.ndim == 2: masks_np = masks_np[None, ...]

    results = []
    num_masks = masks_np.shape[0]

    for i in range(num_masks):
        single_mask = masks_np[i]
        score = float(scores_np[i]) if (scores_np is not None and i < len(scores_np)) else None

        if (score is not None) and (score < score_threshold):
            continue

        bin_mask = (single_mask > 0.5).astype(np.uint8)
        area = int(bin_mask.sum())

        if area < min_area:
            continue

        obb = mask_to_oriented_bbox(single_mask, debug=debug)
        if obb is None:
            continue

        obb_reordered = reorder_corners_clockwise(obb)

        results.append({
            "index": int(i),
            "score": float(score) if score is not None else 0.0,
            "area": int(area),
            "obb_corners": [[int(float(x)), int(float(y))] for x, y in obb_reordered]
        })

    # Sort candidates by score initially
    results.sort(key=lambda x: -x["score"])
    logger.info(
        "SAM3 postprocessing complete | request_id=%s | prompt=%r | duration_s=%.3f | raw_count=%d | kept=%d",
        current_request_id(), prompt, time.monotonic() - postprocess_started,
        raw_candidate_count, len(results),
    )

    return results, masks_np, raw_candidate_count


def load_clip_context(model_name, pretrained, device, disable_clip=False):
    """
    Lazily load an open_clip model for image-text similarity checks.
    Returns None if CLIP is unavailable or disabled.
    """
    if disable_clip:
        print("CLIP filtering disabled via flag.")
        return None

    if open_clip is None:
        print("Warning: open_clip is not installed. Skipping CLIP-based filtering.")
        return None

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
    except Exception as e:
        print(f"Warning: Unable to initialize CLIP model '{model_name}/{pretrained}': {e}")
        return None

    model.to(device)
    model.eval()
    return {
        "model": model,
        "preprocess": preprocess,
        "tokenizer": tokenizer,
        "model_name": model_name,
        "pretrained": pretrained,
    }


def _crop_instance(image, obb_corners, padding_ratio=0.05):
    """
    Crop an axis-aligned patch around the OBB (with optional padding).
    """
    if not obb_corners:
        return None

    width, height = image.size
    pts = np.array(obb_corners)
    min_x = int(np.floor(pts[:, 0].min()))
    min_y = int(np.floor(pts[:, 1].min()))
    max_x = int(np.ceil(pts[:, 0].max()))
    max_y = int(np.ceil(pts[:, 1].max()))

    pad_w = int((max_x - min_x) * padding_ratio)
    pad_h = int((max_y - min_y) * padding_ratio)

    min_x = max(0, min_x - pad_w)
    min_y = max(0, min_y - pad_h)
    max_x = min(width, max_x + pad_w)
    max_y = min(height, max_y + pad_h)

    if max_x <= min_x or max_y <= min_y:
        return None

    return image.crop((min_x, min_y, max_x, max_y))


def score_instances_with_clip(image_path, instances, text_prompt, clip_context,
                              device, padding_ratio=0.05):
    """
    Compute CLIP similarity for each instance crop against the provided text prompt.
    Returns a list of scores aligned with `instances`.
    """
    if not instances or clip_context is None:
        return [None] * len(instances)

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Warning: Could not open image for CLIP scoring ({e}).")
        return [None] * len(instances)

    model = clip_context["model"]
    preprocess = clip_context["preprocess"]
    tokenizer = clip_context["tokenizer"]
    safe_prompt = (text_prompt or "").strip() or "object"

    try:
        text_tokens = tokenizer([safe_prompt])
    except Exception as e:
        print(f"Warning: Failed to tokenize prompt for CLIP ({e}).")
        return [None] * len(instances)

    text_tokens = text_tokens.to(device)

    with torch.no_grad():
        try:
            text_features = model.encode_text(text_tokens)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
        except Exception as e:
            print(f"Warning: Failed to encode text for CLIP scoring ({e}).")
            return [None] * len(instances)

    scores = []
    with torch.no_grad():
        for inst in instances:
            crop = _crop_instance(image, inst.get("obb_corners"), padding_ratio=padding_ratio)
            if crop is None:
                scores.append(None)
                continue

            try:
                image_tensor = preprocess(crop).unsqueeze(0).to(device)
                image_features = model.encode_image(image_tensor)
                image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
                score = (image_features @ text_features.T).squeeze().item()
                scores.append(float(score))
            except Exception as e:
                print(f"Warning: Failed to encode crop for CLIP scoring ({e}).")
                scores.append(None)

    return scores


def visualize_result(image_path, all_results, original_mask_list, output_path="output.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        return

    overlay = img.copy()

    for r in all_results:
        idx = r["index"]
        # Get color from result (should be assigned before calling this function)
        # Color is stored as RGB in JSON, but OpenCV uses BGR
        color = r.get("color", [0, 255, 0])  # Default to green if not set (RGB)
        if isinstance(color, (list, tuple)) and len(color) == 3:
            # Convert RGB to BGR for OpenCV
            color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        else:
            color_bgr = (0, 255, 0)  # Default green in BGR
        
        if original_mask_list is not None and idx < len(original_mask_list):
            mask = original_mask_list[idx]
            mask_bin = (mask > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(img)
            if mask_bin.ndim == 3: mask_bin = mask_bin[:,:,0]
            mask_resized = cv2.resize(mask_bin.astype(np.uint8), (colored_mask.shape[1], colored_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Use the same color for the mask overlay
            colored_mask[mask_resized == 1] = color_bgr
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)

        pts = np.array(r["obb_corners"], dtype=np.int32)
        if pts.shape[0] == 4:
            cv2.drawContours(overlay, [pts], 0, color_bgr, 2)
            cv2.circle(overlay, tuple(pts[0]), 4, color_bgr, -1)
            clip_score = r.get("clip_score")
            if clip_score is not None:
                label = f"{r['score']:.2f}|{clip_score:.2f}"
            else:
                label = f"{r['score']:.2f}"
            cv2.putText(overlay, label, tuple(pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    out = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    cv2.imwrite(output_path, out)
    print(f"Visualization saved to {output_path}")


# -----------------------------------------------------------------------------
# Referring object extractor (LLM prompt tidy-up)
# -----------------------------------------------------------------------------


class ReferringObjectExtractor:
        """Extract referring objects with descriptive adjectives from natural language queries using LLM."""

        def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", device: str = "auto"):
                print(f"Loading model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map=device,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                )
                self.model.eval()
                print("Model loaded successfully!")

        def create_prompt(self, query: str) -> str:
                prompt = f"""Extract the object being referred to from the following text, including any adjectives that describe it. Remove only spatial descriptions and locations (like 'on the left', 'in the corner', 'behind', etc.). Keep color, size, and other descriptive adjectives. Return the object with its adjectives.

Examples:
- Input: "football field on the top left corner of the image"
    Output: football field

- Input: "the red car in the middle of the street"
    Output: red car

- Input: "person wearing blue shirt on the right side"
    Output: person wearing blue shirt

- Input: "large tree behind the building"
    Output: large tree

- Input: "small white cat sitting on the floor"
    Output: small white cat

- Input: "wooden chair in the dining room corner"
    Output: wooden chair

- Input: "bright yellow umbrella held by the person"
    Output: bright yellow umbrella

Now extract the object from this text:
- Input: "{query}"
    Output:"""
                return prompt

        def extract(self, query: str, max_new_tokens: int = 20, temperature: float = 0.1) -> str:
                prompt = self.create_prompt(query)
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                with torch.no_grad():
                        outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                do_sample=True if temperature > 0 else False,
                                pad_token_id=self.tokenizer.eos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
                        )
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "Output:" in generated_text:
                        parts = generated_text.split("Output:")
                        answer = parts[-1].strip()
                        answer = answer.split("\n")[0].strip()
                        return answer
                return generated_text.strip()


# -----------------------------------------------------------------------------
# Qwen helper
# -----------------------------------------------------------------------------

class QwenVisionAgent:
    def __init__(self, model_name="qwen3-vl:30b"):
    # def _init_(self, model_name="qwen3-vl:8b-instruct-bf16"):
        self.model_name = model_name

    def generate(self, prompt, image_path, stream=False):
        """
        Sends a prompt and image to Ollama.
        Returns the full text response string.
        """
        try:
            # We use stream=False by default for programmatic use 
            # so we can return the whole string at once.
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }],
                stream=stream
            )

            # If streaming is requested, return the generator
            if stream:
                return response
            
            # Otherwise return the full content string
            return response['message']['content']

        except ollama.ResponseError as e:
            print(f"Error: {e.error}")
            return None

    def generate_json(self, prompt, image_path):
        """
        Helper that calls generate(), strips Markdown code blocks,
        and returns a Python dictionary (parsed JSON).
        """
        raw_text = self.generate(prompt, image_path)
        if not raw_text:
            return None

        # Clean up Markdown code blocks (e.g. ⁠ json ...  ⁠)
        clean_text = re.sub(r"⁠  json\s*", "", raw_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"  ⁠\s*$", "", clean_text)
        clean_text = clean_text.strip()

        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            print(f"JSON Parsing Failed. Raw output:\n{raw_text}")
            pattern = r"⁠  json\s*(\{.*?\})\s*  ⁠"
            match = re.search(pattern, clean_text, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    print("2nd attempt also failed")
                    return None

            return None


# -----------------------------------------------------------------------------
# Pipeline helpers
# -----------------------------------------------------------------------------


def run_refiner(args: argparse.Namespace) -> Dict[str, Any]:
    if args.disable_refiner:
        refined = args.sam_prompt or args.query
        target = args.target_prompt or args.query
        return {"prompt": refined, "target": target, "used": False}
    print("Running referring object extractor ...")
    extractor = ReferringObjectExtractor(
        model_name=args.refiner_model,
        device=args.refiner_device,
    )
    refined_query = extractor.extract(args.query, temperature=args.refiner_temperature)
    sam_prompt = args.sam_prompt or refined_query or args.query
    target_prompt = args.target_prompt or (refined_query if args.use_refiner_for_target else args.query)
    return {"prompt": sam_prompt, "target": target_prompt, "used": True}


def run_sam_pipeline(
    args: argparse.Namespace,
    device: str,
    sam_prompt: str,
    target_prompt: str,
):
    model = load_sam3_model(args.ckpt, device)
    processor = Sam3Processor(model)
    candidates, masks_np, _raw_candidate_count = get_prediction(
        model,
        processor,
        args.image,
        sam_prompt,
        device,
        score_threshold=args.score_threshold,
        min_area=args.min_area,
    )
    if not candidates:
        return [], masks_np
    final_results = apply_nms(candidates, iou_threshold=args.nms_threshold)
    clip_context = load_clip_context(
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        device=device,
        disable_clip=args.disable_clip,
    )
    if clip_context is None:
        print("CLIP filtering skipped (context unavailable).")
    if clip_context and final_results:
        clip_scores = score_instances_with_clip(
            image_path=args.image,
            instances=final_results,
            text_prompt=target_prompt,
            clip_context=clip_context,
            device=device,
            padding_ratio=max(0.0, args.clip_crop_padding),
        )
        for inst, clip_score in zip(final_results, clip_scores):
            inst["clip_score"] = float(clip_score) if clip_score is not None else None
            printable = f"{inst['clip_score']:.3f}" if inst["clip_score"] is not None else "None"
            print(f"CLIP score -> idx {inst['index']}: {printable}")
        keep = [
            inst
            for inst in final_results
            if inst.get("clip_score") is not None and inst["clip_score"] >= args.clip_threshold
        ]
        if not keep and final_results and args.clip_topk > 0:
            keep = sorted(
                final_results,
                key=lambda r: (r.get("clip_score") if r.get("clip_score") is not None else float("-inf")),
                reverse=True,
            )[: args.clip_topk]
            print(f"CLIP filtering removed all candidates; keeping top {len(keep)} as fallback.")
        final_results = keep
    else:
        for inst in final_results:
            inst["clip_score"] = None
    if not final_results:
        return [], masks_np
    palette = generate_color_palette(len(final_results))
    for idx, inst in enumerate(final_results):
        bgr = palette[idx]
        inst["color"] = [int(bgr[2]), int(bgr[1]), int(bgr[0])]
    return final_results, masks_np


def build_vlm_prompt(image_path: str, clip_threshold: float, target_prompt: str, json_payload: Dict[str, Any]) -> str:
    # Remove 'score' from each instance but keep the rest of the payload unchanged
    sanitized_payload = json.loads(json.dumps(json_payload))  # deep copy
    for inst in sanitized_payload.get("instances", []):
        inst.pop("score", None)
        inst.pop("clip_score", None)
    json_payload = sanitized_payload
    payload_literal = json.dumps(json_payload, indent=2)
    target_literal = json.dumps(target_prompt)
    
    return f"""
SYSTEM:
You are an exacting visual-language adjudicator. Inputs already attached:
1) Rendered image at {image_path}.
2) JSON payload of oriented bounding boxes with index, RGB color, SAM score, and area.

Protocol:
- Treat the JSON as immutable evidence—never fabricate boxes, keys, or coordinates.
- Inspect each box against the target concept {target_literal}; reject mismatched categories, colors, or pure background regions.
- If the concept spans adjacent boxes, include every contributing index.
- When no box qualifies, return selected_instances: [] and set best_guess.index to null with reason "NO SUITABLE BOX".
- Reasons must cite concrete visual cues from the image in ≤2 sentences and may not reference hidden metadata.
- Produce strict JSON only (no prose, markdown fences, or trailing commas).

USER:
Image source (already attached): {image_path}

Literal OBB payload (do NOT modify):
<OBB_JSON_START>
{payload_literal}
<OBB_JSON_END>

Goal:
Identify the bounding box(es) that best match the target concept: {target_literal}

Output schema (STRICT JSON, exact keys only):
{{
    "target_concept": {target_literal},
    "selected_instances": [
    {{
        "index": <int>,
        "reason": "<short 1-2 sentence visual evidence>"
    }}
    ],
    "best_guess": {{
        "index": <int|null>,
        "reason": "<short 1-2 sentence visual evidence>"
    }}
}}

Rules for best_guess:
- If selected_instances is non-empty, best_guess.index must match the single most compelling box (e.g., first element) and its reason must summarize why it best fits.
- If no box fits, set best_guess.index to null with reason "NO SUITABLE BOX".

Respond with JSON only."""


# -----------------------------------------------------------------------------
# CLI + main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified SAM3 + CLIP + Qwen inference")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--query", required=True, help="Original natural-language query")
    parser.add_argument("--sam-prompt", help="Override for SAM prompt")
    parser.add_argument("--target-prompt", help="Override for CLIP/Qwen prompt")
    parser.add_argument("--ckpt", default="sam.pt", help="Path to SAM3 checkpoint")
    parser.add_argument("--output-image", default="output.jpg", help="Visualization path")
    parser.add_argument("--json-out", default="intermediate_output.json", help="Intermediate JSON path")
    parser.add_argument("--json-vlm-out", default="final_output.json", help="VLM JSON output path")

    parser.add_argument("--score-threshold", type=float, default=0.8)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--nms-threshold", type=float, default=0.6)
    parser.add_argument("--clip-model", default="ViT-H-14")
    parser.add_argument("--clip-pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--clip-threshold", type=float, default=0.24)
    parser.add_argument("--clip-topk", type=int, default=2)
    parser.add_argument("--clip-crop-padding", type=float, default=0.15)
    parser.add_argument("--disable-clip", action="store_true")

    parser.add_argument("--disable-refiner", action="store_true")
    parser.add_argument("--refiner-model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--refiner-device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--refiner-temperature", type=float, default=0.1)
    parser.add_argument("--use-refiner-for-target", action="store_true")

    parser.add_argument("--disable-vlm", action="store_true")
    parser.add_argument("--qwen-model", default="qwen3-vl:30b")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    refiner_result = run_refiner(args)
    sam_prompt = refiner_result["prompt"]
    target_prompt = refiner_result["target"]
    print(f"SAM prompt: {sam_prompt}")
    print(f"Target prompt: {target_prompt}")

    results, masks_np = run_sam_pipeline(args, device, sam_prompt, target_prompt)

    if not results:
        print("No candidates remained after filtering; writing empty outputs.")
        empty_payload = {"instances": [], "meta": {"target_prompt": target_prompt}}
        with open(args.json_out, "w") as f:
            json.dump(empty_payload, f, indent=2)
        if not args.disable_vlm:
            with open(args.json_vlm_out, "w") as f:
                json.dump({"selected_instances": [], "best_guess": {"index": None}}, f, indent=2)
        return

    json_payload = {
        "instances": results,
        "meta": {
            "target_prompt": target_prompt,
            "clip_model": None if args.disable_clip else args.clip_model,
            "clip_pretrained": None if args.disable_clip else args.clip_pretrained,
            "clip_threshold": args.clip_threshold,
            "nms_threshold": args.nms_threshold,
            "refiner_used": refiner_result["used"],
        },
    }

    with open(args.json_out, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"Wrote {len(results)} instances to {args.json_out}")

    final_visual_instances = results

    if args.disable_vlm:
        print("Skipping VLM reasoning step as requested.")
        visualize_result(args.image, final_visual_instances, masks_np, args.output_image)
        return

    try:
        agent = QwenVisionAgent(model_name=args.qwen_model)
    except RuntimeError as exc:
        print(str(exc))
        visualize_result(args.image, final_visual_instances, masks_np, args.output_image)
        return

    vlm_prompt = build_vlm_prompt(
        image_path=args.output_image,
        clip_threshold=args.clip_threshold,
        target_prompt=target_prompt,
        json_payload=json_payload,
    )
    result = None
    filtered_instances = []

    try:
        result = agent.generate_json(vlm_prompt, args.output_image)
        with open(args.json_vlm_out, "w") as f:
            json.dump(result, f, indent=2)
        
        if result and "selected_instances" in result:
            # Find the candidate with the matching index
            for filtered in result["selected_instances"]:
                best_index = filtered["index"]
                for candidate in final_visual_instances:
                    if candidate["index"] == best_index:
                        filtered_instances.append(candidate)
                        break
        
    except Exception as e:
        print(f"Warning: VLM reasoning failed ({e})")
        if candidates:
            return max(candidates, key=lambda x: x.get("score", x.get("score", 0.0)))
        return None

    visualize_result(args.image, filtered_instances, masks_np, args.output_image)

    return filtered_instances


if __name__ == "__main__":
    main()
