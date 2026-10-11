import json
import logging
import os
import time
from datetime import datetime, timezone
from io import BytesIO
from urllib.parse import urlparse

from flask import Blueprint, request, jsonify
from werkzeug.exceptions import BadRequest

from backend.core.model import get_model
from backend.utils.image_ops import (
    load_image_from_url,
    load_image_from_file,
    convert_fcc_to_rgb,
    save_converted_fcc_image,
)
from backend.services.sam3_runner import run_sam3_grounding
from backend.services.sar_converter import convert_sar_to_rgb
from backend.services.request_context import current_request_id

api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)


def _timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _error(message, status, details=None):
    payload = {'success': False, 'error': message, 'request_id': current_request_id(), 'timestamp': _timestamp()}
    return jsonify(payload), status


def _public_inference_error(exc):
    message = str(exc)
    if isinstance(exc, TimeoutError) or 'time limit' in message.lower():
        return 'Vision-language inference exceeded its time limit. Retry or configure a smaller compatible model.'
    if 'out of memory' in message.lower() and ('cuda' in message.lower() or 'gpu' in message.lower()):
        return 'Inference ran out of GPU memory. Close GPU-heavy applications or use a smaller image, then retry.'
    logger.error("Inference exception | request_id=%s | type=%s | message=%s",
                 current_request_id(), type(exc).__name__, message)
    return 'Inference failed. Check the server logs using the request ID for details.'

@api_bp.route('/eval', methods=['POST'])
def analyze():
    request_started = time.monotonic()
    logger.info("/geoNLI/eval request | request_id=%s | content_type=%s", current_request_id(), request.content_type)
    try:
        # ---------------------------------------------------------
        # 1. Handle JSON Request (Evaluation Bot)
        # ---------------------------------------------------------
        if request.is_json:
            try:
                data = request.get_json()
            except BadRequest:
                logger.warning("Invalid JSON payload received")
                return _error('Invalid JSON payload', 400)
            response_data = data.copy()
            
            # A. Load Image
            input_image_data = data.get('input_image', {})
            image_url = input_image_data.get('image_url')
            if not image_url:
                return _error('Missing input_image.image_url', 400)
            
            # Extract spatial resolution from metadata
            metadata = input_image_data.get('metadata', {})
            spatial_resolution_m = metadata.get('spatial_resolution_m', 1.0)  # Default to 1.0m if not provided
            
            image_mode = (input_image_data.get('mode') or input_image_data.get('image_type') or 'RGB').upper()
            image = load_image_from_url(image_url)
            converted_path = None
            orig_name = os.path.basename(urlparse(image_url).path)
            try:
                if image_mode == 'FCC':
                    image = convert_fcc_to_rgb(image)
                    converted_path = save_converted_fcc_image(image, orig_name)
                    image_mode = 'RGB'
                elif image_mode == 'SAR':
                    image, converted_path = convert_sar_to_rgb(image, orig_name)
                    image_mode = 'RGB'
            except Exception as exc:
                logger.exception("Failed to convert %s imagery", image_mode)
                message = f'SAR conversion unavailable: {exc}' if image_mode == 'SAR' else f'{image_mode} conversion failed'
                return _error(message, 503 if image_mode == 'SAR' else 500)
            
            # B. Process Queries
            queries = response_data.get('queries', {})
            
            # Handle grounding query separately using SAM3 (like UI requests)
            if 'grounding_query' in queries:
                grounding_instruction = queries['grounding_query'].get('instruction', '')
                if grounding_instruction.strip():
                    try:
                        # Convert PIL image to bytes for SAM3
                        buffer = BytesIO()
                        image.save(buffer, format='PNG')
                        image_bytes = buffer.getvalue()
                        
                        sam3_payload = run_sam3_grounding(
                            image_bytes,
                            grounding_instruction,
                            f"{orig_name}_grounding.png",
                            spatial_resolution_m=spatial_resolution_m,
                        )
                        
                        # Create mock results structure with SAM3 data
                        results = {'grounding': {
                            'bboxes': sam3_payload.get('bboxes', []),
                            'answer': f"Found {len(sam3_payload.get('bboxes', []))} objects"
                        }}
                        
                        logger.info("SAM3 grounding for JSON completed | query=%s | objects=%d", 
                                  grounding_instruction, len(sam3_payload.get('bboxes', [])))
                    except Exception as exc:
                        logger.exception("Inference failed | task=grounding | model=SAM3 | request=json")
                        return _error(_public_inference_error(exc), 503)
                else:
                    results = {'grounding': {'bboxes': [], 'answer': 'No grounding instruction provided'}}
            else:
                results = {}
            
            # Run other inference tasks (caption, VQA) through model
            other_queries = {k: v for k, v in queries.items() if k != 'grounding_query'}
            if other_queries:
                model = get_model()
                model_results = model.process_all_queries(image, image_mode, other_queries, spatial_resolution_m=spatial_resolution_m)
                results.update(model_results)

            # C. Map Results back to Schema
            
            # Caption
            if 'caption' in results and 'caption_query' in queries:
                queries['caption_query']['response'] = results['caption']['answer']

            # Grounding
            if 'grounding' in results and 'grounding_query' in queries:
                # Check if we have SAM3 grounding results with 8-coordinate bboxes
                if 'grounding' in results and results['grounding'].get('bboxes'):
                    # SAM3 provides 8-coordinate corner points [x1,y1,x2,y2,x3,y3,x4,y4]
                    raw_bboxes = results['grounding'].get('bboxes', [])
                    formatted_bboxes = []
                    for idx, bbox in enumerate(raw_bboxes):
                        if len(bbox) == 8:
                            # Use 8-coordinate format directly for corners
                            formatted_bboxes.append({
                                "object-id": str(idx + 1),
                                "obbox": bbox  # Keep 8 coordinates as-is
                            })
                        else:
                            # Fallback for 5-param format [cx, cy, w, h, angle]
                            formatted_bbox = bbox if len(bbox) == 5 else bbox + [0.0]
                            formatted_bboxes.append({
                                "object-id": str(idx + 1),
                                "obbox": formatted_bbox
                            })
                    queries['grounding_query']['response'] = formatted_bboxes
                else:
                    queries['grounding_query']['response'] = []

            # Attributes
            if 'attribute_query' in queries:
                attr_q = queries['attribute_query']
                if 'attr_binary' in results and 'binary' in attr_q:
                    binary_result = results['attr_binary']
                    attr_q['binary']['response'] = binary_result.get('answer', '')
                    attr_q['binary']['route'] = binary_result.get('route', 'VLM_ONLY')
                    # Format bboxes as objects with object-id and obbox (8-coordinates)
                    raw_bboxes = binary_result.get('bboxes', [])
                    formatted_bboxes = []
                    for idx, bbox in enumerate(raw_bboxes):
                        if len(bbox) >= 8:
                            formatted_bboxes.append({
                                "object-id": str(idx + 1),
                                "obbox": bbox[:8]  # Ensure 8 coordinates
                            })
                    attr_q['binary']['bboxes'] = formatted_bboxes
                    attr_q['binary']['filtered_instances'] = binary_result.get('filtered_instances', [])
                if 'attr_numeric' in results and 'numeric' in attr_q:
                    numeric_result = results['attr_numeric']
                    attr_q['numeric']['response'] = numeric_result.get('answer', '')
                    attr_q['numeric']['route'] = numeric_result.get('route', 'VLM_ONLY')
                    attr_q['numeric']['numeric_type'] = numeric_result.get('numeric_type', 'NONE')
                    attr_q['numeric']['numeric_value'] = numeric_result.get('numeric_value')
                    # Format bboxes as objects with object-id and obbox (8-coordinates)
                    raw_bboxes = numeric_result.get('bboxes', [])
                    formatted_bboxes = []
                    for idx, bbox in enumerate(raw_bboxes):
                        if len(bbox) >= 8:
                            formatted_bboxes.append({
                                "object-id": str(idx + 1),
                                "obbox": bbox[:8]  # Ensure 8 coordinates
                            })
                    attr_q['numeric']['bboxes'] = formatted_bboxes
                    attr_q['numeric']['filtered_instances'] = numeric_result.get('filtered_instances', [])
                if 'attr_semantic' in results and 'semantic' in attr_q:
                    attr_q['semantic']['response'] = results['attr_semantic']['answer']
            
            if converted_path:
                response_data.setdefault('input_image', {})['converted_rgb_path'] = converted_path

            response_data.update({'success': True, 'timestamp': _timestamp()})
            response_data['request_id'] = current_request_id()
            logger.info("JSON pipeline completed successfully")
            return jsonify(response_data), 200

        # ---------------------------------------------------------
        # 2. Handle Frontend Upload (Simplified)
        # ---------------------------------------------------------
        elif 'image' in request.files:
            file = request.files['image']
            if not file.filename:
                return _error('No image file was selected', 400)
            image, image_bytes = load_image_from_file(file, return_bytes=True)
            decode_duration = time.monotonic() - request_started
            logger.info("Decoded upload | request_id=%s | dimensions=%sx%s | decode_s=%.3f | mode=%s",
                        current_request_id(), image.width, image.height, decode_duration, image.mode)
            converted_path = None
            
            # Frontend sends simple key-value pairs, not the complex schema.
            # We mock a query structure to reuse the parallel engine.
            task_type = request.form.get('task_type', 'vqa')
            vqa_type = request.form.get('vqa_type', 'semantic')
            image_type = (request.form.get('image_type', 'RGB') or 'RGB').upper()
            query_text = request.form.get('query', '')
            spatial_resolution_m = float(request.form.get('spatial_resolution_m', 1.0))
            if not spatial_resolution_m > 0:
                return _error('spatial_resolution_m must be greater than zero', 400)
            if task_type not in {'captioning', 'vqa', 'grounding'}:
                return _error('task_type must be captioning, vqa, or grounding', 400)
            if vqa_type not in {'semantic', 'binary', 'numeric'}:
                return _error('vqa_type must be semantic, binary, or numeric', 400)
            if task_type != 'captioning' and not query_text.strip():
                return _error('query is required for VQA and grounding', 400)

            conversion_applied = False
            try:
                if image_type == 'FCC':
                    image = convert_fcc_to_rgb(image)
                    converted_path = save_converted_fcc_image(image, getattr(file, 'filename', None))
                    image_type = 'RGB'
                    conversion_applied = True
                elif image_type == 'SAR':
                    image, converted_path = convert_sar_to_rgb(image, getattr(file, 'filename', None))
                    image_type = 'RGB'
                    conversion_applied = True
            except Exception as exc:
                logger.exception("Failed to convert %s imagery", image_type)
                message = f'SAR conversion unavailable: {exc}' if image_type == 'SAR' else f'{image_type} conversion failed'
                return _error(message, 503)

            if conversion_applied:
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()

            logger.info(
                "Processing upload | request_id=%s | dimensions=%sx%s | task=%s | vqa_type=%s | query=%r | filename=%s",
                current_request_id(), image.width, image.height,
                task_type,
                vqa_type,
                query_text[:160],
                getattr(file, 'filename', 'unknown'),
            )
            
            if task_type == 'grounding':
                started = time.monotonic()
                try:
                    sam3_payload = run_sam3_grounding(
                        image_bytes,
                        query_text,
                        getattr(file, 'filename', None),
                        spatial_resolution_m=spatial_resolution_m,
                    )
                    if converted_path:
                        sam3_payload['converted_rgb_path'] = converted_path
                    logger.info("Inference complete | request_id=%s | task=grounding | model=SAM3 | duration_s=%.3f | detections=%d",
                                current_request_id(), time.monotonic() - started, len(sam3_payload.get('bboxes', [])))
                    sam3_payload.update({'success': True, 'task': 'grounding', 'response': sam3_payload.get('answer', ''), 'request_id': current_request_id(), 'timestamp': _timestamp()})
                    return jsonify(sam3_payload), 200
                except Exception as exc:
                    logger.exception("Inference failed | task=grounding | model=SAM3")
                    return _error(_public_inference_error(exc), 503)

            mock_queries = {}
            
            if task_type == 'captioning':
                mock_queries['caption_query'] = {'instruction': 'Describe'}
            else: # vqa
                # Use vqa_type to target the correct sub-model
                mock_queries['attribute_query'] = {vqa_type: {'instruction': query_text}}

            started = time.monotonic()
            logger.info("Inference started | request_id=%s | task=%s | model=shared Qwen3-VL", current_request_id(), task_type)
            model = get_model()
            results = model.process_all_queries(image, image_type, mock_queries, spatial_resolution_m=spatial_resolution_m)
            result_key = 'caption' if task_type == 'captioning' else f'attr_{vqa_type}'
            if result_key not in results:
                raise RuntimeError(f"{task_type} inference did not return a result")
            if results[result_key].get('error'):
                if results[result_key].get('error_type') == 'TimeoutError':
                    return _error(_public_inference_error(TimeoutError(results[result_key]['error'])), 504)
                raise RuntimeError(results[result_key]['error'])
            if not str(results[result_key].get('answer', '')).strip():
                raise RuntimeError(f"{task_type} inference returned an empty answer")
            
            # Flatten results for frontend
            flat_response = {"success": True, "task": task_type, "answer": "", "response": "", "bboxes": [], "request_id": current_request_id(), "timestamp": _timestamp()}
            
            if 'caption' in results: flat_response['answer'] = results['caption']['answer']
            if 'grounding' in results: flat_response['bboxes'] = results['grounding'].get('bboxes', [])
            # Extract specific VQA result
            attr_key = f'attr_{vqa_type}'
            if attr_key in results: 
                flat_response['answer'] = results[attr_key].get('answer', '')
                flat_response['bboxes'] = results[attr_key].get('bboxes', flat_response['bboxes'])

            if converted_path:
                flat_response['converted_rgb_path'] = converted_path

            flat_response['response'] = flat_response['answer']

            logger.info("Inference complete | request_id=%s | task=%s | model=shared Qwen3-VL | duration_s=%.3f | status=200",
                        current_request_id(), task_type, time.monotonic() - started)
            return jsonify(flat_response), 200

        else:
            logger.warning("Unsupported media type: %s", request.content_type)
            return _error('Expected multipart image upload or JSON request', 415)

    except Exception as e:
        logger.exception("Server error while handling request")
        status = 400 if isinstance(e, ValueError) else 500
        message = str(e) if status == 400 else _public_inference_error(e)
        if status == 400 and str(e).startswith('Failed to process uploaded file:'):
            message = 'Could not decode the uploaded image. Please upload a valid PNG, JPEG, or supported image.'
        return _error(message, status)

@api_bp.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "active", "success": True, "timestamp": _timestamp()}), 200
