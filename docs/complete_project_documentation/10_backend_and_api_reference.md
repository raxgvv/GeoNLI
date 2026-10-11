# 10 — Backend and API Reference

## Application Startup

### Entry Point
```bash
python -m backend.app
```

### Startup Sequence
1. `create_app()` is called [Code: backend/app.py:50]
2. `_configure_logging()` sets up file + console logging [Code: backend/app.py:13-48]
3. Flask app created with CORS enabled [Code: backend/app.py:54-61]
4. `Config.init_app()` creates upload/weights directories [Code: backend/config.py:59-62]
5. Request hooks registered (timing, context) [Code: backend/app.py:63-91]
6. Blueprint registered at `/geoNLI` [Code: backend/app.py:94]
7. Flask runs on `BACKEND_HOST:BACKEND_PORT` [Code: backend/app.py:106-110]

**Note:** Models are NOT loaded at startup. They load lazily on first use.

---

## API Endpoints

### `GET /geoNLI/health`

**Purpose:** Liveness probe.

**Request:** No body required.

**Response (200):**
```json
{
  "status": "active",
  "success": true,
  "timestamp": "2026-10-10T12:00:00+00:00"
}
```

**Implementation:** [Code: backend/api/routes.py:341-343]

---

### `POST /geoNLI/eval`

**Purpose:** Main inference endpoint. Accepts both JSON and multipart requests.

---

#### Multipart Request (Frontend)

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Values |
|---|---|---|---|
| `image` | File | Yes | PNG/JPEG upload (≤16 MB) |
| `task_type` | String | Yes | `captioning`, `vqa`, `grounding` |
| `vqa_type` | String | For `vqa` | `semantic`, `binary`, `numeric` |
| `image_type` | String | Yes | `RGB`, `FCC`, `SAR` |
| `query` | String | For `vqa`/`grounding` | Natural language |
| `spatial_resolution_m` | Float | Yes | > 0 |

**Example Request:**
```bash
curl -X POST http://localhost:8000/geoNLI/eval \
  -F "image=@satellite.png" \
  -F "task_type=vqa" \
  -F "vqa_type=binary" \
  -F "image_type=RGB" \
  -F "query=Is there a river?" \
  -F "spatial_resolution_m=1.57"
```

**Success Response (200):**
```json
{
  "success": true,
  "task": "vqa",
  "answer": "Yes",
  "response": "Yes",
  "bboxes": [],
  "request_id": "a1b2c3d4e5f6",
  "timestamp": "2026-10-10T12:00:05+00:00"
}
```

**Error Response (400/415/503/504):**
```json
{
  "success": false,
  "error": "task_type must be captioning, vqa, or grounding",
  "request_id": "a1b2c3d4e5f6",
  "timestamp": "2026-10-10T12:00:05+00:00"
}
```

---

#### JSON Request (Evaluation Harness)

**Content-Type:** `application/json`

**Request Schema:**
```json
{
  "input_image": {
    "image_url": "https://example.com/scene.jpg",
    "mode": "RGB",
    "metadata": {
      "spatial_resolution_m": 1.57
    }
  },
  "queries": {
    "caption_query": {
      "instruction": "Describe"
    },
    "grounding_query": {
      "instruction": "Locate aircraft"
    },
    "attribute_query": {
      "binary": {
        "instruction": "Is there water visible?"
      },
      "numeric": {
        "instruction": "How many vehicles are there?"
      },
      "semantic": {
        "instruction": "What terrain is shown?"
      }
    }
  }
}
```

**Response Schema:**
The endpoint echoes the input payload with each query's `response` populated:

```json
{
  "input_image": {
    "image_url": "https://example.com/scene.jpg",
    "mode": "RGB",
    "metadata": {"spatial_resolution_m": 1.57},
    "converted_rgb_path": "/tmp/inter-iit/fcc_rgb/scene_uuid.png"
  },
  "queries": {
    "caption_query": {
      "instruction": "Describe",
      "response": "A satellite view of an urban area."
    },
    "grounding_query": {
      "instruction": "Locate aircraft",
      "response": [
        {"object-id": "1", "obbox": [0.1, 0.2, 0.4, 0.2, 0.4, 0.5, 0.1, 0.5]}
      ]
    },
    "attribute_query": {
      "binary": {
        "instruction": "Is there water visible?",
        "response": "Yes",
        "route": "VLM_ONLY",
        "bboxes": [],
        "filtered_instances": []
      },
      "numeric": {
        "instruction": "How many vehicles are there?",
        "response": 42,
        "route": "VLM_ONLY",
        "numeric_type": "NONE",
        "numeric_value": 42.0,
        "bboxes": [],
        "filtered_instances": []
      },
      "semantic": {
        "instruction": "What terrain is shown?",
        "response": "Urban residential area"
      }
    }
  },
  "success": true,
  "timestamp": "2026-10-10T12:00:05+00:00",
  "request_id": "a1b2c3d4e5f6"
}
```

---

## Validation Rules

| Rule | Error | Status |
|---|---|---|
| Missing `input_image.image_url` (JSON) | "Missing input_image.image_url" | 400 |
| Invalid JSON payload | "Invalid JSON payload" | 400 |
| Missing image file (multipart) | "No image file was selected" | 400 |
| Invalid image data | "Could not decode the uploaded image" | 400 |
| Invalid `task_type` | "task_type must be captioning, vqa, or grounding" | 400 |
| Invalid `vqa_type` | "vqa_type must be semantic, binary, or numeric" | 400 |
| Missing query for VQA/grounding | "query is required for VQA and grounding" | 400 |
| Invalid `spatial_resolution_m` | "spatial_resolution_m must be greater than zero" | 400 |
| Unsupported media type | "Expected multipart image upload or JSON request" | 415 |
| SAM3 failure | "Inference failed..." | 503 |
| SAR conversion failure | "SAR conversion unavailable..." | 503 |
| VLM timeout | "Vision-language inference exceeded its time limit..." | 504 |
| VLM OOM | "Inference ran out of GPU memory..." | 500 |

---

## Model Calls by Endpoint

| task_type | Model Called | GPU Lock |
|---|---|---|
| `captioning` | Qwen3-VL-2B (CaptioningBackend) | Yes |
| `vqa` (binary) | Qwen3-VL-2B or fine-tuned 8B | Yes |
| `vqa` (numeric) | Qwen3-VL-2B or fine-tuned 8B | Yes |
| `vqa` (semantic) | Qwen3-VL-2B or fine-tuned 8B | Yes |
| `grounding` | SAM3 | Yes |

---

## Error Handling

### Error Sanitization
Internal errors are converted to user-friendly messages [Code: backend/api/routes.py:36-44]:

```python
def _public_inference_error(exc):
    message = str(exc)
    if isinstance(exc, TimeoutError) or 'time limit' in message.lower():
        return 'Vision-language inference exceeded its time limit. Retry or configure a smaller compatible model.'
    if 'out of memory' in message.lower() and ('cuda' in message.lower() or 'gpu' in message.lower()):
        return 'Inference ran out of GPU memory. Close GPU-heavy applications or use a smaller image, then retry.'
    logger.error("Inference exception | request_id=%s | type=%s | message=%s",
                 current_request_id(), type(exc).__name__, message)
    return 'Inference failed. Check the server logs using the request ID for details.'
```

**Security:** Internal error details (stack traces, file paths) are never exposed to the client. The request_id allows server-side correlation.

---

## Logging

### Log Format
```
2026-10-10 12:00:05 | INFO | backend.http | Request started | request_id=a1b2c3d4e5f6 | method=POST | path=/geoNLI/eval
```

### Log Destinations
- **File:** `GEONLI_LOG` env var (default: `geonli.log`)
- **Console:** stdout

### Key Log Events

| Event | Level | Location |
|---|---|---|
| Request started | INFO | backend/app.py:69-72 |
| HTTP response | INFO | backend/app.py:76-83 |
| Inference started | INFO | backend/core/model.py:104-105 |
| Inference complete | INFO | backend/core/model.py:107-108 |
| VLM placement changed | INFO | models/captioning.py:149-150 |
| SAM3 placement changed | INFO | models/sam3/runtime_service.py:138-139 |
| Prediction failed | ERROR | backend/core/model.py:248 |
| Inference exception | ERROR | backend/api/routes.py:42-43 |

---

## Timeout Behavior

| Operation | Timeout | Evidence |
|---|---|---|
| Image URL download | 10 seconds | [Code] backend/utils/image_ops.py:14 |
| Caption generation | 50 seconds | [Code] backend/config.py:36 |
| VLM max_time | Passed to generate() | [Code] models/captioning.py:97 |
| Frontend API call | 90 seconds | [Code] ui/frontend/src/services/api.js:38 |

---

## Resource Implications

| Task | GPU Memory | CPU RAM | Disk I/O |
|---|---|---|---|
| Captioning | ~4-6 GB (VLM 2B bf16) | ~8 GB | None |
| VQA | ~4-6 GB (VLM 2B) or ~18 GB (8B) | ~8-20 GB | None |
| Grounding | ~6-10 GB (SAM3) | ~12 GB | Write upload + output |
| SAR conversion | ~2-4 GB (Pix2Pix) | ~4 GB | Write output |

---

*Next: [Novelty and Research Contributions](11_novelty_and_research_contributions.md)*
