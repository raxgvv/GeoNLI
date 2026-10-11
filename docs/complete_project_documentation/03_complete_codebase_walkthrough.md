# 03 — Complete Codebase Walkthrough

This document walks through every relevant source file in the repository, explaining its purpose, structure, and role in the pipeline.

---

## Repository Structure

```
Inter-IIT/
├── backend/                    # Flask inference server
│   ├── app.py                  # Application factory, logging, CORS
│   ├── config.py               # Environment-driven configuration
│   ├── api/routes.py           # /geoNLI/eval and /geoNLI/health endpoints
│   ├── core/model.py           # Model manager, VLM/VQA wrappers, routing
│   ├── services/
│   │   ├── inference_lock.py   # Fair GPU lock + placement scheduler
│   │   ├── query_router.py     # Mistral-7B routing (disabled by default)
│   │   ├── request_context.py  # Per-request ID propagation
│   │   ├── sam3_runner.py      # Persistent SAM3 grounding client
│   │   └── sar_converter.py    # Lazy Pix2Pix SAR→RGB converter
│   ├── utils/image_ops.py      # URL/file loading, FCC conversion
│   └── tests/                  # Contract tests + benchmark
├── models/
│   ├── captioning.py           # Qwen3-VL captioning/VLM backend
│   ├── vqa.py                  # Fine-tuned Qwen3-VL VQA backend
│   ├── device.py               # CUDA/MPS/CPU selection
│   ├── captioning-model/       # LoRA adapter + processor
│   ├── vqa-model/              # Fine-tuned VQA checkpoint
│   ├── sam3/                   # SAM3 grounding pipeline (vendored)
│   └── SAR2Optical/            # Pix2Pix SAR→optical code
├── ui/frontend/                # React 19 + Vite 7 + Tailwind 4
│   ├── src/
│   │   ├── GeoNLIDashboard.jsx
│   │   ├── components/         # 10 React components
│   │   └── services/api.js     # Backend HTTP client
│   └── api/geoNLI/eval.js      # Vercel serverless proxy
├── VQA-fine-tuning/            # Training code for VQA checkpoint
├── start_geonli.py             # Launcher for backend + frontend
├── stop_geonli.py              # Safe shutdown script
├── setup_gpu.sh                # One-shot environment setup
└── requirements.txt            # Pinned Python dependencies
```

---

## Backend Files

### `backend/app.py` — Application Factory

**Purpose:** Creates and configures the Flask application with logging, CORS, request timing, and blueprint registration.

**Key Components:**

| Element | Lines | Purpose |
|---|---|---|
| `_configure_logging()` | 13-48 | Sets up dual logging (file + console) with timestamp format |
| `create_app()` | 50-96 | Flask factory: CORS, config, request hooks, blueprint |
| `_start_request_timing()` | 63-72 | `before_request` hook: assigns request_id, starts timer |
| `_log_response_status()` | 74-84 | `after_request` hook: logs status code + duration |
| `_clear_request_context()` | 86-91 | `teardown_request` hook: resets ContextVar |
| `__main__` block | 98-111 | Entry point: runs Flask on configured host/port |

**Important Details:**
- Request IDs are 12-char hex strings stored in Flask's `g` object and a `ContextVar` [Code: backend/app.py:67-68]
- The `BACKEND_HOST` environment variable controls bind address (default `0.0.0.0`) [Code: backend/app.py:102]
- Logging goes to both `GEONLI_LOG` file and stdout [Code: backend/app.py:15-42]

---

### `backend/config.py` — Configuration

**Purpose:** Centralizes all environment-driven configuration with sensible defaults.

**Key Settings:**

| Setting | Default | Source |
|---|---|---|
| `PORT` | 8000 | `BACKEND_PORT` env |
| `ADV_VLM_MODEL_ID` | `Qwen/Qwen3-VL-2B-Instruct` | `ADV_VLM_MODEL_ID` env |
| `ADV_VLM_TEMPERATURE` | 0.2 | `ADV_VLM_TEMPERATURE` env |
| `ADV_VLM_MAX_NEW_TOKENS` | 64 | `ADV_VLM_MAX_NEW_TOKENS` env |
| `CAPTIONING_MODEL_DIR` | `models/captioning-model` | `CAPTIONING_MODEL_DIR` env |
| `VQA_MODEL_DIR` | `models/vqa-model` | `VQA_MODEL_DIR` env |
| `MAX_CONTENT_LENGTH` | 16 MB | Hardcoded |

**Notable:** The `_resolve_model_dir` function allows optional model directories to be overridden via environment variables without preventing the service from starting [Code: backend/config.py:9-12].

---

### `backend/api/routes.py` — API Endpoints

**Purpose:** Implements the two API endpoints: `/geoNLI/eval` (POST) and `/geoNLI/health` (GET).

**Key Functions:**

| Function | Lines | Purpose |
|---|---|---|
| `_timestamp()` | 27-28 | UTC ISO timestamp |
| `_error()` | 31-33 | Standardized error response |
| `_public_inference_error()` | 36-44 | Sanitizes internal errors for client |
| `analyze()` | 46-339 | Main eval endpoint (JSON + multipart) |
| `health()` | 341-343 | Liveness probe |

**Request Flow (Multipart):**
1. Validate `task_type` ∈ {captioning, vqa, grounding} [Code: backend/api/routes.py:231-232]
2. Validate `vqa_type` ∈ {semantic, binary, numeric} [Code: backend/api/routes.py:233-234]
3. Validate `spatial_resolution_m` > 0 [Code: backend/api/routes.py:229-230]
4. Load image from file upload [Code: backend/api/routes.py:216]
5. Convert FCC/SAR if needed [Code: backend/api/routes.py:239-252]
6. Route to SAM3 (grounding) or ModelManager (captioning/VQA) [Code: backend/api/routes.py:268-298]
7. Flatten results for frontend [Code: backend/api/routes.py:310-327]

**Request Flow (JSON):**
1. Extract `image_url` from `input_image` [Code: backend/api/routes.py:63-66]
2. Extract `spatial_resolution_m` from metadata (default 1.0) [Code: backend/api/routes.py:70]
3. Load image from URL [Code: backend/api/routes.py:73]
4. Convert FCC/SAR if needed [Code: backend/api/routes.py:76-87]
5. Process grounding query via SAM3 [Code: backend/api/routes.py:93-123]
6. Process other queries via ModelManager [Code: backend/api/routes.py:126-130]
7. Map results back to input schema [Code: backend/api/routes.py:132-207]

---

### `backend/core/model.py` — Model Manager

**Purpose:** Central orchestrator that manages all model backends and routes queries.

**Key Classes:**

| Class | Lines | Responsibility |
|---|---|---|
| `AdvancedVLMAdapter` | 45-109 | Wraps CaptioningBackend with GPU lock + scheduler |
| `CaptioningModel` | 120-137 | Captioning task wrapper |
| `GroundingModel` | 140-152 | Text grounding via VLM (text-only) |
| `VQAModel` | 155-194 | VQA task wrapper (binary/numeric/semantic) |
| `ModelManager` | 197-459 | Singleton manager, routes all queries |

**Key Methods:**

| Method | Lines | Purpose |
|---|---|---|
| `ModelManager.__init__()` | 198-225 | Loads adapter, VQA backend, captioner, grounder |
| `_vqa_checkpoint_complete()` | 227-241 | Validates sharded checkpoint integrity |
| `_process_routed_query()` | 346-412 | Routes binary/numeric through SAM3+VLM or VLM-only |
| `_compute_numeric_value()` | 307-344 | Computes count/area/perimeter/length from OBBs |
| `_polygon_area()` | 299-305 | Shoelace formula for polygon area |
| `process_all_queries()` | 414-459 | Main entry: processes all query types |
| `get_model()` | 466-476 | Singleton factory with thread-safe lazy init |

**Critical Design Decisions:**
- Router is disabled by default (`self.router = None`) to avoid loading a second large model [Code: backend/core/model.py:225]
- VQA checkpoint loading validates all shards exist before attempting load [Code: backend/core/model.py:227-241]
- Numeric values computed deterministically from SAM3 detections using spatial resolution [Code: backend/core/model.py:307-344]
- Binary answers canonicalized: "yes" without "no" → "Yes", "no" without "yes" → "No" [Code: backend/core/model.py:276-286]

---

### `backend/services/inference_lock.py` — GPU Scheduler

**Purpose:** Ensures VLM and SAM3 never hold full GPU weights simultaneously.

**Key Classes:**

| Class | Lines | Responsibility |
|---|---|---|
| `FairReentrantLock` | 10-52 | FIFO reentrant lock (thread-safe) |
| `GPUModelScheduler` | 58-112 | Exclusive CPU↔GPU placement |

**FairReentrantLock:**
- Uses `threading.Condition` with a ticket queue [Code: backend/services/inference_lock.py:13-18]
- Reentrant: same thread can acquire multiple times [Code: backend/services/inference_lock.py:23-25]
- FIFO: waiters served in ticket order [Code: backend/services/inference_lock.py:26-34]

**GPUModelScheduler:**
- `activate_vlm()`: moves SAM3 to CPU, VLM to GPU [Code: backend/services/inference_lock.py:76-85]
- `prepare_sam3()`: moves VLM to CPU before SAM3 loads [Code: backend/services/inference_lock.py:87-93]
- `activate_sam3()`: moves VLM to CPU, SAM3 to GPU [Code: backend/services/inference_lock.py:95-105]

---

### `backend/services/sam3_runner.py` — SAM3 Client

**Purpose:** Manages the persistent SAM3 grounding service and normalizes outputs.

**Key Functions:**

| Function | Lines | Purpose |
|---|---|---|
| `_resolve_sam3_paths()` | 39-46 | Finds unified_inference.py |
| `_get_sam3_service()` | 56-61 | Lazy-initializes SAM3 service |
| `_normalize_corners()` | 64-77 | Converts pixel corners to normalized 0-1 |
| `_encode_output_image()` | 80-85 | Base64 encodes visualization |
| `run_sam3_grounding()` | 88-150 | Main grounding pipeline |
| `run_sam3_grounding_from_image()` | 153-156 | Convenience wrapper for PIL images |

**Pipeline:**
1. Write uploaded image to `user_uploads/` [Code: backend/services/sam3_runner.py:94-98]
2. Acquire GPU lock [Code: backend/services/sam3_runner.py:101]
3. Prepare SAM3 (move VLM to CPU) [Code: backend/services/sam3_runner.py:104]
4. Activate SAM3 (move to GPU) [Code: backend/services/sam3_runner.py:109-110]
5. Run grounding [Code: backend/services/sam3_runner.py:112]
6. Normalize corners to 0-1 [Code: backend/services/sam3_runner.py:121-128]
7. Encode visualization as base64 [Code: backend/services/sam3_runner.py:119]

---

### `backend/services/sar_converter.py` — SAR Converter

**Purpose:** Lazy-loaded Pix2Pix converter for SAR-to-optical imagery.

**Key Class:** `Sar2OpticalConverter` [Code: backend/services/sar_converter.py:19-75]

**Pipeline:**
1. Load Pix2Pix model with `c_in=3, c_out=3, is_train=False` [Code: backend/services/sar_converter.py:47-64]
2. Preprocess: Resize to 256×256, normalize to [-1, 1] [Code: backend/services/sar_converter.py:39-45]
3. Generate: `model.generate(tensor, is_scaled=True)` [Code: backend/services/sar_converter.py:70]
4. Post-process: clamp to [-1,1], scale to [0,255] [Code: backend/services/sar_converter.py:71-73]

**Lazy loading:** The converter is only instantiated when first needed, using a thread-safe singleton pattern [Code: backend/services/sar_converter.py:78-88].

---

### `backend/services/query_router.py` — Query Router

**Purpose:** LLM-based router for deciding between VLM-only and SAM+VLM pipelines.

**Status:** Disabled by default (`router = None` in ModelManager) [Code: backend/core/model.py:225]

**Key Class:** `QueryRouter` [Code: backend/services/query_router.py:24-400]

**Model:** `mistralai/Mistral-7B-Instruct-v0.2` [Code: backend/services/query_router.py:29]

**Routing Logic:**
- Binary questions: Returns `VLM_ONLY` or `SAM_PLUS_VLM` [Code: backend/services/query_router.py:54-151]
- Numeric questions: Returns route + numeric_type (COUNT/LENGTH/AREA/PERIMETER) [Code: backend/services/query_router.py:153-337]
- Uses few-shot prompting with 9-12 examples [Code: backend/services/query_router.py:97-143]

**Geometry Helpers:**
- `get_largest_side()`: Maximum side length of 4-point polygon [Code: backend/services/query_router.py:407-419]
- `get_perimeter()`: Sum of all side lengths [Code: backend/services/query_router.py:422-434]
- `get_area()`: Pixel area × scale [Code: backend/services/query_router.py:437-439]

---

### `backend/services/request_context.py` — Request Context

**Purpose:** Propagates request IDs across async boundaries using Python ContextVars.

**Implementation:** Single `ContextVar` with default "-" [Code: backend/services/request_context.py:4]

---

### `backend/utils/image_ops.py` — Image Operations

**Purpose:** Image loading, FCC conversion, and file persistence.

**Key Functions:**

| Function | Lines | Purpose |
|---|---|---|
| `load_image_from_url()` | 11-18 | Downloads image from URL with 10s timeout |
| `load_image_from_file()` | 20-31 | Loads from Flask file storage |
| `fcc_to_rgb_array()` | 34-55 | FCC→RGB band remapping |
| `convert_fcc_to_rgb()` | 58-62 | PIL wrapper for FCC conversion |
| `save_converted_fcc_image()` | 83-91 | Persists converted FCC image |
| `save_converted_sar_image()` | 94-102 | Persists converted SAR image |

**FCC Conversion Formula:**
```
blue_synth = 13.9053 + 0.873393×green + (−0.138254)×red + (−0.100616)×NIR
```
[Code: backend/utils/image_ops.py:44-48]

---

## Model Files

### `models/captioning.py` — Captioning Backend

**Purpose:** Qwen3-VL vision-language model backend for captioning and VQA.

**Key Class:** `CaptioningBackend` [Code: models/captioning.py:32-281]

**Constructor Parameters:**
- `base_model_id`: Defaults to `Qwen/Qwen3-VL-2B-Instruct` [Code: models/captioning.py:17]
- `adapter_path`: LoRA adapter directory [Code: models/captioning.py:24]
- `min_pixels`: 256×256 (bounded for 6 GiB cards) [Code: models/captioning.py:21]
- `max_pixels`: 384×384 [Code: models/captioning.py:22]

**Model Loading:**
1. Check for LoRA adapter weights [Code: models/captioning.py:53-58]
2. Validate adapter config matches base model [Code: models/captioning.py:73-85]
3. Load processor with min/max pixels [Code: models/captioning.py:89-97]
4. Load base model with bf16/fp16/fp32 [Code: models/captioning.py:99-115]
5. Apply LoRA adapter if present [Code: models/captioning.py:117-124]
6. Register with GPU scheduler [Code: models/captioning.py:128]

**Generation Pipeline:**
1. Build chat message with image + text [Code: models/captioning.py:175-183]
2. Apply chat template [Code: models/captioning.py:184-188]
3. Process with Qwen3-VL processor [Code: models/captioning.py:191-195]
4. Move inputs to model device [Code: models/captioning.py:199-203]
5. Activate VLM on GPU [Code: models/captioning.py:225]
6. Generate with autocast [Code: models/captioning.py:231-244]
7. Check timeout [Code: models/captioning.py:248-251]
8. Decode tokens [Code: models/captioning.py:253-269]

---

### `models/vqa.py` — VQA Backend

**Purpose:** Fine-tuned Qwen3-VL VQA model backend.

**Key Class:** `VQABackend` [Code: models/vqa.py:68-157]

**Differences from CaptioningBackend:**
- Uses `Qwen3VLForConditionalGeneration` directly (not `AutoModelForImageTextToText`) [Code: models/vqa.py:10]
- Loads from local checkpoint directory [Code: models/vqa.py:77]
- Injects `rope_scaling` config if missing [Code: models/vqa.py:92-100]
- Uses custom prompt templates with `<|im_start|>` format [Code: models/vqa.py:23-48]
- Validates output (binary must be yes/no, numeric must contain number) [Code: models/vqa.py:51-65]

**Prompt Templates:**
- Binary: "Answer with yes or no." [Code: models/vqa.py:26]
- Numeric: "Return only a number in digits, no units, rounded to one decimal place." [Code: models/vqa.py:30]
- Semantic: No suffix [Code: models/vqa.py:34]

---

### `models/device.py` — Device Selection

**Purpose:** Simple utility for selecting the best available device.

**Logic:** CUDA → MPS → CPU [Code: models/device.py:3-8]

---

### `models/sam3/runtime_service.py` — SAM3 Service

**Purpose:** Persistent SAM3 grounding service that keeps models alive.

**Key Class:** `Sam3GroundingService` [Code: models/sam3/runtime_service.py:45-293]

**Constructor Parameters:**
- `score_threshold`: 0.4 [Code: models/sam3/runtime_service.py:53]
- `min_area`: 100 pixels [Code: models/sam3/runtime_service.py:54]
- `nms_threshold`: 0.6 [Code: models/sam3/runtime_service.py:55]
- `disable_clip`: True (CLIP disabled by default) [Code: models/sam3/runtime_service.py:57]
- `refiner`: None (Mistral refiner disabled) [Code: models/sam3/runtime_service.py:105]

**Grounding Pipeline:**
1. Normalize query to grounding phrase [Code: models/sam3/runtime_service.py:165-168]
2. Run SAM3 inference [Code: models/sam3/runtime_service.py:170-178]
3. Apply NMS [Code: models/sam3/runtime_service.py:199]
4. Optional CLIP scoring (disabled) [Code: models/sam3/runtime_service.py:202-212]
5. Optional VLM filtering (disabled) [Code: models/sam3/runtime_service.py:228-252]
6. Visualize results [Code: models/sam3/runtime_service.py:254]

**Query Normalization (`_grounding_phrase`):**
- Strips location words ("where is", "locate", "find") [Code: models/sam3/runtime_service.py:266-274]
- Removes trailing "in/on/at this image/scene" [Code: models/sam3/runtime_service.py:273]
- Singularizes nouns ("airplanes" → "airplane") [Code: models/sam3/runtime_service.py:277-286]

---

### `models/sam3/unified_inference.py` — SAM3 Pipeline

**Purpose:** Core SAM3 inference pipeline with mask-to-OBB conversion.

**Key Functions:**

| Function | Lines | Purpose |
|---|---|---|
| `load_sam3_model()` | 48-69 | Loads SAM3 checkpoint |
| `mask_to_oriented_bbox()` | 72-123 | Converts mask to 4-corner OBB |
| `reorder_corners_clockwise()` | 126-134 | Orders corners consistently |
| `apply_nms()` | 174-235 | Non-maximum suppression on OBBs |
| `get_prediction()` | 238-352 | Main inference + post-processing |
| `visualize_result()` | 471-512 | Draws boxes + masks on image |

**Mask-to-OBB Algorithm:**
1. Find contours in binary mask [Code: models/sam3/unified_inference.py:110]
2. Select largest contour [Code: models/sam3/unified_inference.py:115]
3. Compute minimum area rectangle [Code: models/sam3/unified_inference.py:120]
4. Get 4 corner points [Code: models/sam3/unified_inference.py:121]
5. Reorder clockwise [Code: models/sam3/unified_inference.py:126-134]

---

### `models/SAR2Optical/` — Pix2Pix SAR Conversion

**Purpose:** Pix2Pix UNet for SAR-to-optical image translation.

**Key Files:**
- `inference.py`: Standalone inference script [Code: models/SAR2Optical/inference.py]
- `src/networks.py`: UNet encoder/decoder architecture [Code: models/SAR2Optical/src/networks.py]
- `src/pix2pix.py`: Pix2Pix model wrapper
- `src/layers.py`: Downsampling/Upsampling blocks

**Architecture:** UNet with 8 encoder blocks (C64→C512) and 7 decoder blocks [Code: models/SAR2Optical/src/networks.py:7-80]

---

## Frontend Files

### `ui/frontend/src/GeoNLIDashboard.jsx` — Main Dashboard

**Purpose:** Root component managing all state and coordinating child components.

**State Management:**
- `sessions`: Array of session objects with images and messages [Code: ui/frontend/src/GeoNLIDashboard.jsx:17-19]
- `activeSessionId`: Currently selected session [Code: ui/frontend/src/GeoNLIDashboard.jsx:20]
- `selectedTask`: "grounding" or "vqa" [Code: ui/frontend/src/GeoNLIDashboard.jsx:22]
- `vqaType`: "semantic", "binary", or "numeric" [Code: ui/frontend/src/GeoNLIDashboard.jsx:23]
- `imageType`: "RGB", "FCC", or "SAR" [Code: ui/frontend/src/GeoNLIDashboard.jsx:24]
- `spatialResolution`: Meters per pixel [Code: ui/frontend/src/GeoNLIDashboard.jsx:25]

**Key Handlers:**
- `handleFileUpload()`: Creates blob URL, updates session [Code: ui/frontend/src/GeoNLIDashboard.jsx:193-213]
- `handleGenerateCaption()`: Calls API with task_type="captioning" [Code: ui/frontend/src/GeoNLIDashboard.jsx:215-242]
- `handleSendMessage()`: Calls API with selected task [Code: ui/frontend/src/GeoNLIDashboard.jsx:244-313]

---

### `ui/frontend/src/components/Stage.jsx` — Image Stage

**Purpose:** Full-viewport image display with SVG bounding box overlay.

**Key Features:**
- Drag-and-drop file upload [Code: ui/frontend/src/components/Stage.jsx:22-67]
- SVG overlay with `viewBox="0 0 100 100"` for normalized coordinates [Code: ui/frontend/src/components/Stage.jsx:156-168]
- Polygon rendering for 8-point OBBs [Code: ui/frontend/src/components/Stage.jsx:96-118]
- ResizeObserver for responsive overlay sizing [Code: ui/frontend/src/components/Stage.jsx:70-93]

---

### `ui/frontend/src/components/CommandBar.jsx` — Query Input

**Purpose:** Bottom command bar with task mode switch, VQA subtype pills, and text input.

**Key Features:**
- Grounding/VQA mode toggle [Code: ui/frontend/src/components/CommandBar.jsx:119-152]
- VQA subtype selector (binary/numeric/semantic) [Code: ui/frontend/src/components/CommandBar.jsx:87-111]
- Quick prompt chips [Code: ui/frontend/src/components/CommandBar.jsx:3-10]
- Answer toast display [Code: ui/frontend/src/components/CommandBar.jsx:53-66]

---

### `ui/frontend/src/components/SceneInspector.jsx` — Scene Controls

**Purpose:** Image type selector, spatial resolution input, and simulation toggle.

**Key Features:**
- Responsive: floating panel on desktop, bottom sheet on mobile [Code: ui/frontend/src/components/SceneInspector.jsx:135-239]
- Image type dropdown (RGB/SAR/FCC) [Code: ui/frontend/src/components/SceneInspector.jsx:73-83]
- Spatial resolution number input [Code: ui/frontend/src/components/SceneInspector.jsx:90-99]
- Simulation mode toggle [Code: ui/frontend/src/components/SceneInspector.jsx:103-120]

---

### `ui/frontend/src/components/SessionStrip.jsx` — Session Tabs

**Purpose:** Top navigation bar with session management.

**Key Features:**
- Session tabs with thumbnails [Code: ui/frontend/src/components/SessionStrip.jsx:36-87]
- Create new session button [Code: ui/frontend/src/components/SessionStrip.jsx:89-97]
- Terminate session with hover X [Code: ui/frontend/src/components/SessionStrip.jsx:63-84]
- Theme toggle, debug panel toggle, log toggle [Code: ui/frontend/src/components/SessionStrip.jsx:104-141]

---

### `ui/frontend/src/components/CaptionCard.jsx` — Caption Display

**Purpose:** Bottom-left card for scene caption generation and display.

**Key Features:**
- Generate caption button [Code: ui/frontend/src/components/CaptionCard.jsx:82-99]
- Copy to clipboard [Code: ui/frontend/src/components/CaptionCard.jsx:14-23]
- Collapsible panel [Code: ui/frontend/src/components/CaptionCard.jsx:47-60]

---

### `ui/frontend/src/components/LogSheet.jsx` — Conversation Log

**Purpose:** Right-side panel showing conversation history.

**Key Features:**
- Auto-scroll to latest message [Code: ui/frontend/src/components/LogSheet.jsx:14-18]
- User/bot message bubbles [Code: ui/frontend/src/components/LogSheet.jsx:68-114]
- Bounding box count indicator [Code: ui/frontend/src/components/LogSheet.jsx:82-87]
- Loading animation [Code: ui/frontend/src/components/LogSheet.jsx:116-124]

---

### `ui/frontend/src/components/DebugPanel.jsx` — Debug Info

**Purpose:** Floating panel showing current state for debugging.

**Displays:** status, task, type, subtask, image name, last response [Code: ui/frontend/src/components/DebugPanel.jsx:17-33]

---

### `ui/frontend/src/components/StatusBadge.jsx` — Status Indicator

**Purpose:** Shows backend connection status.

**States:** ready, loading, error, inference-error [Code: ui/frontend/src/components/StatusBadge.jsx:1-22]

---

### `ui/frontend/src/components/HideUiToggle.jsx` — UI Toggle

**Purpose:** Hides/shows all floating UI with H key.

**Implementation:** Keyboard listener for "h" key [Code: ui/frontend/src/GeoNLIDashboard.jsx:128-140]

---

### `ui/frontend/src/components/ConfirmDialog.jsx` — Confirmation Dialog

**Purpose:** Modal confirmation for destructive actions (session termination).

**Features:** Escape key handling, focus management [Code: ui/frontend/src/components/ConfirmDialog.jsx:15-28]

---

### `ui/frontend/src/services/api.js` — API Client

**Purpose:** HTTP client for backend communication.

**Key Functions:**
- `analyzeImage()`: Main API call with simulation mode [Code: ui/frontend/src/services/api.js:13-27]
- `fetchRealBackend()`: Multipart POST with 90s timeout [Code: ui/frontend/src/services/api.js:29-74]
- `checkBackendHealth()`: GET /geoNLI/health [Code: ui/frontend/src/services/api.js:76-82]
- `simulateBackendResponse()`: Mock responses for demo [Code: ui/frontend/src/services/api.js:84-124]

**Backend URL:** `VITE_BACKEND_BASE_URL` env var, default `http://127.0.0.1:8000` [Code: ui/frontend/src/services/api.js:7]

---

## VQA Fine-tuning Files

### `VQA-fine-tuning/vrsbench_finetune/train.py` — Training Script

**Purpose:** Full fine-tuning of Qwen3-VL-8B on VRSBench dataset.

**Key Features:**
- Mixed precision (bf16) training [Code: VQA-fine-tuning/vrsbench_finetune/train.py:103-106]
- Gradient accumulation [Code: VQA-fine-tuning/vrsbench_finetune/train.py:182-184]
- Task-weighted loss (binary/numeric/semantic) [Code: VQA-fine-tuning/vrsbench_finetune/train.py:23-70]
- Separate learning rates for backbone vs lm_head [Code: VQA-fine-tuning/vrsbench_finetune/train.py:160-179]
- Cosine LR scheduler with warmup [Code: VQA-fine-tuning/vrsbench_finetune/train.py:186-191]
- Checkpoint saving every N steps [Code: VQA-fine-tuning/vrsbench_finetune/train.py:257-258]

---

### `VQA-fine-tuning/vrsbench_finetune/model.py` — Model Wrapper

**Purpose:** Loads Qwen3-VL with gradient checkpointing.

**Key:** `get_model()` enables `gradient_checkpointing_enable()` [Code: VQA-fine-tuning/vrsbench_finetune/model.py:21]

---

### `VQA-fine-tuning/vrsbench_finetune/dataset.py` — Dataset

**Purpose:** VRSBench dataset loader with tokenization.

**Key Features:**
- Prompt templates matching inference [Code: VQA-fine-tuning/vrsbench_finetune/dataset.py:9-17]
- Label masking (prompt tokens = -100) [Code: VQA-fine-tuning/vrsbench_finetune/dataset.py:79-80]
- Dynamic vision key handling in collator [Code: VQA-fine-tuning/vrsbench_finetune/dataset.py:116-131]

---

### `VQA-fine-tuning/vrsbench_finetune/preprocess.py` — Data Preprocessing

**Purpose:** Converts VRSBench annotations to training JSONL.

**Task Detection:**
- Binary: "yes"/"no" answers [Code: VQA-fine-tuning/vrsbench_finetune/preprocess.py:18-22]
- Numeric: Parses numbers including scientific notation [Code: VQA-fine-tuning/vrsbench_finetune/preprocess.py:24-45]
- Semantic: Everything else [Code: VQA-fine-tuning/vrsbench_finetune/preprocess.py:95-96]

---

### `VQA-fine-tuning/vrsbench_finetune/eval.py` — Evaluation

**Purpose:** Evaluates fine-tuned model on validation set.

**Metrics:**
- Binary: Accuracy (overall, yes-only, no-only) [Code: VQA-fine-tuning/vrsbench_finetune/eval.py:109-129]
- Numeric: MAE, RMSE, normalized score [Code: VQA-fine-tuning/vrsbench_finetune/eval.py:131-149]
- Semantic: BERT-BLEU score [Code: VQA-fine-tuning/vrsbench_finetune/eval.py:26-88]

---

### `VQA-fine-tuning/configs/` — Training Configs

**dev.yaml:** 3 epochs, batch_size=4, grad_accum=4 [Code: VQA-fine-tuning/configs/dev.yaml]
**prod_2L40S.yaml:** 5 epochs, batch_size=8, grad_accum=4, 2×L40S [Code: VQA-fine-tuning/configs/prod_2L40S.yaml]

---

## Launcher Scripts

### `start_geonli.py` — Service Launcher

**Purpose:** Starts backend and frontend together with process management.

**Key Features:**
- Checks conda env `py310`, node, npm [Code: start_geonli.py:267-278]
- Port availability check with owner identification [Code: start_geonli.py:118-155]
- Process group management with unique launch ID [Code: start_geonli.py:313-322]
- Health check polling with 600s timeout [Code: start_geonli.py:231-245]
- Safe shutdown with SIGTERM → SIGKILL escalation [Code: start_geonli.py:167-196]

---

### `stop_geonli.py` — Service Stopper

**Purpose:** Safely stops launcher-managed services.

**Key Features:**
- Process identity verification (PID reuse protection) [Code: stop_geonli.py:19-29]
- Launch ID environment marker verification [Code: stop_geonli.py:32-45]
- Graceful shutdown with 8s grace period [Code: stop_geonli.py:64-113]

---

### `setup_gpu.sh` — Environment Setup

**Purpose:** One-shot setup for fresh machines.

**Steps:** apt update → git config → miniconda → conda env → nvm/node → clone repo → install deps [Code: setup_gpu.sh]

---

*Next: [Line-by-Line Code Reference](04_line_by_line_code_reference.md)*
