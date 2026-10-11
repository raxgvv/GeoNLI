# 02 — System Architecture

## High-Level Architecture

```mermaid
flowchart TB
    subgraph Frontend["React Frontend (ui/frontend)"]
        UI[GeoNLIDashboard.jsx]
        STAGE[Stage.jsx<br/>Image + SVG Overlay]
        CMD[CommandBar.jsx<br/>Query Input]
        SCENE[SceneInspector.jsx<br/>Image Type + Resolution]
        CAP[CaptionCard.jsx]
        LOG[LogSheet.jsx<br/>Conversation History]
        API[services/api.js<br/>HTTP Client]
    end

    subgraph Backend["Flask Backend (backend/)"]
        APP[app.py<br/>Application Factory]
        ROUTES[api/routes.py<br/>/geoNLI/eval, /geoNLI/health]
        CONFIG[config.py<br/>Env-driven Config]
        LOCK[services/inference_lock.py<br/>Fair GPU Lock + Scheduler]
        CTX[services/request_context.py<br/>Request ID]
        IMG[utils/image_ops.py<br/>Image Loading + FCC Conversion]
    end

    subgraph Models["Model Layer"]
        MM[core/model.py<br/>ModelManager]
        CAP_MOD[models/captioning.py<br/>CaptioningBackend]
        VQA_MOD[models/vqa.py<br/>VQABackend]
        SAM3[models/sam3/<br/>SAM3 Grounding Service]
        SAR_MOD[models/SAR2Optical/<br/>Pix2Pix Converter]
        ROUTER[services/query_router.py<br/>Mistral-7B Router<br/>Disabled by default]
    end

    subgraph External["External Services"]
        HF[Hugging Face Hub<br/>Qwen3-VL-2B Download]
        SAM3_CKPT[SAM3 Checkpoint<br/>facebook/sam3]
        GPU[CUDA GPU<br/>6+ GiB VRAM]
    end

    UI --> STAGE & CMD & SCENE & CAP & LOG
    CMD --> API
    SCENE --> API
    CAP --> API
    API -->|"HTTP POST multipart"| ROUTES
    API -->|"HTTP GET health"| ROUTES
    ROUTES --> MM
    ROUTES --> IMG
    ROUTES --> SAM3
    ROUTES --> SAR_MOD
    MM --> CAP_MOD
    MM --> VQA_MOD
    MM --> LOCK
    CAP_MOD --> LOCK
    SAM3 --> LOCK
    LOCK --> GPU
    CAP_MOD -->|"First request"| HF
    SAM3 -->|"First request"| SAM3_CKPT
```

---

## Application Startup Sequence

```mermaid
sequenceDiagram
    participant User
    participant Launcher as start_geonli.py
    participant Flask as Flask App
    participant VLM as CaptioningBackend
    participant SAM3 as Sam3GroundingService
    participant GPU as GPU Scheduler

    User->>Launcher: python start_geonli.py
    Launcher->>Launcher: Check conda env py310
    Launcher->>Launcher: Check node + npm
    Launcher->>Launcher: Check ports 8000, 5173
    Launcher->>Flask: conda run -n py310 python -m backend.app
    Launcher->>User: Wait for /geoNLI/health (timeout 600s)
    Flask->>Flask: create_app() — configure logging, CORS
    Flask->>Flask: Register blueprint /geoNLI
    Note over Flask: Model initialization is LAZY<br/>VLM loads on first inference<br/>SAM3 loads on first grounding
    Flask-->>Launcher: 200 OK {"status": "active"}
    Launcher->>User: Backend ready
    Launcher->>User: Frontend ready
```

**Key detail:** The backend starts quickly because models are loaded lazily. The VLM (`CaptioningBackend`) loads on the first inference request, and SAM3 loads on the first grounding request [Code: models/sam3/runtime_service.py:159, backend/core/model.py:471-475].

---

## Request Lifecycle

```mermaid
sequenceDiagram
    participant FE as Frontend
    participant Flask as Flask Routes
    participant Lock as GPU Lock
    participant Scheduler as GPU Scheduler
    participant Model as VLM / SAM3

    FE->>Flask: POST /geoNLI/eval (multipart)
    Flask->>Flask: Validate task_type, image_type, query
    Flask->>Flask: Load image (file or URL)
    alt FCC image
        Flask->>Flask: convert_fcc_to_rgb()
    else SAR image
        Flask->>Model: convert_sar_to_rgb()
    end
    Flask->>Lock: acquire()
    Lock->>Scheduler: activate_vlm() or prepare_sam3()
    Scheduler->>Model: Move models CPU↔GPU exclusively
    Model->>Model: Inference
    Model-->>Flask: Result
    Scheduler->>Model: Move back to CPU
    Lock->>Lock: release()
    Flask-->>FE: JSON {success, answer, bboxes, ...}
```

---

## AI/ML Inference Pipeline

```mermaid
flowchart TB
    INPUT[Input Image] --> DECODE[Decode to PIL RGB]
    DECODE --> TYPE{Image Type?}

    TYPE -->|RGB| PREPROC[Qwen3-VL Processor<br/>Chat Template + Tokenize]
    TYPE -->|FCC| FCC[FCC→RGB Conversion<br/>Band Remap]
    TYPE -->|SAR| SAR[Pix2Pix Conversion<br/>UNet Generator]
    FCC --> PREPROC
    SAR --> PREPROC

    PREPROC --> VLM[Qwen3-VL-2B-Instruct<br/>Autoregressive Generation]
    VLM --> DECODE_OUT[Decode Tokens]
    DECODE_OUT --> POST[Post-process]

    POST --> BINARY{Binary?}
    BINARY -->|Yes| CANON[Canonicalize to Yes/No]
    BINARY -->|No| NUM{Numeric?}
    NUM -->|Yes| EXTRACT[Extract First Float]
    NUM -->|No| SEMANTIC[Return as-is]

    CANON --> RESPONSE[JSON Response]
    EXTRACT --> RESPONSE
    SEMANTIC --> RESPONSE

    GROUND[Grounding Query] --> SAM3[SAM3 Model]
    SAM3 --> MASKS[Instance Masks]
    MASKS --> OBB[Oriented Bounding Boxes]
    OBB --> NMS[Non-Maximum Suppression]
    NMS --> NORM[Normalize 0-1]
    NORM --> RESPONSE
```

---

## Model Loading and Resource Management Flow

```mermaid
flowchart TD
    START[First Inference Request] --> LOCK_GET[Acquire GPU_INFERENCE_LOCK]
    LOCK_GET --> SCHED[VLM: activate_vlm]
    SCHED --> CHECK_SAM3{SAM3 on GPU?}
    CHECK_SAM3 -->|Yes| MOVE_SAM3_CPU[Move SAM3 to CPU]
    CHECK_SAM3 -->|No| MOVE_VLM_GPU[Move VLM to GPU]
    MOVE_SAM3_CPU --> MOVE_VLM_GPU
    MOVE_VLM_GPU --> INFER[VLM Inference]
    INFER --> SYNC[torch.cuda.synchronize]
    SYNC --> RELEASE[Release GPU_INFERENCE_LOCK]
    RELEASE --> DONE[Result Returned]

    GROUND_REQ[First Grounding Request] --> LOCK_GET2[Acquire GPU_INFERENCE_LOCK]
    LOCK_GET2 --> PREP[SAM3: prepare_sam3]
    PREP --> MOVE_VLM_CPU[Move VLM to CPU]
    MOVE_VLM_CPU --> ACTIVATE[SAM3: activate_sam3]
    ACTIVATE --> MOVE_SAM3_GPU[Move SAM3 to GPU]
    MOVE_SAM3_GPU --> GROUND_INFER[SAM3 Inference]
    GROUND_INFER --> GROUND_DONE[Result + Visualization]
    GROUND_DONE --> RELEASE2[Release GPU_INFERENCE_LOCK]
```

**Critical invariant:** VLM and SAM3 never hold full GPU weights simultaneously. The `GPUModelScheduler` ensures exclusive placement [Code: backend/services/inference_lock.py:58-112].

---

## Error Handling Flow

```mermaid
flowchart TD
    REQUEST[Incoming Request] --> VALIDATE{Validation}
    VALIDATE -->|Invalid task_type| E400[400 Bad Request]
    VALIDATE -->|Missing query| E400
    VALIDATE -->|Invalid image| E400
    VALIDATE -->|Valid| PROCESS[Process Request]

    PROCESS --> CONVERT{Image Conversion}
    CONVERT -->|FCC success| INFER[Inference]
    CONVERT -->|SAR no checkpoint| E503[503 Service Unavailable]
    CONVERT -->|SAR success| INFER

    INFER --> TIMEOUT{Timeout?}
    TIMEOUT -->|Yes| E504[504 Gateway Timeout]
    TIMEOUT -->|No| OOM{Out of Memory?}
    OOM -->|Yes| E500[500 + GPU Memory Error]
    OOM -->|No| SUCCESS{Success?}
    SUCCESS -->|Yes| RESP[200 OK + JSON]
    SUCCESS -->|No| E500[500 Internal Error]

    E400 --> LOG[Log with request_id]
    E503 --> LOG
    E504 --> LOG
    E500 --> LOG
    RESP --> LOG
```

**Error sanitization:** The `_public_inference_error` function converts internal errors to user-friendly messages without leaking implementation details [Code: backend/api/routes.py:36-44].

---

## Component Interaction Summary

| Component | Talks To | Protocol |
|---|---|---|
| Frontend (React) | Flask Backend | HTTP (multipart/form-data or JSON) |
| Flask Routes | ModelManager | Python function calls |
| ModelManager | CaptioningBackend, VQABackend | Python function calls |
| CaptioningBackend | Hugging Face Hub | HTTPS (first load only) |
| CaptioningBackend | GPU Scheduler | Python (move_to) |
| SAM3 Runner | SAM3 Service | Python (in-process) |
| SAM3 Service | SAM3 Checkpoint | File I/O |
| SAR Converter | Pix2Pix Model | Python (in-process) |
| GPU Scheduler | CUDA | PyTorch API |

---

## External Dependencies and Services

| Service | Purpose | Required? | Network |
|---|---|---|---|
| Hugging Face Hub | Download Qwen3-VL-2B-Instruct | Yes (first run) | HTTPS |
| facebook/sam3 | SAM3 checkpoint download | Yes (grounding) | HTTPS |
| CUDA GPU | Model inference | Highly recommended | Local |
| Vercel (optional) | Frontend hosting + proxy | No | HTTPS |

---

## Configuration and Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `BACKEND_PORT` | 8000 | Flask listen port |
| `BACKEND_HOST` | 0.0.0.0 | Flask bind address |
| `ADV_VLM_MODEL_ID` | Qwen/Qwen3-VL-2B-Instruct | Shared VLM backbone |
| `ADV_VLM_TEMPERATURE` | 0.2 | Sampling temperature |
| `ADV_VLM_MAX_NEW_TOKENS` | 64 | Generation cap |
| `CAPTIONING_MODEL_DIR` | models/captioning-model | LoRA adapter location |
| `VQA_MODEL_DIR` | models/vqa-model | Fine-tuned VQA checkpoint |
| `SAM3_ROOT` | models/sam3 | SAM3 pipeline location |
| `SAR2OPTICAL_CHECKPOINT` | models/SAR2Optical/pix2pix_gen_180.pth | Pix2Pix weights |
| `HF_TOKEN` | — | Hugging Face auth (if needed) |
| `GEONLI_LOG` | geonli.log | Backend log file |

---

*Next: [Complete Codebase Walkthrough](03_complete_codebase_walkthrough.md)*
