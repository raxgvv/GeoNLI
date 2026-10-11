# 14 — Dependencies and Technology Choices

## Python Dependencies

### Web Backend

| Package | Version | Purpose | Used In |
|---|---|---|---|
| Flask | 3.1.2 | Web framework | backend/app.py, api/routes.py |
| flask-cors | 6.0.1 | CORS support | backend/app.py:8 |
| python-multipart | 0.0.20 | Multipart form handling | Flask file uploads |

### Deep Learning

| Package | Version | Purpose | Used In |
|---|---|---|---|
| torch | 2.9.1 | Deep learning framework | All model files |
| torchvision | 0.24.1 | Vision models + transforms | sar_converter.py |
| transformers | 4.57.3 | Qwen3-VL, processors | captioning.py, vqa.py |
| accelerate | 1.12.0 | Distributed training | VQA-fine-tuning/ |
| peft | 0.18.0 | LoRA/PEFT adapters | captioning.py:9 |
| huggingface-hub | 0.36.2 | Model downloads | runtime_service.py:67 |
| safetensors | 0.7.0 | Safe model loading | transformers |
| sentencepiece | 0.2.1 | Tokenization | transformers |
| protobuf | 6.33.1 | Protocol buffers | transformers |

### Image Processing

| Package | Version | Purpose | Used In |
|---|---|---|
| pillow | 12.0.0 | Image I/O | Throughout |
| opencv-python | 4.12.0.88 | SAM3 OBB extraction | unified_inference.py:18 |
| numpy | 2.2.6 | Array operations | image_ops.py, unified_inference.py |
| requests | 2.32.5 | HTTP for image download | image_ops.py:8 |

### SAM3 Pipeline

| Package | Version | Purpose | Used In |
|---|---|---|
| decord | 0.6.0 | Video decoding | SAM3 (not used in inference) |
| einops | 0.8.1 | Tensor operations | SAM3 model |
| ftfy | 6.3.1 | Text normalization | SAM3 |
| iopath | 0.1.10 | I/O utilities | SAM3 |
| portalocker | 3.2.0 | File locking | SAM3 |
| pycocotools | 2.0.10 | COCO evaluation | SAM3 eval |
| timm | 1.0.22 | Vision models | SAM3 |
| regex | 2025.11.3 | Regex | SAM3 |

### Training (VQA-fine-tuning)

| Package | Version | Purpose |
|---|---|---|
| deepspeed | >=0.13.0 | Distributed training |
| wandb | >=0.16.0 | Experiment tracking |
| python-dotenv | — | Env file loading |
| scikit-learn | — | Metrics |
| pandas | — | Data handling |
| pyyaml | — | Config parsing |
| tqdm | — | Progress bars |

---

## Frontend Dependencies

| Package | Version | Purpose |
|---|---|---|
| react | ^19.2.0 | UI framework |
| react-dom | ^19.2.0 | DOM rendering |
| vite | ^7.2.4 | Build tool |
| @vitejs/plugin-react-swc | ^4.2.2 | React plugin |
| tailwindcss | ^4.1.17 | CSS framework |
| @tailwindcss/vite | ^4.1.17 | Tailwind Vite plugin |
| lucide-react | ^0.555.0 | Icons |
| @fontsource-variable/fraunces | ^5.3.0 | Display font |
| @fontsource-variable/dm-sans | ^5.3.0 | Body font |
| @fontsource/ibm-plex-mono | ^5.3.0 | Mono font |
| eslint | ^9.39.1 | Linting |

---

## Technology Choice Rationale

### Why Flask over FastAPI?

| Factor | Flask | FastAPI |
|---|---|---|
| Simplicity | Simpler | More features |
| Async | No | Yes |
| Existing code | Already used | — |

**Note:** The code comments mention "parity with FastAPI" [Code: backend/app.py:14], suggesting a previous FastAPI implementation. Flask was likely chosen for simplicity. [Inferred]

### Why React over Vue/Svelte?

| Factor | React | Vue | Svelte |
|---|---|---|---|
| Ecosystem | Largest | Large | Growing |
| Team familiarity | Likely | — | — |

### Why Tailwind over CSS modules?

| Factor | Tailwind | CSS Modules |
|---|---|---|
| Speed | Faster development | More verbose |
| Bundle size | Smaller (purged) | Similar |

### Why PyTorch over JAX?

| Factor | PyTorch | JAX |
|---|---|---|
| Ecosystem | Larger | Smaller |
| Transformers support | Native | Limited |

---

## Potentially Unnecessary Dependencies

| Package | Necessity | Reason |
|---|---|---|
| decord | Uncertain | Video decoding not used in inference |
| pycocotools | Uncertain | COCO eval not used in runtime |
| iopath | Uncertain | May be SAM3 internal |
| portalocker | Uncertain | May be SAM3 internal |
| cors (frontend) | Uncertain | Listed in frontend package.json but Vite handles CORS |

---

## Version Compatibility

| Combination | Status |
|---|---|
| torch 2.9.1 + transformers 4.57.3 | Tested together |
| Python 3.10 + torch 2.9.1 | Supported |
| Node 20 + Vite 7 | Supported |
| CUDA 12.8 + torch 2.9.1 | Supported |

---

## GPU Libraries

| Library | Version | Purpose |
|---|---|---|
| CUDA | 12.x | GPU compute |
| cuDNN | — | Deep learning primitives |
| NCCL | — | Multi-GPU communication (training) |

---

*Next: [Limitations and Future Work](15_limitations_and_future_work.md)*
