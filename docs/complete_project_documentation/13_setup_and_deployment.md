# 13 — Installation, Startup, and Deployment

## Operating System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Linux (Ubuntu 20.04+) | Linux (Ubuntu 22.04+) |
| Python | 3.10 | 3.10 |
| Node.js | 20+ | 20+ |
| npm | 10+ | 10+ |
| GPU | NVIDIA with CUDA 12.x | NVIDIA with 6+ GiB VRAM |
| RAM | 16 GB | 16+ GB |
| Disk | 10 GB free | 15+ GB free |

---

## Python Environment Setup

### Conda (Recommended)

```bash
conda create -n py310 python=3.10 -y
conda activate py310
```

### Install PyTorch with CUDA

```bash
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Node.js Setup

### Using nvm (Recommended)

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm install 20
nvm alias default 20
```

### Install Frontend Dependencies

```bash
cd ui/frontend
npm install
```

---

## Model Downloads

### Automatic Downloads

| Model | Trigger | Location |
|---|---|---|
| Qwen3-VL-2B-Instruct | First inference | Hugging Face Hub cache |
| SAM3 processor | First inference | Hugging Face Hub cache |

### Manual Downloads (Required for Some Features)

| Model | Purpose | How to Obtain |
|---|---|---|
| SAM3 checkpoint (`sam.pt`) | Grounding | Download from facebook/sam3 |
| Pix2Pix checkpoint (`pix2pix_gen_180.pth`) | SAR conversion | Train separately or obtain from team |
| VQA fine-tuned checkpoint | VQA | Train with VQA-fine-tuning/ or obtain from team |
| Captioning LoRA adapter | Captioning | Train separately or obtain from team |

### Environment Variables for Models

```bash
export SAM3_ROOT="/path/to/models/sam3"
export SAM3_CHECKPOINT="/path/to/sam.pt"
export SAR2OPTICAL_CHECKPOINT="/path/to/pix2pix_gen_180.pth"
export VQA_MODEL_DIR="/path/to/models/vqa-model"
export CAPTIONING_MODEL_DIR="/path/to/models/captioning-model"
```

---

## Backend Startup

### Using the Launcher (Recommended)

```bash
cd ~/Inter-IIT
python start_geonli.py
```

This starts both backend and frontend with process management.

### Manual Start

```bash
cd ~/Inter-IIT
conda run -n py310 python -m backend.app
```

### Health Check

```bash
curl http://localhost:8000/geoNLI/health
```

Expected response:
```json
{"status": "active", "success": true, "timestamp": "2026-10-10T12:00:00+00:00"}
```

---

## Frontend Startup

### Development

```bash
cd ui/frontend
npm run dev
```

URL: http://localhost:5173

### Production Build

```bash
cd ui/frontend
npm run build
npm run preview
```

### Vercel Deployment

The frontend includes a Vercel configuration and serverless proxy:
- `vercel.json` — deployment config
- `api/geoNLI/eval.js` — serverless function proxying to backend

---

## Environment Variables

### Backend

| Variable | Default | Purpose |
|---|---|---|
| `BACKEND_PORT` | 8000 | Flask port |
| `BACKEND_HOST` | 0.0.0.0 | Flask bind address |
| `BACKEND_DEBUG` | 0 | Debug mode |
| `ADV_VLM_MODEL_ID` | Qwen/Qwen3-VL-2B-Instruct | VLM backbone |
| `ADV_VLM_TEMPERATURE` | 0.2 | Sampling temperature |
| `ADV_VLM_MAX_NEW_TOKENS` | 64 | Generation cap |
| `CAPTION_GENERATION_TIMEOUT_S` | 50 | Caption timeout |
| `CAPTIONING_MODEL_DIR` | models/captioning-model | LoRA adapter |
| `VQA_MODEL_DIR` | models/vqa-model | VQA checkpoint |
| `SAM3_ROOT` | models/sam3 | SAM3 location |
| `SAM3_CHECKPOINT` | — | SAM3 checkpoint path |
| `SAR2OPTICAL_CHECKPOINT` | models/SAR2Optical/pix2pix_gen_180.pth | Pix2Pix weights |
| `GEONLI_LOG` | geonli.log | Log file path |
| `HF_TOKEN` | — | Hugging Face auth |

### Frontend

| Variable | Default | Purpose |
|---|---|---|
| `VITE_BACKEND_BASE_URL` | http://127.0.0.1:8000 | Backend URL |

---

## Test Execution

### Backend Tests

```bash
python -m pytest backend/tests/test_eval_api.py -q
python -m pytest backend/tests/test_sam3_postprocessing.py -q
```

### Frontend Checks

```bash
cd ui/frontend
npm run lint
npm run build
```

### Grounding Benchmark

```bash
python -m backend.tests.benchmark_grounding
```

---

## Shutdown

### Using the Stop Script

```bash
python stop_geonli.py
```

### Manual Stop

Press Ctrl+C in the terminal running `start_geonli.py`.

---

## Logs

| Log File | Location | Content |
|---|---|---|
| Backend app | `logs/backend-app.log` | Flask + model logs |
| Backend launcher | `logs/backend.log` | Launcher output |
| Frontend | `logs/frontend.log` | Vite output |
| Legacy | `geonli.log` | Old log location |

---

## Troubleshooting

### GPU Out of Memory

```bash
nvidia-smi  # Check GPU usage
```

Solutions:
- Close other GPU applications
- Use smaller images (≤16 MB)
- The scheduler should prevent OOM; if it occurs, check logs

### SAR Conversion Fails (503)

```
Error: SAR conversion unavailable
```

Solution: Set `SAR2OPTICAL_CHECKPOINT` to valid Pix2Pix weights.

### Caption Times Out (504)

```
Error: Vision-language inference exceeded its time limit
```

Solutions:
- Lower `ADV_VLM_MAX_NEW_TOKENS`
- Raise `CAPTION_GENERATION_TIMEOUT_S`
- Use smaller images

### VLM Invalid Yes/No

```
Error: The VLM did not return a valid yes/no answer
```

Solution: Retry or rephrase the question.

### Frontend Cannot Reach Backend

```
Error: Cannot reach backend at http://127.0.0.1:8000
```

Solution: Set `VITE_BACKEND_BASE_URL` in `ui/frontend/.env`.

### SAM3 Checkpoint Missing

```
Error: SAM3 checkpoint was not found
```

Solution: Set `SAM3_CHECKPOINT` env var or place `sam.pt` in `models/sam3/`.

---

## GPU Memory Diagnostics

```bash
# Check current GPU memory
nvidia-smi

# Check PyTorch CUDA memory
python -c "import torch; print(f'Allocated: {torch.cuda.memory_allocated()/1024**2:.0f} MiB'); print(f'Reserved: {torch.cuda.memory_reserved()/1024**2:.0f} MiB')"
```

---

## Deployment Options

| Option | Pros | Cons |
|---|---|---|
| Local machine | Simple, no network | Requires GPU |
| Cloud VM (AWS/GCP) | Scalable | Cost |
| Vercel (frontend only) | Free tier | Backend still needs hosting |

---

*Next: [Dependencies and Technology Choices](14_dependencies_and_technology_choices.md)*
