# GeoNLI Backend - AI Model Server

**Flask-based AI model serving backend for satellite image analysis**

## 🌟 Overview

The GeoNLI backend is a Flask-powered API server that provides three core AI capabilities for satellite imagery:

- **🖼️ Image Captioning** - Detailed scene descriptions using Qwen3-VL
- **❓ Visual Question Answering** - Binary, numeric, and semantic queries  
- **🎯 Visual Grounding** - Object detection and localization with SAM3

## 🏗️ Architecture

```
backend/
├── 🚀 app.py                   # Flask application entry point
├── ⚙️ config.py               # Configuration management
├── 📋 requirements.txt        # Python dependencies
│
├── 🔌 api/
│   └── routes.py              # REST API endpoints (/geoNLI/eval, /health)
│
├── 🧠 core/
│   └── model.py               # Model management & inference orchestration
│
├── 🛠️ services/
│   ├── query_router.py        # Intelligent query routing logic
│   └── sam3_runner.py         # SAM3 grounding integration
│
├── 📁 temp_uploads/           # Temporary file storage
├── 🏋️ weights/               # Model checkpoints directory
└── 🔧 utils/
    └── image_ops.py           # Image processing utilities
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend/
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Model paths
export CAPTIONING_MODEL_DIR="/path/to/captioning-model"
export VQA_MODEL_DIR="/path/to/vqa-model"
export SAM3_ROOT="/root/sam3"

# GPU configuration
export CAPTION_DEVICE="cuda:0"
export VQA_DEVICE="cuda:0"
export GROUNDING_DEVICE="cuda:0"

# Server settings
export BACKEND_PORT="8000"
export BACKEND_DEBUG="0"
```

### 3. Start Server

```bash
# From project root
python -m backend.app

# Server starts at: http://localhost:8000
# Health check: curl http://localhost:8000/geoNLI/health
```

## 🔌 API Endpoints

### `/geoNLI/eval` - Main Evaluation Endpoint

Accepts both JSON and multipart form requests for image analysis.

#### JSON Request (Evaluation Bots)
```json
POST /geoNLI/eval
Content-Type: application/json

{
  "input_image": {
    "image_id": "sample.png",
    "image_url": "https://example.com/satellite.jpg",
    "metadata": {"width": 512, "height": 512, "spatial_resolution_m": 1.57}
  },
  "queries": {
    "caption_query": {"instruction": "Describe this satellite image"},
    "grounding_query": {"instruction": "Locate buildings"},
    "attribute_query": {
      "binary": {"instruction": "Is there water?"},
      "numeric": {"instruction": "Count vehicles"}, 
      "semantic": {"instruction": "Terrain type?"}
    }
  }
}
```

#### Multipart Form Request (Frontend)
```
POST /geoNLI/eval
Content-Type: multipart/form-data

image: [uploaded file]
task_type: "captioning" | "vqa" | "grounding"
vqa_type: "binary" | "numeric" | "semantic" 
query: "natural language instruction"
image_type: "RGB" | "SAR" | "FCC"
```

### `/geoNLI/health` - Health Check

```json
GET /geoNLI/health

Response:
{
  "status": "healthy",
  "models_loaded": true,
  "timestamp": "2024-12-05T10:30:00Z"
}
```

## 🧠 Core Components

### Model Manager (`core/model.py`)

Central orchestrator for all AI model operations:

```python
# Model initialization and caching
model = get_model()

# Task routing and execution  
result = model.process_all_queries(
    image=pil_image,
    caption_query="Describe this image",
    vqa_queries={"binary": "Is there water?"},
    grounding_queries=["Find buildings"]
)
```

**Key Features:**
- **Model Caching**: One-time loading with singleton pattern
- **Sequential Processing**: Prevents GPU memory conflicts
- **Query Routing**: Intelligent task distribution
- **Error Handling**: Robust exception management

### API Router (`api/routes.py`)

Flask blueprint handling HTTP requests:

```python
@api_bp.route('/eval', methods=['POST'])
def analyze():
    # Handles both JSON and multipart requests
    # Routes to appropriate model backends
    # Returns structured JSON responses
```

**Request Processing:**
1. **Content Detection**: JSON vs multipart form parsing
2. **Image Loading**: URL fetch or file upload handling
3. **Query Parsing**: Task type and instruction extraction
4. **Model Execution**: Delegation to core model manager
5. **Response Formatting**: Structured JSON output

### Query Router (`services/query_router.py`)

Intelligent routing system for complex multi-part queries:

```python
class QueryRouter:
    def route_queries(self, queries: List[str]) -> List[RouteDecision]
        # Analyzes query complexity and object detection needs
        # Routes between captioning, VQA, and grounding models
        # Optimizes execution order for performance
```

### SAM3 Integration (`services/sam3_runner.py`)

Grounding service integration with SAM3:

```python
def run_sam3_grounding_from_image(image: Image, query: str):
    # Saves image to temporary file
    # Executes SAM3 unified_inference.py
    # Parses bounding box results
    # Returns formatted detection data
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `CAPTIONING_MODEL_DIR` | Qwen3-VL model path | - | `/models/captioning` |
| `VQA_MODEL_DIR` | VQA model directory | - | `/models/vqa` |
| `SAM3_ROOT` | SAM3 installation path | `/root/sam3` | `/opt/sam3` |
| `CAPTION_DEVICE` | Captioning GPU device | `cuda:0` | `cuda:1` |
| `VQA_DEVICE` | VQA GPU device | `cuda:0` | `cpu` |
| `GROUNDING_DEVICE` | Grounding GPU device | `cuda:0` | `cuda:2` |
| `BACKEND_PORT` | Server port | `8000` | `5000` |
| `BACKEND_DEBUG` | Debug mode | `0` | `1` |
| `GEONLI_LOG` | Log file path | `../geonli.log` | `/var/log/geonli.log` |

### Model Configuration

```python
# config.py - Centralized configuration management
class Config:
    CAPTIONING_MODEL_DIR = os.getenv('CAPTIONING_MODEL_DIR')
    VQA_MODEL_DIR = os.getenv('VQA_MODEL_DIR')
    SAM3_ROOT = os.getenv('SAM3_ROOT', '/root/sam3')
    
    # GPU device mapping
    CAPTION_DEVICE = os.getenv('CAPTION_DEVICE', 'cuda:0')
    VQA_DEVICE = os.getenv('VQA_DEVICE', 'cuda:0')
    GROUNDING_DEVICE = os.getenv('GROUNDING_DEVICE', 'cuda:0')
```

## 🎯 Usage Examples

### Python Client

```python
import requests
import json

# Image captioning
response = requests.post('http://localhost:8000/geoNLI/eval', 
    files={'image': open('satellite.jpg', 'rb')},
    data={'task_type': 'captioning'})

caption = response.json()['caption_response']
print(f"Caption: {caption}")

# Visual Question Answering
response = requests.post('http://localhost:8000/geoNLI/eval',
    files={'image': open('satellite.jpg', 'rb')},
    data={
        'task_type': 'vqa',
        'vqa_type': 'binary',
        'query': 'Is there a runway visible?'
    })

answer = response.json()['vqa_response']
print(f"Answer: {answer}")

# Object grounding
response = requests.post('http://localhost:8000/geoNLI/eval',
    files={'image': open('satellite.jpg', 'rb')},
    data={
        'task_type': 'grounding', 
        'query': 'Find all aircraft'
    })

detections = response.json()['grounding_response']
print(f"Found {len(detections)} aircraft")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/geoNLI/health

# Image captioning
curl -X POST http://localhost:8000/geoNLI/eval \
  -F "image=@satellite.jpg" \
  -F "task_type=captioning"

# VQA query
curl -X POST http://localhost:8000/geoNLI/eval \
  -F "image=@satellite.jpg" \
  -F "task_type=vqa" \
  -F "vqa_type=binary" \
  -F "query=Are there any vehicles?"

# Grounding query
curl -X POST http://localhost:8000/geoNLI/eval \
  -F "image=@satellite.jpg" \
  -F "task_type=grounding" \
  -F "query=Locate buildings"
```

## 🔍 Response Format

### Successful Response
```json
{
  "status": "success",
  "input_image": {
    "image_id": "uploaded_image.jpg",
    "width": 1024,
    "height": 1024,
    "format": "JPEG"
  },
  "caption_response": "A satellite view showing urban area with buildings...",
  "vqa_response": "Yes",
  "grounding_response": [
    {
      "bbox": [100, 150, 200, 300, 210, 280, 110, 130], 
      "confidence": 0.89,
      "label": "building"
    }
  ],
  "processing_time": 2.34
}
```

### Error Response
```json
{
  "status": "error", 
  "error": "Invalid image format",
  "details": "Supported formats: JPEG, PNG, TIFF"
}
```

## 🐛 Debugging & Logs

### Log Configuration

Logs are written to both stdout and `geonli.log`:

```python
# Logging setup in app.py
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler = logging.FileHandler("geonli.log")
stream_handler = logging.StreamHandler()
```

### Common Log Messages

```
2024-12-05 10:30:15 | INFO | Server starting on port 8000
2024-12-05 10:30:16 | INFO | Loading captioning model from /models/captioning
2024-12-05 10:30:20 | INFO | All models loaded successfully
2024-12-05 10:30:25 | INFO | /geoNLI/eval request | content_type=multipart/form-data
2024-12-05 10:30:27 | INFO | Processing captioning query: "Describe this image"
2024-12-05 10:30:29 | INFO | Caption generated successfully (2.1s)
```

### Troubleshooting

#### Model Loading Issues
```bash
# Check model directories
ls $CAPTIONING_MODEL_DIR
ls $VQA_MODEL_DIR

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Test model imports
python -c "from models.captioning import CaptioningBackend"
```

#### Memory Issues
```bash
# Monitor GPU memory
nvidia-smi

# Enable CPU offloading if needed
export ENABLE_CPU_OFFLOAD=1

# Reduce batch sizes in model configs
```

#### SAM3 Integration
```bash
# Check SAM3 installation
ls $SAM3_ROOT/unified_inference.py

# Test SAM3 directly
cd $SAM3_ROOT
python unified_inference.py --image test.jpg --query "find buildings"
```

## 🚀 Performance Optimization

### Model Optimization
- **Model Caching**: Models loaded once at startup
- **Sequential Processing**: Prevents GPU memory fragmentation  
- **Device Management**: Configurable GPU allocation per model
- **Memory Clearing**: `torch.cuda.empty_cache()` between requests

### Request Optimization
- **Lazy Loading**: Models loaded on first request if needed
- **Image Preprocessing**: Automatic resizing and format conversion
- **Error Caching**: Fast failure on invalid inputs
- **Connection Pooling**: Efficient HTTP handling with Flask

### Monitoring
```bash
# Server performance
curl http://localhost:8000/geoNLI/health

# GPU utilization
watch -n 1 nvidia-smi

# Process monitoring
htop -p $(pgrep -f "backend.app")
```

## 📦 Dependencies

Key packages in `requirements.txt`:
- **Flask** - Web framework
- **torch** - PyTorch deep learning
- **transformers** - Hugging Face models
- **Pillow** - Image processing  
- **flask-cors** - CORS handling
- **requests** - HTTP client

## 🔒 Security Notes

- **File Upload Validation**: Images validated before processing
- **Temporary File Cleanup**: Auto-deletion of uploaded files
- **Error Sanitization**: No stack traces in API responses
- **Resource Limits**: Request size limits configured

## 📈 Scaling Considerations

### Horizontal Scaling
- **Stateless Design**: No session storage, safe for load balancing
- **Model Replication**: Each instance loads own model copies
- **Shared Storage**: Use network storage for model weights

### Vertical Scaling  
- **Multi-GPU**: Configure different models on different GPUs
- **Memory Management**: Monitor and tune model memory usage
- **CPU Cores**: Increase workers for better throughput

---

**Part of GeoNLI System - ISRO Inter-IIT Tech Meet 13.0**