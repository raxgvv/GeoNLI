# 04 — Line-by-Line Code Reference

This document provides detailed line-by-line explanations of all active first-party source files.

---

## 4.1 — `backend/app.py`

```python
# Lines 1-11: Imports
import logging
import os
import time
from uuid import uuid4
from pathlib import Path

from flask import Flask, g, has_request_context, request
from flask_cors import CORS
from backend.config import Config
from backend.api.routes import api_bp
from backend.services.request_context import REQUEST_ID
```

**Purpose:** Sets up dependencies. Uses `uuid4` for request IDs, `g` for Flask request-scoped globals, `has_request_context` for safe logging, and `REQUEST_ID` ContextVar for async-safe request tracking.

---

```python
# Lines 13-48: _configure_logging()
def _configure_logging():
    """Send backend logs to both stdout and geonli.log for parity with FastAPI."""
    log_path = os.environ.get(
        "GEONLI_LOG",
        str(Path(__file__).resolve().parents[1] / "geonli.log"),
    )
```

Resolves log path from `GEONLI_LOG` env var, defaulting to repo root.

```python
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
```

Creates log directory if needed. Clears existing handlers to prevent duplicates on re-import.

```python
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s)")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    logging.getLogger('backend').setLevel(logging.INFO)
    logging.getLogger('models').setLevel(logging.INFO)
```

Configures dual logging: file + console. Sets `backend` and `models` module loggers to INFO.

---

```python
# Lines 50-96: create_app()
def create_app():
    _configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("GeoNLI Backend starting up...")
    app = Flask(__name__)
    Config.init_app(app)
    app.config.from_object(Config)
    CORS(app)
```

Flask factory pattern. `Config.init_app` creates upload/weights directories. CORS enabled for frontend access.

```python
    @app.before_request
    def _start_request_timing():
        from flask import g
        g.request_started = time.monotonic()
        g.request_id = uuid4().hex[:12]
        g.request_id_token = REQUEST_ID.set(g.request_id)
        logging.getLogger('backend.http').info(
            "Request started | request_id=%s | method=%s | path=%s",
            g.request_id, request.method, request.path,
        )
```

**Critical:** Every request gets a 12-char hex ID stored in both `g.request_id` and a `ContextVar`. The ContextVar token is saved for later reset.

```python
    @app.after_request
    def _log_response_status(response):
        logging.getLogger('backend.http').info(
            "HTTP response | request_id=%s | method=%s | path=%s | status=%s | total_s=%.3f",
            getattr(g, 'request_id', '-'),
            request.method if has_request_context() else 'unknown',
            request.path if has_request_context() else 'unknown',
            response.status_code,
            time.monotonic() - getattr(g, 'request_started', time.monotonic()) if has_request_context() else 0.0,
        )
        return response
```

Logs every response with status code and duration. Uses `getattr` with defaults for safety outside request context.

```python
    @app.teardown_request
    def _clear_request_context(_error):
        from flask import g
        token = getattr(g, 'request_id_token', None)
        if token is not None:
            REQUEST_ID.reset(token)
```

Resets the ContextVar after each request to prevent leakage between requests.

```python
    app.register_blueprint(api_bp, url_prefix='/geoNLI')
    return app
```

Registers the API blueprint under `/geoNLI` prefix.

---

```python
# Lines 98-111: __main__ block
if __name__ == '__main__':
    app = create_app()
    logger = logging.getLogger(__name__)
    host = os.environ.get('BACKEND_HOST', '0.0.0.0')
    logger.info(f"Starting GeoNLI server on http://{host}:{Config.PORT} (debug={Config.DEBUG})")
    app.run(host=host, port=Config.PORT, debug=Config.DEBUG, use_reloader=False)
```

Entry point. `BACKEND_HOST` env var controls bind address. `use_reloader=False` prevents double model loading.

---

## 4.2 — `backend/config.py`

```python
# Lines 4-6: Constants
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DEFAULT_ADV_VLM_MODEL_ID = 'Qwen/Qwen3-VL-2B-Instruct'
_DEFAULT_GROUNDING_DETECTOR_WEIGHTS = os.path.join(_REPO_ROOT, 'models', 'yolov8n-obb.pt')
```

Repo root resolved relative to config file. Default VLM is Qwen3-VL-2B-Instruct. Grounding detector weights path defined but not used in current code.

---

```python
# Lines 9-12: _resolve_model_dir()
def _resolve_model_dir(env_key: str, default_path: str) -> str:
    candidate = os.path.expanduser(os.environ.get(env_key, default_path))
    return candidate
```

Allows env vars to override model directories. `expanduser` handles `~` in paths.

---

```python
# Lines 15-62: Config class
class Config:
    PORT = int(os.environ.get('BACKEND_PORT', 8000))
    DEBUG = os.environ.get('BACKEND_DEBUG', '0') == '1'
```

All config values are class-level constants resolved at import time from environment variables.

```python
    ADV_VLM_MODEL_ID = os.environ.get('ADV_VLM_MODEL_ID', _DEFAULT_ADV_VLM_MODEL_ID)
    ADV_VLM_TEMPERATURE = float(os.environ.get('ADV_VLM_TEMPERATURE', 0.2))
    ADV_VLM_MAX_NEW_TOKENS = int(os.environ.get('ADV_VLM_MAX_NEW_TOKENS', 64))
    CAPTION_GENERATION_TIMEOUT_S = float(os.environ.get('CAPTION_GENERATION_TIMEOUT_S', 50))
```

VLM generation parameters: temperature 0.2, max 64 tokens, 50s timeout.

```python
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
```

16 MB upload limit enforced by Flask.

```python
    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(os.path.join(Config.BASE_DIR, 'weights'), exist_ok=True)
```

Creates required directories on startup.

---

## 4.3 — `backend/api/routes.py`

```python
# Lines 46-339: analyze() endpoint
@api_bp.route('/eval', methods=['POST'])
def analyze():
    request_started = time.monotonic()
    logger.info("/geoNLI/eval request | request_id=%s | content_type=%s", current_request_id(), request.content_type)
```

Logs request start with content type for debugging.

---

### JSON Path (Lines 54-207)

```python
        if request.is_json:
            try:
                data = request.get_json()
            except BadRequest:
                logger.warning("Invalid JSON payload received")
                return _error('Invalid JSON payload', 400)
            response_data = data.copy()
```

Copies input data to use as response template. This allows echoing back with responses populated.

```python
            input_image_data = data.get('input_image', {})
            image_url = input_image_data.get('image_url')
            if not image_url:
                return _error('Missing input_image.image_url', 400)
```

Extracts and validates image URL.

```python
            metadata = input_image_data.get('metadata', {})
            spatial_resolution_m = metadata.get('spatial_resolution_m', 1.0)
```

Extracts spatial resolution with 1.0m default.

```python
            image_mode = (input_image_data.get('mode') or input_image_data.get('image_type') or 'RGB').upper()
            image = load_image_from_url(image_url)
```

Supports both `mode` and `image_type` keys for flexibility.

```python
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
```

FCC conversion returns 500 on failure. SAR conversion returns 503 (service unavailable) since it's optional.

---

### Grounding Query (Lines 93-123)

```python
            if 'grounding_query' in queries:
                grounding_instruction = queries['grounding_query'].get('instruction', '')
                if grounding_instruction.strip():
                    try:
                        buffer = BytesIO()
                        image.save(buffer, format='PNG')
                        image_bytes = buffer.getvalue()
                        sam3_payload = run_sam3_grounding(
                            image_bytes, grounding_instruction,
                            f"{orig_name}_grounding.png",
                            spatial_resolution_m=spatial_resolution_m,
                        )
```

Converts PIL image to PNG bytes for SAM3. The `orig_name` prefix helps identify the source image in SAM3's upload directory.

```python
                        results = {'grounding': {
                            'bboxes': sam3_payload.get('bboxes', []),
                            'answer': f"Found {len(sam3_payload.get('bboxes', []))} objects"
                        }}
```

Creates a simple answer string with detection count.

---

### Multipart Path (Lines 212-327)

```python
        elif 'image' in request.files:
            file = request.files['image']
            if not file.filename:
                return _error('No image file was selected', 400)
            image, image_bytes = load_image_from_file(file, return_bytes=True)
```

Loads image and raw bytes simultaneously. Raw bytes are needed for SAM3.

```python
            task_type = request.form.get('task_type', 'vqa')
            vqa_type = request.form.get('vqa_type', 'semantic')
            image_type = (request.form.get('image_type', 'RGB') or 'RGB').upper()
            query_text = request.form.get('query', '')
            spatial_resolution_m = float(request.form.get('spatial_resolution_m', 1.0))
```

Extracts form fields with defaults.

```python
            if not spatial_resolution_m > 0:
                return _error('spatial_resolution_m must be greater than zero', 400)
            if task_type not in {'captioning', 'vqa', 'grounding'}:
                return _error('task_type must be captioning, vqa, or grounding', 400)
            if vqa_type not in {'semantic', 'binary', 'numeric'}:
                return _error('vqa_type must be semantic, binary, or numeric', 400)
            if task_type != 'captioning' and not query_text.strip():
                return _error('query is required for VQA and grounding', 400)
```

Validation chain: spatial resolution > 0, valid task type, valid VQA type, query present for non-captioning tasks.

---

### Task Routing (Lines 268-298)

```python
            if task_type == 'grounding':
                started = time.monotonic()
                try:
                    sam3_payload = run_sam3_grounding(
                        image_bytes, query_text,
                        getattr(file, 'filename', None),
                        spatial_resolution_m=spatial_resolution_m,
                    )
                    if converted_path:
                        sam3_payload['converted_rgb_path'] = converted_path
                    sam3_payload.update({'success': True, 'task': 'grounding', 'response': sam3_payload.get('answer', ''), 'request_id': current_request_id(), 'timestamp': _timestamp()})
                    return jsonify(sam3_payload), 200
                except Exception as exc:
                    logger.exception("Inference failed | task=grounding | model=SAM3")
                    return _error(_public_inference_error(exc), 503)
```

Grounding queries bypass the ModelManager entirely and go straight to SAM3. Errors return 503.

```python
            mock_queries = {}
            if task_type == 'captioning':
                mock_queries['caption_query'] = {'instruction': 'Describe'}
            else:
                mock_queries['attribute_query'] = {vqa_type: {'instruction': query_text}}
```

**Key design:** The frontend sends simple form fields, but the backend constructs the internal query schema to reuse `ModelManager.process_all_queries()`.

---

### Result Flattening (Lines 310-327)

```python
            flat_response = {"success": True, "task": task_type, "answer": "", "response": "", "bboxes": [], "request_id": current_request_id(), "timestamp": _timestamp()}
            if 'caption' in results: flat_response['answer'] = results['caption']['answer']
            if 'grounding' in results: flat_response['bboxes'] = results['grounding'].get('bboxes', [])
            attr_key = f'attr_{vqa_type}'
            if attr_key in results:
                flat_response['answer'] = results[attr_key].get('answer', '')
                flat_response['bboxes'] = results[attr_key].get('bboxes', flat_response['bboxes'])
            if converted_path:
                flat_response['converted_rgb_path'] = converted_path
            flat_response['response'] = flat_response['answer']
```

Flattens the nested result structure into a simple frontend-friendly format. The `response` field mirrors `answer` for compatibility.

---

## 4.4 — `backend/core/model.py`

```python
# Lines 27-42: ensure_pil_image()
def ensure_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, Path):
        return ensure_pil_image(str(image))
    if isinstance(image, str):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image path not found: {image}")
        with Image.open(path) as img:
            return img.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")
```

Normalizes any image input to RGB PIL. Supports PIL, Path, and string paths.

---

```python
# Lines 45-109: AdvancedVLMAdapter
class AdvancedVLMAdapter:
    def __init__(self, config):
        self.config = config
        self.temperature = getattr(config, 'ADV_VLM_TEMPERATURE', 0.0)
        self.max_new_tokens = getattr(config, 'ADV_VLM_MAX_NEW_TOKENS', 64)
        self.top_p = getattr(config, 'ADV_VLM_TOP_P', 0.9)
        default_do_sample = self.temperature > 0
        self.do_sample = getattr(config, 'ADV_VLM_DO_SAMPLE', default_do_sample)
```

Uses `getattr` with defaults for resilience. `do_sample` defaults to True only when temperature > 0.

```python
        captioning_model_dir = getattr(config, 'CAPTIONING_MODEL_DIR', None)
        if captioning_model_dir:
            os.environ.setdefault('captioning_model_dir', str(captioning_model_dir))
            os.environ.setdefault('CAPTIONING_MODEL_DIR', str(captioning_model_dir))
```

Sets environment variables for the captioning module to discover the adapter.

```python
        self._lock = threading.Lock()
        self._load_backend()
```

Thread lock prevents concurrent model access.

---

```python
    def generate(self, image, prompt, *, temperature=None, max_new_tokens=None, max_time=None):
        pil_image = ensure_pil_image(image)
        prompt_clean = (prompt or "").strip()
        if prompt is not None and not prompt_clean:
            raise ValueError("Prompt must be a non-empty string when provided for VLM inference.")
        text_input = prompt_clean or "Describe the remote sensing scene in detail."
```

Validates prompt. Falls back to a default description prompt if none provided.

```python
        waiting = time.monotonic()
        with GPU_INFERENCE_LOCK, self._lock:
            lock_wait_s = time.monotonic() - waiting
            inference_started = time.monotonic()
            placement_wait_s = GPU_MODEL_SCHEDULER.activate_vlm()
```

Acquires both the global GPU lock and the instance lock. Measures wait times for logging.

```python
            result = self.model.generate(text_input, pil_image, **generate_kwargs)
```

Delegates to CaptioningBackend.generate().

---

```python
# Lines 197-225: ModelManager.__init__
class ModelManager:
    def __init__(self, config):
        logger.info("Initializing Advanced VLM Model Manager...")
        self.config = config
        self.adapter = AdvancedVLMAdapter(config)

        self.vqa_backend = None
        vqa_dir = getattr(config, 'VQA_MODEL_DIR', None)
        if vqa_dir and self._vqa_checkpoint_complete(vqa_dir):
            try:
                self.vqa_backend = VQABackend(
                    checkpoint_dir=vqa_dir,
                    device=getattr(config, 'VQA_DEVICE', None),
                    max_new_tokens=getattr(config, 'VQA_MAX_NEW_TOKENS', 128),
                )
            except Exception as exc:
                logger.error("Failed to load dedicated VQA model (%s). Falling back to captioning adapter.", exc)
```

VQA backend is optional. If checkpoint is incomplete or fails to load, falls back gracefully.

```python
        self.router = None
```

**Router is always None in the current implementation.** This is a deliberate choice to avoid loading a second large model.

---

```python
# Lines 227-241: _vqa_checkpoint_complete()
    @staticmethod
    def _vqa_checkpoint_complete(checkpoint_dir) -> bool:
        import json
        directory = Path(checkpoint_dir)
        index_path = directory / 'model.safetensors.index.json'
        if index_path.exists():
            try:
                weight_map = json.loads(index_path.read_text()).get('weight_map', {})
                shards = set(weight_map.values())
                return bool(shards) and all((directory / shard).is_file() for shard in shards)
            except (OSError, ValueError, TypeError):
                return False
        return any((directory / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin'))
```

Validates sharded checkpoints by checking every shard file exists. This prevents loading incomplete downloads.

---

```python
# Lines 307-344: _compute_numeric_value()
    def _compute_numeric_value(self, instances, numeric_type):
        if numeric_type == "COUNT":
            return float(len(instances))
        spatial_resolution = getattr(self, '_current_spatial_resolution', 1.0)
        values = []
        for inst in instances:
            corners = inst.get("obb_corners")
            if not corners or len(corners) != 4:
                continue
            if numeric_type == "LENGTH":
                pixel_length = get_largest_side(corners)
                values.append(pixel_length * spatial_resolution)
            elif numeric_type == "PERIMETER":
                pixel_perimeter = get_perimeter(corners)
                values.append(pixel_perimeter * spatial_resolution)
            elif numeric_type == "AREA":
                pixel_area = inst.get("area")
                if pixel_area is None:
                    pixel_area = self._polygon_area(corners)
                real_area = pixel_area * (spatial_resolution ** 2)
                values.append(real_area)
```

**Key formula:** Area uses `spatial_resolution²` (square meters), perimeter/length use `spatial_resolution` (meters). This correctly accounts for the 2D nature of area.

```python
        if numeric_type == "AREA":
            return float(sum(values))
        if numeric_type == "PERIMETER":
            return float(sum(values))
        if numeric_type == "LENGTH":
            return max(values)
```

Area and perimeter sum across all instances. Length takes the maximum (longest single object).

---

```python
# Lines 346-412: _process_routed_query()
    def _process_routed_query(self, image, instruction, task_type):
        query = (instruction or "").strip()
        if not query:
            raise ValueError("Attribute query must include an instruction.")
        if task_type not in {"binary", "numeric"}:
            return self._vlm_only_answer(image, query, task_type)
        if self.router is None:
            return self._vlm_only_answer(image, query, task_type)
```

Since `self.router` is always None, all binary/numeric queries go VLM-only. The SAM+VLM path is dead code in the current configuration.

---

```python
# Lines 414-459: process_all_queries()
    def process_all_queries(self, image, image_type, queries, spatial_resolution_m=1.0):
        pil_image = ensure_pil_image(image)
        results = {}
        self._current_spatial_resolution = spatial_resolution_m
```

Stores spatial resolution as instance state for numeric computations.

```python
        if 'caption_query' in queries:
            results['caption'] = self._safe_predict(self.captioner.predict, pil_image)
        if 'grounding_query' in queries:
            instr = queries['grounding_query'].get('instruction', '')
            results['grounding'] = self._safe_predict(self.grounder.predict, pil_image, instr)
        if 'attribute_query' in queries:
            attr_q = queries['attribute_query']
            if 'binary' in attr_q:
                results['attr_binary'] = self._safe_predict(
                    self._process_routed_query, pil_image,
                    attr_q['binary'].get('instruction', ''), 'binary',
                )
```

Processes each query type independently. `_safe_predict` catches exceptions to prevent one failure from crashing the batch.

---

```python
# Lines 462-476: get_model() singleton
_manager_instance = None
_manager_lock = threading.Lock()

def get_model(config=None):
    global _manager_instance
    if config is None:
        from backend.config import Config
        config = Config
    if _manager_instance is None and config:
        with _manager_lock:
            if _manager_instance is None:
                with GPU_INFERENCE_LOCK:
                    _manager_instance = ModelManager(config)
    return _manager_instance
```

Thread-safe singleton with double-checked locking. The GPU lock prevents model initialization from competing with inference.

---

## 4.5 — `backend/services/inference_lock.py`

```python
# Lines 10-52: FairReentrantLock
class FairReentrantLock:
    def __init__(self):
        self._condition = threading.Condition()
        self._owner = None
        self._depth = 0
        self._next_ticket = 0
        self._waiters = deque()
```

Uses a Condition variable with a ticket queue for FIFO ordering.

```python
    def acquire(self):
        ident = threading.get_ident()
        with self._condition:
            if self._owner == ident:
                self._depth += 1
                return True
            ticket = self._next_ticket
            self._next_ticket += 1
            self._waiters.append(ticket)
            while self._owner is not None or self._waiters[0] != ticket:
                self._condition.wait()
            self._waiters.popleft()
            self._owner = ident
            self._depth = 1
            return True
```

**Reentrant:** If the same thread already owns the lock, just increment depth. **FIFO:** Each waiter gets a ticket; only the thread with the lowest ticket can proceed.

---

```python
# Lines 58-112: GPUModelScheduler
class GPUModelScheduler:
    def __init__(self):
        self._vlm = None
        self._sam3 = None
        self._active = None

    def activate_vlm(self) -> float:
        import torch
        started = time.monotonic()
        if torch.cuda.is_available():
            if self._sam3 is not None and self._sam3.device != "cpu":
                self._sam3.move_to("cpu")
            if self._vlm is not None:
                self._vlm.move_to("cuda")
        self._active = "vlm"
        return time.monotonic() - started
```

Moves SAM3 to CPU first (freeing GPU memory), then moves VLM to GPU. Returns duration for logging.

```python
    def prepare_sam3(self) -> float:
        started = time.monotonic()
        if self._vlm is not None and self._vlm.device != "cpu":
            self._vlm.move_to("cpu")
        self._active = "sam3_loading"
        return time.monotonic() - started
```

Called before SAM3 initialization to ensure GPU is free.

---

## 4.6 — `models/captioning.py`

```python
# Lines 17-27: Constants
BASE_MODEL_ID = os.environ.get("ADV_VLM_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
DEFAULT_CAPTION_PROMPT = "Describe the main visible content in this image in one concise sentence."
MIN_PIXELS = 256 * 256
MAX_PIXELS = 384 * 384
CAPTIONING_MODEL_DIR = Path(os.environ.get("CAPTIONING_MODEL_DIR", Path(__file__).resolve().parent / "captioning-model")).expanduser()
PROCESSOR_DIR = CAPTIONING_MODEL_DIR / "processor"
ADAPTER_DIR = CAPTIONING_MODEL_DIR / "model"
```

Pixel bounds (256² to 384²) keep vision activations bounded for 6 GiB cards.

---

```python
# Lines 53-58: Adapter detection
        self.use_adapter = (
            (self.adapter_path / "adapter_model.safetensors").exists()
            or (self.adapter_path / "adapter_model.bin").exists()
        )
        if not self.use_adapter:
            logger.warning("No PEFT adapter weights found at %s; captioning uses the base model without fine-tuning", self.adapter_path)
```

Gracefully handles missing adapter. The system works without fine-tuning.

---

```python
# Lines 73-85: Adapter validation
        if self.use_adapter:
            import json
            adapter_config_path = self.adapter_path / "adapter_config.json"
            if not adapter_config_path.is_file():
                raise RuntimeError(f"Adapter weights exist but adapter_config.json is missing: {self.adapter_path}")
            adapter_config = json.loads(adapter_config_path.read_text())
            adapter_base = adapter_config.get("base_model_name_or_path")
            for adapter_base and adapter_base.rstrip("/") != self.base_model_id.rstrip("/"):
                raise RuntimeError(f"Caption adapter targets {adapter_base}, but configured base model is {self.base_model_id}")
```

Validates adapter matches base model. The adapter targets `Qwen/Qwen3-VL-8B-Instruct` but the default base is `Qwen/Qwen3-VL-2B-Instruct`, so this check would fail if both were defaults. In practice, `ADV_VLM_MODEL_ID` must match the adapter's base.

---

```python
# Lines 99-124: Model loading
        model_kwargs = {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        }
        base_model = AutoModelForImageTextToText.from_pretrained(self.base_model_id, **model_kwargs)
        if self.use_adapter:
            self.model = PeftModel.from_pretrained(base_model, str(self.adapter_path), torch_dtype=self.dtype)
        else:
            self.model = base_model
        self.model.eval()
        self._model_device = "cpu"
        GPU_MODEL_SCHEDULER.register_vlm(self)
```

Uses `sdpa` (scaled dot-product attention) for memory efficiency. Model starts on CPU. Registers with scheduler for exclusive placement.

---

```python
# Lines 172-281: _generate()
    def _generate(self, user_prompt, image, **overrides):
        prompt = (user_prompt or DEFAULT_CAPTION_PROMPT).strip()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        chat_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

Builds chat message and applies Qwen3-VL's chat template.

```python
        inputs = self.processor(images=[image], text=[chat_prompt], return_tensors="pt")
        input_device = next(self.model.parameters()).device
        inputs = {k: v.to(input_device) if torch.is_tensor(v) else v for k, v in inputs.items()}
```

Moves inputs to wherever the model currently lives (CPU or GPU).

```python
        placement_s = GPU_MODEL_SCHEDULER.activate_vlm()
        input_device = next(self.model.parameters()).device
        inputs = {k: v.to(input_device) if torch.is_tensor(v) else v for k, v in inputs.items()}
```

After activation, re-resolves device (model may have moved to GPU).

```python
        with torch.inference_mode():
            autocast_enabled = self.torch_device.type == "cuda"
            with torch.autocast(device_type=self.torch_device.type, dtype=self.autocast_dtype, enabled=autocast_enabled):
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, ...)
```

Uses autocast for mixed precision on CUDA. `inference_mode` disables gradient tracking.

```python
        if max_time is not None and generation_duration >= max_time:
            raise TimeoutError(f"Vision-language generation exceeded its {max_time:g}-second time limit; no partial caption was returned.")
```

Timeout check after generation. No partial results returned.

---

## 4.7 — `models/vqa.py`

```python
# Lines 39-48: _build_prompt()
def _build_prompt(task_type, question):
    template = _TEMPLATES.get(task_type, _TEMPLATES["semantic"])
    return (
        "<|im_start|>system\n"
        f"{template['system']}<|im_end|>\n"
        "<|im_start|>user\n"
        "Picture 1: <|vision_start|><|image_pad|><|vision_end|>\n"
        f"{template['user_suffix']} {question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
```

Manually constructs Qwen chat format with vision tokens. This matches the training format.

---

```python
# Lines 92-100: rope_scaling injection
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None and getattr(text_cfg, "rope_scaling", None) is None:
            rope_params = getattr(text_cfg, "rope_parameters", {}) or {}
            text_cfg.rope_scaling = {
                "mrope_interleaved": rope_params.get("mrope_interleaved", True),
                "mrope_section": rope_params.get("mrope_section", [24, 20, 20]),
                "rope_type": rope_params.get("rope_type", "default"),
            }
```

Injects M-RoPE (multi-resolution rotary position embedding) config that may be missing from saved checkpoints.

---

## 4.8 — `models/sam3/runtime_service.py`

```python
# Lines 20-31: Import handling
_SAM3_DIR = Path(__file__).resolve().parent
_ORIG_CWD = os.getcwd()
try:
    os.chdir(str(_SAM3_DIR))
    if str(_SAM3_DIR) not in sys.path:
        sys.path.insert(0, str(_SAM3_DIR))
    try:
        from . import unified_inference as sam3_pipeline
    except ImportError as exc:
        raise RuntimeError("Could not import models.sam3.unified_inference") from exc
finally:
    os.chdir(_ORIG_CWD)
```

Temporarily changes CWD to import SAM3 modules, then restores. This is necessary because SAM3 uses relative imports.

---

```python
# Lines 48-101: __init__
    def __init__(self, sam3_root, *, score_threshold=0.4, min_area=100, nms_threshold=0.6,
                 clip_threshold=0.24, disable_clip=True, refiner_temperature=0.1,
                 qwen_model="qwen3-vl:30b", vlm_agent=None):
```

Defaults: CLIP disabled, refiner disabled, VLM agent disabled. This is the minimal configuration.

```python
        configured_checkpoint = os.environ.get("SAM3_CHECKPOINT")
        self.ckpt_path = Path(configured_checkpoint).expanduser() if configured_checkpoint else sam3_root / "sam.pt"
        if not self.ckpt_path.is_file():
            try:
                from huggingface_hub import hf_hub_download
                cached = hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt", local_files_only=True)
                self.ckpt_path = Path(cached)
            except Exception as exc:
                raise FileNotFoundError("SAM3 checkpoint was not found...") from exc
```

Checkpoint resolution: env var → local `sam.pt` → HF cache. Fails if none found.

---

```python
# Lines 263-286: _grounding_phrase()
    @staticmethod
    def _grounding_phrase(query):
        phrase = re.sub(r"[?!.,;:]+$", "", query.strip().lower())
        phrase = re.sub(r"^(?:where\s+(?:is|are)|locate|find|detect|identify|show(?:\s+me)?|ground|segment)\s+", "", phrase)
        phrase = re.sub(r"^(?:(?:the|a|an|any|all)\s+)+", "", phrase)
        phrase = re.sub(r"\s+(?:in|on|at)\s+(?:this|the)\s+(?:image|scene)$", "", phrase)
        phrase = re.sub(r"\s+(?:visible|here)$", "", phrase).strip()
```

Strips location verbs and articles. "where are airplanes?" → "airplanes".

```python
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
```

Singularizes the last word. "airplanes" → "airplane". This improves SAM3's text encoder matching.

---

## 4.9 — Frontend: `ui/frontend/src/services/api.js`

```javascript
// Lines 13-27: analyzeImage()
export const analyzeImage = async (taskType, query, imageFile, isSimulating, vqaType, imageType, spatialResolutionM) => {
    if (isSimulating) {
        return simulateBackendResponse(taskType, query, vqaType, imageType);
    } else {
        return fetchRealBackend(taskType, query, imageFile, vqaType, imageType, spatialResolutionM);
    }
};
```

Simulation mode allows frontend development without a running backend.

---

```javascript
// Lines 29-74: fetchRealBackend()
const fetchRealBackend = async (taskType, query, imageFile, vqaType, imageType, spatialResolutionM) => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 90000);
```

90-second timeout accounts for model inference time.

```javascript
    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("task_type", taskType);
    formData.append("image_type", imageType);
    formData.append("spatial_resolution_m", spatialResolutionM.toString());
    if (taskType === "vqa") {
        formData.append("vqa_type", vqaType);
    }
    formData.append("query", taskType === "captioning" ? "" : query);
```

Builds multipart form. Captioning sends empty query (backend uses default prompt).

```javascript
    const response = await fetch(`${BACKEND_BASE_URL}/geoNLI/eval`, {
        method: "POST", body: formData, signal: controller.signal,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload.success === false) {
        const requestRef = payload.request_id ? ` (request ${payload.request_id})` : "";
        const message = `${payload.error || payload.details || `HTTP ${response.status}`}${requestRef}`;
        throw new Error(`Request failed (HTTP ${response.status}): ${message}`);
    }
    if (!payload.answer && taskType !== "grounding") {
        throw new Error("Backend returned an empty answer.");
    }
    return payload;
```

Error handling includes request ID for server-side log correlation.

---

## 4.10 — Coverage Checklist

| File | Status | Documented In |
|---|---|---|
| backend/app.py | Fully inspected | 4.1 |
| backend/config.py | Fully inspected | 4.2 |
| backend/api/routes.py | Fully inspected | 4.3 |
| backend/core/model.py | Fully inspected | 4.4 |
| backend/services/inference_lock.py | Fully inspected | 4.5 |
| backend/services/sam3_runner.py | Fully inspected | 03 |
| backend/services/sar_converter.py | Fully inspected | 03 |
| backend/services/query_router.py | Fully inspected | 03 |
| backend/services/request_context.py | Fully inspected | 03 |
| backend/utils/image_ops.py | Fully inspected | 03 |
| models/captioning.py | Fully inspected | 4.6 |
| models/vqa.py | Fully inspected | 4.7 |
| models/device.py | Fully inspected | 03 |
| models/sam3/runtime_service.py | Fully inspected | 4.8 |
| models/sam3/unified_inference.py | Fully inspected | 03 |
| models/SAR2Optical/inference.py | Fully inspected | 03 |
| models/SAR2Optical/src/networks.py | Fully inspected | 03 |
| ui/frontend/src/GeoNLIDashboard.jsx | Fully inspected | 03 |
| ui/frontend/src/services/api.js | Fully inspected | 4.9 |
| ui/frontend/src/components/*.jsx (10 files) | Fully inspected | 03 |
| VQA-fine-tuning/vrsbench_finetune/train.py | Fully inspected | 03 |
| VQA-fine-tuning/vrsbench_finetune/model.py | Fully inspected | 03 |
| VQA-fine-tuning/vrsbench_finetune/dataset.py | Fully inspected | 03 |
| VQA-fine-tuning/vrsbench_finetune/preprocess.py | Fully inspected | 03 |
| VQA-fine-tuning/vrsbench_finetune/eval.py | Fully inspected | 03 |
| VQA-fine-tuning/vrsbench_finetune/utils.py | Fully inspected | 03 |
| start_geonli.py | Fully inspected | 03 |
| stop_geonli.py | Fully inspected | 03 |
| setup_gpu.sh | Fully inspected | 03 |

---

*Next: [Models and Model Selection](05_models_and_model_selection.md)*
