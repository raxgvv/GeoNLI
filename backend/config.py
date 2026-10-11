import os
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DEFAULT_ADV_VLM_MODEL_ID = 'Qwen/Qwen3-VL-2B-Instruct'
_DEFAULT_GROUNDING_DETECTOR_WEIGHTS = os.path.join(_REPO_ROOT, 'models', 'yolov8n-obb.pt')


def _resolve_model_dir(env_key: str, default_path: str) -> str:
    candidate = os.path.expanduser(os.environ.get(env_key, default_path))
    # Optional model directories must not prevent the core service from starting.
    return candidate


class Config:
    # Server Settings
    PORT = int(os.environ.get('BACKEND_PORT', 8000))
    DEBUG = os.environ.get('BACKEND_DEBUG', '0') == '1'
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FINAL_MODEL_DIR = os.environ.get('FINAL_MODEL_DIR', os.path.join(_REPO_ROOT, 'Final_model'))
    CAPTIONING_MODEL_DIR = _resolve_model_dir('CAPTIONING_MODEL_DIR', os.path.join(_REPO_ROOT, 'models', 'captioning-model'))
    VQA_MODEL_DIR = _resolve_model_dir('VQA_MODEL_DIR', os.path.join(_REPO_ROOT, 'models', 'vqa-model'))
    
    # Model Weights Paths
    CAPTION_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'captioning_model.pth')
    GROUNDING_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'grounding_model.pth')
    VQA_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'vqa_model.pth')

    # Advanced VLM backbone (BLIP-2 with larger text decoder)
    ADV_VLM_MODEL_ID = os.environ.get('ADV_VLM_MODEL_ID', _DEFAULT_ADV_VLM_MODEL_ID)
    ADV_VLM_DEVICE = os.environ.get('ADV_VLM_DEVICE', 'auto')
    ADV_VLM_TEMPERATURE = float(os.environ.get('ADV_VLM_TEMPERATURE', 0.2))
    ADV_VLM_MAX_NEW_TOKENS = int(os.environ.get('ADV_VLM_MAX_NEW_TOKENS', 64))
    CAPTION_GENERATION_TIMEOUT_S = float(os.environ.get('CAPTION_GENERATION_TIMEOUT_S', 50))
    ADV_VLM_TOP_P = float(os.environ.get('ADV_VLM_TOP_P', 0.9))
    ADV_VLM_DO_SAMPLE = os.environ.get('ADV_VLM_DO_SAMPLE', '0') == '1'

    # Grounding detector (YOLO-based) for bounding boxes
    GROUNDING_DETECTOR_ENABLED = os.environ.get('GROUNDING_DETECTOR_ENABLED', '1') == '1'
    GROUNDING_DETECTOR_WEIGHTS = os.environ.get(
        'GROUNDING_DETECTOR_WEIGHTS',
        _DEFAULT_GROUNDING_DETECTOR_WEIGHTS,
    )
    
    # DEVICE MAPPING (Assign separate GPUs here if available)
    # Example: 'cuda:0', 'cuda:1', 'cpu'
    # If you have 1 GPU, set all to 'cuda:0'. If 2, split them up.
    CAPTION_DEVICE = os.environ.get('CAPTION_DEVICE', 'auto')
    GROUNDING_DEVICE = os.environ.get('GROUNDING_DEVICE', 'auto')
    VQA_DEVICE = os.environ.get('VQA_DEVICE', 'auto')
    VQA_MAX_NEW_TOKENS = int(os.environ.get('VQA_MAX_NEW_TOKENS', 128))
    
    # Upload Settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'temp_uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(os.path.join(Config.BASE_DIR, 'weights'), exist_ok=True)
