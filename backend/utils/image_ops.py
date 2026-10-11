import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image

def load_image_from_url(url):
    """Downloads image from a URL (For Evaluation Bot)"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to load image from URL: {str(e)}")

def load_image_from_file(file_storage, return_bytes=False):
    """Loads image from Flask file upload (For Frontend). Optionally returns raw bytes."""
    try:
        file_storage.stream.seek(0)
        data = file_storage.read()
        image = Image.open(BytesIO(data)).convert('RGB')
        file_storage.stream.seek(0)
        if return_bytes:
            return image, data
        return image
    except Exception as e:
        raise ValueError(f"Failed to process uploaded file: {str(e)}")


def fcc_to_rgb_array(fcc_rgb: np.ndarray) -> np.ndarray:
    """Convert a false-colour composite array (NIR, R, G) to natural RGB."""
    if fcc_rgb.ndim != 3 or fcc_rgb.shape[-1] != 3:
        raise ValueError("FCC images must be HxWx3 arrays")

    fcc = fcc_rgb.astype(np.float32)
    nir = fcc[..., 0]
    red = fcc[..., 1]
    green = fcc[..., 2]

    aB = 13.9053
    bB = 0.873393
    cB = -0.138254
    dB = -0.100616
    blue_synth = aB + bB * green + cB * red + dB * nir
    blue_synth = np.clip(blue_synth, 0, 255).astype(np.uint8)

    rgb = np.stack(
        [red.astype(np.uint8), green.astype(np.uint8), blue_synth],
        axis=-1,
    )
    return rgb


def convert_fcc_to_rgb(image: Image.Image) -> Image.Image:
    """Apply FCC->RGB conversion on a Pillow image."""
    rgb_base = image.convert('RGB')
    rgb_array = fcc_to_rgb_array(np.array(rgb_base))
    return Image.fromarray(rgb_array, mode='RGB')


def _save_converted_image(
    image: Image.Image,
    original_name: Optional[str],
    *,
    env_key: str,
    default_subdir: str,
    prefix: str,
) -> str:
    output_dir = Path(os.environ.get(env_key, f'/tmp/inter-iit/{default_subdir}'))
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(original_name).stem if original_name else prefix
    filename = f"{stem}_{uuid.uuid4().hex}.png"
    save_path = output_dir / filename
    image.save(save_path, format='PNG')
    return str(save_path)


def save_converted_fcc_image(image: Image.Image, original_name: Optional[str] = None) -> str:
    """Persist the converted FCC image and return the saved path."""
    return _save_converted_image(
        image,
        original_name,
        env_key='FCC_RGB_OUTPUT_DIR',
        default_subdir='fcc_rgb',
        prefix='fcc_rgb',
    )


def save_converted_sar_image(image: Image.Image, original_name: Optional[str] = None) -> str:
    """Persist the SAR->RGB conversion and return its path."""
    return _save_converted_image(
        image,
        original_name,
        env_key='SAR_RGB_OUTPUT_DIR',
        default_subdir='sar_rgb',
        prefix='sar_rgb',
    )