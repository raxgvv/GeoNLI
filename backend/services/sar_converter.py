import logging
import os
import threading
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from backend.utils.image_ops import save_converted_sar_image

logger = logging.getLogger(__name__)

DEFAULT_SAR_CHECKPOINT = os.environ.get(
    'SAR2OPTICAL_CHECKPOINT',
    str(Path(__file__).resolve().parents[2] / 'models' / 'SAR2Optical' / 'pix2pix_gen_180.pth'),
)


class Sar2OpticalConverter:
    """Lazy-loaded Pix2Pix converter for SAR -> optical imagery."""

    def __init__(self, checkpoint_path: str, device: Optional[str] = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAR2Optical checkpoint not found at {self.checkpoint_path}. "
                "Set SAR2OPTICAL_CHECKPOINT or place the weights at the default path."
            )

        from torchvision import transforms
        from models.SAR2Optical.inference import Pix2Pix
        self._transforms = transforms
        self._pix2pix_class = Pix2Pix

        resolved_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(resolved_device)
        logger.info("[SAR2Optical] Loading Pix2Pix generator (%s)", self.device)
        self.model = self._load_model()
        self.preprocess = self._transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _load_model(self):
        model = (
            self._pix2pix_class(
                c_in=3,
                c_out=3,
                is_train=False,
                use_upsampling=False,
                mode='nearest',
            )
            .to(self.device)
            .eval()
        )
        try:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        model.gen.load_state_dict(state_dict, strict=False)
        return model

    def convert(self, image: Image.Image) -> Image.Image:
        pil_image = image.convert('RGB')
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model.generate(tensor, is_scaled=True)
            prediction = torch.clamp(prediction, -1, 1)
            prediction = (prediction + 1) / 2.0
            prediction = (prediction * 255).to(torch.uint8)
        array = prediction.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(array, mode='RGB')


_CONVERTER: Optional[Sar2OpticalConverter] = None
_CONVERTER_LOCK = threading.Lock()


def _get_converter() -> Sar2OpticalConverter:
    global _CONVERTER
    if _CONVERTER is None:
        with _CONVERTER_LOCK:
            if _CONVERTER is None:
                _CONVERTER = Sar2OpticalConverter(DEFAULT_SAR_CHECKPOINT)
    return _CONVERTER


def convert_sar_to_rgb(
    image: Image.Image,
    original_name: Optional[str] = None,
    *,
    save_output: bool = True,
) -> Tuple[Image.Image, Optional[str]]:
    """Convert SAR imagery to RGB and optionally persist the output."""
    converter = _get_converter()
    rgb_image = converter.convert(image)
    saved_path = None
    if save_output:
        try:
            saved_path = save_converted_sar_image(rgb_image, original_name)
        except Exception:
            logger.exception("Failed to persist SAR conversion output")
    return rgb_image, saved_path
