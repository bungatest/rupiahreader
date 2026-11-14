"""
Preprocessing pipeline for Rupiah banknote classification.
Matches the exact training pipeline:
- EXIF transpose
- Pad resize maintaining aspect ratio with black background to (224, 224)
- Normalization per model specification:
  - Custom CNN: [0, 255] float32 (Rescaling(1./255) is inside model)
  - MobileNetV2: [0, 1] float32 (Rescaling(1./255) is done outside model)
  - FNN: [0, 255] float32 (Rescaling(1./255) is inside model)
"""

import io
import base64
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Union, List

TARGET_SIZE = (224, 224)


def decode_image_bytes(data: Union[bytes, str]) -> Image.Image:
    """Decode raw bytes or base64 data-URI/string into a PIL RGB Image."""
    if isinstance(data, str):
        if "," in data:  # data:image/jpeg;base64,...
            data = data.split(",", 1)[1]
        raw_bytes = base64.b64decode(data)
    else:
        raw_bytes = data
    
    with Image.open(io.BytesIO(raw_bytes)) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    return img


def pad_resize(img: Image.Image, target_size: Tuple[int, int] = TARGET_SIZE) -> Image.Image:
    """
    Resize image to target_size maintaining aspect ratio with black padding.
    Matches ImageOps.contain + black padding from Rupiah_CustomCNN.ipynb and Rupiah_FNN.ipynb.
    """
    resized = ImageOps.contain(img, target_size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, (0, 0, 0))
    paste_pos = ((target_size[0] - resized.width) // 2, (target_size[1] - resized.height) // 2)
    canvas.paste(resized, paste_pos)
    return canvas


def preprocess_for_model(img: Image.Image, model_id: str) -> np.ndarray:
    """
    Transforms PIL Image to batch tensor (1, 224, 224, 3) float32 with appropriate pixel scaling.
    """
    padded = pad_resize(img, TARGET_SIZE)
    arr = np.array(padded, dtype=np.float32)  # shape (224, 224, 3), range [0, 255]

    if model_id == "mobilenet" or model_id == "mobilenetv2":
        # MobileNetV2 training had rescale=1./255 outside the graph
        arr = arr / 255.0
    else:
        # CNN and FNN have layers.Rescaling(1./255) inside the model graph,
        # so they expect [0, 255] input.
        pass

    return np.expand_dims(arr, axis=0)


def aggregate_frame_probabilities(prob_list: List[np.ndarray]) -> np.ndarray:
    """Average probability distributions across multiple frames (multi-frame capture)."""
    if not prob_list:
        raise ValueError("prob_list cannot be empty")
    stacked = np.stack(prob_list, axis=0)
    return np.mean(stacked, axis=0)

