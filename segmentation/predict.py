"""Predict a mask for one image with the trained U-Net."""
from pathlib import Path

import cv2
import torch

from segmentation.evaluate_model import load_model, predict_unet


def predict_image(image_path: Path, weights: Path, output_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)
    mask = predict_unet(model, image, device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)
    return mask