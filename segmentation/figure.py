"""
Comparison figure: image | Otsu | U-Net, each against the ground truth.

Yellow contour = ground truth, red = Otsu, green = U-Net.
"""
from pathlib import Path

import cv2
import numpy as np

from segmentation.metrics import dice_score
from segmentation.segment import predict_mask_downscaled

PANEL_W, PANEL_H = 320, 240
YELLOW = (0, 255, 255)  # BGR
RED = (0, 0, 255)
GREEN = (0, 200, 0)


def _fit(image: np.ndarray, interpolation: int) -> np.ndarray:
    """Resize to fit a PANEL_W x PANEL_H canvas, keeping the aspect ratio."""
    h, w = image.shape[:2]
    scale = min(PANEL_W / w, PANEL_H / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    canvas = np.zeros((PANEL_H, PANEL_W) + image.shape[2:], image.dtype)
    top, left = (PANEL_H - new_h) // 2, (PANEL_W - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def _draw(panel: np.ndarray, mask: np.ndarray, color: tuple) -> None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, color, 2, cv2.LINE_AA)


def _title(panel: np.ndarray, text: str) -> None:
    cv2.rectangle(panel, (0, 0), (PANEL_W, 24), (0, 0, 0), -1)
    cv2.putText(panel, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


def make_row(image: np.ndarray, target: np.ndarray, unet: np.ndarray,
             name: str) -> np.ndarray:
    """One row of three panels: image, Otsu and U-Net against the truth."""
    if unet.shape != target.shape:
        unet = cv2.resize(unet, (target.shape[1], target.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    otsu = predict_mask_downscaled(image, target.shape)

    base = _fit(image, cv2.INTER_AREA)
    truth = _fit(target, cv2.INTER_NEAREST)

    first = base.copy()
    _title(first, name)
    panels = [first]
    for label, pred, color in (("Otsu", otsu, RED), ("U-Net", unet, GREEN)):
        panel = base.copy()
        _draw(panel, truth, YELLOW)
        _draw(panel, _fit(pred, cv2.INTER_NEAREST), color)
        _title(panel, f"{label}  Dice {dice_score(pred, target):.2f}")
        panels.append(panel)
    return np.hstack(panels)


def save_comparison(names: list[str], images_dir: Path, masks_dir: Path,
                    weights: Path, output_path: Path) -> None:
    """Save one row per image name into a single PNG."""
    # Imported here so that make_row works (and is testable) without torch
    import torch

    from segmentation.evaluate_model import load_model, predict_unet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)

    rows = []
    for name in names:
        image = cv2.imread(str(images_dir / f"{name}.jpg"))
        target = cv2.imread(str(masks_dir / f"{name}_segmentation.png"),
                            cv2.IMREAD_GRAYSCALE)
        if image is None or target is None:
            raise FileNotFoundError(f"Cannot read image or mask for {name}")
        unet = predict_unet(model, image, device)
        rows.append(make_row(image, target, unet, name))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack(rows))
