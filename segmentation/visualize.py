"""
Overlay a mask contour on an image, for quick visual checks.
"""
from pathlib import Path

import cv2


def save_overlay(image_path: Path, mask_path: Path, output_path: Path,
                 scale: float = 0.25,
                 color: tuple[int, int, int] = (0, 255, 0)) -> None:
    """
    Draw the contour of `mask_path` on the image and save a reduced copy.
    """
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    thickness = max(2, image.shape[1] // 400)  # scales with the resolution
    cv2.drawContours(image, contours, -1, color, thickness)

    small = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), small)