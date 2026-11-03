"""
Baseline segmentation: Gaussian blur + Otsu threshold in Lab space.
"""
from pathlib import Path

import cv2
import numpy as np

# Size (in pixels) of the morphological kernel.
OTSU_KERNEL = 15


def predict_mask(image: np.ndarray, kernel_size: int = OTSU_KERNEL) -> np.ndarray:
    """
    Return a binary mask (0/255, uint8) from a BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    channel = cv2.GaussianBlur(lab[:, :, 0], (7, 7), 0)

    # Moles are darker than the skin -> inverted threshold
    _, mask = cv2.threshold(channel, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def predict_mask_downscaled(image: np.ndarray, out_shape: tuple[int, ...],
                            scale: float = 0.25,
                            kernel_size: int = OTSU_KERNEL) -> np.ndarray:
    """
    Otsu on a reduced image (faster), resized back to `out_shape`.
    `out_shape` is (height, width[, ...]), e.g. the shape of the target mask.
    """
    small = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    mask = predict_mask(small, kernel_size)
    return cv2.resize(mask, (out_shape[1], out_shape[0]),
                      interpolation=cv2.INTER_NEAREST)


def segment_image(image_path: Path, output_path: Path) -> np.ndarray:
    """
    Segment one image with Otsu and save the mask.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    mask = predict_mask(image)
    cv2.imwrite(str(output_path), mask)
    return mask