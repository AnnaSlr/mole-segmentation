"""
Evaluate the Otsu baseline on a labelled dataset.

Otsu runs on images reduced to 1/4 size; the mask is resized back
to the resolution of the ground truth before computing the metrics.
"""
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from segmentation.metrics import dice_score, iou_score
from segmentation.segment import predict_mask_downscaled


def evaluate_dataset(images_dir: Path, masks_dir: Path,
                     limit: int | None = None,
                     csv_path: Path | None = None) -> dict:
    image_paths = sorted(images_dir.glob("*.jpg"))
    if limit:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found in {images_dir}")

    rows = []
    for img_path in tqdm(image_paths):
        mask_path = masks_dir / f"{img_path.stem}_segmentation.png"
        image = cv2.imread(str(img_path))
        target = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or target is None:
            tqdm.write(f"Skipping {img_path.name}: image or mask not readable")
            continue

        pred = predict_mask_downscaled(image, target.shape)
        rows.append((img_path.name,
                     dice_score(pred, target),
                     iou_score(pred, target)))

    if not rows:
        raise RuntimeError("No image could be evaluated")

    dice = np.array([r[1] for r in rows])
    iou = np.array([r[2] for r in rows])
    summary = {
        "n_images": len(rows),
        "dice_mean": float(dice.mean()), "dice_std": float(dice.std()),
        "dice_median": float(np.median(dice)),
        "iou_mean": float(iou.mean()), "iou_std": float(iou.std()),
        "iou_median": float(np.median(iou)),
    }

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "dice", "iou"])
            writer.writerows(rows)

    return summary