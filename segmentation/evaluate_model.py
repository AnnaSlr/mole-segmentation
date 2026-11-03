"""
Evaluate Otsu baseline and trained U-Net on the same split, same metrics.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from segmentation.dataset import get_transforms
from segmentation.metrics import dice_score, iou_score
from segmentation.model import build_model
from segmentation.segment import predict_mask_downscaled

SIZE = 256  # input size of the U-Net (same as in preprocessing)


def load_model(weights: Path, device: torch.device):
    """
    Build the U-Net and load trained weights (inference mode).
    """
    model = build_model(pretrained=False)
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def predict_unet(model, image_bgr: np.ndarray, device) -> np.ndarray:
    """
    Return a 0/255 mask at the ORIGINAL image resolution.
    """
    h, w = image_bgr.shape[:2]
    small = cv2.resize(image_bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    x = get_transforms(train=False)(image=rgb)["image"]
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)

    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8) * 255
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def evaluate_on_split(images_dir: Path, masks_dir: Path, splits_path: Path,
                      weights: Path, split: str = "test",
                      csv_path: Path | None = None) -> dict:
    """
    Dice/IoU of Otsu and U-Net on the same images of a split.
    Both are scored at the resolution of the ground-truth mask. Otsu runs on
    images reduced to 1/4 size, with a kernel that is not rescaled (see
    `segment.OTSU_KERNEL`).
    """
    names = json.loads(splits_path.read_text())[split]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)

    rows = []
    for name in tqdm(names):
        image = cv2.imread(str(images_dir / f"{name}.jpg"))
        target = cv2.imread(str(masks_dir / f"{name}_segmentation.png"),
                            cv2.IMREAD_GRAYSCALE)
        if image is None or target is None:
            tqdm.write(f"Skipping {name}")
            continue

        otsu = predict_mask_downscaled(image, target.shape)
        unet = predict_unet(model, image, device)
        if unet.shape != target.shape:
            unet = cv2.resize(unet, (target.shape[1], target.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        rows.append((name,
                     dice_score(otsu, target), iou_score(otsu, target),
                     dice_score(unet, target), iou_score(unet, target)))

    if not rows:
        raise RuntimeError(f"No image could be evaluated in split '{split}'")

    arr = np.array([r[1:] for r in rows])
    summary = {}
    for i, method in enumerate(["otsu", "unet"]):
        dice, iou = arr[:, 2 * i], arr[:, 2 * i + 1]
        summary[method] = {
            "dice_mean": float(dice.mean()), "dice_std": float(dice.std()),
            "dice_median": float(np.median(dice)),
            "iou_mean": float(iou.mean()), "iou_median": float(np.median(iou)),
        }
    summary["n_images"] = len(rows)

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "otsu_dice", "otsu_iou",
                             "unet_dice", "unet_iou"])
            writer.writerows(rows)
    return summary