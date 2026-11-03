"""Resize images and masks once, to speed up training."""
from pathlib import Path

import cv2
from tqdm import tqdm


def preprocess(images_dir: Path, masks_dir: Path, out_dir: Path,
               size: int = 256) -> None:
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(sorted(images_dir.glob("*.jpg"))):
        mask_path = masks_dir / f"{img_path.stem}_segmentation.png"
        image = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print(f"Skipping {img_path.name}")
            continue

        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(out_dir / "images" / f"{img_path.stem}.png"), image)
        cv2.imwrite(str(out_dir / "masks" / f"{img_path.stem}.png"), mask)