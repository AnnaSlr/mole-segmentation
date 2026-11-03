"""
PyTorch dataset reading the preprocessed images and masks.
"""

import json
from pathlib import Path

import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(train: bool) -> A.Compose:
    """
    Augmentations (train only) followed by ImageNet normalisation.
    """
    normalize = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            normalize,
        ])
    return A.Compose([normalize])


class MoleDataset(Dataset):
    """
    Images and masks of one split, read from the preprocessed folder.
    """

    def __init__(self, processed_dir: Path, splits_path: Path,
                 split: str, train: bool = False, limit: int | None = None):
        splits = json.loads(splits_path.read_text())
        self.names = splits[split][:limit] if limit else splits[split]
        self.images_dir = processed_dir / "images"
        self.masks_dir = processed_dir / "masks"
        self.transform = get_transforms(train)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        image_path = self.images_dir / f"{name}.png"
        mask_path = self.masks_dir / f"{name}.png"

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(
                f"Cannot read {image_path} or {mask_path}. "
                "Did you run `python main.py preprocess`?")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # model expects RGB
        mask = (mask > 127).astype("float32")

        out = self.transform(image=image, mask=mask)
        image = torch.from_numpy(out["image"]).permute(2, 0, 1).float()
        mask = torch.from_numpy(out["mask"]).unsqueeze(0).float()
        return image, mask