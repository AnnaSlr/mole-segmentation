"""Training loop: U-Net with BCE + Dice loss."""
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmentation.dataset import MoleDataset
from segmentation.model import build_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def validate(model, loader, device) -> float:
    model.eval()
    scores = []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = (torch.sigmoid(model(images)) > 0.5).float()
        inter = (preds * masks).sum(dim=(1, 2, 3))
        total = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        scores.append(((2 * inter + 1e-7) / (total + 1e-7)).cpu())
    return torch.cat(scores).mean().item()


def train(processed_dir: Path, splits_path: Path, out_dir: Path,
          epochs: int = 30, batch_size: int = 16, lr: float = 1e-3,
          limit: int | None = None) -> None:
    device = get_device()
    print(f"Device: {device}")

    train_ds = MoleDataset(processed_dir, splits_path, "train",
                           train=True, limit=limit)
    val_ds = MoleDataset(processed_dir, splits_path, "val", limit=limit)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2)

    model = build_model().to(device)
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = bce_loss(logits, masks) + dice_loss(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        val_dice = validate(model, val_loader, device)
        scheduler.step(val_dice)
        print(f"Epoch {epoch}: loss={running / len(train_loader):.4f} "
              f"val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            print(f"  -> new best model saved (val_dice={best_dice:.4f})")