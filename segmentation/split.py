"""Create a reproducible train/val/test split."""
import json
import random
from pathlib import Path


def make_split(images_dir: Path, out_path: Path,
               val_frac: float = 0.15, test_frac: float = 0.15,
               seed: int = 42) -> dict:
    names = sorted(p.stem for p in images_dir.glob("*.jpg"))
    if not names:
        raise FileNotFoundError(f"No .jpg images found in {images_dir}")

    rng = random.Random(seed)
    rng.shuffle(names)

    n_test = int(len(names) * test_frac)
    n_val = int(len(names) * val_frac)
    split = {
        "test": sorted(names[:n_test]),
        "val": sorted(names[n_test:n_test + n_val]),
        "train": sorted(names[n_test + n_val:]),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split, indent=2))
    return split