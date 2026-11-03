"""
Run the colour analysis on a whole split and write one CSV row per image.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from analysis.color import analyze_color
from segmentation.evaluate_model import load_model, predict_unet

NAN = float("nan")


def _features(result: dict) -> dict:
    """
    Pick the descriptors exported to the CSV from one analysis result.
    """
    skin = result.get("skin_lab_mean", [NAN, NAN, NAN])
    core, peri = result.get("core"), result.get("periphery")
    d_l = (peri["lab_mean"][0] - core["lab_mean"][0]) if core and peri else NAN
    return {
        "n_colors": result["n_colors"],
        "skin_L": skin[0], "skin_a": skin[1], "skin_b": skin[2],
        "delta_e": result.get("delta_e_vs_skin", NAN),
        "periphery_minus_core_L": d_l,
    }


def run_batch(images_dir: Path, splits_path: Path, weights: Path,
              out_csv: Path, split: str = "test", limit: int | None = None,
              k: int = 8, merge: float = 12.0, l_weight: float = 0.5,
              erode_frac: float = 0.03) -> list[dict]:
    """
    U-Net mask + colour analysis (raw and normalised) for each image.
    """
    names = json.loads(splits_path.read_text())[split]
    if limit:
        names = names[:limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)
    kw = {"k": k, "merge_delta_e": merge, "l_weight": l_weight,
          "erode_frac": erode_frac}

    rows = []
    for name in tqdm(names):
        image = cv2.imread(str(images_dir / f"{name}.jpg"))
        if image is None:
            tqdm.write(f"Skipping {name}: unreadable")
            continue
        mask = predict_unet(model, image, device)
        try:
            raw = analyze_color(image, mask, normalize=False, **kw)
            norm = analyze_color(image, mask, normalize=True, **kw)
        except ValueError:
            tqdm.write(f"Skipping {name}: empty mask")
            continue

        row = {"image": name, "area_fraction": raw["lesion_area_fraction"]}
        row.update({f"{key}_raw": v for key, v in _features(raw).items()})
        row.update({f"{key}_norm": v for key, v in _features(norm).items()})
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No image could be analysed in split '{split}'")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize(rows: list[dict]) -> None:
    """
    Print mean/std of skin and contrast features, raw vs normalised.
    """
    def col(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows], dtype=float)

    print(f"\n{len(rows)} images analysed")
    for feat in ("skin_b", "skin_a", "skin_L", "delta_e"):
        r, n = col(f"{feat}_raw"), col(f"{feat}_norm")
        print(f"{feat:8s} raw  mean {np.nanmean(r):7.2f}  std {np.nanstd(r):6.2f}"
              f" | norm mean {np.nanmean(n):7.2f}  std {np.nanstd(n):6.2f}")
    for suffix in ("raw", "norm"):
        counts = np.bincount(col(f"n_colors_{suffix}").astype(int))
        dist = {i: int(c) for i, c in enumerate(counts) if c}
        print(f"n_colors ({suffix}): {dist}")