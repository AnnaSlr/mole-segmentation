"""
Export visual reports for the extreme cases of a colour batch CSV.
Reports use the raw (non-normalised) image, to show what is really measured.
"""
import csv
from pathlib import Path

import cv2
import torch

from analysis.color import analyze_color, save_report
from segmentation.evaluate_model import load_model, predict_unet

# (output folder, CSV column, sort desc)
CRITERIA = [
    ("skin_b_lowest",  "skin_b_raw",    False),  # bluest
    ("skin_b_highest", "skin_b_raw",    True),   # yellowest
    ("most_colors",    "n_colors_raw",  True),
    ("single_color",   "n_colors_raw",  False),
    ("largest_area",   "area_fraction", True),
    ("smallest_area",  "area_fraction", False),
]


def export_extremes(csv_path: Path, images_dir: Path, weights: Path,
                    out_dir: Path, n: int = 5, k: int = 8,
                    merge: float = 12.0, l_weight: float = 0.5,
                    erode_frac: float = 0.03) -> None:
    """
    Write a visual report for the `n` most extreme images per criterion.
    The analysis parameters should match those used to build the CSV.
    """
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)

    for folder, column, descending in CRITERIA:
        ranked = sorted(rows, key=lambda r: float(r[column]),
                        reverse=descending)[:n]
        for rank, r in enumerate(ranked, 1):
            name = r["image"]
            image = cv2.imread(str(images_dir / f"{name}.jpg"))
            if image is None:
                continue
            mask = predict_unet(model, image, device)
            result = analyze_color(image, mask, k=k, normalize=False,
                                   merge_delta_e=merge, l_weight=l_weight,
                                   erode_frac=erode_frac)
            label = f"{rank:02d}_{float(r[column]):.2f}_{name}"
            save_report(image, mask, result, out_dir / folder, label)
            print(f"{folder}: {name} ({column}={float(r[column]):.2f})")