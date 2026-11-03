"""
Entry point of the project: dispatches to the different modules.
"""
import argparse
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
IMAGES_DIR = DATA_DIR / "raw" / "ISIC2018" / "images"
MASKS_DIR = DATA_DIR / "raw" / "ISIC2018" / "masks"
SPLITS_PATH = DATA_DIR / "splits.json"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"
WEIGHTS_PATH = OUTPUT_DIR / "best_model.pth"
DOCS_DIR = Path(__file__).parent / "docs"

# Handlers

def cmd_segment(args: argparse.Namespace) -> None:
    from segmentation.segment import segment_image

    args.output.parent.mkdir(parents=True, exist_ok=True)
    segment_image(args.input, args.output)
    print(f"Mask saved to {args.output}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from segmentation.evaluate import evaluate_dataset

    s = evaluate_dataset(args.images, args.masks, args.limit, args.csv)
    print(f"Images evaluated: {s['n_images']}")
    print(f"Dice: {s['dice_mean']:.3f} ± {s['dice_std']:.3f} "
          f"(median {s['dice_median']:.3f})")
    print(f"IoU : {s['iou_mean']:.3f} ± {s['iou_std']:.3f} "
          f"(median {s['iou_median']:.3f})")
    print(f"Per-image scores saved to {args.csv}")


def cmd_split(args: argparse.Namespace) -> None:
    from segmentation.split import make_split

    split = make_split(args.images, args.out, seed=args.seed)
    print({name: len(items) for name, items in split.items()}, "->", args.out)


def cmd_preprocess(args: argparse.Namespace) -> None:
    from segmentation.preprocess import preprocess

    preprocess(args.images, args.masks, args.out, args.size)
    print(f"Processed dataset saved to {args.out}")


def cmd_train(args: argparse.Namespace) -> None:
    from segmentation.train import train

    train(args.processed, args.splits, args.out, args.epochs,
          args.batch_size, limit=args.limit)


def cmd_evaluate_model(args: argparse.Namespace) -> None:
    from segmentation.evaluate_model import evaluate_on_split

    s = evaluate_on_split(args.images, args.masks, args.splits,
                          args.weights, args.split, args.csv)
    print(f"Split '{args.split}': {s['n_images']} images")
    for method in ("otsu", "unet"):
        r = s[method]
        print(f"{method:5s} Dice {r['dice_mean']:.3f} ± {r['dice_std']:.3f} "
              f"(median {r['dice_median']:.3f}) | "
              f"IoU {r['iou_mean']:.3f} (median {r['iou_median']:.3f})")


def cmd_predict(args: argparse.Namespace) -> None:
    from segmentation.predict import predict_image

    predict_image(args.input, args.weights, args.output)
    print(f"U-Net mask saved to {args.output}")


def cmd_overlay(args: argparse.Namespace) -> None:
    from segmentation.visualize import save_overlay

    save_overlay(args.input, args.mask, args.output, args.scale)
    print(f"Overlay saved to {args.output}")


def cmd_figure(args: argparse.Namespace) -> None:
    import json

    from segmentation.figure import save_comparison

    names = args.names
    if not names:
        names = json.loads(args.splits.read_text())[args.split][:args.n]
    save_comparison(names, args.images, args.masks, args.weights, args.output)
    print(f"Figure saved to {args.output} ({len(names)} images)")


def cmd_color(args: argparse.Namespace) -> None:
    import cv2

    from analysis.color import (
        analyze_color,
        color_constancy,
        marker_mask,
        save_report,
    )

    image = cv2.imread(str(args.input))
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise FileNotFoundError("Cannot read the image or the mask")

    result = analyze_color(image, mask, k=args.k, normalize=args.normalize,
                           merge_delta_e=args.merge, l_weight=args.l_weight,
                           erode_frac=args.erode)

    # Show the image that was actually analysed
    shown = image
    if args.normalize:
        shown = color_constancy(image, exclude=marker_mask(image))
    save_report(shown, mask, result, args.out, args.input.stem)

    print(f"Number of distinct colours: {result['n_colors']}")
    print(f"ΔE vs skin: {result.get('delta_e_vs_skin')}")
    for c in result["dominant_colors"]:
        print(f"  {c['proportion']:.0%}  RGB {c['rgb']}")
    print(f"Report saved in {args.out}")


def cmd_color_batch(args: argparse.Namespace) -> None:
    from analysis.batch import run_batch, summarize

    rows = run_batch(args.images, args.splits, args.weights, args.csv,
                     args.split, args.limit, args.k, args.merge,
                     args.l_weight, args.erode)
    summarize(rows)
    print(f"CSV saved to {args.csv}")


def cmd_inspect(args: argparse.Namespace) -> None:
    from analysis.inspect_extremes import export_extremes

    export_extremes(args.csv, args.images, args.weights, args.out, args.n)
    print(f"Reports saved in {args.out}")


# Argument parser

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description="Mole segmentation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(handler=handler)
        return p

    # Otsu baseline
    p = add("segment", cmd_segment, "Segment a single image (Otsu baseline)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=OUTPUT_DIR / "mask.png")

    p = add("evaluate", cmd_evaluate, "Dice/IoU of the Otsu baseline on a dataset")
    p.add_argument("--images", type=Path, default=IMAGES_DIR)
    p.add_argument("--masks", type=Path, default=MASKS_DIR)
    p.add_argument("--limit", type=int, default=None,
                   help="Only evaluate the first N images")
    p.add_argument("--csv", type=Path, default=OUTPUT_DIR / "scores.csv")

    # Data preparation
    p = add("split", cmd_split, "Create a train/val/test split")
    p.add_argument("--images", type=Path, default=IMAGES_DIR)
    p.add_argument("--out", type=Path, default=SPLITS_PATH)
    p.add_argument("--seed", type=int, default=42)

    p = add("preprocess", cmd_preprocess, "Resize the dataset once")
    p.add_argument("--images", type=Path, default=IMAGES_DIR)
    p.add_argument("--masks", type=Path, default=MASKS_DIR)
    p.add_argument("--out", type=Path, default=PROCESSED_DIR)
    p.add_argument("--size", type=int, default=256)

    # U-Net
    p = add("train", cmd_train, "Train the U-Net")
    p.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    p.add_argument("--splits", type=Path, default=SPLITS_PATH)
    p.add_argument("--out", type=Path, default=OUTPUT_DIR)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None,
                   help="Use only N images per split (for quick tests)")

    p = add("evaluate-model", cmd_evaluate_model,
            "Compare Otsu and U-Net on a split")
    p.add_argument("--images", type=Path, default=IMAGES_DIR)
    p.add_argument("--masks", type=Path, default=MASKS_DIR)
    p.add_argument("--splits", type=Path, default=SPLITS_PATH)
    p.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    p.add_argument("--split", default="test")
    p.add_argument("--csv", type=Path, default=OUTPUT_DIR / "scores_compare.csv")

    p = add("predict", cmd_predict, "Segment one image with the trained U-Net")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_DIR / "mask_unet.png")

    p = add("overlay", cmd_overlay, "Draw a mask contour on an image")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--mask", type=Path, default=OUTPUT_DIR / "mask_unet.png")
    p.add_argument("--output", type=Path, default=OUTPUT_DIR / "overlay.jpg")
    p.add_argument("--scale", type=float, default=0.25,
                   help="Size of the saved image relative to the original")

    p = add("figure", cmd_figure, "Comparison figure: image | Otsu | U-Net")
    p.add_argument("--names", nargs="*", default=None,
                   help="Image names without extension "
                        "(default: first N images of the split)")
    p.add_argument("--split", default="test")
    p.add_argument("-n", type=int, default=3)
    p.add_argument("--images", type=Path, default=IMAGES_DIR)
    p.add_argument("--masks", type=Path, default=MASKS_DIR)
    p.add_argument("--splits", type=Path, default=SPLITS_PATH)
    p.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    p.add_argument("--output", type=Path,
                   default=DOCS_DIR / "segmentation_examples.png")

    # Colour analysis
    p = add("color", cmd_color, "Colour analysis of one lesion")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--mask", type=Path, default=OUTPUT_DIR / "mask_unet.png")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "color")
    p.add_argument("--k", type=int, default=8,
                   help="Initial number of k-means clusters (before merging)")
    p.add_argument("--merge", type=float, default=12.0,
                   help="Merge colours closer than this ΔE (0 = no merge)")
    p.add_argument("--l-weight", type=float, default=0.5,
                   help="Weight of lightness in clustering (1 = normal)")
    p.add_argument("--erode", type=float, default=0.03,
                   help="Mask erosion for the dominant colours (0 = off)")
    p.add_argument("--normalize", action="store_true",
                   help="Apply colour constancy before the analysis")

    p = add("color-batch", cmd_color_batch, "Colour analysis on a whole split")
    p.add_argument("--images", type=Path, default=IMAGES_DIR)
    p.add_argument("--splits", type=Path, default=SPLITS_PATH)
    p.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--merge", type=float, default=12.0)
    p.add_argument("--l-weight", type=float, default=0.5)
    p.add_argument("--erode", type=float, default=0.03,
                   help="Mask erosion for the dominant colours (0 = off)")
    p.add_argument("--csv", type=Path, default=OUTPUT_DIR / "color_batch.csv")

    p = add("inspect", cmd_inspect, "Visual reports for extreme cases")
    p.add_argument("--csv", type=Path, default=OUTPUT_DIR / "color_batch.csv")
    p.add_argument("--images", type=Path, default=IMAGES_DIR)
    p.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "extremes")
    p.add_argument("-n", type=int, default=5)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
