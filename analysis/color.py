"""
Colour analysis of a lesion, given an image and its binary mask.
"""

import json
from pathlib import Path

import cv2
import numpy as np

MIN_REGION = 200          # minimum number of pixels for a usable region
MAX_KMEANS_PIXELS = 5000  # k-means runs on at most this many pixels
DARK_THRESHOLD = 30       # max(B, G, R) below this = black corner / border


# Helpers

def _ellipse(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def marker_mask(image_bgr: np.ndarray, s_min: int = 90) -> np.ndarray:
    """
    Pixels that look like blue/green/violet pen ink (True = ink).
    The HSV thresholds are empirical: they cover most, not all, of a stroke.
    """

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)  # H: 0-180, S/V: 0-255
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    ink = (h > 85) & (h < 150) & (s > s_min) & (v > 40)
    
    # Remove isolated pixels, then grow to cover the soft edges of strokes
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN,
                           np.ones((3, 3), np.uint8))
    r = max(2, int(0.006 * max(image_bgr.shape[:2])))
    return cv2.dilate(ink, _ellipse(2 * r + 1)) > 0


def color_constancy(image_bgr: np.ndarray, p: int = 6,
                    exclude: np.ndarray | None = None) -> np.ndarray:
    """
    Shades of Gray: estimate the illuminant and neutralise it.
    Black corners and `exclude` pixels (pen ink) are ignored in the estimate.
    """
    img = image_bgr.astype(np.float32)
    flat = img.reshape(-1, 3)
    keep = flat.max(axis=1) > DARK_THRESHOLD
    if exclude is not None:
        keep &= ~exclude.reshape(-1)
    flat = flat[keep]
    if len(flat) < 100:
        return image_bgr
    illum = (flat ** p).mean(axis=0) ** (1.0 / p)
    illum /= np.linalg.norm(illum)
    out = img / (illum * np.sqrt(3))
    return np.clip(out, 0, 255).astype(np.uint8)


def merge_close_colors(centers: np.ndarray, props: np.ndarray,
                       threshold: float,
                       l_weight: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Repeatedly merge the two closest colours while their distance is below
    `threshold`.
    Lightness is down-weighted so that two browns of different brightness
    merge more easily than two different hues. The merged colour is the
    proportion-weighted mean.
    """

    w = np.array([l_weight, 1.0, 1.0])
    centers = [c.astype(np.float64) for c in centers]
    props = [float(p) for p in props]
    while len(centers) > 1:
        best, best_d = None, threshold
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = np.linalg.norm((centers[i] - centers[j]) * w)
                if d < best_d:
                    best, best_d = (i, j), d
        if best is None:
            break
        i, j = best
        total = props[i] + props[j]
        centers[i] = (centers[i] * props[i] + centers[j] * props[j]) / total
        props[i] = total
        del centers[j], props[j]
    return np.array(centers), np.array(props)


def _region_stats(lab: np.ndarray, hsv: np.ndarray, region: np.ndarray) -> dict:
    lab_px = lab[region].astype(np.float64)
    hsv_px = hsv[region].astype(np.float64)
    mean = lab_px.mean(axis=0)
    return {
        "lab_mean": mean.round(2).tolist(),
        "lab_std": lab_px.std(axis=0).round(2).tolist(),
        # Hue as an angle in Lab (a, b): averages correctly, unlike HSV hue
        "chroma": round(float(np.hypot(mean[1], mean[2])), 2),
        "hue_angle_deg": round(
            float(np.degrees(np.arctan2(mean[2], mean[1])) % 360), 1),
        "saturation_mean": round(float(hsv_px[:, 1].mean()), 3),
        "value_mean": round(float(hsv_px[:, 2].mean()), 3),
    }


def _pick(region: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """
    Use `region` unless it is almost empty.
    """
    return region if region.sum() >= MIN_REGION else fallback


def _skin_ring(lesion: np.ndarray) -> np.ndarray:
    """
    Ring of skin around the lesion, separated from it by a small gap.
    """
    radius = max(5, int(0.5 * np.sqrt(lesion.sum() / np.pi)))
    gap = max(3, radius // 5)
    lesion_u8 = lesion.astype(np.uint8)
    outer = cv2.dilate(lesion_u8, _ellipse(2 * (gap + radius) + 1)) > 0
    inner = cv2.dilate(lesion_u8, _ellipse(2 * gap + 1)) > 0
    return outer & ~inner


def _dominant_colors(pixels: np.ndarray, k: int, merge_delta_e: float,
                     l_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """
    K-means in Lab, then merge perceptually close clusters.
    Returns (centres in Lab, proportions), sorted by decreasing proportion.
    """
    if len(pixels) > MAX_KMEANS_PIXELS:
        rng = np.random.default_rng(0)
        pixels = pixels[rng.choice(len(pixels), MAX_KMEANS_PIXELS,
                                   replace=False)]
    k = min(k, len(pixels))

    # Down-weight L so clusters follow hue differences more than brightness
    weighted = pixels.copy()
    weighted[:, 0] *= l_weight
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, _ = cv2.kmeans(weighted, k, None, criteria, 5,
                              cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    # Centres and proportions in the real (unweighted) Lab space
    used = [i for i in range(k) if (labels == i).any()]
    centers = np.array([pixels[labels == i].mean(axis=0) for i in used])
    props = np.array([(labels == i).mean() for i in used])

    centers, props = merge_close_colors(centers, props, merge_delta_e, l_weight)
    order = np.argsort(-props)
    return centers[order], props[order]


# Analysis

def analyze_color(image_bgr: np.ndarray, mask: np.ndarray,
                  k: int = 8, max_side: int = 512,
                  normalize: bool = False,
                  merge_delta_e: float = 12.0,
                  l_weight: float = 0.5,
                  erode_frac: float = 0.0,
                  exclude_ink: bool = True) -> dict:
    """
    Colour descriptors of the lesion defined by `mask` (non-zero = lesion).

    Args:
        image_bgr: BGR uint8 image
        mask: binary mask, same size as the image
        k: initial number of k-means clusters, before merging
        max_side: the image is reduced so its longest side is at most this
        normalize: apply Shades of Gray colour constancy first
        merge_delta_e: merge dominant colours closer than this distance
        l_weight: weight of lightness in clustering and merging (1 = normal)
        erode_frac: erosion of the mask for the dominant colours, as a
            fraction of sqrt(area). 0 = off. (The CLI uses 0.03 by default.)
        exclude_ink: ignore pen-ink pixels in the measurements

    Raises:
        ValueError: if the mask is (almost) empty.
    """
    # Work on a reduced image: faster and enough for colour statistics
    scale = max_side / max(mask.shape[:2])
    if scale < 1:
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    lesion = mask > 0
    if lesion.sum() < 50:
        raise ValueError("Mask is (almost) empty: nothing to analyse")

    # Pen ink and black corners, detected on the ORIGINAL colours
    ink = marker_mask(image_bgr) if exclude_ink else np.zeros(mask.shape, bool)
    dark = image_bgr.max(axis=2) <= DARK_THRESHOLD

    if normalize:
        image_bgr = color_constancy(image_bgr, exclude=ink)

    # float32 input -> true Lab (L: 0-100) and HSV (S/V: 0-1)
    img = image_bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Lesion pixels without ink (fallback to the full lesion if too few left)
    lesion_valid = _pick(lesion & ~ink, lesion)

    result = {
        "lesion_area_fraction": round(float(lesion.mean()), 4),
        "ink_fraction_image": round(float(ink.mean()), 4),
        "lesion": _region_stats(lab, hsv, lesion_valid),
    }

    # Centre vs periphery
    dist = cv2.distanceTransform(lesion.astype(np.uint8), cv2.DIST_L2, 5)
    core = lesion_valid & (dist > 0.5 * dist.max())
    periphery = lesion_valid & ~core
    if core.sum() > 0 and periphery.sum() > 0:
        result["core"] = _region_stats(lab, hsv, core)
        result["periphery"] = _region_stats(lab, hsv, periphery)

    # Surrounding skin and contrast (ink and black corners excluded)
    ring = _skin_ring(lesion)
    if ring.sum() > 0:
        result["ink_fraction_ring"] = round(
            float((ring & ink).sum() / ring.sum()), 4)
    ring = ring & ~ink & ~dark
    if ring.sum() > 50:
        skin_lab = lab[ring].astype(np.float64).mean(axis=0)
        lesion_lab = lab[lesion_valid].astype(np.float64).mean(axis=0)
        result["skin_lab_mean"] = skin_lab.round(2).tolist()
        result["delta_e_vs_skin"] = round(
            float(np.linalg.norm(lesion_lab - skin_lab)), 2)

    # Region used for the dominant colours (optionally eroded)
    color_region = lesion_valid
    if erode_frac > 0:
        erode_px = max(2, int(erode_frac * np.sqrt(lesion.sum())))
        inner = (cv2.erode(lesion.astype(np.uint8),
                           _ellipse(2 * erode_px + 1)) > 0) & ~ink
        color_region = _pick(inner, lesion_valid)

    # Dominant colours
    centers, props = _dominant_colors(lab[color_region], k,
                                      merge_delta_e, l_weight)
    centers_bgr = cv2.cvtColor(centers.astype(np.float32).reshape(1, -1, 3),
                               cv2.COLOR_LAB2BGR)
    centers_rgb = (np.clip(centers_bgr[0], 0, 1) * 255).round().astype(int)
    centers_rgb = centers_rgb[:, ::-1]  # BGR -> RGB

    result["n_colors"] = len(props)
    result["dominant_colors"] = [
        {"proportion": round(float(props[i]), 3),
         "lab": centers[i].round(2).tolist(),
         "rgb": centers_rgb[i].tolist()}
        for i in range(len(props))
    ]
    return result



# Report

def _rgb_to_bgr(color: dict) -> tuple[int, int, int]:
    r, g, b = (int(v) for v in color["rgb"])
    return b, g, r


def save_report(image_bgr: np.ndarray, mask: np.ndarray, result: dict,
                out_dir: Path, name: str) -> None:
    """
    Save the JSON results and a visual: contour + colour bar + legend.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_color.json").write_text(json.dumps(result, indent=2))

    width = 600

    # Image with lesion contour
    vis = image_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0),
                     max(2, vis.shape[1] // 300))
    vis = cv2.resize(vis, (width, int(width * vis.shape[0] / vis.shape[1])))

    colors = result["dominant_colors"]

    # Proportional bar with the % written in each segment
    bar = np.full((60, width, 3), 255, np.uint8)
    x = 0
    for i, c in enumerate(colors):
        is_last = i == len(colors) - 1
        seg = width - x if is_last else max(1, int(width * c["proportion"]))
        bgr = _rgb_to_bgr(c)
        bar[:, x:x + seg] = bgr
        if seg > 50:
            brightness = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
            txt_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)
            cv2.putText(bar, f"{c['proportion']:.0%}", (x + 8, 37),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, txt_color, 2,
                        cv2.LINE_AA)
        x += seg

    # Legend: one row per colour
    row_h = 36
    legend = np.full((row_h * len(colors) + 10, width, 3), 255, np.uint8)
    for i, c in enumerate(colors):
        y = 5 + i * row_h
        cv2.rectangle(legend, (10, y), (46, y + 26), _rgb_to_bgr(c), -1)
        cv2.rectangle(legend, (10, y), (46, y + 26), (0, 0, 0), 1)
        r, g, b = c["rgb"]
        cv2.putText(legend, f"{c['proportion']:.1%}   RGB ({r}, {g}, {b})",
                    (60, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_dir / f"{name}_color.png"),
                np.vstack([vis, bar, legend]))