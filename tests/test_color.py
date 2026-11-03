"""Tests for the colour analysis, on small synthetic images."""
import cv2
import numpy as np
import pytest

from analysis.color import (
    analyze_color,
    color_constancy,
    marker_mask,
    merge_close_colors,
)

SKIN_BGR = (140, 160, 200)
BROWN_BGR = (30, 50, 90)
TAN_BGR = (90, 130, 180)


def _lesion_image(two_tone: bool = False):
    """Skin-coloured 200x200 image with a round brown lesion, and its mask."""
    image = np.full((200, 200, 3), SKIN_BGR, np.uint8)
    mask = np.zeros((200, 200), np.uint8)
    cv2.circle(image, (100, 100), 40, BROWN_BGR, -1)
    cv2.circle(mask, (100, 100), 40, 255, -1)
    if two_tone:
        right_half = mask.copy()
        right_half[:, :100] = 0
        image[right_half > 0] = TAN_BGR
    return image, mask


# --- merge_close_colors ---------------------------------------------------
def test_close_colors_are_merged_with_weighted_mean():
    centers = np.array([[50.0, 10.0, 10.0], [52.0, 10.0, 10.0]])
    props = np.array([0.75, 0.25])
    merged, merged_props = merge_close_colors(centers, props, threshold=5,
                                              l_weight=1.0)
    assert len(merged) == 1
    assert abs(merged_props[0] - 1.0) < 1e-9
    assert abs(merged[0][0] - 50.5) < 1e-9  # 0.75 * 50 + 0.25 * 52


def test_distant_colors_are_kept():
    centers = np.array([[20.0, 5.0, 5.0], [80.0, 30.0, -20.0]])
    merged, props = merge_close_colors(centers, np.array([0.4, 0.6]),
                                       threshold=12)
    assert len(merged) == 2
    assert abs(props.sum() - 1.0) < 1e-9


def test_lightness_weight_controls_merging_of_same_hue():
    centers = np.array([[20.0, 10.0, 10.0], [60.0, 10.0, 10.0]])
    props = np.array([0.5, 0.5])
    full, _ = merge_close_colors(centers, props, threshold=12, l_weight=1.0)
    light, _ = merge_close_colors(centers, props, threshold=12, l_weight=0.2)
    assert len(full) == 2   # distance 40 > 12
    assert len(light) == 1  # distance 40 * 0.2 = 8 < 12


# --- marker_mask ----------------------------------------------------------
def test_marker_mask_detects_blue_ink_but_not_skin():
    image = np.full((200, 200, 3), SKIN_BGR, np.uint8)
    image[50:55, 20:180] = (255, 0, 0)  # saturated blue stroke (BGR)
    ink = marker_mask(image)
    assert ink[52, 100]
    assert not ink[150, 100]


def test_marker_mask_ignores_brown_lesion():
    image, _ = _lesion_image()
    assert not marker_mask(image).any()


# --- color_constancy ------------------------------------------------------
def test_color_constancy_removes_a_uniform_cast():
    cast = np.full((100, 100, 3), (166, 128, 102), np.uint8)  # blue-ish grey
    out = color_constancy(cast)
    mean = out.reshape(-1, 3).mean(axis=0)
    assert np.ptp(mean) < 3  # channels are now (almost) equal


def test_color_constancy_can_ignore_ink_pixels():
    image = np.full((100, 100, 3), 128, np.uint8)
    image[:, :20] = (255, 0, 0)  # blue ink block
    ink = np.zeros((100, 100), bool)
    ink[:, :20] = True

    with_exclusion = color_constancy(image, exclude=ink)[50, 80]
    without = color_constancy(image)[50, 80]
    assert int(with_exclusion.max()) - int(with_exclusion.min()) <= 1
    assert int(without.max()) - int(without.min()) > 20


# --- analyze_color --------------------------------------------------------
def test_single_color_lesion():
    image, mask = _lesion_image()
    result = analyze_color(image, mask)
    assert result["n_colors"] == 1
    assert abs(result["lesion_area_fraction"] - 0.1257) < 0.01
    assert result["delta_e_vs_skin"] > 20
    assert result["skin_lab_mean"][0] > result["lesion"]["lab_mean"][0]


def test_two_tone_lesion_gives_two_colors():
    image, mask = _lesion_image(two_tone=True)
    result = analyze_color(image, mask)
    assert result["n_colors"] == 2
    smallest = min(c["proportion"] for c in result["dominant_colors"])
    assert abs(smallest - 0.5) < 0.1


def test_dominant_color_proportions_sum_to_one():
    image, mask = _lesion_image(two_tone=True)
    result = analyze_color(image, mask)
    total = sum(c["proportion"] for c in result["dominant_colors"])
    assert abs(total - 1.0) < 0.01


def test_normalize_and_erosion_options_run():
    image, mask = _lesion_image()
    result = analyze_color(image, mask, normalize=True, erode_frac=0.03)
    assert result["n_colors"] >= 1


def test_empty_mask_raises():
    image, _ = _lesion_image()
    with pytest.raises(ValueError):
        analyze_color(image, np.zeros((200, 200), np.uint8))
