"""Tests for the segmentation metrics."""
import numpy as np

from segmentation.metrics import dice_score, iou_score


def _square(top: int, left: int, size: int = 2, shape=(10, 10)) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    mask[top:top + size, left:left + size] = 255
    return mask


def test_identical_masks_score_one():
    m = _square(2, 2)
    assert dice_score(m, m) > 0.999999
    assert iou_score(m, m) > 0.999999


def test_disjoint_masks_score_zero():
    a, b = _square(0, 0), _square(6, 6)
    assert dice_score(a, b) < 1e-6
    assert iou_score(a, b) < 1e-6


def test_partial_overlap_known_values():
    # Two 2x2 squares sharing 2 pixels: |P| = |G| = 4, inter = 2, union = 6
    a, b = _square(2, 2), _square(2, 3)
    assert abs(dice_score(a, b) - 0.5) < 1e-6
    assert abs(iou_score(a, b) - 2 / 6) < 1e-6


def test_both_empty_masks_score_one():
    empty = np.zeros((10, 10), np.uint8)
    assert dice_score(empty, empty) > 0.999999
    assert iou_score(empty, empty) > 0.999999


def test_mask_encoding_does_not_matter():
    a, b = _square(2, 2), _square(2, 3)
    assert dice_score(a, b) == dice_score(a // 255, b // 255)


def test_dice_is_never_below_iou():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.integers(0, 2, (16, 16))
        b = rng.integers(0, 2, (16, 16))
        assert dice_score(a, b) >= iou_score(a, b) - 1e-9
