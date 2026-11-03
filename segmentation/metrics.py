"""Segmentation metrics."""
import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred > 0
    target = target > 0
    intersection = np.logical_and(pred, target).sum()
    return float((2 * intersection + eps) / (pred.sum() + target.sum() + eps))


def iou_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred > 0
    target = target > 0
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((intersection + eps) / (union + eps))