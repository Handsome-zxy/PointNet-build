"""Point-level binary metrics (no sklearn dependency)."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _prf(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def average_precision(labels: np.ndarray, probs: np.ndarray) -> float:
    """Average precision for the positive (building) class.

    Equal scores are one threshold.  The previous implementation accumulated
    per-row precision/recall after a stable sort, which made AP depend on the
    input row order when many scores were tied (the cache is class-contiguous,
    so this matters).  Prefer sklearn; the fallback groups ties explicitly.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    valid = labels >= 0
    y = labels[valid].astype(np.int8)
    p = probs[valid].astype(np.float64)
    if len(y) == 0:
        return float("nan")
    total_pos = int((y == 1).sum())
    if total_pos == 0:
        return float("nan")

    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y, p))
    except Exception:
        pass

    # Tie-grouped fallback: one precision/recall point per distinct score.
    order = np.argsort(-p, kind="mergesort")
    y = y[order]
    p = p[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(p) != 0) + 1, len(p)]
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tp += int((y[start:end] == 1).sum())
        fp += int((y[start:end] == 0).sum())
        recall = tp / total_pos
        precision = tp / max(tp + fp, 1)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return float(ap)


def threshold_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Metrics at one threshold without computing AP."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    valid = labels >= 0
    y = labels[valid].astype(np.int8)
    p = probs[valid].astype(np.float64)
    pred = (p >= float(threshold)).astype(np.int8)
    tp = float(np.sum((pred == 1) & (y == 1)))
    tn = float(np.sum((pred == 0) & (y == 0)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    building = _prf(tp, fp, fn)
    nonbuilding = _prf(tn, fn, fp)
    oa = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    balanced_acc = 0.5 * (building["recall"] + nonbuilding["recall"])
    macro_f1 = 0.5 * (building["f1"] + nonbuilding["f1"])
    miou = 0.5 * (building["iou"] + nonbuilding["iou"])
    return {
        "threshold": float(threshold),
        "oa": float(oa),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "miou": float(miou),
        "building_precision": float(building["precision"]),
        "building_recall": float(building["recall"]),
        "building_f1": float(building["f1"]),
        "building_iou": float(building["iou"]),
        "nonbuilding_precision": float(nonbuilding["precision"]),
        "nonbuilding_recall": float(nonbuilding["recall"]),
        "nonbuilding_f1": float(nonbuilding["f1"]),
        "nonbuilding_iou": float(nonbuilding["iou"]),
    }


def best_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    metric: str = "miou",
    thresholds: np.ndarray | None = None,
) -> tuple[float, Dict[str, float]]:
    """Return (threshold, metrics) maximizing `metric` (miou/macro_f1/building_f1)."""
    if thresholds is None:
        thresholds = np.arange(0.01, 0.991, 0.01)
    best_t, best_score, best_metrics = 0.5, -1.0, None
    for t in thresholds:
        m = threshold_metrics(labels, probs, float(t))
        score = float(m[metric])
        if score > best_score:
            best_t, best_score, best_metrics = float(t), score, m
    assert best_metrics is not None
    return best_t, best_metrics


def best_f1_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    t, m = best_threshold(labels, probs, metric="building_f1", thresholds=thresholds)
    return t, float(m["building_f1"])


def binary_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    valid = labels >= 0
    labels = labels[valid].astype(np.int8)
    probs = probs[valid].astype(np.float64)
    if len(labels) == 0:
        raise ValueError("no valid labels")

    pred = (probs >= threshold).astype(np.int8)
    tp = float(np.sum((pred == 1) & (labels == 1)))
    tn = float(np.sum((pred == 0) & (labels == 0)))
    fp = float(np.sum((pred == 1) & (labels == 0)))
    fn = float(np.sum((pred == 0) & (labels == 1)))

    building = _prf(tp, fp, fn)
    nonbuilding = _prf(tn, fn, fp)
    oa = (tp + tn) / (tp + tn + fp + fn)
    recall_b = building["recall"]
    recall_n = nonbuilding["recall"]
    balanced_acc = 0.5 * (recall_b + recall_n)
    macro_f1 = 0.5 * (building["f1"] + nonbuilding["f1"])
    miou = 0.5 * (building["iou"] + nonbuilding["iou"])

    return {
        "n_valid": int(len(labels)),
        "n_building": int((labels == 1).sum()),
        "n_nonbuilding": int((labels == 0).sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "oa": float(oa),
        "balanced_accuracy": float(balanced_acc),
        "building_precision": float(building["precision"]),
        "building_recall": float(recall_b),
        "building_f1": float(building["f1"]),
        "building_iou": float(building["iou"]),
        "nonbuilding_precision": float(nonbuilding["precision"]),
        "nonbuilding_recall": float(recall_n),
        "nonbuilding_f1": float(nonbuilding["f1"]),
        "nonbuilding_iou": float(nonbuilding["iou"]),
        "macro_f1": float(macro_f1),
        "miou": float(miou),
        "building_ap": average_precision(labels, probs),
    }


def weighted_ce_dice_from_probs(
    labels: np.ndarray,
    probs: np.ndarray,
    class_weights: np.ndarray,
    dice_weight: float = 1.0,
) -> Dict[str, float]:
    """Validation loss from averaged probabilities."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    valid = labels >= 0
    y = labels[valid].astype(np.int32)
    p = probs[valid].astype(np.float64)
    p_true = np.where(y == 1, p, 1.0 - p)
    ce = -np.mean(np.log(np.clip(p_true, 1e-7, 1.0)) * class_weights[y])
    inter = np.sum(p * (y == 1))
    denom = np.sum(p) + np.sum(y == 1)
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    return {"ce": float(ce), "dice": float(dice), "loss": float(ce + dice_weight * dice)}


def confusion_matrix(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    labels = np.asarray(labels)
    valid = labels >= 0
    labels = labels[valid].astype(np.int64)
    pred = (np.asarray(probs)[valid] >= threshold).astype(np.int64)
    flat = np.bincount(labels * 2 + pred, minlength=4)
    return flat.reshape(2, 2).astype(np.int64)
