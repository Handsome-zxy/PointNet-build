"""Standalone evaluation and shared split-inference helpers."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from .config import load_config, resolve_path
from .data_prep import SUBSET_VAL
from .dataset import BuildingData, iter_eval_batches
from .logging_utils import TeeLogger, collect_env, write_json
from .metrics import binary_metrics, confusion_matrix
from .model import get_model, placeholder_inputs


def build_inference_ops(cfg: Dict[str, Any], batch_size: int, num_point: int, gpu: int = 0):
    with tf.device("/gpu:" + str(gpu)):
        pointclouds_pl, labels_pl = placeholder_inputs(batch_size, num_point)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        logits, end_points = get_model(
            pointclouds_pl,
            is_training_pl,
            use_input_tnet=bool(cfg["model"]["use_input_tnet"]),
            use_feature_tnet=bool(cfg["model"].get("use_feature_tnet", True)),
            dropout=float(cfg["model"].get("dropout", 0.5)),
            num_classes=int(cfg["model"].get("num_classes", 2)),
        )
        prob = tf.nn.softmax(logits, axis=-1)
        saver = tf.train.Saver()
    return {
        "pointclouds_pl": pointclouds_pl,
        "labels_pl": labels_pl,
        "is_training_pl": is_training_pl,
        "logits": logits,
        "prob": prob,
        "saver": saver,
        "end_points": end_points,
    }


def make_session(cfg: Dict[str, Any]) -> tf.Session:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    return tf.Session(config=config)


def predict_split(
    sess: tf.Session,
    ops: Dict[str, Any],
    data: BuildingData,
    set_name: str,
    batch_blocks: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (evaluation_indices, averaged_building_prob, hit_count)."""
    if set_name == "val":
        cache = data.small
        if cache.subset is None:
            raise ValueError("small cache has no subset array")
        eval_idx = np.flatnonzero(cache.subset == SUBSET_VAL).astype(np.int64)
    elif set_name == "test":
        cache = data.test
        eval_idx = np.arange(len(cache), dtype=np.int64)
    else:
        raise ValueError("set_name must be val or test")

    pos = np.full(len(cache), -1, dtype=np.int32)
    pos[eval_idx] = np.arange(len(eval_idx), dtype=np.int32)

    prob_sum = np.zeros(len(eval_idx), dtype=np.float64)
    count = np.zeros(len(eval_idx), dtype=np.int32)

    for feats, _labels, idxs, valid in iter_eval_batches(data, set_name, batch_blocks):
        probs = sess.run(
            ops["prob"],
            feed_dict={
                ops["pointclouds_pl"]: feats,
                ops["is_training_pl"]: False,
            },
        )
        for b in range(len(valid)):
            length = int(valid[b])
            if length == 0:
                continue
            ids = idxs[b, :length]
            local = pos[ids]
            if np.any(local < 0):
                raise RuntimeError("window point is not part of the evaluation split")
            prob_sum[local] += probs[b, :length, 1]
            count[local] += 1

    if np.any(count == 0):
        missing = int((count == 0).sum())
        raise RuntimeError(f"{set_name}: {missing} points were never predicted")

    return eval_idx, (prob_sum / count).astype(np.float32), count


def evaluate_split(
    cfg: Dict[str, Any],
    checkpoint: str,
    split: str,
    output_dir: Path,
    batch_size: int,
    threshold: float,
    gpu: int,
    write_las: bool = False,
) -> Dict[str, Any]:
    data = BuildingData(cfg)
    num_point = int(cfg["data"]["points_per_block"])
    log = TeeLogger(output_dir / "eval.log")
    log(f"evaluating split={split} checkpoint={checkpoint}")

    with tf.Graph().as_default():
        ops = build_inference_ops(cfg, batch_size, num_point, gpu=gpu)
        sess = make_session(cfg)
        try:
            ops["saver"].restore(sess, checkpoint)
            log("checkpoint restored")
            eval_idx, probs, count = predict_split(sess, ops, data, split, batch_size)
        finally:
            sess.close()

    cache = data.test if split == "test" else data.small
    labels = cache.label[eval_idx].astype(np.int32)
    metrics = binary_metrics(labels, probs, threshold=threshold)
    cm = confusion_matrix(labels, probs, threshold=threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "prob.npy", probs)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "point_rows.npy", cache.row[eval_idx])
    np.savetxt(
        output_dir / "confusion_matrix.csv",
        cm,
        fmt="%d",
        delimiter=",",
        header="pred_0,pred_1",
        comments="",
    )
    result = {
        "split": split,
        "checkpoint": str(checkpoint),
        "threshold": float(threshold),
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "n_eval_indices": int(len(eval_idx)),
        "n_predictions": int(count.sum()),
        "avg_hits_per_point": float(count.mean()),
    }
    write_json(output_dir / "metrics.json", result)
    write_json(output_dir / "env.json", collect_env(cfg, " ".join(sys.argv)))
    log(
        "OA=%.4f building_F1=%.4f building_IoU=%.4f nonbuilding_F1=%.4f mIoU=%.4f"
        % (
            metrics["oa"],
            metrics["building_f1"],
            metrics["building_iou"],
            metrics["nonbuilding_f1"],
            metrics["miou"],
        )
    )

    if write_las and split == "test":
        input_path = resolve_path(cfg, Path(cfg["data"]["raw_dir"]) / cfg["data"]["test"])
        output_path = output_dir / "test_pred.las"
        classification = (probs >= threshold).astype(np.uint8)
        try:
            from .las_utils import write_predictions_las

            write_predictions_las(input_path, output_path, classification, probs)
            log(f"wrote {output_path}")
        except Exception as exc:  # laspy may be unavailable
            log(f"WARNING: could not write prediction LAS: {exc!r}")

    log.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--write-las", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate_split(
        cfg,
        checkpoint=args.checkpoint,
        split=args.split,
        output_dir=resolve_path(cfg, args.output_dir),
        batch_size=args.batch_size,
        threshold=args.threshold,
        gpu=args.gpu,
        write_las=args.write_las,
    )


if __name__ == "__main__":
    main()
