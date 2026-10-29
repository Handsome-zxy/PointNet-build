"""Train one BuildingPointNet arm."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from .config import load_config, resolve_path, save_config
from .dataset import BuildingData, TrainSampler
from .evaluate import make_session, predict_split
from .logging_utils import (
    CsvLogger,
    TeeLogger,
    collect_env,
    hash_files,
    make_run_dir,
    write_json,
)
from .metrics import (
    best_threshold,
    binary_metrics,
    confusion_matrix,
    weighted_ce_dice_from_probs,
)
from .model import get_loss, get_model, placeholder_inputs


def _learning_rate(step: int, cfg: Dict[str, Any]) -> float:
    base = float(cfg["train"]["learning_rate"])
    end = float(cfg["train"]["end_learning_rate"])
    warmup = int(cfg["train"]["warmup_steps"])
    max_steps = int(cfg["train"]["max_steps"])
    if step < warmup:
        return base * float(step + 1) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, max_steps - warmup))
    progress = min(max(progress, 0.0), 1.0)
    return end + 0.5 * (base - end) * (1.0 + math.cos(math.pi * progress))


def build_graph(cfg: Dict[str, Any], batch_size: int, num_point: int, class_weights: np.ndarray):
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
    weights = tf.constant(class_weights, dtype=tf.float32)
    total, ce, dice, reg = get_loss(
        logits,
        labels_pl,
        end_points,
        weights,
        dice_weight=float(cfg["train"].get("dice_weight", 1.0)),
        reg_weight=float(cfg["model"].get("reg_weight", 0.001)),
    )
    prob = tf.nn.softmax(logits, axis=-1)

    global_step = tf.Variable(0, trainable=False, name="global_step")
    lr_pl = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(lr_pl)
    tvars = tf.trainable_variables()
    grads = tf.gradients(total, tvars)
    grads, _ = tf.clip_by_global_norm(grads, float(cfg["train"].get("grad_clip", 1.0)))
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    tf.summary.scalar("train/total_loss", total)
    tf.summary.scalar("train/ce_loss", ce)
    tf.summary.scalar("train/dice_loss", dice)
    tf.summary.scalar("train/reg_loss", reg)
    tf.summary.scalar("train/learning_rate", lr_pl)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=2)

    return {
        "pointclouds_pl": pointclouds_pl,
        "labels_pl": labels_pl,
        "is_training_pl": is_training_pl,
        "logits": logits,
        "prob": prob,
        "total": total,
        "ce": ce,
        "dice": dice,
        "reg": reg,
        "train_op": train_op,
        "global_step": global_step,
        "lr_pl": lr_pl,
        "merged": merged,
        "saver": saver,
        "end_points": end_points,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Evaluate the frozen test set after training. Development runs must omit this.",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Deprecated; the test set is not evaluated by default.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.pilot:
        cfg["train"]["max_steps"] = min(int(cfg["train"]["max_steps"]), 2000)
        cfg["train"]["val_every_steps"] = 500
        cfg["train"]["early_stop_patience"] = 4
        cfg["run"]["pilot"] = True
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = int(args.max_steps)
    if args.val_every is not None:
        cfg["train"]["val_every_steps"] = int(args.val_every)

    run_test = bool(args.run_test) and not bool(args.no_test)
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    run_root = resolve_path(cfg, args.run_root)
    arm = str(cfg["run"].get("arm", "base"))
    suffix = "pilot" if args.pilot else ""
    run_dir = make_run_dir(run_root, arm, seed, suffix)
    log = TeeLogger(run_dir / "train.log")
    log(f"run_dir={run_dir}")
    log(f"config={json.dumps(cfg, ensure_ascii=False)}")

    data = BuildingData(cfg)
    sampler = TrainSampler(data)
    batch_size = int(cfg["train"]["batch_size"])
    num_point = int(cfg["data"]["points_per_block"])
    max_steps = int(cfg["train"]["max_steps"])
    if len(sampler) < max_steps * batch_size:
        raise RuntimeError(
            f"schedule has {len(sampler)} blocks, need {max_steps * batch_size}; "
            "regenerate cache with a larger train.max_steps"
        )

    class_weights = np.asarray(data.manifest["class_weights"], dtype=np.float32)
    weight_mode = str(cfg["train"].get("class_weight_mode", "none")).lower()
    counts = np.asarray(data.manifest["train_class_counts"], dtype=np.float64)
    if weight_mode == "none":
        expected_weights = np.ones(2, dtype=np.float64)
    elif weight_mode == "inverse":
        expected_weights = float(counts.sum()) / (2.0 * counts)
        expected_weights = expected_weights / expected_weights.mean()
    else:
        raise ValueError(f"unknown class_weight_mode: {weight_mode}")
    if not np.allclose(class_weights.astype(np.float64), expected_weights, rtol=1e-6, atol=1e-6):
        raise RuntimeError(
            "cache manifest class_weights are stale for "
            f"class_weight_mode={weight_mode}: cache={class_weights.tolist()}, "
            f"expected={expected_weights.tolist()}. Rerun scripts/prepare_data.py."
        )
    manifest_mode = data.manifest.get("class_weight_mode")
    if manifest_mode is not None and str(manifest_mode).lower() != weight_mode:
        raise RuntimeError(
            f"cache class_weight_mode={manifest_mode!r} does not match config {weight_mode!r}"
        )
    z_ref = data.z_ref
    log(
        f"arm={arm} seed={seed} max_steps={max_steps} "
        f"class_weights={class_weights.tolist()} mode={weight_mode} z_ref={z_ref:.4f}"
    )

    save_config(cfg, run_dir / "config.json")
    write_json(run_dir / "data_manifest.json", data.manifest)
    write_json(run_dir / "env.json", collect_env(cfg, " ".join(sys.argv)))

    project_root = Path(cfg["_project_root"])
    hash_targets = [Path(cfg["_config_path"]), project_root / "cache" / "manifest.json"]
    hash_targets += sorted((project_root / "src" / "building_pointnet").glob("*.py"))
    write_json(run_dir / "code_hashes.json", hash_files(hash_targets))

    with tf.Graph().as_default():
        tf.set_random_seed(seed)
        ops = build_graph(cfg, batch_size, num_point, class_weights)
        n_params = int(sum(int(np.prod(v.get_shape().as_list())) for v in tf.trainable_variables()))
        log(f"trainable parameters: {n_params:,}")
        sess = make_session(cfg)
        writer = tf.summary.FileWriter(str(run_dir / "tb"), sess.graph)
        sess.run(tf.global_variables_initializer())

        fields = [
            "step", "epoch", "lr", "train_loss", "train_ce", "train_dice",
            "val_loss", "val_ce", "val_dice", "val_oa", "val_balanced_accuracy",
            "val_building_precision", "val_building_recall", "val_building_f1",
            "val_building_iou", "val_nonbuilding_precision", "val_nonbuilding_recall",
            "val_nonbuilding_f1", "val_nonbuilding_iou", "val_macro_f1", "val_miou",
            "val_building_ap", "val_best_threshold", "val_building_f1_tuned",
            "val_building_iou_tuned", "val_miou_tuned", "val_nonbuilding_f1_tuned",
            "val_n_valid", "best_building_iou", "best_val_miou", "time_sec",
        ]
        csv_log = CsvLogger(run_dir / "metrics.csv", fields)

        val_every = int(cfg["train"]["val_every_steps"])
        patience = int(cfg["train"]["early_stop_patience"])
        threshold = float(cfg["eval"].get("threshold", 0.5))
        dice_weight = float(cfg["train"].get("dice_weight", 1.0))
        best_score = -1.0
        best_step = -1
        best_tuned_threshold = None
        best_use_tuned = False
        best_val_metrics_05: Dict[str, Any] | None = None
        no_improve = 0
        started = time.time()
        loss_sum = ce_sum = dice_sum = 0.0
        loss_count = 0
        best_val_metrics: Dict[str, Any] | None = None
        stopped_early = False

        try:
            for step in range(max_steps):
                feats = []
                labels = []
                for b in range(batch_size):
                    f, l, _wid = sampler.sample(step * batch_size + b)
                    feats.append(f)
                    labels.append(l)
                feed = {
                    ops["pointclouds_pl"]: np.stack(feats),
                    ops["labels_pl"]: np.stack(labels),
                    ops["is_training_pl"]: True,
                    ops["lr_pl"]: _learning_rate(step, cfg),
                }
                summary, _, loss_val, ce_val, dice_val, gstep = sess.run(
                    [
                        ops["merged"],
                        ops["train_op"],
                        ops["total"],
                        ops["ce"],
                        ops["dice"],
                        ops["global_step"],
                    ],
                    feed_dict=feed,
                )
                writer.add_summary(summary, gstep)
                loss_sum += float(loss_val)
                ce_sum += float(ce_val)
                dice_sum += float(dice_val)
                loss_count += 1

                if step % 20 == 0:
                    log(
                        f"step {step:6d}/{max_steps} loss={loss_val:.4f} "
                        f"ce={ce_val:.4f} dice={dice_val:.4f} lr={_learning_rate(step, cfg):.2e}"
                    )

                if (step + 1) % val_every == 0 or step == max_steps - 1:
                    eval_idx, probs, count = predict_split(
                        sess, ops, data, "val", batch_size
                    )
                    labels_val = data.small.label[eval_idx].astype(np.int32)
                    val_metrics = binary_metrics(labels_val, probs, threshold=threshold)
                    tuned_threshold, tuned_metrics = best_threshold(
                        labels_val, probs, metric="miou"
                    )
                    tuned_f1 = float(tuned_metrics["building_f1"])
                    tuned_miou = float(tuned_metrics["miou"])
                    val_loss = weighted_ce_dice_from_probs(
                        labels_val, probs, class_weights, dice_weight=dice_weight
                    )
                    avg_loss = loss_sum / max(1, loss_count)
                    avg_ce = ce_sum / max(1, loss_count)
                    avg_dice = dice_sum / max(1, loss_count)
                    loss_sum = ce_sum = dice_sum = 0.0
                    loss_count = 0

                    elapsed = time.time() - started
                    log(
                        f"VAL step {step} epoch={(step + 1) / int(cfg['train']['steps_per_epoch']):.2f} "
                        f"val_loss={val_loss['loss']:.4f} OA={val_metrics['oa']:.4f} "
                        f"bF1={val_metrics['building_f1']:.4f} bIoU={val_metrics['building_iou']:.4f} "
                        f"nbF1={val_metrics['nonbuilding_f1']:.4f} mIoU={val_metrics['miou']:.4f} "
                        f"AP={val_metrics['building_ap']:.4f} hits={count.mean():.2f} "
                        f"tuned_t={tuned_threshold:.2f} tuned_mIoU={tuned_miou:.4f} "
                        f"tuned_bF1={tuned_f1:.4f} tuned_nbF1={tuned_metrics['nonbuilding_f1']:.4f} "
                        f"tuned_bIoU={tuned_metrics['building_iou']:.4f} time={elapsed:.1f}s"
                    )

                    improved = tuned_miou > best_score + 1e-6
                    if improved:
                        best_score = tuned_miou
                        best_step = step
                        no_improve = 0
                        best_val_metrics = dict(tuned_metrics)
                        best_val_metrics["threshold"] = tuned_threshold
                        best_val_metrics["selection_metric"] = "miou"
                        best_val_metrics_05 = dict(val_metrics)
                        best_use_tuned = tuned_miou > val_metrics["miou"] + 1e-6
                        best_tuned_threshold = tuned_threshold if best_use_tuned else None
                        ops["saver"].save(sess, str(run_dir / "best_model.ckpt"))
                    else:
                        no_improve += 1

                    csv_log.write(
                        {
                            "step": step,
                            "epoch": (step + 1) / int(cfg["train"]["steps_per_epoch"]),
                            "lr": _learning_rate(step, cfg),
                            "train_loss": avg_loss,
                            "train_ce": avg_ce,
                            "train_dice": avg_dice,
                            "val_loss": val_loss["loss"],
                            "val_ce": val_loss["ce"],
                            "val_dice": val_loss["dice"],
                            "val_oa": val_metrics["oa"],
                            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                            "val_building_precision": val_metrics["building_precision"],
                            "val_building_recall": val_metrics["building_recall"],
                            "val_building_f1": val_metrics["building_f1"],
                            "val_building_iou": val_metrics["building_iou"],
                            "val_nonbuilding_precision": val_metrics["nonbuilding_precision"],
                            "val_nonbuilding_recall": val_metrics["nonbuilding_recall"],
                            "val_nonbuilding_f1": val_metrics["nonbuilding_f1"],
                            "val_nonbuilding_iou": val_metrics["nonbuilding_iou"],
                            "val_macro_f1": val_metrics["macro_f1"],
                            "val_miou": val_metrics["miou"],
                            "val_building_ap": val_metrics["building_ap"],
                            "val_best_threshold": tuned_threshold,
                            "val_building_f1_tuned": tuned_f1,
                            "val_building_iou_tuned": tuned_metrics["building_iou"],
                            "val_miou_tuned": tuned_miou,
                            "val_nonbuilding_f1_tuned": tuned_metrics["nonbuilding_f1"],
                            "val_n_valid": val_metrics["n_valid"],
                            "best_building_iou": (
                                best_val_metrics["building_iou"] if best_val_metrics else 0.0
                            ),
                            "best_val_miou": best_score,
                            "time_sec": elapsed,
                        }
                    )

                    if patience > 0 and no_improve >= patience:
                        log(f"early stopping at step {step} (no improvement for {no_improve} validations)")
                        stopped_early = True
                        break

            ops["saver"].save(sess, str(run_dir / "last_model.ckpt"))
            log("saved last checkpoint")

            test_metrics = None
            test_metrics_05 = None
            threshold_used = float(threshold)
            test_metrics_path = None
            if run_test:
                best_path = run_dir / "best_model.ckpt"
                if tf.train.checkpoint_exists(str(best_path)):
                    ops["saver"].restore(sess, str(best_path))
                    log(f"restored best checkpoint at step {best_step}")
                else:
                    log("WARNING: no best checkpoint found; evaluating the last state")
                eval_idx, probs, count = predict_split(sess, ops, data, "test", batch_size)
                labels_test = data.test.label[eval_idx].astype(np.int32)
                test_metrics_05 = binary_metrics(labels_test, probs, threshold=threshold)
                threshold_used = (
                    float(best_tuned_threshold)
                    if best_use_tuned and best_tuned_threshold is not None
                    else float(threshold)
                )
                test_metrics = binary_metrics(labels_test, probs, threshold=threshold_used)
                cm = confusion_matrix(labels_test, probs, threshold=threshold_used)
                np.save(run_dir / "test_prob.npy", probs)
                np.save(run_dir / "test_labels.npy", labels_test)
                np.save(run_dir / "test_rows.npy", data.test.row[eval_idx])
                np.savetxt(
                    run_dir / "test_confusion_matrix.csv",
                    cm,
                    fmt="%d",
                    delimiter=",",
                    header="pred_0,pred_1",
                    comments="",
                )
                test_metrics_path = run_dir / "test_metrics.json"
                write_json(
                    test_metrics_path,
                    {
                        "metrics": test_metrics,
                        "metrics_at_0.5": test_metrics_05,
                        "confusion_matrix": cm.tolist(),
                        "threshold_used": threshold_used,
                        "threshold_0.5": threshold,
                        "threshold_tuned": best_tuned_threshold,
                        "use_tuned_threshold": bool(best_use_tuned),
                        "n_predictions": int(count.sum()),
                        "avg_hits_per_point": float(count.mean()),
                        "checkpoint": str(run_dir / "best_model.ckpt"),
                        "best_val_step": best_step,
                    },
                )
                log(
                    f"TEST thr={threshold_used:.2f} OA={test_metrics['oa']:.4f} "
                    f"bF1={test_metrics['building_f1']:.4f} bIoU={test_metrics['building_iou']:.4f} "
                    f"nbF1={test_metrics['nonbuilding_f1']:.4f} mIoU={test_metrics['miou']:.4f} "
                    f"AP={test_metrics['building_ap']:.4f} | @0.5 bF1={test_metrics_05['building_f1']:.4f} "
                    f"bIoU={test_metrics_05['building_iou']:.4f}"
                )

                try:
                    from .las_utils import write_predictions_las

                    input_path = resolve_path(
                        cfg, Path(cfg["data"]["raw_dir"]) / cfg["data"]["test"]
                    )
                    rows = data.test.row[eval_idx]
                    n_test = len(data.test)
                    full_prob = np.zeros(n_test, dtype=np.float32)
                    full_cls = np.zeros(n_test, dtype=np.uint8)
                    full_prob[rows] = probs
                    full_cls[rows] = (probs >= threshold_used).astype(np.uint8)
                    write_predictions_las(
                        input_path, run_dir / "test_pred.las", full_cls, full_prob
                    )
                    log("wrote test_pred.las")
                except Exception as exc:
                    log(f"WARNING: could not write prediction LAS: {exc!r}")

            summary = {
                "arm": arm,
                "seed": seed,
                "max_steps": max_steps,
                "best_step": best_step,
                "best_val_metrics": best_val_metrics,
                "best_val_metrics_at_0.5": best_val_metrics_05,
                "test_metrics": test_metrics,
                "test_metrics_at_0.5": test_metrics_05,
                "threshold_used": threshold_used,
                "threshold_tuned": best_tuned_threshold,
                "use_tuned_threshold": bool(best_use_tuned),
                "test_metrics_path": str(test_metrics_path) if test_metrics_path else None,
                "n_parameters": n_params,
                "class_weights": class_weights.tolist(),
                "z_ref": z_ref,
                "batch_size": batch_size,
                "num_point": num_point,
                "stopped_early": stopped_early,
                "test_evaluated": bool(run_test),
                "runtime_sec": time.time() - started,
                "run_dir": str(run_dir),
            }
            write_json(run_dir / "summary.json", summary)
        finally:
            csv_log.close()
            writer.close()
            sess.close()
            log.close()


if __name__ == "__main__":
    main()
