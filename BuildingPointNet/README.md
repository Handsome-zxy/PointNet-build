# BuildingPointNet

LAS 点云逐点二分类（建筑 / 非建筑），基于 TensorFlow 2.21 `compat.v1`。

完整设计见 [DESIGN.md](DESIGN.md)。

## 数据

原始数据只读，位于：

```
../Data/Train_Data/train_data/
```

- `non_ground_points_build_small.las` → 建筑（1）
- `non_ground_points_noise_small.las` → 非建筑（0）
- `non_ground_points_big_validation.las` → 固定测试集；标签由 `Original cloud index` 与
  `non_ground_points_build_big.las` / `non_ground_points_noise_big.las` 的坐标追溯得到
  （0 = 建筑，1 = 非建筑）。

固定空间留出与 5 m 缓冲区见 `configs/base.json` 的 `data.val_box` / `data.buffer_margin`。

## 环境

复用 `../PointNet/.venv`。运行前：

```bash
source ../PointNet/env.sh
source ../PointNet/.venv/bin/activate
export PYTHONPATH=src
```

## 1. 生成缓存与窗口

```bash
python scripts/prepare_data.py --config configs/base.json
```

输出：

- `cache/small/*.npy`, `cache/test/*.npy`
- `cache/windows/{train,val,test}_{indices,offsets}.npy` 与 `*_meta.json`
- `cache/windows/schedule.npz`
- `cache/manifest.json`
- `reports/data_report.md`
- `reports/conflicting_noise_points.csv`

## 2. Pilot

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py --pilot --gpu 1
```

## 3. 正式实验

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py --gpu 1
```

默认跑两组 × 3 seeds：

- `tnet_on`：保留 3×3 输入 T-Net
- `tnet_off`：去除 3×3 输入 T-Net（feature T-Net 保留）

结果在 `runs/<run_id>/`，汇总在 `runs/summary.csv` / `runs/summary.md`。

## 4. 单次训练 / 评估 / 推理

```bash
# 训练
python -m building_pointnet.train --config configs/tnet_on.json --seed 1234 --gpu 0

# 评估（val 或 test）
python -m building_pointnet.evaluate \
  --config configs/tnet_on.json \
  --checkpoint runs/<run_id>/best_model.ckpt \
  --split test --output-dir runs/<run_id>/eval_test --write-las

# 对任意 LAS/LAZ 推理（输出 classification + pred_prob）
python -m building_pointnet.predict \
  --config configs/tnet_on.json \
  --checkpoint runs/<run_id>/best_model.ckpt \
  --input ../Data/Train_Data/train_data/non_ground_points_big_validation.las \
  --output runs/<run_id>/predict/test_pred.las
```

## 记录文件

每个 run 目录包含：

- `config.json`、`env.json`、`data_manifest.json`
- `train.log`、`metrics.csv`
- `best_model.ckpt.*`、`last_model.ckpt.*`
- `summary.json`
- `test_metrics.json`、`test_confusion_matrix.csv`、`test_pred.las`（训练结束时自动生成）

## 重要约束

- 训练窗口只读取 train 点；val/buffer 点不会进入训练输入。
- 验证/测试使用**固定、与标签无关的窗口内随机排列**后再切 4096 点块，避免缓存中
  “建筑在前、非建筑在后”的顺序造成纯类块；任意 LAS 推理使用同一规则。
- 验证/测试保证每个点至少被预测一次；重叠窗口按点 ID 平均。
- 测试集不参与模型选择、早停或阈值选择；训练默认**不评估 test**，只有显式
  `--run-test` 才执行。现有 600-step pilot 已查看过 test，见
  `reports/test_set_usage.md`。
- AP 使用 sklearn / 同分组实现，保证相同分数不依赖行顺序。
- `Classification`、`Intensity`、`Return Number`、`Original cloud index` 不作为输入特征。
- 输出 LAS 的 `classification` 0/1 是本项目约定；同时写 `pred_building` 额外字段。

## 5. 审查修正后的重评

```bash
python scripts/audit_eval_grouping.py
python scripts/reevaluate_checkpoint.py \
  --config configs/tnet_on.json \
  --checkpoint runs/<run_id>/best_model.ckpt \
  --split val \
  --output-dir runs/<run_id>/reeval_val_corrected
```

结果写入 `reeval.json` / `reeval.md`；同时给出 0.5、验证集 mIoU 最优阈值和常数预测基线。

## 6. 验证错误导出与最终测试候选

```bash
# 验证错误 LAS + 俯视图（FP/FN）
python scripts/export_val_errors.py \
  --config configs/tnet_on.json \
  --checkpoint runs/<run_id>/best_model.ckpt \
  --output-dir runs/<run_id>/val_errors_corrected

# 生成并核验最终测试候选（不冻结）
python scripts/prepare_final_test_candidate.py

# 全部模型选定后冻结候选
python scripts/freeze_final_test_candidate.py \
  --meta reports/final_test_candidate_2026-10-28.json \
  --las cache/final_test_candidate_2026-10-28.las

# 冻结后统一评价（阈值必须来自验证集）
python scripts/evaluate_final_test.py \
  --config configs/tnet_on.json \
  --checkpoint runs/<run_id>/best_model.ckpt \
  --candidate-las cache/final_test_candidate_2026-10-28.las \
  --candidate-meta reports/final_test_candidate_2026-10-28.json \
  --threshold 0.61 --output-dir runs/<run_id>/final_test
```

候选核验报告：`reports/final_test_candidate_2026-10-28.md`。
