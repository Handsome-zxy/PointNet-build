# BuildingPointNet 实验设计（第一版）

> 任务：LAS 点云逐点二分类（建筑 / 非建筑）
> 项目目录：`/mnt/whu_stg/zxy/projects/pointcloud/BuildingPointNet`
> 原始数据只读；所有缓存、配置、日志、权重、指标均在本目录内。
> 当前状态：设计已确认，开始实现数据管线与 pilot。

---

## 1. 已核验的数据事实

| 文件 | 点数 | XY 范围 (m) | Z p1/p50/p99 (m) | 标签来源 |
| --- | ---: | --- | --- | --- |
| `non_ground_points_noise_small.las` | 3,113,465 | X 405420–405677<br>Y 3112639–3112732 | 51.5 / 56.2 / 64.0 | 非建筑（文件级） |
| `non_ground_points_build_small.las` | 18,426,342 | X 405430–405661<br>Y 3112638–3112728 | 53.8 / 103.1 / 154.4 | 建筑（文件级） |
| `non_ground_points_big_validation.las` | 4,059,791 | X 405544–405638<br>Y 3112906–3112977 | 50.6 / 64.8 / 112.8 | 额外字段 + 大文件坐标追溯 |

共同属性：LAS 1.4、point format 7、EPSG:4547、scale 0.01 m、offset (405000, 3112000, 0)。

已完成的核验：

1. **留出区点数与用户统计完全一致**：
   - 验证集建筑 2,443,136；非建筑 518,182；合计 2,961,318。
   - 缓冲区 114,735；训练集 18,463,754；总计 21,539,807。
2. **留出区主建筑完整**：最大建筑连通分量 2,439,399 点，XY 包围盒 X[405431.82, 405465.57]、Y[3112696.46, 3112722.59]，全部位于验证框内。
3. **一个 460 点低矮构件跨 Y 边界**：X[405452.27, 405461.14]、Y[3112684.74, 3112685.85]、Z 53.9–58.7 m；298 点在验证框内、162 点在缓冲区。它不是高层建筑，且缓冲区已阻止其进入训练，不构成泄漏；第一版保留原框、不调整。
4. **RGB 编码全量确认**：三份文件所有 RGB 值均为 256 的整数倍，范围 0–65280；统一使用 `RGB / 65280`。
5. **验证集标签可恢复**：额外字段 `Original cloud index` 与两个大文件坐标 100% 对应：
   - 0 组 2,476,850 点 → `non_ground_points_build_big.las`（建筑）
   - 1 组 1,582,941 点 → `non_ground_points_noise_big.las`（非建筑）
   - 两组互不交叉，未匹配点为 0。
   - 因此验证/测试标签：`label = 1 - Original_cloud_index`。
6. **重复点**：验证文件 4,059,791 行中有 117,727 个重复坐标；评价按原始点行逐行计分，不去重。
7. **类别冲突**：`noise_small` 中有约 2,122 个坐标与 `build_small` 完全重合；按“建筑优先”删除这些 noise 点并记录 CSV，其余点保留。

不可用字段：`Classification`（全 0）、`Intensity`（全 0）、`Return Number/Number of Returns`（全 1）、`Scanner Channel`（全 0）。它们既不作标签也不作特征。

---

## 2. 标签与数据划分

### 2.1 标签

- `build_small`：全部为 1（建筑，含屋顶/立面/墙体/阳台，按用户人工检查结果）。
- `noise_small`：全部为 0（非建筑）。
- 两文件之间 2,122 个坐标冲突：删除 noise 侧，保留 build 侧。
- `Classification` 与 `Original cloud index` 不作为训练输入特征。
- 验证/测试文件的标签由坐标追溯得到（见 1.5），不把大区域点加入训练。

### 2.2 固定空间留出

原始坐标：

- 验证框：`405420 <= X < 405480`，`3112685 <= Y < 3112733`，Z 全保留。
- 缓冲区：验证框向四周扩 5 m：`405415 <= X < 405485`，`3112680 <= Y < 3112738`，去掉验证框本身。
- 训练集：small 点中位于扩展框之外的点。

集合归属：

| 集合 | 规则 | 用途 |
| --- | --- | --- |
| train | small 点且不在扩展框内 | 训练 |
| val | small 点且在验证框内 | 模型选择、早停 |
| buffer | small 点且在扩展框内、验证框外 | 不训练、不计分 |
| test | `non_ground_points_big_validation.las` 全部点 | 最终测试，只使用一次 |

第一版不做 5 折交叉验证，不扩充大文件训练数据。

### 2.3 防止泄漏

- 训练窗口只取 `subset == train` 的点；即使窗口框与验证/缓冲区相交，也不读取其中的 val/buffer 点。
- 验证窗口只取 `subset == val` 的点。
- 测试窗口只取测试文件的点。
- 归一化参数 `z_ref`、类别权重只由训练集计算。
- 测试文件不参与模型选择、阈值选择、早停或任何调参。

---

## 3. 预处理与窗口采样

### 3.1 不做固定体素下采样

保留原始点。第一版不删除离群点，不做统计滤波；只做数据校验、冲突处理、窗口采样。0.1/0.2 m 体素作为后续对照实验。

### 3.2 窗口

- XY 窗口：30 m × 30 m。
- 步长：15 m（50% 重叠）。
- Z：保留窗口内全部实际高度。
- 每块输入点数：4096。
- 窗口内点数 ≥ 4096：无放回随机抽 4096；< 4096：有放回补齐。
- 训练窗口顺序打乱；同一窗口在不同 step 可以抽到不同点。
- 验证/测试：对每个窗口内的全部点做确定性分块，保证每个点至少被预测一次；重叠窗口的概率按原始点 ID 求平均。

### 3.3 输入与归一化

输入 6 维：`[x, y, z, R, G, B]`。

```
x_in = (X - window_center_X) / 30
y_in = (Y - window_center_Y) / 30
z_in = (Z - z_ref) / 30
RGB_in = RGB_raw / 65280
```

- `z_ref`：训练集全部点 Z 的中位数，只计算一次，验证/测试复用。
- 三轴同一尺度；保留米制坐标用于裁块、邻域计算与结果回写。
- 不使用“块内低分位数”作为真实地面；第一版不加入相对 Z 通道。
- 输入 T-Net 只作用于前 3 维 XYZ；RGB 在 T-Net 之后拼接。

---

## 4. 模型

### 4.1 结构

逐点二分类 PointNet 分割网络：

1. 输入 `[B, 4096, 6]`。
2. 拆出 XYZ `[B,N,3]` 与 RGB `[B,N,3]`。
3. **实验组 A**：`input_transform_net` 生成 3×3 矩阵，作用于 XYZ；**实验组 B**：不做该变换。
4. XYZ（变换后）与 RGB 拼接为 6 维。
5. 共享 MLP：conv `[1,6]→64`、`[1,1]→64`。
6. feature T-Net（64×64，两组都保留）→ 变换 64 维逐点特征。
7. conv `[1,1]→64`、`→128`、`→1024`；对 N 做全局 max pooling。
8. 全局特征复制到每个点，与 64 维逐点特征拼接为 1088 维。
9. MLP：512 → 256 → 128 → 2（逐点 logits）。
10. BN + ReLU；512/256 后 dropout 0.5。

### 4.2 对照设置

- A（`tnet_on`）：保留 3×3 输入 T-Net。
- B（`tnet_off`）：仅去除 3×3 输入 T-Net。
- 两组都保留 64×64 feature T-Net 及其正交正则。
- 除 `use_input_tnet` 外，数据、窗口、采样顺序、增强、超参数、随机种子、评价协议完全一致。

### 4.3 损失

```
loss = weighted_CE + 1.0 * Dice + 0.001 * feature_transform_orthogonality
```

- CE 类别权重：第一版使用 `[1.0, 1.0]`（不加权），由 Dice 处理类别不平衡。
- Dice 作用于 building 概率与 building 标签。
- **Pilot 调整记录**：初版使用反频率权重 `[1.731, 0.269]`，2.5 个 epoch 后验证集
  building AP=0.9976，但 0.5 阈值下 building recall 仅 0.004（概率整体被压低）。
  因此改为无类别权重；这是 pilot 后唯一调整的超参数，数据划分、窗口、网络、增强均未改动。
- 仅对 feature T-Net 的 64×64 变换做正交正则（与原始 PointNet 一致）。

---

## 5. 训练

| 项目 | 值 |
| --- | --- |
| 框架 | TensorFlow 2.21 `compat.v1`（复用 PointNet/.venv） |
| GPU | 单卡 A100；通过 `CUDA_VISIBLE_DEVICES` 指定，默认 1 号卡 |
| batch | 8 个块 × 4096 点 |
| 优化器 | Adam |
| 初始学习率 | 1e-3，500 step 线性 warmup，cosine 衰减到 1e-5 |
| 训练步数 | 8,000 step（40 epochs × 200 step） |
| 验证 | 每 1,000 step 全量验证一次 |
| 早停 | 验证集 mIoU（最优阈值下），连续 4 次验证无提升则停止 |
| 梯度裁剪 | 全局范数 1.0 |
| 权重衰减 | 第一版不加 |
| 数据增强 | 绕 Z 随机旋转、XY 平移 ±3 m、坐标抖动 σ=0.01 m；两组完全一致 |
| 随机种子 | 1234 / 2345 / 3456；先 pilot 再全量 |

**关于 epoch 的定义**：30 m/15 m 窗口下，训练区完整扫一遍只有约 90 个窗口 × 4096 点，单轮样本量远小于 18.5M 点。因此这里把 1 epoch 定义为 200 个 optimizer step（1,600 个块），而不是“窗口全扫一遍”。窗口、步长、每块点数仍严格按约定执行。

**关于 8,000 step**：pilot 实测一次全量验证约 4–5 分钟；20,000 step × 每 1,000 step 验证的成本过高。600-step 校准 pilot 显示 500 step 时 building AP 已很高，因此正式实验取 8,000 step（约 14 个数据 pass），并用验证集 mIoU 早停。该调整只影响训练长度，不影响数据、模型和对照公平性。

**公平性**：

- 预先固定训练 schedule：每个 block 记录 `(window_id, block_seed)`，两组读取同一份 schedule，保证看到的点、增强、顺序完全一致。
- 同一空间划分、同一验证集、同一指标、同一天度。
- 固定 Python/NumPy/TF 种子；开启 TF deterministic ops（如可用）。
- 记录参数量、训练时间、推理时间、显存峰值。
- 每组 3 个种子，报告 mean ± std 与逐种子结果。

---

## 6. 评价

### 6.1 指标

逐点、只在有效标签上计算：

- OA、balanced accuracy
- building / non-building 的 Precision、Recall、F1、IoU
- macro F1、mIoU
- 2×2 混淆矩阵
- building 概率的 PR-AUC
- 可选块级 mean IoU（按 30 m 窗口）

验证/测试都按原始点 ID 合并重叠窗口预测；每个点只计分一次。测试标签缺失/冲突的点记为 ignore，并在报告中给出数量。

### 6.2 选择与测试

- 模型选择、早停、阈值选择只用验证集。
- 每个种子用验证集 mIoU（在 0.01–0.99 阈值扫描下的最优值）选最佳 checkpoint。
- 测试集只在训练结束后用最佳 checkpoint 评估一次。
- 阈值默认 0.5；同时在验证集上扫描 0.01–0.99，选择 mIoU 最优阈值（同时兼顾 building / non-building）。
  若该阈值比 0.5 的验证 mIoU 高至少 0.001，测试使用该阈值；无论是否使用，都同时报告
  0.5 与调优阈值的测试指标，并输出 PR-AUC。

---

## 7. 推理

`predict.py`：

- 输入任意 LAS/LAZ，自动按 30 m/15 m 窗口分块、批量推理。
- 每个窗口内全部点确定性分块，保证覆盖；重叠窗口概率平均。
- 输出新 LAS：
  - `classification`：1 = 建筑，0 = 非建筑
  - 额外字段 `pred_prob`：building 概率
  - 保留原 CRS、scale/offset、点顺序、RGB 与其他属性
- 同时输出 CSV/JSON 摘要：点数、类别比例、概率统计、耗时。
- 第一版内存内读写；超大文件后续加分块流式写入。

---

## 8. 实验矩阵

| 实验 | 输入 T-Net | 种子 | 步数 | 用途 |
| --- | --- | --- | ---: | --- |
| pilot_on | 有 | 1234 | 2,000 | 管线/收敛 sanity |
| pilot_off | 无 | 1234 | 2,000 | 管线/收敛 sanity |
| full_on_s1234 | 有 | 1234 | 8,000 | 正式 |
| full_on_s2345 | 有 | 2345 | 8,000 | 正式 |
| full_on_s3456 | 有 | 3456 | 8,000 | 正式 |
| full_off_s1234 | 无 | 1234 | 8,000 | 正式 |
| full_off_s2345 | 无 | 2345 | 8,000 | 正式 |
| full_off_s3456 | 无 | 3456 | 8,000 | 正式 |

报告：

- 两组的 mean ± std（OA、building F1、building IoU、mIoU、non-building F1 等）
- 参数量、训练时长、推理时长、GPU 显存
- 每种子逐项结果
- 验证集与测试集分开报告

---

## 9. 目录与实验记录

```
BuildingPointNet/
├── DESIGN.md
├── README.md
├── requirements.txt
├── configs/
│   ├── base.json
│   ├── tnet_on.json
│   └── tnet_off.json
├── src/building_pointnet/
│   ├── las_utils.py
│   ├── data_prep.py
│   ├── dataset.py
│   ├── tf_util.py
│   ├── transform_nets.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── logging_utils.py
├── scripts/
│   ├── prepare_data.py
│   └── run_experiment.py
├── cache/                 # 派生缓存，可重建
├── runs/<run_id>/
│   ├── config.json
│   ├── env.json
│   ├── data_manifest.json
│   ├── metrics.csv
│   ├── train.log
│   ├── best_model.ckpt.*
│   ├── last_model.ckpt.*
│   ├── test_metrics.json
│   ├── test_confusion_matrix.csv
│   └── test_pred.las
└── reports/
    ├── data_report.md
    ├── split_report.json
    └── summary.csv / summary.md
```

---

## 10. 训练前验证门

1. 数据点数与用户统计一致（已通过）。
2. 主建筑完整（已通过）；460 点跨边界构件已记录。
3. RGB 全量编码一致（已通过）。
4. 验证/测试标签坐标追溯 100% 覆盖（已通过）。
5. 窗口覆盖：验证/测试每个点至少进一个窗口。
6. 集合隔离：train 窗口只含 train 点；val 窗口只含 val 点。
7. pilot：loss 下降、无 NaN、验证指标高于随机。
8. 测试集在最终评估前不读取标签。

---

## 11. 风险与备注

- `build_small` 的 0 组中有少量远离建筑轮廓的点；按用户人工检查结果整体标为建筑，不再清洗。
- 训练/验证建筑占比约 86.6%/82.4%，测试建筑占比约 61.0%；只看 OA 会偏乐观，必须报告 building IoU/F1 与两类指标。
- RGB 有少量全黑点（603/1832/451），保留，不做特殊处理。
- 输入 T-Net 的对照结论可能受旋转增强影响；第一版两组使用相同增强，不额外做无旋转消融；若结果差异不明显，再追加该消融。
- 第一版不做体素下采样与离群点删除，后续作为独立对照实验。

---

## 12. 审查修正记录（2026-10-28）

审查报告见 `BuildingPointNet实验审查报告_2026-10-28.md`，处理结果见
`BuildingPointNet实验审查报告处理结果_2026-10-28.md`。

已修正：

1. **验证/推理分组**：缓存按类别连续存储（先 building 后 noise），旧 `iter_eval_batches`
   直接 `idx[a:a+N]`，导致验证块几乎全是纯类；训练却是窗口内随机抽 4096 点。现改为
   每个窗口先做固定、与标签无关的 `window_permutation` 再切块；`predict.py` 同步使用。
2. **AP 同分处理**：改用 `sklearn.metrics.average_precision_score`，并提供同分分组的
   fallback；增加恒定分数/行重排不变性测试。
3. **测试集使用**：训练默认不评估 test，只有显式 `--run-test` 才执行；600-step pilot
   已查看 test，记录为开发期已查看。正式最终测试规则见 `reports/test_set_usage.md`。
4. **缓存权重一致性**：manifest 记录 `class_weight_mode`；训练启动时校验
   `class_weights` 与当前配置一致，防止旧缓存。
5. **LAS header**：`read_header` 只读 375 字节，不再重复读取整个文件。
6. **输出字段**：LAS 额外写 `pred_building`（1/0），明确 `classification` 0/1 是本项目约定。
7. **代码/配置冻结**：每次 run 写 `code_hashes.json`；正式实验前写
   `reports/protocol_freeze_*.json`。

修正后的重评（同一 checkpoint，仅改评估管线）：

| checkpoint | 阈值 | OA | building F1 | building IoU | non-building F1 | mIoU | building AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 600-step | 0.50 | 0.9473 | 0.9675 | 0.9371 | 0.8597 | 0.8455 | 0.9962 |
| 1k-step | 0.50 | 0.9564 | 0.9735 | 0.9484 | 0.8768 | 0.8645 | 0.9981 |
| 1k-step | 0.61 (val mIoU) | 0.9586 | 0.9745 | 0.9502 | 0.8901 | 0.8761 | 0.9981 |

结论：之前 0.5 阈值下的低指标主要是评估分组错误，不是模型没有学到信息。暂不改变
损失、学习率或网络结构，正式比较应以修正后的评估协议重新训练/选择。
