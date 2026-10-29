# 测试集使用记录（2026-10-28）

## 已发生的测试集评估

| 运行目录 | 内容 | 状态 |
| --- | --- | --- |
| `20261028_135939_tnet_on_s1234_pilot` | 600-step 校准 pilot，已生成 `test_metrics.json`、`test_prob.npy`、`test_labels.npy`、`test_pred.las` | **测试集已被开发期查看** |
| `20261028_135306_tnet_on_s1234_pilot` | 首轮 pilot，在 test 前停止 | 未评估 test |
| `20261028_140625_tnet_on_s1234` | 第一次 8k run，在 test 前停止 | 未评估 test |
| `20261028_141535_tnet_on_s1234` | 第二次 8k run，step 1620 左右停止 | 未评估 test |

现有 `non_ground_points_big_validation.las` 已不再满足“完全未查看的最终测试集”这一严格条件。
600-step pilot 的测试指标只能作为开发期证据，不能用于最终选择或方法调整。

## 代码策略（已修改）

- `train.py` 默认**不评估 test**；只有显式传入 `--run-test` 才会在训练末尾评估。
- pilot、调参、对照实验期间一律不传 `--run-test`。
- 正式最终测试应在以下条件全部冻结后，**只执行一次**：
  - 代码哈希、配置、数据 manifest 已写入 `protocol_freeze_*.json`；
  - 两组的训练/验证/早停/阈值规则已完成；
  - 不再根据 test 结果修改任何方法。
- 最终测试建议使用独立的 `evaluate.py` / `reevaluate_checkpoint.py` 调用，并记录完整命令与哈希。

## 若需要严格未查看测试集

当前数据中没有第三个已标注、且完全未查看的区域。可选方案（需要用户确认后才能执行）：

1. 从 `non_ground_points_build_big.las` / `non_ground_points_noise_big.las` 中，
   在现有小训练区、现有 big_validation 之外，冻结一个新的空间区域作为 test；
   按大文件来源构造标签，并做坐标追溯核验。
2. 接受现有 test 作为“开发期已查看”的评估集，在论文/报告中明确披露；
   所有方法选择仍只用 small 的验证集。

在用户确认前，不创建新测试区域，也不再用现有 test 做任何阈值/模型选择。

## 新最终测试候选（2026-10-28）

已准备并核验一个空间独立候选：

- 区域：X [405185, 405262)、Y [3112900, 3112968)
- 2,550,737 点（建筑 1,532,959 / 非建筑 1,017,778，建筑比例 0.6010）
- 与 small 训练区最近距离 167.81 m；与现有 big_validation 最近距离 281.57 m
- 5 个建筑连通分量与候选相交，**0 个被切断**
- 跨类坐标冲突 0；类内重复点已记录
- 候选文件：`cache/final_test_candidate_2026-10-28.las`（含 `label`/`source` 字段）
- 核验报告：`reports/final_test_candidate_2026-10-28.md`
- 状态：`frozen=false`，尚未冻结；不参与训练/验证/模型选择

待 seed 1234 配对流程确认、全部模型选定后，再冻结该候选并统一评价一次。
