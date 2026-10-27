# PointNet 运行环境说明

## 1. 环境概览

| 项目 | 说明 |
| --- | --- |
| 系统 | Debian 13 / Linux 6.12，无 conda |
| Python | 3.13.5（虚拟环境位于 `PointNet/.venv`） |
| TensorFlow | 2.21.0（GPU 版；pip 自带 CUDA 12.9 / cuDNN 9.27 运行库，无需系统 CUDA） |
| GPU | 2 × NVIDIA A100 80GB |
| 代码兼容 | 原始 TF1.x 代码通过 `tensorflow.compat.v1` + `disable_v2_behavior()` 运行 |
| 数据 | ModelNet40 HDF5，已解压到 `data/modelnet40_ply_hdf5_2048/` |

原始 PointNet 代码写于 TensorFlow 1.0 / Python 2.7 时代（使用了
`tf.placeholder`、`tf.contrib`、`tf.Session` 等 API）。官方 TF 1.x 无法在
Python 3.13 + A100（sm_80）上运行，因此这里采用 TensorFlow 2.x 的
compat.v1 图模式，并对少量不兼容 API 做了等价替换（见第 4 节）。

## 2. 快速开始

```bash
cd /mnt/whu_stg/zxy/projects/pointcloud/PointNet
source .venv/bin/activate

# 训练（默认 250 epoch；先用 1 个 epoch 验证流程）
python train.py --gpu 1 --num_point 1024 --batch_size 32 --max_epoch 1

# 完整训练
python train.py --gpu 1 --num_point 1024 --batch_size 32

# 评估（读取 log/model.ckpt）
python evaluate.py --gpu 1 --model_path log/model.ckpt

# 错误样本可视化（图片输出到 dump/）
python evaluate.py --gpu 1 --model_path log/model.ckpt --visu
```

`--gpu N` 选择物理 GPU 编号。当前 0 号卡显存占用很高（约 75GB），建议使用
`--gpu 1`；也可以通过 `CUDA_VISIBLE_DEVICES=1 python train.py --gpu 0` 的方式
隔离显卡。

训练日志与 checkpoint 默认保存在 `log/`，评估日志在 `dump/log_evaluate.txt`。

## 3. 数据

- 路径：`data/modelnet40_ply_hdf5_2048/`（5 个训练分片、2 个测试分片，共 40 类）。
- 来源：官方 Stanford 下载地址在当前网络不可达，已改用 HuggingFace 镜像
  `Msun/modelnet40` 中的同名 `modelnet40_ply_hdf5_2048.zip`（与原版一致）。
- `provider.py` 已改为：数据不存在时用 Python `urllib` 从该镜像自动下载并解压，
  可用环境变量 `MODELNET40_URL` 覆盖下载地址。

## 4. 为适配 TF2 所做的代码修改

| 文件 | 修改 |
| --- | --- |
| `train.py`, `evaluate.py`, `models/*.py`, `utils/tf_util.py` | `import tensorflow as tf` → `import tensorflow.compat.v1 as tf` + `tf.disable_v2_behavior()` |
| `utils/tf_util.py` | `tf.contrib.layers.xavier_initializer()` → `tf.glorot_uniform_initializer()`（TF2 无 `tf.contrib`） |
| `models/*.py`, `utils/tf_util.py` | `tensor.get_shape()[i].value` → `tensor.get_shape()[i]`（TF2 返回 int） |
| `train.py`, `evaluate.py` | `dtype=np.float` → `np.float32`（NumPy ≥1.24 已移除 `np.float`） |
| `evaluate.py` | `scipy.misc.imsave` → `PIL.Image.fromarray(...).save`（SciPy ≥1.3 已移除） |
| `provider.py` | 数据下载改为可用镜像；`getDataFiles` 返回基于脚本目录的绝对路径 |
| `models/pointnet_seg.py` | 旧版 `tf.concat(3, ...)` / `tf.scalar_summary` → 新 API |

模型结构与训练超参数均未改动，仅做 API 层面的等价替换。

## 5. 依赖清单

见 `requirements.txt`：

```
tensorflow[and-cuda]==2.21.0
numpy
h5py
matplotlib
pillow
scipy
```

环境已创建在 `PointNet/.venv`，激活后即可运行：

```bash
source .venv/bin/activate
```

如需重建：

```bash
bash setup_env.sh
```

> 网络提示：当前网络下部分 pip 镜像对 HTTP/1.1（pip/urllib）单连接限速明显，
> 而 curl 的 HTTP/2 下载可达 15MB/s 以上。首次安装 TensorFlow + CUDA 轮子
> （约 3.4GB）时如果 pip 下载过慢，可先用 curl 下载 wheel 再用
> `pip install --find-links <目录> ...` 安装。实测可用的镜像：
> `https://mirrors.aliyun.com/pypi/simple/`、`https://repo.huaweicloud.com/repository/pypi/simple/`。

## 6. 已知限制

- `part_seg/`（部件分割）与 `sem_seg/`（室内语义分割）目录未做完整适配：
  它们各自的 `train.py` 仍使用 `import tensorflow as tf`，且依赖的 ShapeNetPart /
  S3DIS 数据集未下载。需要时可按第 4 节的方式同样适配。
- `tensorboard` 未安装（TF 2.21 不再强制依赖）；如需可视化，可
  `.venv/bin/pip install tensorboard` 后运行 `tensorboard --logdir log`。

## 7. 验证记录

（见下方“验证结果”）
