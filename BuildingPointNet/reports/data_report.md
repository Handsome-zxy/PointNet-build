# BuildingPointNet 数据核验报告

## 点数

- build_small: 18,426,342
- noise_small: 3,113,465
- small 合计: 21,539,807
- 删除的跨类冲突 noise 点: 2,122
- build_small 类内重复坐标: 513,230
- noise_small 类内重复坐标: 103,220

## 划分

| 集合 | 总数 | 建筑 | 非建筑 |
| --- | ---: | ---: | ---: |
| train | 18,463,748 | 15,982,332 | 2,481,416 |
| val | 2,959,202 | 2,443,136 | 516,066 |
| buffer | 114,735 | 874 | 113,861 |
| ignore | 2,122 | 0 | 2,122 |

- val_box: [405420.0, 405480.0, 3112685.0, 3112733.0]
- buffer_box: [405415.0, 405485.0, 3112680.0, 3112738.0]
- z_ref (train median): 93.7600 m
- class_weights: [1.0, 1.0]
- train class counts: [2481416, 15982332]

## 测试集标签（来自 Original cloud index 坐标追溯）

- building: 2,476,850
- non-building: 1,582,941
- ignore: 0

## 窗口

| 集合 | 窗口数 | 覆盖率 | 采样点次 |
| --- | ---: | ---: | ---: |
| train | 90 | 1.0000 | 73,284,562 |
| val | 9 | 1.0000 | 12,242,307 |
| test | 35 | 1.0000 | 17,493,735 |

- window_size=30.0 m, stride=15.0 m, points_per_block=4096
- schedule: {'n_blocks': 64000, 'seed': 20261028}
