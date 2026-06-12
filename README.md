# IMU Standardized Training Data Format

本仓库用于保存已经标准化、可直接用于大规模训练的 IMU 片段数据。

本规范只约束最终处理后的 `processed/` 数据，不约束原始数据的存储方式。原始数据、临时文件、中间处理结果、检查日志和失败样本不属于最终数据集，可以在数据处理阶段临时存在，但不应作为训练数据规范的一部分进入本仓库。

## 目标

标准化后的数据需要满足以下目标：

- 区分不同设备位置：`phone`、`watch`、`ring`、`other`
- 保留三轴加速度计；如果数据源包含三轴陀螺仪，也一并保留
- 统一单位制：
  - 加速度：`m/s^2`
  - 角速度：`rad/s`
- 统一坐标系：右手系 `x/y/z`
- 统一时间戳：片段内从 `0` 开始，单位 `ms`
- 输出两种切片模式：
  - `10s`：每段约 10 秒
  - `1000f`：每段 1000 帧
- 使用 `.npz` 保存片段数据，便于 PyTorch 训练时快速读取
- 使用 JSONL manifest 记录每个训练片段的元信息

## 最终目录结构

```text
processed/
├── sources/
│   ├── source_a/
│   │   ├── segments/
│   │   │   ├── 10s/
│   │   │   │   ├── phone/
│   │   │   │   │   ├── 00000001.npz
│   │   │   │   │   └── 00000002.npz
│   │   │   │   ├── watch/
│   │   │   │   │   └── 00000001.npz
│   │   │   │   ├── ring/
│   │   │   │   └── other/
│   │   │   └── 1000f/
│   │   │       ├── phone/
│   │   │       │   └── 00000001.npz
│   │   │       ├── watch/
│   │   │       ├── ring/
│   │   │       └── other/
│   │   └── manifests/
│   │       ├── 10s.jsonl
│   │       └── 1000f.jsonl
│   │
│   └── source_b/
│       ├── segments/
│       │   ├── 10s/
│       │   │   ├── phone/
│       │   │   ├── watch/
│       │   │   ├── ring/
│       │   │   └── other/
│       │   └── 1000f/
│       │       ├── phone/
│       │       ├── watch/
│       │       ├── ring/
│       │       └── other/
│       └── manifests/
│           ├── 10s.jsonl
│           └── 1000f.jsonl
│
└── manifests/
    ├── all_10s.jsonl
    ├── all_1000f.jsonl
    ├── phone_10s.jsonl
    ├── phone_1000f.jsonl
    ├── watch_10s.jsonl
    ├── watch_1000f.jsonl
    ├── ring_10s.jsonl
    ├── ring_1000f.jsonl
    ├── other_10s.jsonl
    └── other_1000f.jsonl
```

`processed/sources/{src}/` 是一个完整、可独立上传、下载、替换、删除的数据源包。后续如果某个数据源需要重跑，只应替换对应的 `processed/sources/{src}/`，然后重新生成 `processed/manifests/` 下的全局索引文件。

## 路径规范

每个 IMU 片段文件的路径固定为：

```text
processed/sources/{src}/segments/{mode}/{device}/{segment_id}.npz
```

字段含义：

| 字段 | 含义 | 允许值或建议 |
| --- | --- | --- |
| `src` | 数据来源 ID | 英文、小写、稳定，推荐 `snake_case`，例如 `source_a`、`realworld` |
| `mode` | 切片模式 | `10s` 或 `1000f` |
| `device` | 设备位置 | `phone`、`watch`、`ring`、`other` |
| `segment_id` | 片段编号 | 推荐 8 位递增编号，例如 `00000001.npz` |

`segment_id` 只要求在同一个 `{src}/{mode}/{device}` 目录内唯一。

## NPZ 片段格式

每个 `.npz` 文件必须包含 `acc` 数组。如果数据源包含陀螺仪，还必须包含 `gyro` 数组；如果数据源没有陀螺仪，则不写入 `gyro`，并在 manifest 中设置 `has_gyro: false`。

| Key | Required | Shape | Dtype | Columns |
| --- | --- | --- | --- | --- |
| `acc` | yes | `[T, 4]` | `float32` | `time_ms, acc_x, acc_y, acc_z` |
| `gyro` | no | `[T, 4]` | `float32` | `time_ms, gyro_x, gyro_y, gyro_z` |

数组列顺序固定为：

```text
[:, 0] = time_ms
[:, 1] = x
[:, 2] = y
[:, 3] = z
```

`time_ms` 是片段内部时间戳，单位为毫秒，必须从 `0` 开始。若原数据有可用时间戳，应转换为相对当前片段起点的毫秒时间戳。若原数据没有可用时间戳，`time_ms` 列必须全部填 `0`，并在 manifest 中设置 `has_timestamp: false`。

`acc[:, 1:4]` 的单位为 `m/s^2`，坐标轴为右手系 `x/y/z`。如果存在 `gyro`，`gyro[:, 1:4]` 的单位为 `rad/s`，坐标轴为右手系 `x/y/z`。

暂时不保存磁力计或其他传感器。如果后续需要加入其他信号，应先更新本规范，再生成新的标准数据。

处理过程内部可以使用 `float64` 进行单位转换、重采样或坐标变换，但写入最终 `.npz` 前必须转换为 `float32`。选择 `float32` 是为了降低存储体积、提高读取速度，并与 PyTorch 默认训练张量类型保持一致。

## 切片规范

默认使用不重叠窗口切片。除非某个数据集的专门处理文档明确说明，否则不得使用 overlap sliding window。

### `10s` 模式

`10s` 模式的目标帧数为：

```text
T = ceil(freq * 10)
```

要求：

- 每个片段的 `num_frames` 必须等于 `ceil(freq * 10)`
- 尾段不足 `T` 帧时必须丢弃
- 不允许 padding
- `duration_sec` 建议记录为 `num_frames / freq`

示例：

```text
freq = 100.0 Hz  -> T = ceil(1000.0) = 1000
freq = 99.8 Hz   -> T = ceil(998.0) = 998
freq = 100.1 Hz  -> T = ceil(1001.0) = 1001
freq = 59.94 Hz  -> T = ceil(599.4) = 600
```

### `1000f` 模式

`1000f` 模式的目标帧数固定为：

```text
T = 1000
```

要求：

- 每个片段的 `num_frames` 必须等于 `1000`
- 尾段不足 `1000` 帧时必须丢弃
- 不允许 padding
- `duration_sec` 建议记录为 `1000 / freq`

## Manifest 规范

Manifest 使用 JSONL 格式，一行表示一个训练片段。所有 manifest 文件使用相同的 entry schema。

### Source Manifest

每个数据源内部维护自己的权威 manifest：

```text
processed/sources/{src}/manifests/10s.jsonl
processed/sources/{src}/manifests/1000f.jsonl
```

source manifest 只记录当前 `src` 的片段，不记录其他数据源。

示例：

```json
{"dir":"sources/source_a/segments/10s/phone/00000001.npz","src":"source_a","device":"phone","freq":100.0,"mode":"10s","num_frames":1000,"duration_sec":10.0,"has_timestamp":true,"has_gyro":true,"label":{"raw":"walking"}}
```

### Global Manifest

全局 manifest 放在：

```text
processed/manifests/
```

它们是从各个 source manifest 合并或过滤生成的派生索引，不是手工维护的权威数据。

```text
processed/manifests/all_10s.jsonl
= all sources/*/manifests/10s.jsonl

processed/manifests/all_1000f.jsonl
= all sources/*/manifests/1000f.jsonl
```

按设备过滤的 manifest：

```text
processed/manifests/phone_10s.jsonl
processed/manifests/phone_1000f.jsonl
processed/manifests/watch_10s.jsonl
processed/manifests/watch_1000f.jsonl
processed/manifests/ring_10s.jsonl
processed/manifests/ring_1000f.jsonl
processed/manifests/other_10s.jsonl
processed/manifests/other_1000f.jsonl
```

例如：

```text
phone_10s.jsonl = all_10s.jsonl entries where device == "phone"
ring_1000f.jsonl = all_1000f.jsonl entries where device == "ring"
```

权威关系：

```text
source manifest 是权威清单。
all manifest 是全局训练入口。
device manifest 是按设备过滤后的便捷训练入口。
```

### Manifest Entry Schema

每条 JSONL 记录必须至少包含以下字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `dir` | string | yes | 相对 `processed/` 的 `.npz` 路径 |
| `src` | string | yes | 数据来源 ID |
| `device` | string | yes | `phone`、`watch`、`ring`、`other` |
| `freq` | number | yes | 采样频率，单位 Hz |
| `mode` | string | yes | `10s` 或 `1000f` |
| `num_frames` | integer | yes | 片段实际帧数 |
| `duration_sec` | number | yes | 片段时长，单位秒 |
| `has_timestamp` | boolean | yes | `acc` 和可用的 `gyro` 是否包含真实时间戳 |
| `has_gyro` | boolean | yes | `.npz` 中是否包含 `gyro` 数组 |
| `label` | object | yes | 原数据集标签和其他标签信息 |

`dir` 必须相对 `processed/`，不能使用绝对路径，也不能相对仓库根目录。例如：

```json
{"dir":"sources/source_a/segments/1000f/ring/00000001.npz","src":"source_a","device":"ring","freq":50.0,"mode":"1000f","num_frames":1000,"duration_sec":20.0,"has_timestamp":true,"has_gyro":false,"label":{"raw":"sitting"}}
```

`label` 至少建议包含原始标签：

```json
{"label":{"raw":"walking"}}
```

如果某个数据源有更多可用信息，可以继续扩展：

```json
{"label":{"raw":"walking","standard":"walk","subject":"S01","trial":"trial_03"}}
```

扩展字段不得破坏已有必填字段的含义。

## 数据处理流程规范

不同数据源可以有各自的具体处理脚本和处理说明，但进入 `processed/` 前必须遵循同一套通用流程：

1. 读取原始 IMU 数据。
2. 识别设备位置，并映射到 `phone`、`watch`、`ring`、`other` 之一。
3. 识别或确定采样频率 `freq`。
4. 将加速度转换为 `m/s^2`。
5. 如果存在陀螺仪，将角速度转换为 `rad/s`。
6. 将坐标轴转换为右手系 `x/y/z`。
7. 根据 `10s` 和 `1000f` 两种模式切片。
8. 丢弃不足目标长度的尾段，不做 padding。
9. 将每个片段的时间戳转换为从 `0` 开始的 `time_ms`；若无可用时间戳，则填 `0`。
10. 将 `acc` 和可用的 `gyro` 转为 `[T, 4]` 的 `float32` 数组。
11. 写入 `.npz` 文件。
12. 写入当前 source 的 JSONL manifest，并记录 `has_timestamp` 和 `has_gyro`。
13. 通过验收检查后，才允许加入最终 `processed/` 数据集。
14. 重新生成全局 manifest 和按设备 manifest。

本 README 不规定每个原始数据集的具体字段映射、单位换算来源、坐标轴换算矩阵或标签映射规则。这些细节应在后续数据集专门处理文档中说明。

## 验收规范

能进入 `processed/` 的数据默认已经验收通过。不在 `processed/` 中保留验收报告、失败样本、临时日志或中间文件。

每个 source/mode 在加入最终数据前必须检查：

- source manifest 文件存在，且每一行都是合法 JSON
- 每条 manifest 的 `dir` 都是相对 `processed/` 的路径
- 每条 manifest 指向的 `.npz` 文件都存在
- `src` 与所在目录 `processed/sources/{src}/` 一致
- `device` 只允许 `phone`、`watch`、`ring`、`other`
- `mode` 只允许 `10s`、`1000f`
- `.npz` 文件必须包含 `acc`
- 如果 `has_gyro == true`，`.npz` 文件必须包含 `gyro`
- 如果 `has_gyro == false`，`.npz` 文件不应包含伪造的 `gyro` 数据
- `acc` 和可用的 `gyro` 都是 `float32`
- `acc` 和可用的 `gyro` 都是二维数组，shape 为 `[T, 4]`
- `acc.shape[0] == num_frames`
- 如果存在 `gyro`，`gyro.shape[0] == num_frames`
- `acc[:, 0]` 和可用的 `gyro[:, 0]` 是 `time_ms`
- 如果 `has_timestamp == true`，`time_ms` 必须从 `0` 开始，单位为毫秒
- 如果 `has_timestamp == false`，`time_ms` 必须全部为 `0`
- 数组值全部有限，不包含 `NaN` 或 `Inf`
- `10s` 模式下 `num_frames == ceil(freq * 10)`
- `1000f` 模式下 `num_frames == 1000`
- `duration_sec` 与 `num_frames / freq` 一致或足够接近
- 数据已经统一为指定单位和右手系 `x/y/z`
- 全局 manifest 能从 source manifest 重新生成
- 按设备 manifest 能从全局 manifest 重新生成

如果任一检查失败，该 source/mode 不得进入最终 `processed/` 数据集。

## 读取规范

训练代码应优先读取 `processed/manifests/` 下的 JSONL 文件。

示例：

```python
from pathlib import Path
import json
import numpy as np
import torch

processed_dir = Path("processed")
manifest_path = processed_dir / "manifests" / "all_10s.jsonl"

with manifest_path.open("r", encoding="utf-8") as f:
    item = json.loads(next(f))

npz_path = processed_dir / item["dir"]
data = np.load(npz_path)

acc = torch.from_numpy(data["acc"])
acc_xyz = acc[:, 1:4]

if item["has_gyro"]:
    gyro = torch.from_numpy(data["gyro"])
    gyro_xyz = gyro[:, 1:4]
    imu = torch.cat([acc_xyz, gyro_xyz], dim=1)  # shape: [T, 6]
else:
    imu = acc_xyz  # shape: [T, 3]
```

按设备训练时，可以直接读取对应的 device manifest：

```text
processed/manifests/ring_10s.jsonl
processed/manifests/watch_1000f.jsonl
```

按数据源调试时，可以读取 source manifest：

```text
processed/sources/{src}/manifests/10s.jsonl
processed/sources/{src}/manifests/1000f.jsonl
```

无论读取哪一种 manifest，entry schema 都相同。

## 版本和扩展

当前规范版本只包含：

- `acc`
- optional `gyro`
- segment-local `time_ms`
- `10s` 切片
- `1000f` 切片
- `phone`、`watch`、`ring`、`other` 四类设备位置

如果未来需要加入磁力计、气压计、GPS、更多设备类型、重叠窗口或新的标签体系，应先更新本 README，再生成对应的新标准数据。
