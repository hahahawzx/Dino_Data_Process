# Capture24 Processing Specification

本文档规定 Capture24 数据源如何转换为仓库根目录 `README.md` 中定义的标准 IMU 训练格式。

Capture24 的原始数据不进入最终 processed 仓库；处理脚本只读取本地下载目录，最终输出写入外置硬盘：

```text
/Volumes/Felix_Backups/Processed
```

## Source Definition

Capture24 的标准 source id 固定为：

```text
capture24
```

本地原始数据目录为：

```text
/Users/zixuanwang/Downloads/capture24
```

最终输出目录为：

```text
/Volumes/Felix_Backups/Processed/sources/capture24
```

Capture24 是腕部佩戴的 Axivity AX3 三轴加速度计数据。按照本仓库的设备位置定义，全部片段映射为：

```text
device = watch
```

Capture24 不包含陀螺仪，因此输出 `.npz` 只包含 `acc`，不写入 `gyro`：

```text
has_gyro = false
```

## External References

处理规范参考 Capture24 官方说明和本地文件实测字段。当前环境访问 GitHub 超时，因此本 README 以本地下载文件为准。

参考链接：

```text
https://github.com/OxWearables/capture24
```

本地确认的信息：

- 共 151 个参与者文件：`P001.csv.gz` 到 `P151.csv.gz`。
- 每个参与者文件为 gzip 压缩 CSV。
- `annotation-label-dictionary.csv` 包含 206 条 annotation 到多套标签体系的映射。
- `metadata.csv` 包含 `pid,age,sex`，但当前处理不使用年龄、性别等人口统计信息。

## Input Dataset Structure

Capture24 本地目录为扁平结构：

```text
capture24/
├── P001.csv.gz
├── P002.csv.gz
├── ...
├── P151.csv.gz
├── annotation-label-dictionary.csv
└── metadata.csv
```

每个 `P*.csv.gz` 是一个参与者的连续腕部加速度记录。处理脚本不得修改 `/Users/zixuanwang/Downloads/capture24` 内的任何文件。

## Input CSV Format

参与者文件字段固定为：

```text
time,x,y,z,annotation
```

样例：

```text
time,x,y,z,annotation
2016-11-13 02:18:00.000000,-0.46669036,-0.5333412,0.65847206,7030 sleeping;MET 0.95
2016-11-13 02:18:00.010000,-0.46669036,-0.5333412,0.65847206,7030 sleeping;MET 0.95
```

字段含义：

| 字段 | 含义 | 处理方式 |
| --- | --- | --- |
| `time` | 绝对时间戳 | 解析为时间轴，输出片段内转换为从 `0` 开始的 `time_ms` |
| `x` | 原始 x 轴加速度 | 原始单位为 `g`，必须转换为 `m/s^2` |
| `y` | 原始 y 轴加速度 | 原始单位为 `g`，必须转换为 `m/s^2` |
| `z` | 原始 z 轴加速度 | 原始单位为 `g`，必须转换为 `m/s^2` |
| `annotation` | 原始细粒度活动标注 | 通过 `annotation-label-dictionary.csv` 映射到标准标签 |

## Sampling And Timestamp Policy

Capture24 的目标输出频率固定为：

```text
freq = 100.0
```

本地 `P001.csv.gz` 实测相邻时间戳间隔为 `0.010000` 秒，即 100 Hz。

处理脚本不能直接假设每个原始 CSV 的时间轴完全稳定。必须先基于 `time` 字段对整个 participant 文件做全局稳定性检查。

当前 Capture24 的处理策略是严格接受策略：

```text
如果整个 participant 的相邻时间戳都是稳定 10 ms，则直接处理，不插值。
如果不是稳定 10 ms，则整个 participant 跳过，不做局部修补，不做插值。
```

因此：

```text
10s mode:   T = ceil(100.0 * 10) = 1000
1000f mode: T = 1000
```

这意味着 Capture24 的 `10s` 和 `1000f` 两种模式在帧数上都是 1000 帧；但仍然分别输出到两个 mode 目录和两个 manifest，保持仓库统一接口。

时间戳列规则：

```text
acc[:, 0] = time_ms
time_ms[0] = 0
time_ms[i] = 当前样本相对片段起点的毫秒时间
```

输出 manifest 中必须设置：

```text
has_timestamp = true
```

### Time Stability Policy

Capture24 输出必须来自稳定 100 Hz 原始时间轴：

```text
required_raw_dt = 10.0 ms
```

处理流程：

1. 将原始 `time` 解析为毫秒时间轴。
2. 保留原始文件顺序，不排序，不去重。
3. 检查原始顺序下相邻时间戳差值是否全部为 `10.0 ms`。
4. 如果存在任何非 `10.0 ms` 间隔，则跳过整个 participant。
5. 如果时间轴稳定，则直接使用原始行生成片段，不做插值。

允许浮点解析误差，建议判断阈值：

```python
np.allclose(np.diff(time_ms), 10.0, rtol=0.0, atol=1e-6)
```

annotation 是离散标签，不能插值。由于当前策略不插值，segment 的标签检查直接基于原始行内 annotation。

## Unit And Axis Policy

Capture24 原始 `x/y/z` 单位为 `g`。处理脚本必须转换为国际单位制：

```python
acc_mps2 = acc_g * 9.80665
```

输出 `.npz` 中：

```text
acc[:, 1:4] -> m/s^2
```

Capture24 是腕部加速度计数据，本 README 暂不额外做坐标轴翻转或重排。处理脚本默认保留原始 `x/y/z` 轴顺序，并将其作为标准输出的 `x/y/z`。如果后续确认 Capture24 原始坐标系需要转换为另一个右手系定义，必须先在本 README 中补充明确的轴映射后再处理。

## Label Dictionary

标签映射文件：

```text
annotation-label-dictionary.csv
```

字段为：

```text
annotation,
label:WillettsSpecific2018,
label:WillettsMET2018,
label:DohertySpecific2018,
label:Willetts2018,
label:Doherty2018,
label:Walmsley2020
```

本地统计：

| 标签列 | 类别数 | 说明 |
| --- | ---: | --- |
| `annotation` | 206 | 原始细粒度标注 |
| `label:WillettsSpecific2018` | 10 | 推荐作为主活动标签 |
| `label:WillettsMET2018` | 11 | 活动和强度混合标签 |
| `label:DohertySpecific2018` | 10 | 另一套活动标签 |
| `label:Willetts2018` | 6 | 粗粒度活动标签 |
| `label:Doherty2018` | 5 | 粗粒度活动/强度标签 |
| `label:Walmsley2020` | 4 | 最粗粒度强度标签 |

## Label Policy

Capture24 的 manifest `label` 使用英文字段，不使用中文字段。

推荐主标签使用：

```text
label:WillettsSpecific2018
```

原因：

- 类别数为 10，粒度适中。
- 类别语义清晰，包含 `sleep`、`sitting`、`standing`、`walking`、`vehicle`、`bicycling` 等常见活动。
- 比 `Walmsley2020` 的 4 类强度标签更适合保留活动类型信息。
- 比原始 206 类 annotation 更稳定，降低训练标签稀疏问题。

`label:WillettsSpecific2018` 的类别为：

```text
bicycling
household-chores
manual-work
mixed-activity
sitting
sleep
sports
standing
vehicle
walking
```

`label:Walmsley2020` 暂不写入 processed manifest。它只作为标签字典中的可选参考列保留在原始数据中；如果后续需要强度分类，可以基于 `label.annotations` 和 `annotation-label-dictionary.csv` 重新生成派生索引。

`label:Walmsley2020` 的类别为：

```text
light
moderate-vigorous
sedentary
sleep
```

每个 manifest entry 的 `label` 推荐格式：

```json
{
  "activity": "sleep",
  "annotations": ["7030 sleeping;MET 0.95"]
}
```

字段含义：

| 字段 | 来源 | 说明 |
| --- | --- | --- |
| `activity` | `label:WillettsSpecific2018` | 主训练标签 |
| `annotations` | 原始 `annotation` | 该 segment 内出现过的原始标注列表，英文，可保留用于追溯 |

如果某个原始 annotation 为空字符串或不在 `annotation-label-dictionary.csv` 中，该样本不得直接进入最终片段。处理脚本应将其视为 unknown label，并在切片阶段丢弃包含 unknown label 的片段。

## Segment Label Rule

Capture24 的 annotation 是逐帧字段。为了避免一个训练片段混入多个主标签，默认要求：

```text
一个保存的 segment 内所有帧映射后的 activity 必须一致。
```

如果一个候选 segment 内出现多个 `activity`，则丢弃该 segment，不做多数投票，不跨标签边界切片。

原始 `annotation` 不要求完全一致。只要它们都能映射到同一个 `activity`，该 segment 可以保存。

```text
all activity labels in segment are identical
raw annotations in segment may differ
```

manifest 中的 `label.annotations` 必须记录该 segment 内出现过的所有唯一原始 annotation。列表顺序应按 annotation 在 segment 内第一次出现的顺序保存。

如果一个 segment 内存在空 annotation，或存在无法在 `annotation-label-dictionary.csv` 中找到映射的 annotation，则丢弃该 segment。

## Quality Filtering

每个参与者文件至少需要满足：

- CSV 文件存在并可解析。
- header 必须为 `time,x,y,z,annotation`。
- `time` 能解析为单调时间轴。
- `x/y/z` 数值有限，不包含 `NaN` 或 `Inf`。
- 原始时间轴必须全局稳定为 100 Hz。
- 所有进入最终片段的 annotation 都能在 `annotation-label-dictionary.csv` 中找到映射。

participant 级时间轴检查：

```text
all diff(time_ms) == 10.0 ms within tolerance
```

如果时间轴检查失败，整个 participant 不进入最终 processed 数据。当前 Capture24 处理不做 1 秒 bin 级修补，不做插值。

segment 级检查：

- segment 内 `activity` 必须唯一。
- segment 内所有非空 `annotation` 都必须能映射到该唯一 `activity`。
- `acc.shape == (num_frames, 4)`。
- `acc.dtype == float32`。
- `acc[0, 0] == 0.0`。
- `acc[:, 0]` 单位为 ms。
- `acc[:, 0]` 必须是稳定 100 Hz 时间网格，即相邻差值为 `10.0 ms`。
- `acc[:, 1:4]` 全部有限。
- `.npz` 不包含 `gyro`。

## Segment Generation

默认使用不重叠窗口切片。

切片基于通过全局稳定性检查后的原始行号。由于只有相邻时间戳全部为 `10.0 ms` 的 participant 才会进入切片阶段，最终 `.npz` 具有稳定 100 Hz 帧率。

```text
10s mode:
  T = 1000
  non-overlapping windows
  drop tail shorter than 1000 frames
  no padding

1000f mode:
  T = 1000
  non-overlapping windows
  drop tail shorter than 1000 frames
  no padding
```

由于 Capture24 的频率是 100 Hz，两个模式的窗口长度相同。处理脚本仍需分别生成两份输出，以满足统一训练入口。

候选 segment 的时间范围为：

```text
segment_start_time = time_ms[start]
segment_end_time = time_ms[end - 1]
```

segment 标签检查使用该窗口内原始行的 annotation。只有满足以下条件才保存：

- 时间范围内不存在空 annotation。
- 时间范围内所有 annotation 都能在标签字典中找到映射。
- 时间范围内所有 annotation 映射得到的 `activity` 完全一致。

## Output Layout

Capture24 输出目录：

```text
/Volumes/Felix_Backups/Processed/
├── sources/
│   └── capture24/
│       ├── segments/
│       │   ├── 10s/
│       │   │   └── watch/
│       │   └── 1000f/
│       │       └── watch/
│       └── manifests/
│           ├── watch_10s.jsonl
│           └── watch_1000f.jsonl
└── manifests/
    ├── all_10s.jsonl
    ├── all_1000f.jsonl
    ├── watch_10s.jsonl
    └── watch_1000f.jsonl
```

片段路径示例：

```text
sources/capture24/segments/10s/watch/00000001.npz
sources/capture24/segments/1000f/watch/00000001.npz
```

manifest entry 示例：

```json
{
  "dir": "sources/capture24/segments/10s/watch/00000001.npz",
  "src": "capture24",
  "device": "watch",
  "freq": 100.0,
  "mode": "10s",
  "num_frames": 1000,
  "duration_sec": 10.0,
  "has_timestamp": true,
  "has_gyro": false,
  "label": {
    "activity": "sleep",
    "annotations": ["7030 sleeping;MET 0.95"]
  }
}
```

## Processing Commands

全量重建 Capture24：

```bash
cd /Users/zixuanwang/Code/research/Dino_Data_Process

python3 Capture24/process_capture24.py \
  --dataset-root "/Users/zixuanwang/Downloads/capture24" \
  --processed-root "/Volumes/Felix_Backups/Processed" \
  --overwrite
```

如果只想先处理排序后的前 50 个 participant，也就是 `P001.csv.gz` 到 `P050.csv.gz`：

```bash
cd /Users/zixuanwang/Code/research/Dino_Data_Process

python3 Capture24/process_capture24.py \
  --dataset-root "/Users/zixuanwang/Downloads/capture24" \
  --processed-root "/Volumes/Felix_Backups/Processed" \
  --overwrite \
  --max-participants 50
```

注意：`--max-participants 50` 生成的是一个只包含前 50 个 participant 的 `sources/capture24`。它不是断点续跑；如果后续要补 `P051` 之后的数据，需要再增加 participant range 或 resume 机制。

## Validation Before Acceptance

Capture24 数据进入 `/Volumes/Felix_Backups/Processed` 前必须检查：

- 每个 manifest entry 的 `src == "capture24"`。
- 每个 manifest entry 的 `device == "watch"`。
- `freq == 100.0`。
- `has_timestamp == true`。
- `has_gyro == false`。
- `.npz` 内只包含 `acc`，不包含 `gyro`。
- `10s` 片段 `num_frames == 1000`。
- `1000f` 片段 `num_frames == 1000`。
- `acc.shape == [1000, 4]`。
- `acc.dtype == float32`。
- `acc[:, 0]` 从 `0` 开始，单位为 ms。
- `acc[:, 1:4]` 单位为 `m/s^2`。
- `acc` 全部为有限值。
- `label` 至少包含 `activity`、`annotations`。
- `label.activity` 必须来自 `label:WillettsSpecific2018`。
- `label.annotations` 必须是非空列表。
- `label.annotations` 中每个原始 annotation 都必须能映射到同一个 `label.activity`。
- 不保留 age、sex 等 metadata 字段。

如果任一检查失败，对应 participant/mode/segment 不得进入最终 processed 数据。

## Confirmed Decisions Before Implementation

正式写处理脚本时遵循以下已确认规则：

1. segment 内 `annotation` 可以不一致，但映射后的 `activity` 必须一致。
2. 对空 annotation 全部丢弃。
3. Capture24 原始坐标轴保留原始 `x/y/z`，只做 `g -> m/s^2` 单位换算。
4. 保留原始 annotation 到 manifest，字段为 `label.annotations`。
5. 只处理全局稳定 10 ms 时间轴的 participant；不稳定则跳过，不做 bin 修补，不插值。
