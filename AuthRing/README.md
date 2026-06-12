# AuthRing Processing Specification

本文档规定 AuthRing 数据源如何转换为仓库根目录 `README.md` 中定义的标准 IMU 训练格式。

AuthRing 的处理脚本、数据源专用说明和后续检查逻辑放在本目录下。最终处理结果不写入本仓库，而是写入外置硬盘：

```text
/Volumes/Felix_Backups/Processed
```

## Source Definition

AuthRing 的标准 source id 固定为：

```text
authring
```

原始干净数据目录为：

```text
/Volumes/Felix_Backups/Root26.5.22/科研与科创/数据记录/AuthRing/Dataset
```

最终输出目录为：

```text
/Volumes/Felix_Backups/Processed/sources/authring
```

本数据源只生成两类 device：

```text
ring
phone
```

不生成 `watch` 或 `other`。

## Input Dataset Structure

AuthRing 的最小读取单位是 session。目录结构为：

```text
Dataset/
└── participantName/
    └── yyyyMMdd_HHmmss_participantName/
        ├── session.json
        ├── ring_imu.csv
        ├── phone_acc.csv
        ├── phone_gyro.csv
        ├── touch.csv
        ├── time_calib.csv
        ├── calibration.json
        ├── post_calibration.json
        ├── stages.jsonl
        ├── ground_truth.jsonl
        ├── trial_touch_details.jsonl
        └── health_events.jsonl
```

处理脚本不得修改 `Dataset/` 内的任何文件。

## Device Mapping

### Ring

`ring_imu.csv` 映射为：

```text
device = ring
```

输入字段：

```text
timestamp, accX, accY, accZ, gyroX, gyroY, gyroZ
```

输出 `.npz`：

```text
acc:  [T, 4] float32 = time_ms, acc_x, acc_y, acc_z
gyro: [T, 4] float32 = time_ms, gyro_x, gyro_y, gyro_z
```

### Phone

`phone_acc.csv` 和 `phone_gyro.csv` 合并后映射为：

```text
device = phone
```

输入字段：

```text
phone_acc.csv:  timestamp, x, y, z
phone_gyro.csv: timestamp, x, y, z
```

输出 `.npz`：

```text
acc:  [T, 4] float32 = time_ms, acc_x, acc_y, acc_z
gyro: [T, 4] float32 = time_ms, gyro_x, gyro_y, gyro_z
```

## Sampling And Timestamp Policy

AuthRing 的标准输出频率固定为：

```text
freq = 200.0
```

因此：

```text
10s mode:  T = ceil(200.0 * 10) = 2000
1000f mode: T = 1000
```

所有输出片段必须包含真实片段内时间戳：

```text
has_timestamp = true
```

时间戳列规则：

```text
time_ms[0] = 0
time_ms[i] = 当前样本相对片段起点的毫秒时间
```

如果某个 session 无法可靠恢复对应 device 的时间轴，则该 device 的该 session 不得进入最终 processed 数据。

### Phone Alignment

Phone acc 和 gyro 都应转换到同一个 200 Hz 时间网格。处理要求：

- 使用 `phone_acc.csv` 和 `phone_gyro.csv` 的 timestamp 建立手机时间轴。
- 对 acc 和 gyro 分别插值或对齐到同一组 200 Hz `time_ms`。
- 输出的 `acc[:, 0]` 和 `gyro[:, 0]` 必须一致。
- 如果 timestamp 非单调，应先按 timestamp 排序，并记录为处理脚本的检查项。

### Ring Alignment

`ring_imu.csv` 已经同时包含戒指侧时间戳、加速度和角速度。由于标准训练数据按 device 独立保存，AuthRing 的 ring 片段不需要转换到手机时间轴，也不需要与 phone acc/gyro 做同一时刻对齐。

注意：`ring_imu.csv` 中的 `timestamp` 原始单位不是毫秒，而是戒指侧 tick。处理脚本必须先转换为毫秒：

```python
ring_time_ms = np.asarray(timestamp_ticks, dtype=np.float64) / 16384.0 * 1000.0
```

处理要求：

- ring 使用 `ring_imu.csv` 自带的 `timestamp` 转换得到的 `ring_time_ms` 作为戒指侧时间轴。
- ring 的 acc 和 gyro 已在同一行内同步，因此不需要额外做 acc/gyro 时间对齐。
- ring 输出必须基于戒指侧时间轴重采样到 200 Hz 时间网格。
- ring 片段的 `time_ms` 必须从 `0` 开始。
- 如果 `ring_time_ms` 非单调，应先按 `ring_time_ms` 排序，并记录为处理脚本的检查项。
- 如果无法基于 `ring_time_ms` 稳定生成 200 Hz 时间网格，则不生成该 session 的 ring 片段。

`calibration.json`、`post_calibration.json` 和 `time_calib.csv` 用于跨设备时间对齐任务；当前标准训练数据不要求 ring 与 phone 对齐，因此 AuthRing 的 ring 片段生成流程不依赖这些文件。

## Unit And Axis Policy

输出单位必须满足根目录 `README.md`：

```text
acc[:, 1:4]  -> m/s^2
gyro[:, 1:4] -> rad/s
```

AuthRing 样例数据中的加速度数值量级接近 `m/s^2`，陀螺仪数值量级接近 `rad/s`。处理脚本仍必须保留单位检查逻辑，不能只依赖字段名。

输出坐标系必须是右手系 `x/y/z`。如果确认 AuthRing 原始坐标已经符合标准，则不做轴变换；如果后续确认需要变换，必须在本 README 中补充明确的轴映射或变换矩阵后再处理。

## Session Filtering

每个 session 至少需要满足：

- `session.json` 存在并可解析。
- `status == "COMPLETED"`。
- 对应 device 所需的 IMU 文件存在。
- 对应 device 的时间轴可以恢复到 200 Hz。
- 对应 device 的 acc/gyro 数值有限，不包含 `NaN` 或 `Inf`。

### CE56 Ring Exclusion

AuthRing 说明文档中标记 CE56 戒指数据为坏数据。

如果 `session.json` 中的 `ringDeviceId` 包含：

```text
BCL603CE56
F8:BC:9A:CD:CE:56
```

处理规则为：

```text
不生成该 session 的 ring 片段。
phone 片段仍可生成，只要 phone 数据通过检查。
```

也就是说，CE56 只排除 `device = ring`，不排除整个 session。

## Segment Generation

AuthRing 的 label 使用佩戴手指，因此允许跨 stage 切片。

切片范围：

- 以每个 session 的可用 IMU 时间范围为基础。
- `phone` 按 phone acc/gyro 的共同有效时间范围切片。
- `ring` 按 ring 可恢复的有效时间范围切片。
- 不要求 stage 边界对齐。
- 不要求 trial 边界对齐。

切片规则：

```text
10s mode:
  T = 2000
  non-overlapping windows
  drop tail shorter than 2000 frames
  no padding

1000f mode:
  T = 1000
  non-overlapping windows
  drop tail shorter than 1000 frames
  no padding
```

## Label Policy

AuthRing 的主 label 为戒指佩戴手指。

`label.raw` 必须来自 `session.json` 的：

```text
ringWearingFinger
```

示例：

```json
{"raw":"右手无名指"}
```

推荐同时保存标准化手指标签：

| 原始值 | 推荐 `label.finger` |
| --- | --- |
| `右手拇指` | `right_thumb` |
| `右手食指` | `right_index` |
| `右手中指` | `right_middle` |
| `右手无名指` | `right_ring` |
| `右手小指` | `right_little` |

`label` 只保存手指标签，不保存 stage、participant、session 或设备 ID。

推荐格式：

```json
{
  "raw": "右手无名指",
  "finger": "right_ring"
}
```

如果某个片段跨越多个 stage，不需要把 stage 写入 `label.raw`，也不要把 stage 信息写入 `label`。

## Output Layout

AuthRing 输出目录：

```text
/Volumes/Felix_Backups/Processed/
├── sources/
│   └── authring/
│       ├── segments/
│       │   ├── 10s/
│       │   │   ├── ring/
│       │   │   └── phone/
│       │   └── 1000f/
│       │       ├── ring/
│       │       └── phone/
│       └── manifests/
│           ├── 10s.jsonl
│           └── 1000f.jsonl
└── manifests/
    ├── all_10s.jsonl
    ├── all_1000f.jsonl
    ├── phone_10s.jsonl
    ├── phone_1000f.jsonl
    ├── ring_10s.jsonl
    └── ring_1000f.jsonl
```

片段路径示例：

```text
sources/authring/segments/10s/ring/00000001.npz
sources/authring/segments/1000f/phone/00000001.npz
```

manifest entry 示例：

```json
{"dir":"sources/authring/segments/10s/ring/00000001.npz","src":"authring","device":"ring","freq":200.0,"mode":"10s","num_frames":2000,"duration_sec":10.0,"has_timestamp":true,"has_gyro":true,"label":{"raw":"右手无名指","finger":"right_ring"}}
```

## Validation Before Acceptance

AuthRing 数据进入 `/Volumes/Felix_Backups/Processed` 前必须检查：

- 每个 manifest entry 的 `src == "authring"`。
- `device` 只能是 `ring` 或 `phone`。
- `freq == 200.0`。
- `has_timestamp == true`。
- `has_gyro == true`。
- `10s` 片段 `num_frames == 2000`。
- `1000f` 片段 `num_frames == 1000`。
- `.npz` 内包含 `acc` 和 `gyro`。
- `acc` 和 `gyro` 都是 `float32`。
- `acc.shape == gyro.shape == [num_frames, 4]`。
- `acc[:, 0]` 和 `gyro[:, 0]` 从 `0` 开始，单位为 ms。
- `acc[:, 1:4]` 和 `gyro[:, 1:4]` 全部为有限值。
- CE56 session 不产生 ring 片段。
- 每个片段的 `label.raw` 来自 `session.json:ringWearingFinger`。

如果任一检查失败，对应 session/device/mode 不得进入最终 processed 数据。
