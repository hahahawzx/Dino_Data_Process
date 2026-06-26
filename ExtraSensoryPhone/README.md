# ExtraSensoryPhone Processing Notes

ExtraSensoryPhone 是写入 `/Volumes/Felix_Backups/Processed2` 的手机 IMU 数据源。这个数据源原始采样率明显低于 WaveHand，因此使用独立的数据源规则。

## Raw Data

原始数据根目录：

```text
/Volumes/Felix_Backups/ExtraSensoryPhone
```

当前看到两个信号目录：

```text
raw_acc/{uuid}/{timestamp}.m_raw_acc.dat
proc_gyro/{uuid}/{timestamp}.m_proc_gyro.dat
```

`.dat` 是 ASCII 文本，每行 4 列，空格分隔：

```text
time x y z
```

只处理 `raw_acc` 和 `proc_gyro` 都存在的 `{uuid}/{timestamp}` 文件对。只有 acc、没有 gyro 的文件暂不进入本 source。

## Source Layout

输出 source ID 使用：

```text
ExtraSensoryPhone
```

设备类型固定为：

```text
phone
```

输出路径：

```text
/Volumes/Felix_Backups/Processed2/sources/ExtraSensoryPhone/
```

片段路径：

```text
sources/ExtraSensoryPhone/segments/10s/phone/{segment_id}.npz
sources/ExtraSensoryPhone/segments/512f/phone/{segment_id}.npz
sources/ExtraSensoryPhone/segments/1024f/phone/{segment_id}.npz
```

source manifest：

```text
sources/ExtraSensoryPhone/manifests/phone_10s.jsonl
sources/ExtraSensoryPhone/manifests/phone_512f.jsonl
sources/ExtraSensoryPhone/manifests/phone_1024f.jsonl
```

## Units

`raw_acc` 的单位不是全局一致，必须按文件判断。判断依据是当前文件 acc 三轴模长的中位数：

| Median acceleration magnitude | Interpretation | Output conversion |
| --- | --- | --- |
| `0.7 <= median <= 1.5` | `g` | multiply by `9.80665` |
| `6.0 <= median <= 15.0` | `m/s^2` | keep unchanged |
| otherwise | anomalous or unclear | skip file |

`proc_gyro` 当前按已经是 `rad/s` 处理，不再做 `deg/s -> rad/s` 转换。

所有输出 `.npz` 都必须满足：

| Key | Shape | Dtype | Columns |
| --- | --- | --- | --- |
| `acc` | `[T, 4]` | `float32` | `time_ms, acc_x, acc_y, acc_z` |
| `gyro` | `[T, 4]` | `float32` | `time_ms, gyro_x, gyro_y, gyro_z` |

## Time Alignment

每个文件对内部使用 acc 和 gyro 的时间交集，不使用绝对时间。处理流程：

1. 读取 acc 和 gyro `.dat`。
2. 丢弃非有限行。
3. 分别按时间排序。
4. 重复时间戳只保留第一条。
5. 取 acc 和 gyro 的重叠时间区间。
6. 在重叠区间内做质量检查。
7. 使用一阶线性插值统一到 `40.0 Hz`。
8. 输出片段内 `time_ms` 从 `0` 开始。

目标时间间隔：

```text
1000 / 40 = 25 ms
```

## Quality Rules

ExtraSensoryPhone 的原始采样率较低，典型情况是：

```text
raw_acc   ~= 34-35 Hz
proc_gyro ~= 40 Hz
```

因此片段级质量检查使用下面的 raw bin 阈值：

```text
acc  >= 30 raw frames / second
gyro >= 35 raw frames / second
```

片段覆盖到的所有 1 秒 raw bin 都必须满足对应阈值，否则不保存该片段。

## Segment Modes

ExtraSensoryPhone 输出三种模式：

| Mode | Target freq | Frames | Duration | Padding |
| --- | ---: | ---: | ---: | --- |
| `10s` | `40.0 Hz` | `400` | `10.0 s` | no |
| `512f` | `40.0 Hz` | `512` | `12.8 s` | no |
| `1024f` | `40.0 Hz` | `1024` | `25.6 s` | yes, tail padding |

`10s` 和 `512f` 不做 padding，不足目标长度则丢弃。

`1024f` 是本数据源的特殊模式：从同一个有效重叠区取最多 `1024` 帧真实数据；如果真实数据不足 `1024` 帧，则尾部补 0 到 `1024` 帧。为了避免 padding 比例过高，真实有效帧数低于 `512` 时不生成 `1024f`。padding 行的 `time_ms` 保持规则 40 Hz 时间网格继续递增，`acc/gyro` 的 xyz 值填 `0`。

manifest 的 `label` 中记录补帧数量：

```json
{"pad_frames": 225}
```

对于非 padding 模式，`pad_frames` 固定为 `0`：

```json
{"pad_frames": 0}
```

## Manifest Example

```json
{"dir":"sources/ExtraSensoryPhone/segments/1024f/phone/00000001.npz","src":"ExtraSensoryPhone","device":"phone","freq":40.0,"mode":"1024f","num_frames":1024,"duration_sec":25.6,"has_timestamp":true,"has_gyro":true,"label":{"pad_frames":225}}
```

## Inspection Notebook

单文件调试 notebook：

```text
ExtraSensoryPhone/inspect_extrasensory_phone_processing.ipynb
```

建议使用 `tsai_env` 运行。该 notebook 默认检查：

```text
0A986513-7828-4D53-AA1F-E02D6DF9561B/1449601597
```

样例输出写到：

```text
ExtraSensoryPhone/inspect_outputs/
```

`inspect_outputs/` 只用于人工检查，不是正式 processed 数据。

## Full Processing Script

完整处理脚本：

```text
ExtraSensoryPhone/process_extrasensory_phone.py
```

正式重建 ExtraSensoryPhone source：

```bash
conda activate tsai_env
python ExtraSensoryPhone/process_extrasensory_phone.py \
  --dataset-root "/Volumes/Felix_Backups/ExtraSensoryPhone" \
  --processed-root "/Volumes/Felix_Backups/Processed2" \
  --overwrite
```

小范围试跑可以指定一个或多个 exact key：

```bash
conda activate tsai_env
python ExtraSensoryPhone/process_extrasensory_phone.py \
  --dataset-root "/Volumes/Felix_Backups/ExtraSensoryPhone" \
  --processed-root "ExtraSensoryPhone/process_smoke_output" \
  --overwrite \
  --keys "0A986513-7828-4D53-AA1F-E02D6DF9561B/1449601597" \
  --progress-every 1
```

也可以按 UUID 或文件数限制：

```bash
conda activate tsai_env
python ExtraSensoryPhone/process_extrasensory_phone.py \
  --dataset-root "/Volumes/Felix_Backups/ExtraSensoryPhone" \
  --processed-root "ExtraSensoryPhone/process_smoke_output" \
  --overwrite \
  --uuids "0A986513-7828-4D53-AA1F-E02D6DF9561B" \
  --limit-files 10 \
  --progress-every 1
```
