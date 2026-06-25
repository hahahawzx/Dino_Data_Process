# WaveHand Processing Notes

WaveHand 是写入 `/Volumes/Felix_Backups/Processed2` 的新批次数据源之一。`Processed2` 的全局切片模式固定为 `10s` + `1024f`；WaveHand 这个数据源的目标频率固定为 `150.0 Hz`。

## Raw Data

原始数据根目录：

```text
/Users/zixuanwang/Desktop/HiSync_publish_anonymous
```

每个数字目录表示一个 group，例如：

```text
/Users/zixuanwang/Desktop/HiSync_publish_anonymous/1
```

当前统计到的 group 为：

```text
1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
```

当前原始目录中未看到 `4` 和 `7`。

只处理以下两个设备目录：

| Raw directory | Processed device |
| --- | --- |
| `IMU/IMU_Ring` | `ring` |
| `IMU/IMU_Wrist` | `watch` |

`IMU/IMU_Palm` 暂不进入本次 processed 数据。每个设备目录下只处理 `calibrated_imu_*.csv`，不处理其他原始或临时 CSV。

## Source Layout

输出 source ID 使用：

```text
WaveHand
```

输出路径：

```text
/Volumes/Felix_Backups/Processed2/sources/WaveHand/
```

片段路径：

```text
sources/WaveHand/segments/10s/{device}/{segment_id}.npz
sources/WaveHand/segments/1024f/{device}/{segment_id}.npz
```

source manifest：

```text
sources/WaveHand/manifests/ring_10s.jsonl
sources/WaveHand/manifests/ring_1024f.jsonl
sources/WaveHand/manifests/watch_10s.jsonl
sources/WaveHand/manifests/watch_1024f.jsonl
```

## Signal Fields

CSV 中读取以下字段：

| Signal | CSV fields | Output unit |
| --- | --- | --- |
| acceleration | `accel_with_gravity_x`, `accel_with_gravity_y`, `accel_with_gravity_z` | `m/s^2` |
| gyroscope | `gyro_x`, `gyro_y`, `gyro_z` | `rad/s` |

`accel_with_gravity_*` 当前已经是国际单位制 `m/s^2`，不再乘以 `9.80665`。`gyro_*` 当前单位是 `deg/s`，写入前必须转换为 `rad/s`：

```python
gyro_rad_s = gyro_deg_s * np.pi / 180.0
```

`.npz` 中写入：

| Key | Shape | Dtype | Columns |
| --- | --- | --- | --- |
| `acc` | `[T, 4]` | `float32` | `time_ms, acc_x, acc_y, acc_z` |
| `gyro` | `[T, 4]` | `float32` | `time_ms, gyro_x, gyro_y, gyro_z` |

`time_ms` 是片段内部从 `0` 开始的毫秒时间戳。WaveHand 有可用时间戳，因此 manifest 中 `has_timestamp` 为 `true`，`has_gyro` 为 `true`。

## Time And Resampling

帧率统计和重采样时间轴优先使用 `deviceTimestamp`。`systemTimestamp` 可能存在重复或非单调，不作为传感器帧率依据。

处理流程：

1. 读取 `calibrated_imu_*.csv`。
2. 丢弃时间戳或 IMU 数值非有限的行。
3. 按时间戳排序。
4. 重复时间戳只保留第一条。
5. 检查原始数据质量。
6. 使用一阶线性插值重采样到 `150.0 Hz`。
7. 不做 extrapolation；目标时间点无法由左右两侧原始样本覆盖时，不保存对应片段。

目标时间间隔为：

```text
1000 / 150 = 6.6666667 ms
```

保存片段时，`time_ms` 必须重新从 `0` 开始。

## Quality Rules

WaveHand 的原始质量阈值为：

```text
120 raw frames / second
```

质量检查分两层：

1. 整文件平均原始帧率低于 `120 Hz` 的 CSV 可以直接跳过。
2. 片段级检查时，片段覆盖到的所有 1 秒 raw bin 都必须满足 `raw frames >= 120`，否则该片段不保存。

进入 `Processed2` 的片段默认已经通过质量检查；不在 processed 目录中保留失败片段或检查报告。

## Segment Modes

WaveHand 输出两种模式：

| Mode | Target freq | Frames | Duration |
| --- | ---: | ---: | ---: |
| `10s` | `150.0 Hz` | `1500` | `10.0 s` |
| `1024f` | `150.0 Hz` | `1024` | `6.8266667 s` |

尾段不足目标帧数时丢弃，不做 padding。默认不使用重叠窗口。

## Manifest Label

WaveHand 暂不写入训练标签：

```json
{"label":{}}
```

group、文件名、采集者等信息不作为训练标签写入 manifest。后续如果需要追溯，可以单独生成调试统计表，而不是混入训练 label。

## Frame Rate Statistics

当前已生成逐文件帧率统计表：

```text
WaveHand/frame_rate_stats.csv
```

统计口径：

- 只统计 `IMU_Ring` 和 `IMU_Wrist`
- 只统计 `calibrated_imu_*.csv`
- 使用 `deviceTimestamp` 计算平均帧率和相邻帧间隔
- `below_120hz == true` 表示整文件平均原始帧率低于 `120 Hz`

当前结果：

| Device | Files | Avg Hz min | Avg Hz max | Files below 120 Hz |
| --- | ---: | ---: | ---: | ---: |
| `ring` | 25 | 47.121 | 195.620 | 5 |
| `watch` | 28 | 67.746 | 199.073 | 12 |

因此，WaveHand 原始数据不是全部天然 150 Hz。处理结果需要通过线性插值统一到 `150.0 Hz`，并用 `120 raw frames/sec` 阈值筛掉质量不足的文件或片段。

## Inspection Notebook

单文件调试 notebook：

```text
WaveHand/inspect_wavehand_processing.ipynb
```

建议使用 `tsai_env` 运行。该 notebook 默认检查 group 1 的 ring 文件，并把样例输出写到：

```text
WaveHand/inspect_outputs/
```

`inspect_outputs/` 只用于人工检查，不是正式 processed 数据。

## Full Processing Script

完整处理脚本：

```text
WaveHand/process_wavehand.py
```

正式重建 WaveHand source：

```bash
conda activate tsai_env
python WaveHand/process_wavehand.py \
  --dataset-root "/Users/zixuanwang/Desktop/HiSync_publish_anonymous" \
  --processed-root "/Volumes/Felix_Backups/Processed2" \
  --overwrite
```

小范围试跑可以限制 group、device 或文件数：

```bash
conda activate tsai_env
python WaveHand/process_wavehand.py \
  --dataset-root "/Users/zixuanwang/Desktop/HiSync_publish_anonymous" \
  --processed-root "WaveHand/process_smoke_output" \
  --overwrite \
  --groups 1 \
  --devices ring \
  --limit-files 1
```
