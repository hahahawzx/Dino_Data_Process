# AuthRing Daily Processing Notes

本目录用于处理 AuthRing 日常采集数据。该数据和 `AuthRing/` 原始处理流程使用同一类采集算法，但目录结构没有 `session.json`，因此不能读取原数据集中的手指、stage 或 session 元信息。

原始数据默认位置：

```text
/Volumes/Felix_Backups/Root26.5.22/科研与科创/数据记录/AuthRing/日常采集数据-解压
```

处理后 source id：

```text
authring_daily
```

## Input Structure

读取单位是：

```text
person/session_time/
```

每个 session 通常包含：

```text
ring_imu.csv
phone_acc.csv
phone_gyro.csv
touch.csv
time_calib.csv
calibration.json
```

当前处理只使用 IMU 文件。

## Processing Policy

- `ring` 和 `phone` 分开处理，分别生成 manifest。
- 输出频率统一为 `200 Hz`。
- 平均原始帧率必须 `>= 150 Hz`。
- 保存的每个 segment 覆盖到的所有 1 秒原始 bin 都必须 `>= 150` 帧。
- 使用一阶线性插值重采样到 200 Hz。
- `10s` 模式输出 `2000` 帧。
- `1000f` 模式输出 `1000` 帧。
- label 统一写为 `{"finger":"right_index"}`。

## Commands

Ring:

```bash
python3 AuthRing_Daily/process_ring.py \
  --dataset-root "/Volumes/Felix_Backups/Root26.5.22/科研与科创/数据记录/AuthRing/日常采集数据-解压" \
  --processed-root "/Volumes/Felix_Backups/Processed" \
  --overwrite
```

Phone:

```bash
python3 AuthRing_Daily/process_phone.py \
  --dataset-root "/Volumes/Felix_Backups/Root26.5.22/科研与科创/数据记录/AuthRing/日常采集数据-解压" \
  --processed-root "/Volumes/Felix_Backups/Processed" \
  --overwrite
```
