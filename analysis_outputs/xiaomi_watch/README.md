# Xiaomi Watch IMU Sample Notes

本目录只保留一次小米手表 IMU 样例数据的快速分析结果，用来判断原始数据大致是否可用。

## Files

```text
per_second_frame_count.svg
magnitude_trace.svg
```

- `per_second_frame_count.svg`: 每秒帧数图，用来观察 200 Hz 采样是否稳定，以及是否有低于 150 fps 的秒。
- `magnitude_trace.svg`: acc 和 gyro 模长图，用来观察数值量级和整体分布。

## Observed Format

本次样例包含两路 IMU CSV：

```text
accel-200hz.csv
gyroscope-200hz.csv
```

观测到的字段结构：

```text
CurrentTimestamp(ms), EventTimestamp(ms), x, y, z
```

其中 `EventTimestamp(ms)` 更适合作为采样时间轴；`CurrentTimestamp(ms)` 存在重复值，不适合直接用于帧率统计。

## Unit Observation

本次样例中：

- acc 模长中位数约为 `9.8`，因此加速度看起来已经是 `m/s^2`，不是 `g`。
- gyro 数值量级符合手表/Android gyroscope 的 `rad/s` 常见输出。

这些结论只针对当前样例；后续新数据仍应先做同样的单位和量级检查。

## Frame-Rate Observation

目标采样率为 `200 Hz`，即相邻帧期望间隔为 `5 ms`。

按完整 1 秒窗口统计：

| Signal | Full Seconds | Seconds `<150 fps` | Ratio |
| --- | ---: | ---: | ---: |
| acc | 13,207 | 1 | 0.00757% |
| gyro | 13,206 | 0 | 0% |

本次样例的帧率整体稳定。acc 的低帧率秒出现在开头，后续主体质量较好。

## Current Caveat

当前只做了单位量级和帧率稳定性观察，尚未确认物理坐标轴是否已经满足统一右手系定义。
