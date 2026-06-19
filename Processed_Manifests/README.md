# Processed Manifest Builder

本目录用于从 `processed/sources/*/manifests/` 下的 source manifest 生成训练入口文件和抽样可视化。

默认 processed 根目录：

```text
/Volumes/Felix_Backups/Processed
```

生成内容：

```text
processed/manifests/
├── all_10s.jsonl
├── all_1000f.jsonl
├── phone_10s.jsonl
├── phone_1000f.jsonl
├── watch_10s.jsonl
├── watch_1000f.jsonl
├── ring_10s.jsonl
├── ring_1000f.jsonl
├── other_10s.jsonl
├── other_1000f.jsonl
├── summary.json
└── visualization/
    └── {src}/
        ├── 10s/*.svg
        └── 1000f/*.svg
```

常用命令：

```bash
python3 Processed_Manifests/build_manifests.py \
  --processed-root "/Volumes/Felix_Backups/Processed" \
  --skip-invalid
```

`--skip-invalid` 会跳过 manifest 中字段格式错误的条目，并在 `summary.json` 中记录跳过数量和示例。构建全局 manifest 时默认信任每个 source manifest，不逐条检查 `.npz` 文件是否存在；只有抽样可视化时才读取少量 `.npz`。

每个 `{src, mode}` 默认抽样 10 个片段生成 SVG：

```bash
python3 Processed_Manifests/build_manifests.py \
  --processed-root "/Volumes/Felix_Backups/Processed" \
  --samples-per-source-mode 10 \
  --skip-invalid
```

只重建 JSONL 和 `summary.json`，不生成图片：

```bash
python3 Processed_Manifests/build_manifests.py \
  --processed-root "/Volumes/Felix_Backups/Processed" \
  --skip-invalid \
  --no-visualization
```
