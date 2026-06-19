"""Build global manifests, summary stats, and sample visualizations.

This script reads source manifests from:
  processed/sources/{src}/manifests/{device}_{mode}.jsonl

and writes derived files into:
  processed/manifests/
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROCESSED_ROOT = Path("/Volumes/Felix_Backups/Processed")
DEVICES = ("phone", "watch", "ring", "other")
MODES = ("10s", "1000f")
REQUIRED_FIELDS = (
    "dir",
    "src",
    "device",
    "freq",
    "mode",
    "num_frames",
    "duration_sec",
    "has_timestamp",
    "has_gyro",
    "label",
)
COLORS = {
    "acc_x": "#2563eb",
    "acc_y": "#16a34a",
    "acc_z": "#dc2626",
    "|acc|": "#7c3aed",
    "gyro_x": "#0f766e",
    "gyro_y": "#ea580c",
    "gyro_z": "#9333ea",
    "|gyro|": "#111827",
}


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(description="Build derived processed manifests and visualizations.")
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--samples-per-source-mode", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip malformed manifest entries instead of failing.",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Only build JSONL manifests and summary.json; do not generate SVG samples.",
    )
    return parser.parse_args()


def manifest_root(processed_root: Path) -> Path:
    """Return processed/manifests."""

    return processed_root / "manifests"


def source_manifest_paths(processed_root: Path) -> list[Path]:
    """Find all source manifests that match {device}_{mode}.jsonl."""

    paths = []
    sources_root = processed_root / "sources"
    if not sources_root.exists():
        raise FileNotFoundError(f"missing sources directory: {sources_root}")
    for src_dir in sorted(path for path in sources_root.iterdir() if path.is_dir()):
        manifests_dir = src_dir / "manifests"
        if not manifests_dir.exists():
            continue
        for device in DEVICES:
            for mode in MODES:
                path = manifests_dir / f"{device}_{mode}.jsonl"
                if path.exists():
                    paths.append(path)
    return paths


def open_derived_writers(root: Path) -> dict[str, Any]:
    """Open all derived manifest writers."""

    root.mkdir(parents=True, exist_ok=True)
    writers = {}
    for mode in MODES:
        writers[f"all_{mode}"] = (root / f"all_{mode}.jsonl").open("w", encoding="utf-8")
        for device in DEVICES:
            writers[f"{device}_{mode}"] = (root / f"{device}_{mode}.jsonl").open("w", encoding="utf-8")
    return writers


def close_writers(writers: dict[str, Any]) -> None:
    """Close manifest writers."""

    for writer in writers.values():
        writer.close()


def source_name_from_manifest(path: Path) -> str:
    """Return {src} from processed/sources/{src}/manifests/file.jsonl."""

    return path.parent.parent.name


def parse_manifest_name(path: Path) -> tuple[str, str]:
    """Return expected device/mode from a source manifest filename."""

    match = re.fullmatch(r"(phone|watch|ring|other)_(10s|1000f)\.jsonl", path.name)
    if not match:
        raise ValueError(f"unexpected source manifest name: {path}")
    return match.group(1), match.group(2)


def compact_json(data: Any) -> str:
    """Return compact JSON with Unicode preserved."""

    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def validate_entry(entry: dict, source_manifest: Path, line_number: int) -> str | None:
    """Return None if valid, otherwise return an error reason."""

    expected_src = source_name_from_manifest(source_manifest)
    expected_device, expected_mode = parse_manifest_name(source_manifest)

    for field in REQUIRED_FIELDS:
        if field not in entry:
            return f"missing field {field}"
    if entry["src"] != expected_src:
        return f"src mismatch: {entry['src']} != {expected_src}"
    if entry["device"] != expected_device:
        return f"device mismatch at {source_manifest.name}:{line_number}"
    if entry["mode"] != expected_mode:
        return f"mode mismatch at {source_manifest.name}:{line_number}"
    if entry["device"] not in DEVICES:
        return f"invalid device: {entry['device']}"
    if entry["mode"] not in MODES:
        return f"invalid mode: {entry['mode']}"
    if not isinstance(entry["label"], dict):
        return "label is not object"
    if Path(entry["dir"]).is_absolute():
        return "dir is absolute"
    if not str(entry["dir"]).startswith(f"sources/{entry['src']}/segments/{entry['mode']}/{entry['device']}/"):
        return f"dir does not match src/mode/device: {entry['dir']}"
    return None


def make_empty_stats() -> dict:
    """Create the mutable summary stats structure."""

    return {
        "total": 0,
        "by_mode": {mode: 0 for mode in MODES},
        "by_device": {device: 0 for device in DEVICES},
        "by_source": {},
        "by_source_mode": {},
        "by_source_device_mode": {},
        "invalid": {
            "total": 0,
            "by_source": {},
            "reasons": {},
            "examples": [],
        },
    }


def add_invalid(stats: dict, source: str, reason: str, source_manifest: Path, line_number: int) -> None:
    """Record one skipped invalid manifest entry."""

    invalid = stats["invalid"]
    invalid["total"] += 1
    invalid["by_source"][source] = invalid["by_source"].get(source, 0) + 1
    invalid["reasons"][reason] = invalid["reasons"].get(reason, 0) + 1
    if len(invalid["examples"]) < 50:
        invalid["examples"].append(
            {
                "source_manifest": str(source_manifest),
                "line_number": line_number,
                "reason": reason,
            }
        )


def add_valid_stats(stats: dict, entry: dict) -> None:
    """Update summary stats for one valid entry."""

    src = entry["src"]
    device = entry["device"]
    mode = entry["mode"]

    stats["total"] += 1
    stats["by_mode"][mode] += 1
    stats["by_device"][device] += 1
    stats["by_source"][src] = stats["by_source"].get(src, 0) + 1

    stats["by_source_mode"].setdefault(src, {mode_name: 0 for mode_name in MODES})
    stats["by_source_mode"][src][mode] += 1

    stats["by_source_device_mode"].setdefault(src, {})
    stats["by_source_device_mode"][src].setdefault(device, {mode_name: 0 for mode_name in MODES})
    stats["by_source_device_mode"][src][device][mode] += 1


def write_derived_entry(writers: dict[str, Any], entry: dict) -> None:
    """Write one entry to all-mode and device-mode manifests."""

    line = compact_json(entry) + "\n"
    writers[f"all_{entry['mode']}"].write(line)
    writers[f"{entry['device']}_{entry['mode']}"].write(line)


def reservoir_add(samples: dict, seen: dict, entry: dict, samples_per_key: int, rng: random.Random) -> None:
    """Reservoir-sample entries per (src, mode)."""

    key = (entry["src"], entry["mode"])
    seen[key] += 1
    bucket = samples[key]
    if len(bucket) < samples_per_key:
        bucket.append(entry.copy())
        return
    replace_idx = rng.randrange(seen[key])
    if replace_idx < samples_per_key:
        bucket[replace_idx] = entry.copy()


def build_manifests(args: argparse.Namespace) -> tuple[dict, dict]:
    """Build JSONL manifests and return summary stats plus sampled entries."""

    root = manifest_root(args.processed_root)
    paths = source_manifest_paths(args.processed_root)
    writers = open_derived_writers(root)
    rng = random.Random(args.seed)
    stats = make_empty_stats()
    samples: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen_for_sampling: dict[tuple[str, str], int] = defaultdict(int)

    try:
        for path in paths:
            src = source_name_from_manifest(path)
            print(f"Read {path}", flush=True)
            with path.open("r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as exc:
                        if args.skip_invalid:
                            add_invalid(stats, src, f"json decode error: {exc.msg}", path, line_number)
                            continue
                        raise ValueError(f"invalid JSON in {path}:{line_number}") from exc

                    reason = validate_entry(entry, path, line_number)
                    if reason is not None:
                        if args.skip_invalid:
                            add_invalid(stats, src, reason, path, line_number)
                            continue
                        raise ValueError(f"{path}:{line_number}: {reason}")

                    add_valid_stats(stats, entry)
                    write_derived_entry(writers, entry)
                    reservoir_add(samples, seen_for_sampling, entry, args.samples_per_source_mode, rng)
    finally:
        close_writers(writers)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "processed_root": str(args.processed_root),
        "source_manifests": [str(path) for path in paths],
        "samples_per_source_mode": args.samples_per_source_mode,
        **stats,
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary, samples


def slugify(value: str) -> str:
    """Convert text into a filename-safe slug."""

    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return value or "unknown"


def svg_text(x: float, y: float, value: str, size: int = 12, weight: int = 400, anchor: str = "start") -> str:
    """Return an SVG text element."""

    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-family="Arial, sans-serif" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="#24292f">{html.escape(value)}</text>'
    )


def y_range(values: np.ndarray) -> tuple[float, float]:
    """Return a padded robust y-range."""

    if len(values) == 0 or not np.all(np.isfinite(values)):
        return -1.0, 1.0
    lo = float(np.percentile(values, 1))
    hi = float(np.percentile(values, 99))
    if hi <= lo:
        pad = max(abs(hi), 1.0) * 0.1
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.12
    return lo - pad, hi + pad


def downsample(time_ms: np.ndarray, values: np.ndarray, max_points: int = 900) -> tuple[np.ndarray, np.ndarray]:
    """Downsample for compact SVG rendering."""

    if len(values) <= max_points:
        return time_ms, values
    idx = np.linspace(0, len(values) - 1, max_points).astype(np.int64)
    return time_ms[idx], values[idx]


def signal_points(time_ms: np.ndarray, values: np.ndarray, x: float, y: float, w: float, h: float) -> str:
    """Map signal data into SVG polyline points."""

    time_ms, values = downsample(time_ms, values)
    ymin, ymax = y_range(values)
    denom_t = max(float(time_ms[-1] - time_ms[0]), 1.0)
    denom_y = max(ymax - ymin, 1e-9)
    pts = []
    for t, value in zip(time_ms, values):
        px = x + w * float(t - time_ms[0]) / denom_t
        clipped = min(max(float(value), ymin), ymax)
        py = y + h - h * (clipped - ymin) / denom_y
        pts.append(f"{px:.2f},{py:.2f}")
    return " ".join(pts)


def panel_svg(name: str, time_ms: np.ndarray, values: np.ndarray, x: float, y: float, w: float, h: float) -> str:
    """Render one signal panel."""

    ymin, ymax = y_range(values)
    color = COLORS.get(name, "#2563eb")
    parts = [
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="#ffffff" stroke="#d0d7de"/>',
        svg_text(x + 8, y + 17, name, 12, 700),
    ]
    for i in range(4):
        yy = y + h * i / 3
        parts.append(f'<line x1="{x:.2f}" y1="{yy:.2f}" x2="{x + w:.2f}" y2="{yy:.2f}" stroke="#eef2f6"/>')
    parts.append(svg_text(x - 8, y + 12, f"{ymax:.2f}", 10, 400, "end"))
    parts.append(svg_text(x - 8, y + h, f"{ymin:.2f}", 10, 400, "end"))
    parts.append(
        f'<polyline points="{signal_points(time_ms, values, x, y, w, h)}" '
        f'fill="none" stroke="{color}" stroke-width="1.25"/>'
    )
    return "\n".join(parts)


def wrap_label_text(value: str, width: int = 120) -> list[str]:
    """Split long JSON label text for SVG display."""

    return [value[i : i + width] for i in range(0, len(value), width)] or [""]


def render_segment_svg(processed_root: Path, entry: dict) -> str:
    """Render one manifest entry as an SVG image."""

    npz_path = processed_root / entry["dir"]
    with np.load(npz_path) as data:
        acc = data["acc"].astype(np.float64, copy=False)
        gyro = data["gyro"].astype(np.float64, copy=False) if "gyro" in data.files else None

    time_ms = acc[:, 0]
    signals = [
        ("acc_x", acc[:, 1]),
        ("acc_y", acc[:, 2]),
        ("acc_z", acc[:, 3]),
        ("|acc|", np.linalg.norm(acc[:, 1:4], axis=1)),
    ]
    if gyro is not None:
        signals.extend(
            [
                ("gyro_x", gyro[:, 1]),
                ("gyro_y", gyro[:, 2]),
                ("gyro_z", gyro[:, 3]),
                ("|gyro|", np.linalg.norm(gyro[:, 1:4], axis=1)),
            ]
        )

    panel_h = 70
    panel_gap = 18
    top = 118
    width = 1160
    height = top + len(signals) * (panel_h + panel_gap) + 34
    x = 86
    w = 1020

    label_json = compact_json(entry.get("label", {}))
    title = f"{entry['src']} / {entry['mode']} / {entry['device']} / {Path(entry['dir']).name}"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(26, 32, title, 17, 700),
        svg_text(26, 56, f"dir: {entry['dir']}", 12),
        svg_text(26, 78, f"freq={entry['freq']}Hz frames={entry['num_frames']} duration={entry['duration_sec']}s has_gyro={entry['has_gyro']}", 12),
    ]
    for idx, line in enumerate(wrap_label_text(f"label: {label_json}")):
        parts.append(svg_text(26, 100 + idx * 16, line, 12))
    y0 = top + max(0, len(wrap_label_text(f"label: {label_json}")) - 1) * 16
    for idx, (name, values) in enumerate(signals):
        parts.append(panel_svg(name, time_ms, values, x, y0 + idx * (panel_h + panel_gap), w, panel_h))
    parts.append(svg_text(x + w / 2, height - 16, "segment-local time (ms)", 12, 400, "middle"))
    parts.append("</svg>")
    return "\n".join(parts)


def prepare_visualization_root(root: Path) -> None:
    """Clear and recreate processed/manifests/visualization."""

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)


def generate_visualizations(processed_root: Path, samples: dict[tuple[str, str], list[dict]]) -> dict:
    """Generate SVG visualizations under processed/manifests/visualization."""

    visualization_root = manifest_root(processed_root) / "visualization"
    prepare_visualization_root(visualization_root)
    counts = {}
    skipped = []
    for (src, mode), entries in sorted(samples.items()):
        out_dir = visualization_root / src / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        counts.setdefault(src, {})[mode] = 0
        for idx, entry in enumerate(entries, start=1):
            label_slug = slugify(compact_json(entry.get("label", {})))[:40]
            filename = f"{idx:02d}_{entry['device']}_{Path(entry['dir']).stem}_{label_slug}.svg"
            try:
                svg = render_segment_svg(processed_root, entry)
            except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
                skipped.append({"dir": entry.get("dir"), "reason": str(exc)})
                continue
            (out_dir / filename).write_text(svg, encoding="utf-8")
            counts[src][mode] += 1
    return {"counts": counts, "skipped": skipped}


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    if args.samples_per_source_mode <= 0:
        raise ValueError("--samples-per-source-mode must be positive")
    summary, samples = build_manifests(args)
    if not args.no_visualization:
        visualization = generate_visualizations(args.processed_root, samples)
        summary["visualization"] = {
            "root": str(manifest_root(args.processed_root) / "visualization"),
            **visualization,
        }
        (manifest_root(args.processed_root) / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
