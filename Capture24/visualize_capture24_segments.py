"""Visualize sampled Capture24 processed segments by activity label.

The script reads a Capture24 source manifest, samples a few segments for each
label.activity, and writes standalone HTML/SVG files for quick inspection.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import shutil
from pathlib import Path

import numpy as np


PROCESSED_ROOT = Path("/Volumes/Felix_Backups/Processed")
SRC = "capture24"
DEVICE = "watch"
MODES = ("10s", "1000f")
COLORS = {
    "acc_x": "#2563eb",
    "acc_y": "#16a34a",
    "acc_z": "#dc2626",
    "|acc|": "#7c3aed",
}


def parse_args() -> argparse.Namespace:
    """Read visualization command-line arguments."""

    parser = argparse.ArgumentParser(description="Visualize sampled Capture24 processed NPZ segments.")
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--mode", choices=MODES, default="10s")
    parser.add_argument("--per-label", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_output_dir(mode: str) -> Path:
    """Return the default visualization output directory for one mode."""

    return Path(__file__).resolve().parent / "visualizations" / f"capture24_watch_{mode}"


def manifest_path(processed_root: Path, mode: str) -> Path:
    """Return the Capture24 source manifest for one mode."""

    return processed_root / "sources" / SRC / "manifests" / f"{DEVICE}_{mode}.jsonl"


def slugify(value: str) -> str:
    """Convert a label into a stable filename stem."""

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "unknown"


def load_manifest(path: Path) -> list[dict]:
    """Read a JSONL manifest into a list of entries."""

    if not path.exists():
        raise FileNotFoundError(f"missing manifest: {path}")

    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} line {line_number}") from exc
            activity = entry.get("label", {}).get("activity")
            if not activity:
                continue
            entries.append(entry)
    if not entries:
        raise ValueError(f"manifest has no entries with label.activity: {path}")
    return entries


def group_by_activity(entries: list[dict]) -> dict[str, list[dict]]:
    """Group manifest entries by label.activity."""

    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        activity = entry["label"]["activity"]
        grouped.setdefault(activity, []).append(entry)
    return dict(sorted(grouped.items()))


def sample_entries(grouped: dict[str, list[dict]], per_label: int, seed: int) -> dict[str, list[dict]]:
    """Sample up to per_label entries for each activity with deterministic randomness."""

    rng = random.Random(seed)
    sampled = {}
    for activity, entries in grouped.items():
        if len(entries) <= per_label:
            sampled[activity] = list(entries)
        else:
            sampled[activity] = sorted(rng.sample(entries, per_label), key=lambda item: item["dir"])
    return sampled


def prepare_output(path: Path, overwrite: bool) -> None:
    """Create a clean output directory."""

    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output exists; use --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def load_acc(processed_root: Path, entry: dict) -> np.ndarray:
    """Load one processed NPZ segment and return its acc array."""

    path = processed_root / entry["dir"]
    if not path.exists():
        raise FileNotFoundError(f"missing segment file: {path}")
    with np.load(path) as data:
        if "acc" not in data:
            raise ValueError(f"missing acc in segment file: {path}")
        acc = data["acc"]
    if acc.ndim != 2 or acc.shape[1] != 4:
        raise ValueError(f"unexpected acc shape {acc.shape} in {path}")
    return acc.astype(np.float64, copy=False)


def svg_text(x: float, y: float, value: str, size: int = 12, anchor: str = "start", weight: int = 400) -> str:
    """Return an SVG text element."""

    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="#24292f">{html.escape(value)}</text>'
    )


def y_range(values: np.ndarray) -> tuple[float, float]:
    """Return a padded y-range robust to constant signals."""

    lo = float(np.nanpercentile(values, 1))
    hi = float(np.nanpercentile(values, 99))
    if not math.isfinite(lo) or not math.isfinite(hi):
        lo, hi = -1.0, 1.0
    if hi <= lo:
        pad = max(abs(hi), 1.0) * 0.1
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.12
    return lo - pad, hi + pad


def polyline_points(time_ms: np.ndarray, values: np.ndarray, x: float, y: float, w: float, h: float) -> str:
    """Map time/value arrays into SVG polyline points."""

    ymin, ymax = y_range(values)
    denom_t = max(float(time_ms[-1] - time_ms[0]), 1.0)
    denom_y = max(ymax - ymin, 1e-9)
    points = []
    for t, value in zip(time_ms, values):
        px = x + w * float(t - time_ms[0]) / denom_t
        clipped = min(max(float(value), ymin), ymax)
        py = y + h - h * (clipped - ymin) / denom_y
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def panel_svg(time_ms: np.ndarray, values: np.ndarray, name: str, x: float, y: float, w: float, h: float) -> str:
    """Render one signal panel."""

    ymin, ymax = y_range(values)
    color = COLORS[name]
    parts = [
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="#fff" stroke="#d0d7de"/>',
        svg_text(x + 8, y + 17, name, 12, weight=700),
    ]
    for i in range(4):
        yy = y + h * i / 3
        parts.append(f'<line x1="{x:.2f}" y1="{yy:.2f}" x2="{x + w:.2f}" y2="{yy:.2f}" stroke="#eef2f6"/>')
    parts.append(svg_text(x - 8, y + 12, f"{ymax:.2f}", 10, anchor="end"))
    parts.append(svg_text(x - 8, y + h, f"{ymin:.2f}", 10, anchor="end"))
    points = polyline_points(time_ms, values, x, y, w, h)
    parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.25"/>')
    return "\n".join(parts)


def segment_svg(acc: np.ndarray, entry: dict, chart_id: int) -> str:
    """Render one segment as a multi-panel SVG."""

    time_ms = acc[:, 0]
    values = {
        "acc_x": acc[:, 1],
        "acc_y": acc[:, 2],
        "acc_z": acc[:, 3],
        "|acc|": np.linalg.norm(acc[:, 1:4], axis=1),
    }
    width, height = 1080, 520
    x, top, panel_w, panel_h = 78, 72, 950, 82
    title = f"{entry['dir']}  |  frames={entry['num_frames']}  freq={entry['freq']} Hz"
    parts = [
        f'<svg id="chart-{chart_id}" xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(28, 34, title, 16, weight=700),
    ]
    for idx, name in enumerate(("acc_x", "acc_y", "acc_z", "|acc|")):
        parts.append(panel_svg(time_ms, values[name], name, x, top + idx * 100, panel_w, panel_h))
    parts.append(svg_text(x + panel_w / 2, height - 22, "segment-local time (ms)", 12, anchor="middle"))
    parts.append("</svg>")
    return "\n".join(parts)


def activity_page(activity: str, entries: list[dict], processed_root: Path, output_dir: Path) -> dict:
    """Write one HTML page for a sampled activity and return index metadata."""

    slug = slugify(activity)
    html_path = output_dir / f"{slug}.html"
    sections = []
    for idx, entry in enumerate(entries, start=1):
        acc = load_acc(processed_root, entry)
        mag = np.linalg.norm(acc[:, 1:4], axis=1)
        annotations = entry.get("label", {}).get("annotations", [])
        sections.append(
            f"<section>\n"
            f"<h2>Sample {idx}</h2>\n"
            f"<p><code>{html.escape(entry['dir'])}</code></p>\n"
            f"<p>annotations: {html.escape(', '.join(annotations))}</p>\n"
            f"<p>|acc| mean={mag.mean():.4f}, std={mag.std():.4f}, min={mag.min():.4f}, max={mag.max():.4f}</p>\n"
            f"{segment_svg(acc, entry, idx)}\n"
            f"</section>\n"
        )
    page = html_document(f"Capture24 {activity}", "\n".join(sections))
    html_path.write_text(page, encoding="utf-8")
    return {"activity": activity, "file": html_path.name, "count": len(entries)}


def html_document(title: str, body: str) -> str:
    """Wrap body HTML with shared styling."""

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 26px; color: #24292f; }}
a {{ color: #0969da; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
section {{ margin: 24px 0 44px; border-top: 1px solid #d0d7de; padding-top: 18px; }}
code {{ background: #f6f8fa; padding: 2px 5px; border-radius: 4px; }}
table {{ border-collapse: collapse; margin-top: 14px; }}
th, td {{ border: 1px solid #d0d7de; padding: 7px 10px; text-align: left; }}
th {{ background: #f6f8fa; }}
svg {{ display: block; max-width: 100%; height: auto; margin-top: 12px; border: 1px solid #d0d7de; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def index_page(items: list[dict], output_dir: Path, mode: str, manifest: Path, per_label: int) -> None:
    """Write index.html linking to every activity page."""

    rows = "\n".join(
        f'<tr><td>{html.escape(item["activity"])}</td><td>{item["count"]}</td>'
        f'<td><a href="{html.escape(item["file"])}">{html.escape(item["file"])}</a></td></tr>'
        for item in items
    )
    body = f"""
<h1>Capture24 Segment Visualizations</h1>
<p>mode: <code>{html.escape(mode)}</code></p>
<p>manifest: <code>{html.escape(str(manifest))}</code></p>
<p>sampled segments per label: <code>{per_label}</code></p>
<table>
<thead><tr><th>activity</th><th>samples</th><th>page</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
"""
    (output_dir / "index.html").write_text(html_document("Capture24 Visualizations", body), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    """Generate visualization pages and return the output directory."""

    if args.per_label <= 0:
        raise ValueError("--per-label must be positive")
    output_dir = args.output_dir or default_output_dir(args.mode)
    prepare_output(output_dir, args.overwrite)

    manifest = manifest_path(args.processed_root, args.mode)
    entries = load_manifest(manifest)
    grouped = group_by_activity(entries)
    sampled = sample_entries(grouped, args.per_label, args.seed)

    pages = []
    for activity, activity_entries in sampled.items():
        pages.append(activity_page(activity, activity_entries, args.processed_root, output_dir))
        print(f"{activity}: {len(activity_entries)} samples", flush=True)

    index_page(pages, output_dir, args.mode, manifest, args.per_label)
    return output_dir


def main() -> None:
    """CLI entry point."""

    output_dir = run(parse_args())
    print(f"wrote {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
