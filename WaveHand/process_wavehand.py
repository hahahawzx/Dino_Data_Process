"""One-off WaveHand IMU processor for Processed2.

Output follows the Processed2 format:
  /Volumes/Felix_Backups/Processed2/sources/WaveHand/segments/{mode}/{device}/*.npz
  /Volumes/Felix_Backups/Processed2/sources/WaveHand/manifests/{device}_{mode}.jsonl

WaveHand-specific rules:
  - modes: 10s and 1024f
  - target frequency: 150 Hz
  - raw quality threshold: 120 frames/sec
  - acceleration fields are already m/s^2
  - gyroscope fields are deg/s and are converted to rad/s
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np


DATASET_ROOT = Path("/Users/zixuanwang/Desktop/HiSync_publish_anonymous")
PROCESSED_ROOT = Path("/Volumes/Felix_Backups/Processed2")

SRC = "WaveHand"
FREQ = 150.0
PERIOD_MS = 1000.0 / FREQ
MIN_RAW_HZ = 120
LABEL = {}

DEVICE_DIRS = {
    "ring": "IMU_Ring",
    "watch": "IMU_Wrist",
}

REQUIRED_COLUMNS = [
    "deviceTimestamp",
    "accel_with_gravity_x",
    "accel_with_gravity_y",
    "accel_with_gravity_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]

MODES = {
    "10s": {"frames": int(math.ceil(FREQ * 10.0)), "duration_sec": 10.0},
    "1024f": {"frames": 1024, "duration_sec": 1024 / FREQ},
}


class SkipFile(Exception):
    """Raised when one WaveHand CSV should not produce output."""


def parse_args() -> argparse.Namespace:
    """Read command-line options for full or partial WaveHand processing."""

    parser = argparse.ArgumentParser(description="Process WaveHand ring/watch IMU into Processed2 NPZ segments.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="Rebuild the whole WaveHand processed source.")
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        default=None,
        help="Optional group IDs to process, for example: --groups 1 2 3",
    )
    parser.add_argument(
        "--devices",
        choices=sorted(DEVICE_DIRS),
        nargs="+",
        default=sorted(DEVICE_DIRS),
        help="Optional devices to process. Default: ring watch.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional limit after filtering by group/device. Useful for smoke tests.",
    )
    return parser.parse_args()


def source_root(processed_root: Path) -> Path:
    """Return processed2/sources/WaveHand."""

    return processed_root / "sources" / SRC


def segment_dir(processed_root: Path, mode: str, device: str) -> Path:
    """Return the output segment directory for one mode/device."""

    return source_root(processed_root) / "segments" / mode / device


def manifest_path(processed_root: Path, mode: str, device: str) -> Path:
    """Return the source manifest path for one mode/device."""

    return source_root(processed_root) / "manifests" / f"{device}_{mode}.jsonl"


def prepare_output(processed_root: Path, overwrite: bool, devices: list[str]) -> None:
    """Create WaveHand output folders; optionally remove previous WaveHand output."""

    root = source_root(processed_root)
    if overwrite and root.exists():
        shutil.rmtree(root)
    elif root.exists():
        raise FileExistsError(f"output exists; use --overwrite: {root}")

    for device in devices:
        for mode in MODES:
            segment_dir(processed_root, mode, device).mkdir(parents=True, exist_ok=True)
            manifest_path(processed_root, mode, device).parent.mkdir(parents=True, exist_ok=True)


def discover_files(dataset_root: Path, groups: set[int] | None, devices: list[str]) -> list[dict]:
    """Find calibrated WaveHand CSV files under selected group/device folders."""

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")

    files: list[dict] = []
    group_dirs = sorted(
        [path for path in dataset_root.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )

    for group_dir in group_dirs:
        group = int(group_dir.name)
        if groups is not None and group not in groups:
            continue

        for device in devices:
            device_dir = group_dir / "IMU" / DEVICE_DIRS[device]
            if not device_dir.exists():
                continue
            for path in sorted(device_dir.glob("calibrated_imu_*.csv")):
                files.append(
                    {
                        "group": group,
                        "device": device,
                        "path": path,
                        "relative_path": path.relative_to(dataset_root),
                    }
                )

    return files


def parse_float(row: dict, column: str) -> float | None:
    """Parse one finite float value from a CSV row."""

    try:
        value = float(row[column])
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def read_wavehand_csv(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read one calibrated CSV and return time_ms plus [acc, gyro] in standard units."""

    columns = {name: [] for name in REQUIRED_COLUMNS}
    total_rows = 0
    bad_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise SkipFile(f"missing columns: {missing}")

        for row in reader:
            total_rows += 1
            values = [parse_float(row, name) for name in REQUIRED_COLUMNS]
            if any(value is None for value in values):
                bad_rows += 1
                continue
            for name, value in zip(REQUIRED_COLUMNS, values):
                columns[name].append(value)

    if len(columns["deviceTimestamp"]) < 2:
        raise SkipFile("too few finite rows")

    original_t = np.asarray(columns["deviceTimestamp"], dtype=np.float64)
    original_dt = np.diff(original_t)
    nonpositive_dt_original_order = int(np.sum(original_dt <= 0))

    acc = np.column_stack(
        [
            columns["accel_with_gravity_x"],
            columns["accel_with_gravity_y"],
            columns["accel_with_gravity_z"],
        ]
    ).astype(np.float64)

    gyro_deg_s = np.column_stack(
        [
            columns["gyro_x"],
            columns["gyro_y"],
            columns["gyro_z"],
        ]
    ).astype(np.float64)
    gyro_rad_s = gyro_deg_s * np.pi / 180.0
    values = np.column_stack([acc, gyro_rad_s])

    order = np.argsort(original_t, kind="stable")
    sorted_t = original_t[order]
    sorted_values = values[order]

    unique_t, first_idx = np.unique(sorted_t, return_index=True)
    values = sorted_values[first_idx]

    if len(unique_t) < 2:
        raise SkipFile("too few unique timestamps")

    time_ms = unique_t - unique_t[0]
    stats = {
        "total_rows": total_rows,
        "bad_rows": bad_rows,
        "finite_rows": int(len(original_t)),
        "clean_rows": int(len(time_ms)),
        "duplicate_timestamps": int(len(sorted_t) - len(unique_t)),
        "nonpositive_dt_original_order": nonpositive_dt_original_order,
    }
    return time_ms, values, stats


def check_average_hz(time_ms: np.ndarray) -> float:
    """Reject one file if its average raw frame rate is below MIN_RAW_HZ."""

    duration_sec = (time_ms[-1] - time_ms[0]) / 1000.0
    if duration_sec <= 0:
        raise SkipFile("non-positive duration")

    hz = (len(time_ms) - 1) / duration_sec
    if hz < MIN_RAW_HZ:
        raise SkipFile(f"raw average hz below {MIN_RAW_HZ}")
    return hz


def build_quality_bins(time_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 1-second raw-frame bins for segment-level quality checks."""

    end = math.ceil(float(time_ms[-1]) / 1000.0) * 1000.0
    bins = np.arange(0.0, end + 1000.0, 1000.0)
    counts, _ = np.histogram(time_ms, bins=bins)
    valid = counts >= MIN_RAW_HZ
    return bins, counts, valid


def resample_150hz(time_ms: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use first-order linear interpolation to resample WaveHand data to 150 Hz."""

    target_t = np.arange(0.0, time_ms[-1] + 1e-9, PERIOD_MS, dtype=np.float64)
    target_t = target_t[target_t <= time_ms[-1]]
    if len(target_t) < MODES["1024f"]["frames"]:
        raise SkipFile("duration shorter than 1024 frames after resampling")

    resampled = np.column_stack([np.interp(target_t, time_ms, values[:, i]) for i in range(values.shape[1])])
    return target_t, resampled


def target_quality_index(target_t: np.ndarray, bins: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Map each target sample to a precomputed raw-quality 1-second bin."""

    idx = np.searchsorted(bins, target_t, side="right") - 1
    return np.clip(idx, 0, len(valid) - 1)


def segment_is_valid(bin_idx: np.ndarray, valid: np.ndarray, start: int, end: int) -> bool:
    """Return True when all 1-second bins touched by a segment are valid."""

    touched = np.unique(bin_idx[start:end])
    return bool(len(touched) > 0 and np.all(valid[touched]))


def build_npz_arrays(target_t: np.ndarray, resampled: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Build [T, 4] acc/gyro arrays with segment-local time_ms in column 0."""

    local_t = target_t[start:end] - target_t[start]
    acc = np.column_stack([local_t, resampled[start:end, 0:3]]).astype(np.float32)
    gyro = np.column_stack([local_t, resampled[start:end, 3:6]]).astype(np.float32)
    return acc, gyro


def arrays_are_valid(acc: np.ndarray, gyro: np.ndarray, frames: int) -> bool:
    """Final in-memory validation before writing one segment."""

    return (
        acc.shape == (frames, 4)
        and gyro.shape == (frames, 4)
        and acc.dtype == np.float32
        and gyro.dtype == np.float32
        and abs(float(acc[0, 0])) < 1e-6
        and abs(float(gyro[0, 0])) < 1e-6
        and np.all(np.isfinite(acc))
        and np.all(np.isfinite(gyro))
        and np.allclose(acc[:, 0], gyro[:, 0])
    )


def write_manifest_line(writer, mode: str, device: str, filename: str, frames: int, duration_sec: float) -> None:
    """Write one JSONL manifest entry for a saved segment."""

    entry = {
        "dir": f"sources/{SRC}/segments/{mode}/{device}/{filename}",
        "src": SRC,
        "device": device,
        "freq": FREQ,
        "mode": mode,
        "num_frames": frames,
        "duration_sec": duration_sec,
        "has_timestamp": True,
        "has_gyro": True,
        "label": LABEL,
    }
    writer.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")


def process_file(
    record: dict,
    processed_root: Path,
    writers: dict[tuple[str, str], object],
    counters: dict[tuple[str, str], int],
    summary: dict,
) -> dict[str, int]:
    """Process one calibrated CSV and directly write all valid segments."""

    device = record["device"]
    before = {mode: summary["segments"][device][mode] for mode in MODES}

    time_ms, values, read_stats = read_wavehand_csv(record["path"])
    avg_hz = check_average_hz(time_ms)
    bins, counts, valid_bins = build_quality_bins(time_ms)
    target_t, resampled = resample_150hz(time_ms, values)
    bin_idx = target_quality_index(target_t, bins, valid_bins)

    if read_stats["bad_rows"]:
        print(f"  dropped malformed rows={read_stats['bad_rows']}", flush=True)
    if read_stats["duplicate_timestamps"] or read_stats["nonpositive_dt_original_order"]:
        print(
            f"  timeline cleanup duplicates={read_stats['duplicate_timestamps']} "
            f"nonpositive_dt={read_stats['nonpositive_dt_original_order']}",
            flush=True,
        )

    invalid_seconds = int((~valid_bins).sum())
    print(
        f"  avg_hz={avg_hz:.3f} raw_seconds={len(counts)} invalid_seconds={invalid_seconds} "
        f"target_frames={len(target_t)}",
        flush=True,
    )

    wrote_any = False
    for mode, spec in MODES.items():
        frames = spec["frames"]
        for start in range(0, len(target_t) - frames + 1, frames):
            end = start + frames
            if not segment_is_valid(bin_idx, valid_bins, start, end):
                continue

            acc, gyro = build_npz_arrays(target_t, resampled, start, end)
            if not arrays_are_valid(acc, gyro, frames):
                summary["invalid_arrays"] += 1
                continue

            key = (device, mode)
            filename = f"{counters[key]:08d}.npz"
            counters[key] += 1
            np.savez(segment_dir(processed_root, mode, device) / filename, acc=acc, gyro=gyro)
            write_manifest_line(writers[key], mode, device, filename, frames, spec["duration_sec"])
            summary["segments"][device][mode] += 1
            wrote_any = True

    if wrote_any:
        summary["processed"] += 1

    return {mode: summary["segments"][device][mode] - before[mode] for mode in MODES}


def open_manifest_writers(processed_root: Path, devices: list[str]) -> dict[tuple[str, str], object]:
    """Open one manifest writer per device/mode."""

    writers = {}
    for device in devices:
        for mode in MODES:
            writers[(device, mode)] = manifest_path(processed_root, mode, device).open("w", encoding="utf-8")
    return writers


def close_manifest_writers(writers: dict[tuple[str, str], object]) -> None:
    """Close all opened manifest writers."""

    for writer in writers.values():
        writer.close()


def run(
    dataset_root: Path,
    processed_root: Path,
    overwrite: bool,
    group_filter: list[int] | None,
    devices: list[str],
    limit_files: int | None,
) -> dict:
    """Run the full WaveHand processing pipeline."""

    groups = set(group_filter) if group_filter is not None else None
    files = discover_files(dataset_root, groups, devices)
    if limit_files is not None:
        files = files[:limit_files]
    if not files:
        raise FileNotFoundError("no WaveHand calibrated CSV files matched the requested filters")

    prepare_output(processed_root, overwrite, devices)

    summary = {
        "seen": len(files),
        "processed": 0,
        "skipped_low_hz": 0,
        "skipped_other": 0,
        "invalid_arrays": 0,
        "segments": {device: {mode: 0 for mode in MODES} for device in devices},
        "skip_reasons": {},
    }
    counters = {(device, mode): 1 for device in devices for mode in MODES}
    writers = open_manifest_writers(processed_root, devices)

    try:
        total = len(files)
        for idx, record in enumerate(files, start=1):
            group = record["group"]
            device = record["device"]
            path = record["path"]
            try:
                written = process_file(record, processed_root, writers, counters, summary)
                print(
                    f"[{idx}/{total}] group={group} {device} {path.name} OK "
                    f"10s={written['10s']} 1024f={written['1024f']} "
                    f"total_{device}_10s={summary['segments'][device]['10s']} "
                    f"total_{device}_1024f={summary['segments'][device]['1024f']}",
                    flush=True,
                )
            except SkipFile as exc:
                reason = str(exc)
                if reason == f"raw average hz below {MIN_RAW_HZ}":
                    summary["skipped_low_hz"] += 1
                else:
                    summary["skipped_other"] += 1
                summary["skip_reasons"][reason] = summary["skip_reasons"].get(reason, 0) + 1
                print(f"[{idx}/{total}] group={group} {device} {path.name} SKIP {reason}", flush=True)
    finally:
        close_manifest_writers(writers)

    return summary


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    summary = run(
        dataset_root=args.dataset_root,
        processed_root=args.processed_root,
        overwrite=args.overwrite,
        group_filter=args.groups,
        devices=args.devices,
        limit_files=args.limit_files,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
