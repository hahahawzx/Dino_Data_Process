"""One-off ExtraSensoryPhone processor for Processed2.

Output follows the Processed2 source format:
  /Volumes/Felix_Backups/Processed2/sources/ExtraSensoryPhone/segments/{mode}/phone/*.npz
  /Volumes/Felix_Backups/Processed2/sources/ExtraSensoryPhone/manifests/phone_{mode}.jsonl

ExtraSensoryPhone-specific rules:
  - device: phone
  - target frequency: 40 Hz
  - modes: 10s, 512f, 1024f
  - 1024f may tail-pad zeros; label.pad_frames records the padded frame count
  - acceleration unit is inferred per file from median magnitude
  - proc_gyro is treated as rad/s
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


DATASET_ROOT = Path("/Volumes/Felix_Backups/ExtraSensoryPhone")
PROCESSED_ROOT = Path("/Volumes/Felix_Backups/Processed2")

SRC = "ExtraSensoryPhone"
DEVICE = "phone"
FREQ = 40.0
PERIOD_MS = 1000.0 / FREQ
ACC_MIN_RAW_HZ = 30
GYRO_MIN_RAW_HZ = 35
G_TO_MPS2 = 9.80665

MODES = {
    "10s": {"frames": 400, "duration_sec": 10.0, "padding": False},
    "512f": {"frames": 512, "duration_sec": 512 / FREQ, "padding": False},
    "1024f": {"frames": 1024, "duration_sec": 1024 / FREQ, "padding": True, "min_real_frames": 512},
}


class SkipFile(Exception):
    """Raised when one acc/gyro file pair should not produce output."""


def parse_args() -> argparse.Namespace:
    """Read CLI options for full or partial ExtraSensoryPhone processing."""

    parser = argparse.ArgumentParser(description="Process ExtraSensoryPhone IMU into Processed2 NPZ segments.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="Rebuild the whole ExtraSensoryPhone source output.")
    parser.add_argument(
        "--uuids",
        nargs="+",
        default=None,
        help="Optional UUID filter. Example: --uuids 0A986513-7828-4D53-AA1F-E02D6DF9561B",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Optional exact key filter in UUID/timestamp form. Example: --keys UUID/1449601597",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional limit after filtering. Useful for smoke tests.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N files. Default: 1000. Use 1 for verbose progress.",
    )
    return parser.parse_args()


def source_root(processed_root: Path) -> Path:
    """Return processed2/sources/ExtraSensoryPhone."""

    return processed_root / "sources" / SRC


def segment_dir(processed_root: Path, mode: str) -> Path:
    """Return the segment output directory for one mode."""

    return source_root(processed_root) / "segments" / mode / DEVICE


def manifest_path(processed_root: Path, mode: str) -> Path:
    """Return the source manifest path for one mode."""

    return source_root(processed_root) / "manifests" / f"{DEVICE}_{mode}.jsonl"


def prepare_output(processed_root: Path, overwrite: bool) -> None:
    """Create source output folders; optionally remove previous source output."""

    root = source_root(processed_root)
    if overwrite and root.exists():
        shutil.rmtree(root)
    elif root.exists():
        raise FileExistsError(f"output exists; use --overwrite: {root}")

    for mode in MODES:
        segment_dir(processed_root, mode).mkdir(parents=True, exist_ok=True)
        manifest_path(processed_root, mode).parent.mkdir(parents=True, exist_ok=True)


def acc_key(path: Path, acc_root: Path) -> str:
    """Return UUID/timestamp key for one raw_acc path."""

    return path.relative_to(acc_root).as_posix().replace(".m_raw_acc.dat", "")


def gyro_key(path: Path, gyro_root: Path) -> str:
    """Return UUID/timestamp key for one proc_gyro path."""

    return path.relative_to(gyro_root).as_posix().replace(".m_proc_gyro.dat", "")


def discover_pairs(
    dataset_root: Path,
    uuids: Optional[set[str]],
    keys: Optional[set[str]],
    limit_files: Optional[int],
) -> tuple[list[dict], dict]:
    """Find paired raw_acc/proc_gyro files and return records plus discovery counts."""

    acc_root = dataset_root / "raw_acc"
    gyro_root = dataset_root / "proc_gyro"
    if not acc_root.exists():
        raise FileNotFoundError(f"missing raw_acc directory: {acc_root}")
    if not gyro_root.exists():
        raise FileNotFoundError(f"missing proc_gyro directory: {gyro_root}")

    acc_files = {acc_key(path, acc_root): path for path in acc_root.glob("*/*.m_raw_acc.dat")}
    gyro_files = {gyro_key(path, gyro_root): path for path in gyro_root.glob("*/*.m_proc_gyro.dat")}
    paired_keys = sorted(set(acc_files) & set(gyro_files))

    if uuids is not None:
        paired_keys = [key for key in paired_keys if key.split("/", 1)[0] in uuids]
    if keys is not None:
        paired_keys = [key for key in paired_keys if key in keys]
    if limit_files is not None:
        paired_keys = paired_keys[:limit_files]

    records = [
        {
            "key": key,
            "uuid": key.split("/", 1)[0],
            "timestamp": key.split("/", 1)[1],
            "acc_path": acc_files[key],
            "gyro_path": gyro_files[key],
        }
        for key in paired_keys
    ]
    counts = {
        "acc_files": len(acc_files),
        "gyro_files": len(gyro_files),
        "paired_files": len(set(acc_files) & set(gyro_files)),
        "acc_only_files": len(set(acc_files) - set(gyro_files)),
        "gyro_only_files": len(set(gyro_files) - set(acc_files)),
        "selected_files": len(records),
    }
    return records, counts


def read_dat(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a four-column .dat text file, clean timestamps, and return time_s/xyz."""

    try:
        data = np.loadtxt(path, dtype=np.float64)
    except ValueError as exc:
        raise SkipFile(f"cannot parse dat: {path.name}: {exc}") from exc

    if data.size == 0:
        raise SkipFile(f"empty dat: {path.name}")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 4:
        raise SkipFile(f"unexpected column count {data.shape[1]}: {path.name}")

    finite = np.all(np.isfinite(data), axis=1)
    dropped_nonfinite = int(np.sum(~finite))
    data = data[finite]
    if len(data) < 2:
        raise SkipFile(f"too few finite rows: {path.name}")

    original_t = data[:, 0]
    original_dt = np.diff(original_t)
    nonpositive_dt_original_order = int(np.sum(original_dt <= 0))

    order = np.argsort(original_t, kind="stable")
    data = data[order]
    sorted_t = data[:, 0]
    sorted_xyz = data[:, 1:4]

    unique_t, first_idx = np.unique(sorted_t, return_index=True)
    xyz = sorted_xyz[first_idx]
    if len(unique_t) < 2:
        raise SkipFile(f"too few unique timestamps: {path.name}")

    stats = {
        "rows": int(len(original_t)),
        "clean_rows": int(len(unique_t)),
        "dropped_nonfinite": dropped_nonfinite,
        "duplicate_timestamps": int(len(sorted_t) - len(unique_t)),
        "nonpositive_dt_original_order": nonpositive_dt_original_order,
    }
    return unique_t, xyz, stats


def infer_acc_unit(acc_xyz: np.ndarray) -> tuple[str, float, float]:
    """Infer acceleration unit from median magnitude."""

    median_mag = float(np.median(np.linalg.norm(acc_xyz, axis=1)))
    if 0.7 <= median_mag <= 1.5:
        return "g", G_TO_MPS2, median_mag
    if 6.0 <= median_mag <= 15.0:
        return "m/s^2", 1.0, median_mag
    raise SkipFile("unknown acc unit")


def overlap_signals(
    acc_t: np.ndarray,
    acc_xyz: np.ndarray,
    gyro_t: np.ndarray,
    gyro_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Trim acc/gyro to their shared time range and shift time to overlap start."""

    start = max(float(acc_t[0]), float(gyro_t[0]))
    end = min(float(acc_t[-1]), float(gyro_t[-1]))
    if end <= start:
        raise SkipFile("no acc/gyro overlap")

    acc_mask = (acc_t >= start) & (acc_t <= end)
    gyro_mask = (gyro_t >= start) & (gyro_t <= end)
    acc_t_rel = acc_t[acc_mask] - start
    gyro_t_rel = gyro_t[gyro_mask] - start
    acc_xyz_overlap = acc_xyz[acc_mask]
    gyro_xyz_overlap = gyro_xyz[gyro_mask]

    if len(acc_t_rel) < ACC_MIN_RAW_HZ:
        raise SkipFile("too few acc overlap rows")
    if len(gyro_t_rel) < GYRO_MIN_RAW_HZ:
        raise SkipFile("too few gyro overlap rows")
    return acc_t_rel, acc_xyz_overlap, gyro_t_rel, gyro_xyz_overlap, end - start


def build_quality_bins(time_s: np.ndarray, min_hz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count raw samples in 1-second bins and mark bins that meet min_hz."""

    end = math.ceil(float(time_s[-1]))
    bins = np.arange(0.0, end + 1.0, 1.0)
    counts, _ = np.histogram(time_s, bins=bins)
    valid = counts >= min_hz
    return bins, counts, valid


def interpolate_columns(time_s: np.ndarray, xyz: np.ndarray, target_s: np.ndarray) -> np.ndarray:
    """Linearly interpolate xyz columns to target_s."""

    out = np.empty((len(target_s), xyz.shape[1]), dtype=np.float64)
    for axis in range(xyz.shape[1]):
        out[:, axis] = np.interp(target_s, time_s, xyz[:, axis])
    return out


def resample_overlap(
    acc_t_s: np.ndarray,
    acc_xyz: np.ndarray,
    gyro_t_s: np.ndarray,
    gyro_xyz: np.ndarray,
    duration_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample acc and gyro overlap to 40 Hz."""

    step_s = 1.0 / FREQ
    target_s = np.arange(0.0, duration_sec + 1e-12, step_s, dtype=np.float64)
    target_s = target_s[target_s <= duration_sec]
    if len(target_s) < MODES["10s"]["frames"]:
        raise SkipFile("duration shorter than 10s after resampling")

    target_ms = target_s * 1000.0
    acc = interpolate_columns(acc_t_s, acc_xyz, target_s)
    gyro = interpolate_columns(gyro_t_s, gyro_xyz, target_s)
    return target_ms, acc, gyro


def target_quality_index(target_ms: np.ndarray, bins_s: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Map target samples to 1-second raw-quality bins."""

    target_s = target_ms / 1000.0
    idx = np.searchsorted(bins_s, target_s, side="right") - 1
    return np.clip(idx, 0, len(valid) - 1)


def segment_is_valid(acc_bin_idx: np.ndarray, gyro_bin_idx: np.ndarray, acc_valid: np.ndarray, gyro_valid: np.ndarray, start: int, end: int) -> bool:
    """Return True if acc and gyro raw bins touched by a segment are valid."""

    acc_touched = np.unique(acc_bin_idx[start:end])
    gyro_touched = np.unique(gyro_bin_idx[start:end])
    return bool(np.all(acc_valid[acc_touched]) and np.all(gyro_valid[gyro_touched]))


def segment_ranges(mode: str, target_len: int) -> Iterable[tuple[int, int]]:
    """Yield real-frame ranges for one mode. 1024f may yield a padded tail."""

    spec = MODES[mode]
    frames = spec["frames"]
    if not spec["padding"]:
        for start in range(0, target_len - frames + 1, frames):
            yield start, start + frames
        return

    min_real = spec["min_real_frames"]
    for start in range(0, target_len, frames):
        end = min(start + frames, target_len)
        if end - start < min_real:
            break
        yield start, end


def build_npz_arrays(mode: str, acc_values: np.ndarray, gyro_values: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Build [T, 4] acc/gyro arrays. Tail-pad 1024f if needed."""

    spec = MODES[mode]
    frames = spec["frames"]
    real_frames = end - start
    pad_frames = frames - real_frames
    if pad_frames < 0:
        raise ValueError("real frame range exceeds mode frame count")
    if pad_frames and not spec["padding"]:
        raise ValueError(f"{mode} does not allow padding")

    local_t = np.arange(real_frames, dtype=np.float64) * PERIOD_MS
    acc_real = np.column_stack([local_t, acc_values[start:end]])
    gyro_real = np.column_stack([local_t, gyro_values[start:end]])

    if pad_frames:
        pad_t = (np.arange(real_frames, frames, dtype=np.float64) * PERIOD_MS).reshape(-1, 1)
        pad_xyz = np.zeros((pad_frames, 3), dtype=np.float64)
        pad_block = np.column_stack([pad_t, pad_xyz])
        acc = np.vstack([acc_real, pad_block])
        gyro = np.vstack([gyro_real, pad_block])
    else:
        acc = acc_real
        gyro = gyro_real

    return acc.astype(np.float32), gyro.astype(np.float32), int(pad_frames)


def arrays_are_valid(acc: np.ndarray, gyro: np.ndarray, frames: int, pad_frames: int) -> bool:
    """Final in-memory validation before writing one segment."""

    ok = (
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
    if not ok:
        return False
    if pad_frames:
        return bool(np.all(acc[-pad_frames:, 1:4] == 0.0) and np.all(gyro[-pad_frames:, 1:4] == 0.0))
    return True


def write_manifest_line(writer, mode: str, filename: str, frames: int, duration_sec: float, pad_frames: int) -> None:
    """Write one JSONL manifest entry."""

    entry = {
        "dir": f"sources/{SRC}/segments/{mode}/{DEVICE}/{filename}",
        "src": SRC,
        "device": DEVICE,
        "freq": FREQ,
        "mode": mode,
        "num_frames": frames,
        "duration_sec": duration_sec,
        "has_timestamp": True,
        "has_gyro": True,
        "label": {"pad_frames": pad_frames},
    }
    writer.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")


def process_pair(record: dict, processed_root: Path, writers: dict[str, object], counters: dict[str, int], summary: dict) -> dict[str, int]:
    """Process one acc/gyro file pair and write all valid segments."""

    before = {mode: summary["segments"][mode] for mode in MODES}

    acc_t, acc_xyz_raw, acc_stats = read_dat(record["acc_path"])
    gyro_t, gyro_xyz, gyro_stats = read_dat(record["gyro_path"])
    acc_unit, acc_multiplier, acc_median_mag = infer_acc_unit(acc_xyz_raw)
    acc_xyz = acc_xyz_raw * acc_multiplier

    acc_t_s, acc_overlap, gyro_t_s, gyro_overlap, duration_sec = overlap_signals(acc_t, acc_xyz, gyro_t, gyro_xyz)
    acc_bins, acc_counts, acc_valid = build_quality_bins(acc_t_s, ACC_MIN_RAW_HZ)
    gyro_bins, gyro_counts, gyro_valid = build_quality_bins(gyro_t_s, GYRO_MIN_RAW_HZ)
    target_ms, acc_resampled, gyro_resampled = resample_overlap(acc_t_s, acc_overlap, gyro_t_s, gyro_overlap, duration_sec)
    acc_bin_idx = target_quality_index(target_ms, acc_bins, acc_valid)
    gyro_bin_idx = target_quality_index(target_ms, gyro_bins, gyro_valid)

    summary["acc_unit_counts"][acc_unit] = summary["acc_unit_counts"].get(acc_unit, 0) + 1
    if acc_stats["dropped_nonfinite"] or gyro_stats["dropped_nonfinite"]:
        summary["dropped_nonfinite_rows"] += acc_stats["dropped_nonfinite"] + gyro_stats["dropped_nonfinite"]
    if acc_stats["duplicate_timestamps"] or gyro_stats["duplicate_timestamps"]:
        summary["duplicate_timestamps"] += acc_stats["duplicate_timestamps"] + gyro_stats["duplicate_timestamps"]
    if acc_stats["nonpositive_dt_original_order"] or gyro_stats["nonpositive_dt_original_order"]:
        summary["nonpositive_dt_original_order"] += acc_stats["nonpositive_dt_original_order"] + gyro_stats["nonpositive_dt_original_order"]

    for mode, spec in MODES.items():
        frames = spec["frames"]
        for start, end in segment_ranges(mode, len(target_ms)):
            if not segment_is_valid(acc_bin_idx, gyro_bin_idx, acc_valid, gyro_valid, start, end):
                summary["quality_rejected_segments"][mode] += 1
                continue

            acc, gyro, pad_frames = build_npz_arrays(mode, acc_resampled, gyro_resampled, start, end)
            if not arrays_are_valid(acc, gyro, frames, pad_frames):
                summary["invalid_arrays"] += 1
                continue

            filename = f"{counters[mode]:08d}.npz"
            counters[mode] += 1
            np.savez(segment_dir(processed_root, mode) / filename, acc=acc, gyro=gyro)
            write_manifest_line(writers[mode], mode, filename, frames, spec["duration_sec"], pad_frames)
            summary["segments"][mode] += 1
            if pad_frames:
                summary["total_pad_frames"] += pad_frames
                summary["padded_1024f_segments"] += 1

    written = {mode: summary["segments"][mode] - before[mode] for mode in MODES}
    if sum(written.values()) == 0:
        raise SkipFile("no valid segments")

    summary["processed"] += 1
    return written


def open_manifest_writers(processed_root: Path) -> dict[str, object]:
    """Open source manifest writers for all modes."""

    return {mode: manifest_path(processed_root, mode).open("w", encoding="utf-8") for mode in MODES}


def close_manifest_writers(writers: dict[str, object]) -> None:
    """Close all manifest writers."""

    for writer in writers.values():
        writer.close()


def should_print_progress(idx: int, total: int, progress_every: int) -> bool:
    """Return True if the runner should print a progress line."""

    if progress_every <= 0:
        return False
    return idx == 1 or idx == total or idx % progress_every == 0


def run(
    dataset_root: Path,
    processed_root: Path,
    overwrite: bool,
    uuids: Optional[list[str]],
    keys: Optional[list[str]],
    limit_files: Optional[int],
    progress_every: int,
) -> dict:
    """Run the full ExtraSensoryPhone processing pipeline."""

    records, discovery = discover_pairs(
        dataset_root=dataset_root,
        uuids=set(uuids) if uuids is not None else None,
        keys=set(keys) if keys is not None else None,
        limit_files=limit_files,
    )
    if not records:
        raise FileNotFoundError("no paired ExtraSensoryPhone acc/gyro files matched the requested filters")

    prepare_output(processed_root, overwrite)

    summary = {
        "discovery": discovery,
        "seen": len(records),
        "processed": 0,
        "skipped": 0,
        "invalid_arrays": 0,
        "segments": {mode: 0 for mode in MODES},
        "quality_rejected_segments": {mode: 0 for mode in MODES},
        "padded_1024f_segments": 0,
        "total_pad_frames": 0,
        "acc_unit_counts": {},
        "dropped_nonfinite_rows": 0,
        "duplicate_timestamps": 0,
        "nonpositive_dt_original_order": 0,
        "skip_reasons": {},
    }
    counters = {mode: 1 for mode in MODES}
    writers = open_manifest_writers(processed_root)

    try:
        total = len(records)
        for idx, record in enumerate(records, start=1):
            try:
                written = process_pair(record, processed_root, writers, counters, summary)
                if should_print_progress(idx, total, progress_every):
                    print(
                        f"[{idx}/{total}] {record['key']} OK "
                        f"10s={written['10s']} 512f={written['512f']} 1024f={written['1024f']} "
                        f"total10s={summary['segments']['10s']} "
                        f"total512f={summary['segments']['512f']} "
                        f"total1024f={summary['segments']['1024f']}",
                        flush=True,
                    )
            except SkipFile as exc:
                summary["skipped"] += 1
                reason = str(exc)
                summary["skip_reasons"][reason] = summary["skip_reasons"].get(reason, 0) + 1
                if should_print_progress(idx, total, progress_every):
                    print(f"[{idx}/{total}] {record['key']} SKIP {reason}", flush=True)
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
        uuids=args.uuids,
        keys=args.keys,
        limit_files=args.limit_files,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
