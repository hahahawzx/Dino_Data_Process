"""One-off Xiaomi Watch IMU processor.

Output follows the repository processed-data format:
  /Volumes/Felix_Backups/Processed/sources/Xiaomi_watch/segments/{mode}/watch/*.npz
  /Volumes/Felix_Backups/Processed/sources/Xiaomi_watch/manifests/watch_{mode}.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np


DATASET_ROOT = Path("/Users/zixuanwang/Downloads/Xiaomi_Watch")
PROCESSED_ROOT = Path("/Volumes/Felix_Backups/Processed")

SRC = "Xiaomi_watch"
DEVICE = "watch"
FREQ = 200.0
PERIOD_MS = 1000.0 / FREQ
MIN_RAW_HZ = 150
LABEL = {}

MODES = {
    "10s": {"frames": 2000, "duration_sec": 10.0},
    "1000f": {"frames": 1000, "duration_sec": 5.0},
}

ONLINE_COLUMNS = ["CurrentTimestamp(ms)", "EventTimestamp(ms)", "x", "y", "z"]
IMU_BINARY_DTYPE = np.dtype(
    [
        ("utc", "<u8"),
        ("ts", "<u8"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("pad", "V4"),
    ]
)


class SkipSession(Exception):
    """Raised when one Xiaomi Watch session should not produce output."""


def parse_args() -> argparse.Namespace:
    """Read command-line arguments for one session or the full Xiaomi Watch dataset."""

    parser = argparse.ArgumentParser(description="Process Xiaomi Watch Online/Offline IMU into NPZ segments.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset-root", type=Path, default=None)
    source.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-start-sec",
        type=float,
        default=5.0,
        help="Drop startup samples before quality checks and segmentation. Default: 5.",
    )
    return parser.parse_args()


def discover_sessions(dataset_root: Path) -> list[tuple[str, Path]]:
    """Find Online and Offline session directories that contain acc and gyro files."""

    sessions: list[tuple[str, Path]] = []
    for collection in ("Online", "Offline"):
        collection_dir = dataset_root / collection
        if not collection_dir.exists():
            continue
        for session_dir in sorted(collection_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            has_acc = (session_dir / "accel-200hz.csv").exists()
            has_gyro = (session_dir / "gyroscope-200hz.csv").exists()
            if has_acc and has_gyro:
                sessions.append((collection.lower(), session_dir))
    return sessions


def get_sessions(dataset_root: Path | None, session_dir: Path | None) -> list[tuple[str, Path]]:
    """Resolve CLI input into a list of collection/session pairs."""

    if session_dir is not None:
        missing = [name for name in ("accel-200hz.csv", "gyroscope-200hz.csv") if not (session_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"missing Xiaomi Watch files in {session_dir}: {missing}")
        collection = session_dir.parent.name.lower()
        if collection not in {"online", "offline"}:
            collection = "online"
        return [(collection, session_dir)]
    return discover_sessions(dataset_root or DATASET_ROOT)


def source_root(processed_root: Path) -> Path:
    """Return processed/sources/Xiaomi_watch."""

    return processed_root / "sources" / SRC


def segment_dir(processed_root: Path, mode: str) -> Path:
    """Return the output segment directory for one mode."""

    return source_root(processed_root) / "segments" / mode / DEVICE


def manifest_path(processed_root: Path, mode: str) -> Path:
    """Return the source manifest path for one mode."""

    return source_root(processed_root) / "manifests" / f"{DEVICE}_{mode}.jsonl"


def prepare_output(processed_root: Path, overwrite: bool) -> None:
    """Create source output folders; optionally rebuild existing Xiaomi Watch output."""

    root = source_root(processed_root)
    if overwrite and root.exists():
        shutil.rmtree(root)
    elif root.exists():
        raise FileExistsError(f"output exists; use --overwrite: {root}")

    for mode in MODES:
        segment_dir(processed_root, mode).mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)


def read_online_csv(path: Path, signal_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read an Online CSV and return EventTimestamp in ms plus xyz values."""

    rows = []
    dropped = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
            header = next(reader)
        except StopIteration as exc:
            raise SkipSession(f"empty {signal_name} online csv") from exc

        if len(header) < 5:
            raise SkipSession(f"unexpected {signal_name} online header: {header}")

        for row in reader:
            if len(row) < 5:
                dropped += 1
                continue
            try:
                values = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]
            except ValueError:
                dropped += 1
                continue
            if not np.all(np.isfinite(values)):
                dropped += 1
                continue
            rows.append(values)

    if not rows:
        raise SkipSession(f"empty {signal_name} online data")
    if dropped:
        print(f"  dropped malformed {signal_name} online rows={dropped}", flush=True)

    data = np.asarray(rows, dtype=np.float64)
    return data[:, 0], data[:, 1:4]


def read_offline_binary(path: Path, signal_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read an Offline binary IMU file and return EventTimestamp in ms plus xyz values."""

    if path.stat().st_size < IMU_BINARY_DTYPE.itemsize:
        raise SkipSession(f"empty {signal_name} offline binary")
    if path.stat().st_size % IMU_BINARY_DTYPE.itemsize != 0:
        raise SkipSession(f"truncated {signal_name} offline binary")

    data = np.fromfile(path, dtype=IMU_BINARY_DTYPE)
    if len(data) == 0:
        raise SkipSession(f"empty {signal_name} offline binary")

    time_ms = data["ts"].astype(np.float64) / 1000.0
    xyz = np.column_stack(
        [
            data["x"].astype(np.float64),
            data["y"].astype(np.float64),
            data["z"].astype(np.float64),
        ]
    )
    finite = np.isfinite(time_ms) & np.all(np.isfinite(xyz), axis=1)
    time_ms = time_ms[finite]
    xyz = xyz[finite]
    if len(time_ms) == 0:
        raise SkipSession(f"no finite {signal_name} offline rows")
    return time_ms, xyz


def read_signal(session_dir: Path, collection: str, filename: str, signal_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one Xiaomi Watch signal from Online CSV or Offline binary format."""

    path = session_dir / filename
    if not path.exists():
        raise SkipSession(f"missing {filename}")
    if collection == "offline":
        return read_offline_binary(path, signal_name)
    return read_online_csv(path, signal_name)


def clean_timeline(time_ms: np.ndarray, xyz: np.ndarray, signal_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Sort by timestamp and keep the first sample for duplicate timestamps."""

    order = np.argsort(time_ms, kind="stable")
    time_ms = time_ms[order]
    xyz = xyz[order]

    time_ms, first_idx = np.unique(time_ms, return_index=True)
    xyz = xyz[first_idx]

    if len(time_ms) < 2:
        raise SkipSession(f"not enough unique {signal_name} samples")
    return time_ms, xyz


def drop_startup(time_ms: np.ndarray, xyz: np.ndarray, skip_start_sec: float, signal_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Drop the initial startup region before quality checks."""

    if skip_start_sec <= 0:
        return time_ms, xyz

    keep_start = time_ms[0] + skip_start_sec * 1000.0
    keep = time_ms >= keep_start
    time_ms = time_ms[keep]
    xyz = xyz[keep]
    if len(time_ms) < MODES["1000f"]["frames"]:
        raise SkipSession(f"{signal_name} too short after startup drop")
    return time_ms, xyz


def check_average_hz(time_ms: np.ndarray, signal_name: str) -> float:
    """Reject one session if a signal's average raw frame rate is below 150 Hz."""

    duration_sec = (time_ms[-1] - time_ms[0]) / 1000.0
    if duration_sec <= 0:
        raise SkipSession(f"non-positive {signal_name} duration")

    hz = len(time_ms) / duration_sec
    if hz < MIN_RAW_HZ:
        raise SkipSession(f"{signal_name} raw average hz below 150")
    return hz


def build_quality_bins(time_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1-second raw-frame bins used by segment-level quality checks."""

    start = math.floor(time_ms[0] / 1000.0) * 1000.0
    end = math.ceil(time_ms[-1] / 1000.0) * 1000.0
    bins = np.arange(start, end + 1000.0, 1000.0)
    counts, _ = np.histogram(time_ms, bins=bins)
    return bins, counts >= MIN_RAW_HZ


def build_common_target_grid(acc_time: np.ndarray, gyro_time: np.ndarray) -> np.ndarray:
    """Build a 200 Hz target grid inside the shared acc/gyro time range."""

    common_start = max(acc_time[0], gyro_time[0])
    common_end = min(acc_time[-1], gyro_time[-1])
    target_t = np.arange(common_start, common_end + 1e-6, PERIOD_MS)
    if len(target_t) < MODES["1000f"]["frames"]:
        raise SkipSession("common duration shorter than 1000 frames")
    return target_t


def interpolate_xyz(source_t: np.ndarray, source_xyz: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """Use first-order linear interpolation without extrapolation."""

    if target_t[0] < source_t[0] or target_t[-1] > source_t[-1]:
        raise SkipSession("target grid outside source timeline")
    return np.column_stack([np.interp(target_t, source_t, source_xyz[:, i]) for i in range(3)])


def target_quality_index(target_t: np.ndarray, bins: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Map each target sample to a precomputed raw-quality bin."""

    idx = np.searchsorted(bins, target_t, side="right") - 1
    return np.clip(idx, 0, len(valid) - 1)


def segment_is_valid(bin_idx: np.ndarray, valid: np.ndarray, start: int, end: int) -> bool:
    """Check whether all 1-second bins touched by this segment are valid."""

    touched = np.unique(bin_idx[start:end])
    return bool(len(touched) > 0 and np.all(valid[touched]))


def build_npz_arrays(
    target_t: np.ndarray,
    acc_interp: np.ndarray,
    gyro_interp: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build acc and gyro arrays with segment-local time_ms in column 0."""

    local_t = target_t[start:end] - target_t[start]
    acc = np.column_stack([local_t, acc_interp[start:end]]).astype(np.float32)
    gyro = np.column_stack([local_t, gyro_interp[start:end]]).astype(np.float32)
    return acc, gyro


def arrays_are_valid(acc: np.ndarray, gyro: np.ndarray, frames: int) -> bool:
    """Final in-memory validation before writing one segment."""

    return (
        acc.shape == (frames, 4)
        and gyro.shape == (frames, 4)
        and acc.dtype == np.float32
        and gyro.dtype == np.float32
        and acc[0, 0] == 0.0
        and gyro[0, 0] == 0.0
        and np.array_equal(acc[:, 0], gyro[:, 0])
        and np.all(np.isfinite(acc))
        and np.all(np.isfinite(gyro))
    )


def write_manifest_line(writer, mode: str, filename: str, frames: int, duration_sec: float) -> None:
    """Write one JSONL manifest row for a saved segment."""

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
        "label": LABEL,
    }
    writer.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")


def process_session(
    collection: str,
    session_dir: Path,
    processed_root: Path,
    writers: dict[str, object],
    counters: dict[str, int],
    summary: dict,
    skip_start_sec: float,
) -> dict[str, int]:
    """Process one Xiaomi Watch session and write all passing segments."""

    before = summary["segments"].copy()

    acc_time, acc_xyz = read_signal(session_dir, collection, "accel-200hz.csv", "acc")
    gyro_time, gyro_xyz = read_signal(session_dir, collection, "gyroscope-200hz.csv", "gyro")
    acc_time, acc_xyz = clean_timeline(acc_time, acc_xyz, "acc")
    gyro_time, gyro_xyz = clean_timeline(gyro_time, gyro_xyz, "gyro")
    acc_time, acc_xyz = drop_startup(acc_time, acc_xyz, skip_start_sec, "acc")
    gyro_time, gyro_xyz = drop_startup(gyro_time, gyro_xyz, skip_start_sec, "gyro")

    check_average_hz(acc_time, "acc")
    check_average_hz(gyro_time, "gyro")

    acc_bins, acc_valid = build_quality_bins(acc_time)
    gyro_bins, gyro_valid = build_quality_bins(gyro_time)
    target_t = build_common_target_grid(acc_time, gyro_time)
    acc_interp = interpolate_xyz(acc_time, acc_xyz, target_t)
    gyro_interp = interpolate_xyz(gyro_time, gyro_xyz, target_t)
    acc_bin_idx = target_quality_index(target_t, acc_bins, acc_valid)
    gyro_bin_idx = target_quality_index(target_t, gyro_bins, gyro_valid)

    wrote_any = False
    for mode, spec in MODES.items():
        frames = spec["frames"]
        for start in range(0, len(target_t) - frames + 1, frames):
            end = start + frames
            if not segment_is_valid(acc_bin_idx, acc_valid, start, end):
                continue
            if not segment_is_valid(gyro_bin_idx, gyro_valid, start, end):
                continue

            acc, gyro = build_npz_arrays(target_t, acc_interp, gyro_interp, start, end)
            if not arrays_are_valid(acc, gyro, frames):
                continue

            filename = f"{counters[mode]:08d}.npz"
            counters[mode] += 1
            np.savez(segment_dir(processed_root, mode) / filename, acc=acc, gyro=gyro)
            write_manifest_line(writers[mode], mode, filename, frames, spec["duration_sec"])
            summary["segments"][mode] += 1
            wrote_any = True

    if wrote_any:
        summary["processed"] += 1
        summary[f"processed_{collection}"] += 1

    return {
        "10s": summary["segments"]["10s"] - before["10s"],
        "1000f": summary["segments"]["1000f"] - before["1000f"],
    }


def run(
    dataset_root: Path | None,
    session_dir: Path | None,
    processed_root: Path,
    overwrite: bool,
    skip_start_sec: float,
) -> dict:
    """Run the Xiaomi Watch processing pipeline."""

    sessions = get_sessions(dataset_root, session_dir)
    prepare_output(processed_root, overwrite)

    summary = {
        "seen": len(sessions),
        "processed": 0,
        "processed_online": 0,
        "processed_offline": 0,
        "skipped_low_hz": 0,
        "skipped_other": 0,
        "segments": {"10s": 0, "1000f": 0},
        "skip_reasons": {},
    }
    counters = {"10s": 1, "1000f": 1}

    with manifest_path(processed_root, "10s").open("w", encoding="utf-8") as man_10s, manifest_path(
        processed_root, "1000f"
    ).open("w", encoding="utf-8") as man_1000f:
        writers = {"10s": man_10s, "1000f": man_1000f}
        total = len(sessions)
        for idx, (collection, session) in enumerate(sessions, start=1):
            try:
                written = process_session(
                    collection,
                    session,
                    processed_root,
                    writers,
                    counters,
                    summary,
                    skip_start_sec,
                )
                print(
                    f"[{idx}/{total}] {collection}/{session.name} OK "
                    f"10s={written['10s']} 1000f={written['1000f']} "
                    f"total10s={summary['segments']['10s']} total1000f={summary['segments']['1000f']}",
                    flush=True,
                )
            except SkipSession as exc:
                reason = str(exc)
                if "below 150" in reason:
                    summary["skipped_low_hz"] += 1
                else:
                    summary["skipped_other"] += 1
                summary["skip_reasons"][reason] = summary["skip_reasons"].get(reason, 0) + 1
                print(f"[{idx}/{total}] {collection}/{session.name} SKIP {reason}", flush=True)

    return summary


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    summary = run(args.dataset_root, args.session_dir, args.processed_root, args.overwrite, args.skip_start_sec)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
