"""One-off AuthRing daily phone IMU processor.

Output follows the repository processed-data format:
  /Volumes/Felix_Backups/Processed/sources/authring_daily/segments/{mode}/phone/*.npz
  /Volumes/Felix_Backups/Processed/sources/authring_daily/manifests/phone_{mode}.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np


DATASET_ROOT = Path("/Volumes/Felix_Backups/Root26.5.22/科研与科创/数据记录/AuthRing/日常采集数据-解压")
PROCESSED_ROOT = Path("/Volumes/Felix_Backups/Processed")

SRC = "authring_daily"
DEVICE = "phone"
FREQ = 200.0
PERIOD_MS = 1000.0 / FREQ
MIN_RAW_HZ = 150
LABEL = {"finger": "right_index"}

PHONE_COLUMNS = ["timestamp", "x", "y", "z"]
MODES = {
    "10s": {"frames": 2000, "duration_sec": 10.0},
    "1000f": {"frames": 1000, "duration_sec": 5.0},
}


class SkipSession(Exception):
    """Raised when a daily session should not produce phone output."""


def parse_args() -> argparse.Namespace:
    """Read command-line arguments for either one session or the whole daily dataset."""

    parser = argparse.ArgumentParser(description="Process AuthRing daily phone IMU into NPZ training segments.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset-root", type=Path, default=None)
    source.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_sessions(dataset_root: Path) -> list[Path]:
    """Find person/session directories that contain phone acc and gyro CSVs."""

    sessions = []
    for participant_dir in sorted(dataset_root.iterdir()):
        if not participant_dir.is_dir():
            continue
        for session_dir in sorted(participant_dir.iterdir()):
            if session_dir.is_dir() and (session_dir / "phone_acc.csv").exists() and (
                session_dir / "phone_gyro.csv"
            ).exists():
                sessions.append(session_dir)
    return sessions


def get_sessions(dataset_root: Path | None, session_dir: Path | None) -> list[Path]:
    """Resolve input mode into a session list."""

    if session_dir is not None:
        missing = [name for name in ("phone_acc.csv", "phone_gyro.csv") if not (session_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"missing phone files in {session_dir}: {missing}")
        return [session_dir]
    return discover_sessions(dataset_root or DATASET_ROOT)


def source_root(processed_root: Path) -> Path:
    """Return processed/sources/authring_daily."""

    return processed_root / "sources" / SRC


def segment_dir(processed_root: Path, mode: str) -> Path:
    """Return the output directory for one segment mode."""

    return source_root(processed_root) / "segments" / mode / DEVICE


def manifest_path(processed_root: Path, mode: str) -> Path:
    """Return the source manifest path for this device and one segment mode."""

    return source_root(processed_root) / "manifests" / f"{DEVICE}_{mode}.jsonl"


def prepare_output(processed_root: Path, overwrite: bool) -> None:
    """Create phone output folders; optionally rebuild existing daily phone outputs."""

    phone_dirs = [segment_dir(processed_root, mode) for mode in MODES]
    manifests = [manifest_path(processed_root, mode) for mode in MODES]

    if overwrite:
        for path in phone_dirs:
            if path.exists():
                shutil.rmtree(path)
        for path in manifests:
            if path.exists():
                path.unlink()
    else:
        existing = [str(path) for path in phone_dirs + manifests if path.exists()]
        if existing:
            raise FileExistsError("output exists; use --overwrite: " + ", ".join(existing))

    for mode in MODES:
        segment_dir(processed_root, mode).mkdir(parents=True, exist_ok=True)
        manifest_path(processed_root, mode).parent.mkdir(parents=True, exist_ok=True)


def read_phone_csv(session_dir: Path, filename: str) -> np.ndarray:
    """Read a phone IMU CSV as float64 and drop malformed or non-finite rows."""

    path = session_dir / filename
    if not path.exists():
        raise SkipSession(f"missing {filename}")

    rows = []
    dropped = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SkipSession(f"empty {filename}") from exc
        if header[: len(PHONE_COLUMNS)] != PHONE_COLUMNS:
            raise SkipSession(f"unexpected {filename} header: {header}")

        for row in reader:
            if len(row) < len(PHONE_COLUMNS):
                dropped += 1
                continue
            try:
                values = [float(value) for value in row[: len(PHONE_COLUMNS)]]
            except ValueError:
                dropped += 1
                continue
            if not np.all(np.isfinite(values)):
                dropped += 1
                continue
            rows.append(values)

    if not rows:
        raise SkipSession(f"empty {filename}")
    if dropped:
        print(f"  {session_dir.parent.name}/{session_dir.name} dropped malformed {filename} rows={dropped}", flush=True)
    return np.asarray(rows, dtype=np.float64)


def clean_phone_timeline(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by phone timestamp and keep the first row for duplicate timestamps."""

    time_ms = data[:, 0]
    values = data[:, 1:4]

    order = np.argsort(time_ms, kind="stable")
    time_ms = time_ms[order]
    values = values[order]

    time_ms, first_idx = np.unique(time_ms, return_index=True)
    values = values[first_idx]

    if len(time_ms) < 2:
        raise SkipSession("not enough unique phone samples")
    return time_ms, values


def check_average_hz(time_ms: np.ndarray, signal_name: str) -> float:
    """Reject a session if one phone signal's average raw frame rate is below 150 Hz."""

    duration_sec = (time_ms[-1] - time_ms[0]) / 1000.0
    if duration_sec <= 0:
        raise SkipSession(f"non-positive {signal_name} duration")

    hz = len(time_ms) / duration_sec
    if hz < MIN_RAW_HZ:
        raise SkipSession("raw average hz below 150")
    return hz


def build_quality_bins(time_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1-second bins; every saved segment must cover only valid bins."""

    start = math.floor(time_ms[0] / 1000.0) * 1000.0
    end = math.ceil(time_ms[-1] / 1000.0) * 1000.0
    bins = np.arange(start, end + 1000.0, 1000.0)
    counts, _ = np.histogram(time_ms, bins=bins)
    return bins, counts >= MIN_RAW_HZ


def build_common_target_grid(acc_time: np.ndarray, gyro_time: np.ndarray) -> np.ndarray:
    """Build a 200 Hz grid inside the shared acc/gyro time range."""

    common_start = max(acc_time[0], gyro_time[0])
    common_end = min(acc_time[-1], gyro_time[-1])
    target_t = np.arange(common_start, common_end, PERIOD_MS)
    if len(target_t) < MODES["1000f"]["frames"]:
        raise SkipSession("common phone duration shorter than 1000 frames")
    return target_t


def interpolate_xyz(source_t: np.ndarray, source_xyz: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """Use first-order linear interpolation without extrapolation."""

    if target_t[0] < source_t[0] or target_t[-1] > source_t[-1]:
        raise SkipSession("target grid outside source timeline")
    return np.column_stack([np.interp(target_t, source_t, source_xyz[:, i]) for i in range(3)])


def target_quality_index(target_t: np.ndarray, bins: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Map each 200 Hz sample to a precomputed raw-quality bin."""

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
    """Final in-memory check before writing one segment."""

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
    """Write one JSONL row for a saved segment."""

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


def process_session(session_dir: Path, processed_root: Path, writers: dict, counters: dict, summary: dict) -> dict:
    """Process one daily session and directly write all passing phone segments."""

    before = summary["segments"].copy()

    acc_raw = read_phone_csv(session_dir, "phone_acc.csv")
    gyro_raw = read_phone_csv(session_dir, "phone_gyro.csv")
    acc_time, acc_xyz = clean_phone_timeline(acc_raw)
    gyro_time, gyro_xyz = clean_phone_timeline(gyro_raw)

    check_average_hz(acc_time, "phone_acc")
    check_average_hz(gyro_time, "phone_gyro")

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
            acc_ok = segment_is_valid(acc_bin_idx, acc_valid, start, end)
            gyro_ok = segment_is_valid(gyro_bin_idx, gyro_valid, start, end)
            if not (acc_ok and gyro_ok):
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

    return {
        "10s": summary["segments"]["10s"] - before["10s"],
        "1000f": summary["segments"]["1000f"] - before["1000f"],
    }


def run(dataset_root: Path | None, session_dir: Path | None, processed_root: Path, overwrite: bool) -> dict:
    """Run the daily phone processing pipeline."""

    sessions = get_sessions(dataset_root, session_dir)
    prepare_output(processed_root, overwrite)

    summary = {
        "seen": len(sessions),
        "processed": 0,
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
        for idx, session in enumerate(sessions, start=1):
            try:
                written = process_session(session, processed_root, writers, counters, summary)
                print(
                    f"[{idx}/{total}] {session.parent.name}/{session.name} OK "
                    f"10s={written['10s']} 1000f={written['1000f']} "
                    f"total10s={summary['segments']['10s']} total1000f={summary['segments']['1000f']}",
                    flush=True,
                )
            except SkipSession as exc:
                reason = str(exc)
                if reason == "raw average hz below 150":
                    summary["skipped_low_hz"] += 1
                else:
                    summary["skipped_other"] += 1
                summary["skip_reasons"][reason] = summary["skip_reasons"].get(reason, 0) + 1
                print(f"[{idx}/{total}] {session.parent.name}/{session.name} SKIP {reason}", flush=True)

    return summary


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    summary = run(args.dataset_root, args.session_dir, args.processed_root, args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
