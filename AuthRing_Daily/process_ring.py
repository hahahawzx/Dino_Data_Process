"""One-off AuthRing daily ring IMU processor.

Output follows the repository processed-data format:
  /Volumes/Felix_Backups/Processed/sources/authring_daily/segments/{mode}/ring/*.npz
  /Volumes/Felix_Backups/Processed/sources/authring_daily/manifests/ring_{mode}.jsonl
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
DEVICE = "ring"
FREQ = 200.0
PERIOD_MS = 1000.0 / FREQ
MIN_RAW_HZ = 150
RING_TICKS_PER_SEC = 16384.0
LABEL = {"finger": "right_index"}

RING_COLUMNS = ["timestamp", "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"]
MODES = {
    "10s": {"frames": 2000, "duration_sec": 10.0},
    "1000f": {"frames": 1000, "duration_sec": 5.0},
}


class SkipSession(Exception):
    """Raised when a daily session should not produce ring output."""


def parse_args() -> argparse.Namespace:
    """Read command-line arguments for either one session or the whole daily dataset."""

    parser = argparse.ArgumentParser(description="Process AuthRing daily ring IMU into NPZ training segments.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset-root", type=Path, default=None)
    source.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_sessions(dataset_root: Path) -> list[Path]:
    """Find person/session directories that contain ring_imu.csv."""

    sessions = []
    for participant_dir in sorted(dataset_root.iterdir()):
        if not participant_dir.is_dir():
            continue
        for session_dir in sorted(participant_dir.iterdir()):
            if session_dir.is_dir() and (session_dir / "ring_imu.csv").exists():
                sessions.append(session_dir)
    return sessions


def get_sessions(dataset_root: Path | None, session_dir: Path | None) -> list[Path]:
    """Resolve input mode into a session list."""

    if session_dir is not None:
        if not (session_dir / "ring_imu.csv").exists():
            raise FileNotFoundError(f"missing ring_imu.csv: {session_dir}")
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
    """Create ring output folders; optionally rebuild existing daily ring outputs."""

    ring_dirs = [segment_dir(processed_root, mode) for mode in MODES]
    manifests = [manifest_path(processed_root, mode) for mode in MODES]

    if overwrite:
        for path in ring_dirs:
            if path.exists():
                shutil.rmtree(path)
        for path in manifests:
            if path.exists():
                path.unlink()
    else:
        existing = [str(path) for path in ring_dirs + manifests if path.exists()]
        if existing:
            raise FileExistsError("output exists; use --overwrite: " + ", ".join(existing))

    for mode in MODES:
        segment_dir(processed_root, mode).mkdir(parents=True, exist_ok=True)
        manifest_path(processed_root, mode).parent.mkdir(parents=True, exist_ok=True)


def read_ring_csv(session_dir: Path) -> np.ndarray:
    """Read ring_imu.csv as float64 and drop malformed or non-finite rows."""

    path = session_dir / "ring_imu.csv"
    if not path.exists():
        raise SkipSession("missing ring_imu.csv")

    rows = []
    dropped = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SkipSession("empty ring_imu.csv") from exc
        if header[: len(RING_COLUMNS)] != RING_COLUMNS:
            raise SkipSession(f"unexpected ring_imu.csv header: {header}")

        for row in reader:
            if len(row) < len(RING_COLUMNS):
                dropped += 1
                continue
            try:
                values = [float(value) for value in row[: len(RING_COLUMNS)]]
            except ValueError:
                dropped += 1
                continue
            if not np.all(np.isfinite(values)):
                dropped += 1
                continue
            rows.append(values)

    if not rows:
        raise SkipSession("empty ring_imu.csv")
    if dropped:
        print(f"  {session_dir.parent.name}/{session_dir.name} dropped malformed ring_imu.csv rows={dropped}", flush=True)
    return np.asarray(rows, dtype=np.float64)


def clean_ring_timeline(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert ring ticks to ms, sort by time, and keep the first duplicate."""

    time_ms = data[:, 0] / RING_TICKS_PER_SEC * 1000.0
    values = data[:, 1:7]

    order = np.argsort(time_ms, kind="stable")
    time_ms = time_ms[order]
    values = values[order]

    time_ms, first_idx = np.unique(time_ms, return_index=True)
    values = values[first_idx]

    if len(time_ms) < 2:
        raise SkipSession("not enough unique ring samples")
    return time_ms, values


def check_average_hz(time_ms: np.ndarray) -> float:
    """Reject a session if raw average ring frame rate is below 150 Hz."""

    duration_sec = (time_ms[-1] - time_ms[0]) / 1000.0
    if duration_sec <= 0:
        raise SkipSession("non-positive ring duration")

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


def resample_200hz(time_ms: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use first-order linear interpolation to resample ring data to 200 Hz."""

    target_t = np.arange(time_ms[0], time_ms[-1], PERIOD_MS)
    if len(target_t) < MODES["1000f"]["frames"]:
        raise SkipSession("duration shorter than 1000 frames after resampling")

    resampled = np.column_stack([np.interp(target_t, time_ms, values[:, i]) for i in range(values.shape[1])])
    return target_t, resampled


def target_quality_index(target_t: np.ndarray, bins: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Map each 200 Hz sample to a precomputed raw-quality bin."""

    idx = np.searchsorted(bins, target_t, side="right") - 1
    return np.clip(idx, 0, len(valid) - 1)


def segment_is_valid(bin_idx: np.ndarray, valid: np.ndarray, start: int, end: int) -> bool:
    """Check whether all 1-second bins touched by this segment are valid."""

    touched = np.unique(bin_idx[start:end])
    return bool(len(touched) > 0 and np.all(valid[touched]))


def build_npz_arrays(target_t: np.ndarray, resampled: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Build acc and gyro arrays with segment-local time_ms in column 0."""

    local_t = target_t[start:end] - target_t[start]
    acc = np.column_stack([local_t, resampled[start:end, 0:3]]).astype(np.float32)
    gyro = np.column_stack([local_t, resampled[start:end, 3:6]]).astype(np.float32)
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
    """Process one daily session and directly write all passing ring segments."""

    before = summary["segments"].copy()

    raw_csv = read_ring_csv(session_dir)
    time_ms, values = clean_ring_timeline(raw_csv)
    check_average_hz(time_ms)

    bins, valid = build_quality_bins(time_ms)
    target_t, resampled = resample_200hz(time_ms, values)
    bin_idx = target_quality_index(target_t, bins, valid)

    wrote_any = False
    for mode, spec in MODES.items():
        frames = spec["frames"]
        for start in range(0, len(target_t) - frames + 1, frames):
            end = start + frames
            if not segment_is_valid(bin_idx, valid, start, end):
                continue

            acc, gyro = build_npz_arrays(target_t, resampled, start, end)
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
    """Run the daily ring processing pipeline."""

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
