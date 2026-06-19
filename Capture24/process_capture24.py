"""One-off Capture24 watch accelerometer processor.

Output follows the repository processed-data format:
  /Volumes/Felix_Backups/Processed/sources/capture24/segments/{mode}/watch/*.npz
  /Volumes/Felix_Backups/Processed/sources/capture24/manifests/watch_{mode}.jsonl
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


DATASET_ROOT = Path("/Users/zixuanwang/Downloads/capture24")
PROCESSED_ROOT = Path("/Volumes/Felix_Backups/Processed")

SRC = "capture24"
DEVICE = "watch"
FREQ = 100.0
PERIOD_MS = 1000.0 / FREQ
TIME_ATOL_MS = 1e-6
G_TO_MPS2 = 9.80665

CSV_HEADER = ["time", "x", "y", "z", "annotation"]
LABEL_COLUMN = "label:WillettsSpecific2018"
MODES = {
    "10s": {"frames": 1000, "duration_sec": 10.0},
    "1000f": {"frames": 1000, "duration_sec": 10.0},
}


class SkipParticipant(Exception):
    """Raised when a participant file should not produce Capture24 output."""


def parse_args() -> argparse.Namespace:
    """Read command-line arguments for either one participant file or the whole dataset."""

    parser = argparse.ArgumentParser(description="Process Capture24 accelerometer CSVs into NPZ training segments.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset-root", type=Path, default=None)
    source.add_argument("--participant-file", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional debug limit on data rows read from each participant. Omit for full processing.",
    )
    parser.add_argument(
        "--max-participants",
        type=int,
        default=None,
        help="Process only the first N sorted participant files. Omit for all participants.",
    )
    parser.add_argument(
        "--progress-every-rows",
        type=int,
        default=5_000_000,
        help="Print row-level progress for long participant files. Use 0 to disable.",
    )
    return parser.parse_args()


def discover_participants(dataset_root: Path) -> list[Path]:
    """Find all Capture24 participant files in P*.csv.gz format."""

    return sorted(path for path in dataset_root.glob("P*.csv.gz") if path.is_file())


def get_participants(
    dataset_root: Path | None,
    participant_file: Path | None,
    max_participants: int | None,
) -> list[Path]:
    """Resolve input mode into a participant file list."""

    if participant_file is not None:
        if not participant_file.exists():
            raise FileNotFoundError(f"missing participant file: {participant_file}")
        return [participant_file]

    participants = discover_participants(dataset_root or DATASET_ROOT)
    if max_participants is not None:
        if max_participants <= 0:
            raise ValueError("--max-participants must be positive")
        participants = participants[:max_participants]
    return participants


def source_root(processed_root: Path) -> Path:
    """Return processed/sources/capture24."""

    return processed_root / "sources" / SRC


def segment_dir(processed_root: Path, mode: str) -> Path:
    """Return the output directory for one segment mode."""

    return source_root(processed_root) / "segments" / mode / DEVICE


def manifest_path(processed_root: Path, mode: str) -> Path:
    """Return the source manifest path for this device and one segment mode."""

    return source_root(processed_root) / "manifests" / f"{DEVICE}_{mode}.jsonl"


def prepare_output(processed_root: Path, overwrite: bool) -> None:
    """Create Capture24 watch output folders; optionally rebuild existing watch outputs."""

    watch_dirs = [segment_dir(processed_root, mode) for mode in MODES]
    manifests = [manifest_path(processed_root, mode) for mode in MODES]

    if overwrite:
        for path in watch_dirs:
            if path.exists():
                shutil.rmtree(path)
        for path in manifests:
            if path.exists():
                path.unlink()
    else:
        existing = [str(path) for path in watch_dirs + manifests if path.exists()]
        if existing:
            raise FileExistsError("output exists; use --overwrite: " + ", ".join(existing))

    for mode in MODES:
        segment_dir(processed_root, mode).mkdir(parents=True, exist_ok=True)
        manifest_path(processed_root, mode).parent.mkdir(parents=True, exist_ok=True)


def load_label_map(dataset_root: Path) -> dict[str, str]:
    """Read annotation-label-dictionary.csv and map raw annotation to main activity."""

    path = dataset_root / "annotation-label-dictionary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing label dictionary: {path}")

    label_map: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "annotation" not in reader.fieldnames or LABEL_COLUMN not in reader.fieldnames:
            raise ValueError(f"unexpected label dictionary header: {reader.fieldnames}")
        for row in reader:
            annotation = (row.get("annotation") or "").strip()
            activity = (row.get(LABEL_COLUMN) or "").strip()
            if annotation and activity:
                label_map[annotation] = activity

    if not label_map:
        raise ValueError(f"empty label dictionary: {path}")
    return label_map


def open_participant_csv(path: Path):
    """Open one gzip participant CSV in text mode."""

    return gzip.open(path, "rt", encoding="utf-8", newline="")


def read_header(reader: csv.reader, participant_file: Path) -> None:
    """Validate the Capture24 participant CSV header."""

    try:
        header = next(reader)
    except StopIteration as exc:
        raise SkipParticipant("empty csv") from exc
    if header != CSV_HEADER:
        raise SkipParticipant(f"unexpected csv header: {header}")


def parse_time_ms(value: str, start_time: datetime | None) -> tuple[datetime, float]:
    """Parse an ISO timestamp and return milliseconds relative to the participant start."""

    try:
        current = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SkipParticipant(f"cannot parse time: {value}") from exc

    if start_time is None:
        return current, 0.0
    return start_time, (current - start_time).total_seconds() * 1000.0


def parse_acc_g(row: list[str], row_number: int) -> tuple[float, float, float]:
    """Parse x/y/z values in g units and reject non-finite rows."""

    try:
        x = float(row[1])
        y = float(row[2])
        z = float(row[3])
    except (IndexError, ValueError) as exc:
        raise SkipParticipant(f"cannot parse acceleration at row {row_number}") from exc

    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        raise SkipParticipant(f"non-finite acceleration at row {row_number}")
    return x, y, z


def validate_participant_file(participant_file: Path, max_rows: int | None, progress_every_rows: int) -> int:
    """Validate raw-order 10 ms timing and finite acceleration before writing output."""

    row_count = 0
    start_time: datetime | None = None
    previous_ms: float | None = None

    with open_participant_csv(participant_file) as f:
        reader = csv.reader(f)
        read_header(reader, participant_file)
        for row_number, row in enumerate(reader, start=2):
            if max_rows is not None and row_count >= max_rows:
                break
            if len(row) != len(CSV_HEADER):
                raise SkipParticipant(f"unexpected column count at row {row_number}")

            start_time, time_ms = parse_time_ms(row[0], start_time)
            parse_acc_g(row, row_number)

            if previous_ms is not None:
                dt = time_ms - previous_ms
                if not np.isclose(dt, PERIOD_MS, rtol=0.0, atol=TIME_ATOL_MS):
                    raise SkipParticipant(f"raw time axis is not strictly stable at 10 ms near row {row_number}")

            previous_ms = time_ms
            row_count += 1
            if progress_every_rows > 0 and row_count % progress_every_rows == 0:
                print(f"  {participant_file.name} validate rows={row_count}", flush=True)

    if row_count < MODES["1000f"]["frames"]:
        raise SkipParticipant("fewer than 1000 rows")
    return row_count


def unique_in_order(values: list[str]) -> list[str]:
    """Return unique strings while preserving first-seen order."""

    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def build_segment_label(annotations: list[str], label_map: dict[str, str]) -> dict | None:
    """Create the manifest label object; return None if the segment label is invalid."""

    if any(not annotation for annotation in annotations):
        return None

    activities = []
    for annotation in annotations:
        activity = label_map.get(annotation)
        if activity is None:
            return None
        activities.append(activity)

    unique_activities = unique_in_order(activities)
    if len(unique_activities) != 1:
        return None

    return {
        "activity": unique_activities[0],
        "annotations": unique_in_order(annotations),
    }


def build_acc_array(acc_rows_g: list[tuple[float, float, float]]) -> np.ndarray:
    """Build a float32 [1000, 4] acc array with segment-local time in milliseconds."""

    frames = len(acc_rows_g)
    local_t = (np.arange(frames, dtype=np.float64) * PERIOD_MS).reshape(frames, 1)
    acc_mps2 = np.asarray(acc_rows_g, dtype=np.float64) * G_TO_MPS2
    return np.column_stack([local_t, acc_mps2]).astype(np.float32)


def acc_array_is_valid(acc: np.ndarray, frames: int) -> bool:
    """Final in-memory check before writing one accelerometer segment."""

    return (
        acc.shape == (frames, 4)
        and acc.dtype == np.float32
        and acc[0, 0] == 0.0
        and np.allclose(np.diff(acc[:, 0]), PERIOD_MS, rtol=0.0, atol=TIME_ATOL_MS)
        and np.all(np.isfinite(acc))
    )


def write_manifest_line(writer, mode: str, filename: str, frames: int, duration_sec: float, label: dict) -> None:
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
        "has_gyro": False,
        "label": label,
    }
    writer.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_segment(
    processed_root: Path,
    writers: dict,
    counters: dict,
    summary: dict,
    acc: np.ndarray,
    label: dict,
) -> dict[str, int]:
    """Write one accepted 1000-frame segment into both Capture24 output modes."""

    written = {mode: 0 for mode in MODES}
    for mode, spec in MODES.items():
        frames = spec["frames"]
        if not acc_array_is_valid(acc, frames):
            continue

        filename = f"{counters[mode]:08d}.npz"
        counters[mode] += 1
        np.savez(segment_dir(processed_root, mode) / filename, acc=acc)
        write_manifest_line(writers[mode], mode, filename, frames, spec["duration_sec"], label)
        summary["segments"][mode] += 1
        written[mode] += 1
    return written


def process_participant(
    participant_file: Path,
    processed_root: Path,
    label_map: dict[str, str],
    writers: dict,
    counters: dict,
    summary: dict,
    max_rows: int | None,
    progress_every_rows: int,
) -> dict[str, int]:
    """Validate and process one participant file in original row order."""

    validate_participant_file(participant_file, max_rows, progress_every_rows)

    before = summary["segments"].copy()
    acc_rows_g: list[tuple[float, float, float]] = []
    annotations: list[str] = []
    candidate_segments = 0
    processed_rows = 0

    with open_participant_csv(participant_file) as f:
        reader = csv.reader(f)
        read_header(reader, participant_file)
        for row_number, row in enumerate(reader, start=2):
            if max_rows is not None and processed_rows >= max_rows:
                break

            processed_rows += 1
            acc_rows_g.append(parse_acc_g(row, row_number))
            annotations.append(row[4].strip())

            if len(acc_rows_g) < 1000:
                continue

            candidate_segments += 1
            label = build_segment_label(annotations, label_map)
            if label is not None:
                acc = build_acc_array(acc_rows_g)
                written = write_segment(processed_root, writers, counters, summary, acc, label)
                for mode, count in written.items():
                    summary["accepted_candidates"][mode] += count
            else:
                summary["dropped_label"] += 1

            acc_rows_g = []
            annotations = []

            if progress_every_rows > 0 and processed_rows % progress_every_rows == 0:
                print(
                    f"  {participant_file.name} write rows={processed_rows} "
                    f"10s_total={summary['segments']['10s']} 1000f_total={summary['segments']['1000f']}",
                    flush=True,
                )

    if acc_rows_g:
        summary["dropped_tail"] += 1

    written_now = {
        "10s": summary["segments"]["10s"] - before["10s"],
        "1000f": summary["segments"]["1000f"] - before["1000f"],
    }
    if written_now["10s"] > 0 or written_now["1000f"] > 0:
        summary["processed"] += 1
    return written_now


def run(
    dataset_root: Path | None,
    participant_file: Path | None,
    processed_root: Path,
    overwrite: bool,
    max_rows: int | None,
    max_participants: int | None,
    progress_every_rows: int,
) -> dict:
    """Run the Capture24 processing pipeline."""

    participants = get_participants(dataset_root, participant_file, max_participants)
    if dataset_root is not None:
        dictionary_root = dataset_root
    elif participant_file is not None:
        dictionary_root = participant_file.parent
    else:
        dictionary_root = DATASET_ROOT
    label_map = load_label_map(dictionary_root)
    prepare_output(processed_root, overwrite)

    summary = {
        "seen": len(participants),
        "processed": 0,
        "skipped_time": 0,
        "skipped_other": 0,
        "dropped_label": 0,
        "dropped_tail": 0,
        "accepted_candidates": {"10s": 0, "1000f": 0},
        "segments": {"10s": 0, "1000f": 0},
        "skip_reasons": {},
    }
    counters = {"10s": 1, "1000f": 1}

    with manifest_path(processed_root, "10s").open("w", encoding="utf-8") as man_10s, manifest_path(
        processed_root, "1000f"
    ).open("w", encoding="utf-8") as man_1000f:
        writers = {"10s": man_10s, "1000f": man_1000f}
        total = len(participants)
        for idx, participant in enumerate(participants, start=1):
            try:
                written = process_participant(
                    participant,
                    processed_root,
                    label_map,
                    writers,
                    counters,
                    summary,
                    max_rows,
                    progress_every_rows,
                )
                print(
                    f"[{idx}/{total}] {participant.name} OK "
                    f"10s={written['10s']} 1000f={written['1000f']} "
                    f"total10s={summary['segments']['10s']} total1000f={summary['segments']['1000f']}",
                    flush=True,
                )
            except SkipParticipant as exc:
                reason = str(exc)
                if "raw time axis" in reason:
                    summary["skipped_time"] += 1
                else:
                    summary["skipped_other"] += 1
                summary["skip_reasons"][reason] = summary["skip_reasons"].get(reason, 0) + 1
                print(f"[{idx}/{total}] {participant.name} SKIP {reason}", flush=True)

    return summary


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    summary = run(
        args.dataset_root,
        args.participant_file,
        args.processed_root,
        args.overwrite,
        args.max_rows,
        args.max_participants,
        args.progress_every_rows,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
