from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from scripts.backfill_entsoe_dataset import compact_error, is_no_data, should_split
from src.eem_intel.extractors.entsoe import EntsoeExtractor
from src.eem_intel.http import HttpClient
from src.eem_intel.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-window-hours", type=int, default=6)
    return parser.parse_args()


def repair_fetch(
    fetch_fn: Callable[[datetime, datetime], pd.DataFrame],
    start: datetime,
    end: datetime,
    *,
    min_window: timedelta,
    depth: int = 0,
) -> pd.DataFrame:
    """Retry one failed range, aggressively splitting ambiguous HTTP 400 responses.

    The original backfill already splits size/rate/server errors. Historical ENTSO-E
    responses sometimes return HTTP 400 acknowledgement documents for a range that
    succeeds once narrowed. For repair only, HTTP 400 is therefore also splittable.
    Confirmed ENTSO-E no-data acknowledgements remain valid empty results.
    """
    try:
        return fetch_fn(start, end)
    except requests.HTTPError as exc:
        if is_no_data(exc):
            return pd.DataFrame()
        can_split = should_split(exc) or (exc.response is not None and exc.response.status_code == 400)
        if not can_split:
            raise
        error: Exception = exc
    except (requests.Timeout, requests.ConnectionError) as exc:
        error = exc

    duration = end - start
    if duration <= min_window:
        raise error

    midpoint = start + duration / 2
    midpoint = midpoint.replace(minute=0, second=0, microsecond=0)
    if midpoint <= start or midpoint >= end:
        raise error

    indent = "  " * depth
    print(f"{indent}repair split {start.isoformat()} -> {end.isoformat()} after {type(error).__name__}", flush=True)
    left = repair_fetch(fetch_fn, start, midpoint, min_window=min_window, depth=depth + 1)
    right = repair_fetch(fetch_fn, midpoint, end, min_window=min_window, depth=depth + 1)
    frames = [frame for frame in (left, right) if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def merge_frame(existing_path: Path, repaired: pd.DataFrame) -> int:
    frames: list[pd.DataFrame] = []
    if existing_path.exists():
        frames.append(pd.read_parquet(existing_path))
    if not repaired.empty:
        frames.append(repaired)
    if not frames:
        return 0

    df = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates()
    sort_cols = [c for c in ["timestamp_utc", "mRID", "position"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(existing_path, index=False, compression="zstd")
    return int(len(df))


def main() -> None:
    args = parse_args()
    settings = load_settings()
    if not settings.entsoe_token:
        raise RuntimeError("ENTSOE_SECURITY_TOKEN is not available")

    cfg = settings.config
    extractor = EntsoeExtractor(
        base_url=settings.entsoe_base_url,
        security_token=settings.entsoe_token,
        http=HttpClient(timeout=90),
    )
    min_window = timedelta(hours=max(1, args.min_window_hours))
    report: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "min_window_hours": args.min_window_hours,
        "attempts": [],
        "summary": {},
    }

    manifests = sorted(args.input_root.glob("entsoe-*/manifest.json"))
    if not manifests:
        raise RuntimeError(f"No manifests found under {args.input_root}")

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_name = manifest["dataset"]
        ds = cfg["entsoe"]["datasets"][dataset_name]
        artifact_dir = manifest_path.parent
        changed = False

        for member, info in manifest.get("members", {}).items():
            original_errors = list(info.get("errors") or [])
            if not original_errors:
                continue

            if ds["scope"] == "zone":
                zone_name = member
                eic = cfg["zones"][zone_name]["eic"]
                fetch_fn = lambda a, b, ds=ds, eic=eic: extractor.fetch_zone(ds, eic, a, b)
                add_identity = lambda df, zone_name=zone_name: df.assign(zone=zone_name)
                file_name = info.get("file") or f"{zone_name}.parquet"
            elif ds["scope"] == "border":
                src, dst = member.split("_to_", 1)
                src_eic = cfg["zones"][src]["eic"]
                dst_eic = cfg["zones"][dst]["eic"]
                fetch_fn = lambda a, b, ds=ds, src_eic=src_eic, dst_eic=dst_eic: extractor.fetch_border(ds, src_eic, dst_eic, a, b)
                add_identity = lambda df, src=src, dst=dst: df.assign(from_zone=src, to_zone=dst)
                file_name = info.get("file") or f"{member}.parquet"
            else:
                raise ValueError(f"Unknown scope {ds['scope']}")

            repaired_frames: list[pd.DataFrame] = []
            unresolved: list[dict] = []
            repaired_no_data = 0

            for error in original_errors:
                start = parse_utc(error["start_utc"])
                end = parse_utc(error["end_utc"])
                attempt = {
                    "artifact": artifact_dir.name,
                    "dataset": dataset_name,
                    "year": manifest["year"],
                    "member": member,
                    "start_utc": start.isoformat(),
                    "end_utc": end.isoformat(),
                    "original_error": error.get("error"),
                }
                print(f"\n=== REPAIR {artifact_dir.name} / {member} / {start} -> {end} ===", flush=True)
                try:
                    frame = repair_fetch(fetch_fn, start, end, min_window=min_window)
                    if frame.empty:
                        repaired_no_data += 1
                        attempt["result"] = "confirmed_no_data"
                        attempt["rows"] = 0
                    else:
                        frame = add_identity(frame)
                        repaired_frames.append(frame)
                        attempt["result"] = "repaired"
                        attempt["rows"] = int(len(frame))
                except Exception as exc:
                    new_error = {
                        "start_utc": start.isoformat(),
                        "end_utc": end.isoformat(),
                        "error": compact_error(exc, limit=1200),
                    }
                    unresolved.append(new_error)
                    attempt["result"] = "unresolved"
                    attempt["rows"] = 0
                    attempt["repair_error"] = new_error["error"]
                report["attempts"].append(attempt)

            if repaired_frames:
                repaired = pd.concat(repaired_frames, ignore_index=True, sort=False).drop_duplicates()
            else:
                repaired = pd.DataFrame()

            output_path = artifact_dir / file_name
            rows = merge_frame(output_path, repaired) if (not repaired.empty or output_path.exists()) else 0
            info["rows"] = rows
            info["file"] = file_name if rows else None
            info["errors"] = unresolved
            info["error_chunks"] = len(unresolved)
            info["no_data_chunks"] = int(info.get("no_data_chunks", 0) or 0) + repaired_no_data
            if rows == 0 and unresolved:
                info["status"] = "error"
            elif rows == 0:
                info["status"] = "empty"
            elif unresolved:
                info["status"] = "partial"
            else:
                info["status"] = "ok"
            changed = True

        if changed:
            manifest["total_rows"] = int(sum(int(x.get("rows", 0) or 0) for x in manifest["members"].values()))
            manifest["total_error_chunks"] = int(sum(int(x.get("error_chunks", 0) or 0) for x in manifest["members"].values()))
            manifest["repair_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    unresolved_attempts = [x for x in report["attempts"] if x["result"] == "unresolved"]
    repaired_attempts = [x for x in report["attempts"] if x["result"] == "repaired"]
    no_data_attempts = [x for x in report["attempts"] if x["result"] == "confirmed_no_data"]
    report["summary"] = {
        "attempted_error_chunks": len(report["attempts"]),
        "repaired_error_chunks": len(repaired_attempts),
        "confirmed_no_data_chunks": len(no_data_attempts),
        "unresolved_error_chunks": len(unresolved_attempts),
        "repaired_rows_fetched": int(sum(int(x.get("rows", 0)) for x in repaired_attempts)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== REPAIR SUMMARY ===", flush=True)
    print(json.dumps(report["summary"], indent=2), flush=True)

    if unresolved_attempts:
        raise RuntimeError(f"ENTSO-E repair left {len(unresolved_attempts)} unresolved error chunks")


if __name__ == "__main__":
    main()
