from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from src.eem_intel.extractors.entsoe import EntsoeExtractor
from src.eem_intel.http import HttpClient
from src.eem_intel.settings import ROOT, load_settings


NO_DATA_MARKERS = (
    "no matching data",
    "no data matching",
    "no data found",
    "no data available",
    "there is no data",
    "no content",
)

SPLIT_MARKERS = (
    "amount of requested data exceeds",
    "requested data exceeds",
    "too much data",
    "too many",
    "maximum allowed",
    "request is too large",
    "data limit",
)


def response_text(exc: requests.HTTPError) -> str:
    if exc.response is None:
        return ""
    return (exc.response.text or "").strip()


def compact_error(exc: Exception, limit: int = 500) -> str:
    if isinstance(exc, requests.HTTPError):
        body = response_text(exc).replace("\n", " ").replace("\r", " ")
        if body:
            return f"{type(exc).__name__}: {exc}; body={body[:limit]}"
    return f"{type(exc).__name__}: {exc}"


def is_no_data(exc: requests.HTTPError) -> bool:
    text = response_text(exc).lower()
    return any(marker in text for marker in NO_DATA_MARKERS)


def should_split(exc: Exception) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        text = response_text(exc).lower()
        if exc.response.status_code in {408, 413, 429, 500, 502, 503, 504}:
            return True
        return any(marker in text for marker in SPLIT_MARKERS)
    return False


def adaptive_fetch(
    fetch_fn: Callable[[datetime, datetime], pd.DataFrame],
    start: datetime,
    end: datetime,
    *,
    min_window: timedelta = timedelta(days=1),
    depth: int = 0,
) -> pd.DataFrame:
    """Fetch one bounded window, splitting only when the API says it is too large/transient."""
    try:
        return fetch_fn(start, end)
    except requests.HTTPError as exc:
        if is_no_data(exc):
            return pd.DataFrame()
        if not should_split(exc):
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
    print(
        f"{indent}Splitting {start.isoformat()} -> {end.isoformat()} "
        f"after {type(error).__name__}",
        flush=True,
    )
    left = adaptive_fetch(fetch_fn, start, midpoint, min_window=min_window, depth=depth + 1)
    right = adaptive_fetch(fetch_fn, midpoint, end, min_window=min_window, depth=depth + 1)
    frames = [frame for frame in (left, right) if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def iter_chunks(start: datetime, end: datetime, days: int):
    cursor = start
    step = timedelta(days=max(1, int(days)))
    while cursor < end:
        chunk_end = min(cursor + step, end)
        yield cursor, chunk_end
        cursor = chunk_end


def fetch_member_chunked(
    fetch_fn: Callable[[datetime, datetime], pd.DataFrame],
    start: datetime,
    end: datetime,
    *,
    chunk_days: int,
) -> tuple[pd.DataFrame, dict]:
    frames: list[pd.DataFrame] = []
    errors: list[dict] = []
    no_data_chunks = 0
    total_chunks = 0

    for chunk_start, chunk_end in iter_chunks(start, end, chunk_days):
        total_chunks += 1
        print(
            f"  chunk {chunk_start.date()} -> {chunk_end.date()}",
            flush=True,
        )
        try:
            chunk = adaptive_fetch(fetch_fn, chunk_start, chunk_end)
            if chunk.empty:
                no_data_chunks += 1
            else:
                frames.append(chunk)
        except Exception as exc:
            message = compact_error(exc)
            print(f"  ERROR {message}", flush=True)
            errors.append(
                {
                    "start_utc": chunk_start.isoformat(),
                    "end_utc": chunk_end.isoformat(),
                    "error": message,
                }
            )

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates()
        sort_cols = [c for c in ["timestamp_utc", "mRID", "position"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        df = pd.DataFrame()

    report = {
        "total_chunks": total_chunks,
        "no_data_chunks": no_data_chunks,
        "error_chunks": len(errors),
        "errors": errors,
    }
    return df, report


def save_frame(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    settings = load_settings()
    if not settings.entsoe_token:
        raise RuntimeError("ENTSOE_SECURITY_TOKEN is not available")

    cfg = settings.config
    datasets = cfg["entsoe"]["datasets"]
    if args.dataset not in datasets:
        raise ValueError(
            f"Unknown dataset {args.dataset!r}. Available: {', '.join(datasets)}"
        )

    year = args.year
    dataset_name = args.dataset
    ds = datasets[dataset_name]

    today_utc = datetime.now(timezone.utc).date()
    start_date = date(year, 1, 1)
    end_date_exclusive = min(date(year + 1, 1, 1), today_utc)
    if end_date_exclusive <= start_date:
        raise RuntimeError(f"No completed UTC days available for {year}")

    start = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(end_date_exclusive, datetime.min.time(), tzinfo=timezone.utc)

    default_chunk_days = int(cfg["entsoe"].get("chunk_days", 31))
    chunk_days = int(ds.get("chunk_days", default_chunk_days))

    extractor = EntsoeExtractor(
        base_url=settings.entsoe_base_url,
        security_token=settings.entsoe_token,
        http=HttpClient(timeout=90),
    )

    root = ROOT / "data" / "entsoe_backfill" / str(year) / dataset_name
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "year": year,
        "dataset": dataset_name,
        "scope": ds["scope"],
        "required": bool(ds.get("required", False)),
        "chunk_days": chunk_days,
        "start_utc": start.isoformat(),
        "end_utc_exclusive": end.isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "ENTSO-E Transparency Platform Web API",
        "members": {},
        "total_rows": 0,
        "total_error_chunks": 0,
    }

    if ds["scope"] == "zone":
        members = [
            (zone_name, zone_cfg["eic"])
            for zone_name, zone_cfg in cfg["zones"].items()
        ]
        for zone_name, eic in members:
            print(f"\n=== {dataset_name} / {zone_name} / {year} ===", flush=True)
            df, report = fetch_member_chunked(
                lambda a, b, ds=ds, eic=eic: extractor.fetch_zone(ds, eic, a, b),
                start,
                end,
                chunk_days=chunk_days,
            )
            if not df.empty:
                df["zone"] = zone_name
                save_frame(df, root / f"{zone_name}.parquet")

            rows = int(len(df))
            status = "ok"
            if rows == 0 and report["error_chunks"]:
                status = "error"
            elif rows == 0:
                status = "empty"
            elif report["error_chunks"]:
                status = "partial"

            manifest["members"][zone_name] = {
                "status": status,
                "rows": rows,
                "file": f"{zone_name}.parquet" if rows else None,
                **report,
            }
            manifest["total_rows"] += rows
            manifest["total_error_chunks"] += report["error_chunks"]

    elif ds["scope"] == "border":
        for zone_a, zone_b in cfg["borders"]:
            for src, dst in ((zone_a, zone_b), (zone_b, zone_a)):
                key = f"{src}_to_{dst}"
                print(f"\n=== {dataset_name} / {src}->{dst} / {year} ===", flush=True)
                df, report = fetch_member_chunked(
                    lambda a, b, ds=ds, src=src, dst=dst: extractor.fetch_border(
                        ds,
                        cfg["zones"][src]["eic"],
                        cfg["zones"][dst]["eic"],
                        a,
                        b,
                    ),
                    start,
                    end,
                    chunk_days=chunk_days,
                )
                if not df.empty:
                    df["from_zone"] = src
                    df["to_zone"] = dst
                    save_frame(df, root / f"{key}.parquet")

                rows = int(len(df))
                status = "ok"
                if rows == 0 and report["error_chunks"]:
                    status = "error"
                elif rows == 0:
                    status = "empty"
                elif report["error_chunks"]:
                    status = "partial"

                manifest["members"][key] = {
                    "status": status,
                    "rows": rows,
                    "file": f"{key}.parquet" if rows else None,
                    **report,
                }
                manifest["total_rows"] += rows
                manifest["total_error_chunks"] += report["error_chunks"]
    else:
        raise ValueError(f"Unknown ENTSO-E scope: {ds['scope']}")

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    (root / "ATTRIBUTION.txt").write_text(
        "Source: ENTSO-E Transparency Platform (https://transparency.entsoe.eu/)\n"
        "Retrieved through the ENTSO-E Transparency Platform Web API.\n"
        "The project must comply with the ENTSO-E Transparency Platform terms and the "
        "applicable re-use licence for each data item.\n",
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===", flush=True)
    print(f"dataset={dataset_name} year={year} rows={manifest['total_rows']}", flush=True)
    print(f"error_chunks={manifest['total_error_chunks']}", flush=True)
    print(f"manifest={manifest_path}", flush=True)

    if manifest["required"] and manifest["total_rows"] == 0:
        raise RuntimeError(
            f"Required dataset {dataset_name} returned zero rows for {year}"
        )


if __name__ == "__main__":
    main()
