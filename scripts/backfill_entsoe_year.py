from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from src.eem_intel.extractors.entsoe import EntsoeExtractor
from src.eem_intel.http import HttpClient
from src.eem_intel.settings import ROOT, load_settings


NO_DATA_MARKERS = (
    "no matching data found",
    "no data found",
    "no data available",
)


def is_no_data(exc: requests.HTTPError) -> bool:
    if exc.response is None:
        return False
    text = (exc.response.text or "").lower()
    return any(marker in text for marker in NO_DATA_MARKERS)


def should_split(exc: Exception) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        text = (exc.response.text or "").lower()
        if exc.response.status_code in {408, 413, 429, 500, 502, 503, 504}:
            return True
        return any(
            phrase in text
            for phrase in (
                "amount of requested data exceeds",
                "requested data exceeds",
                "too many",
                "maximum allowed",
                "limit",
            )
        )
    return False


def adaptive_fetch(fetch_fn, start: datetime, end: datetime, *, depth: int = 0) -> pd.DataFrame:
    try:
        return fetch_fn(start, end)
    except requests.HTTPError as exc:
        if is_no_data(exc):
            return pd.DataFrame()
        if not should_split(exc):
            raise
        error = exc
    except (requests.Timeout, requests.ConnectionError) as exc:
        error = exc

    duration = end - start
    if duration <= timedelta(days=1):
        raise error

    midpoint = start + duration / 2
    midpoint = midpoint.replace(minute=0, second=0, microsecond=0)
    if midpoint <= start or midpoint >= end:
        raise error

    indent = "  " * depth
    print(
        f"{indent}Splitting request {start.isoformat()} -> {end.isoformat()} "
        f"after {type(error).__name__}",
        flush=True,
    )
    left = adaptive_fetch(fetch_fn, start, midpoint, depth=depth + 1)
    right = adaptive_fetch(fetch_fn, midpoint, end, depth=depth + 1)
    frames = [df for df in (left, right) if not df.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_cols = [
        c
        for c in ["timestamp_utc", "mRID", "position"]
        if c in df.columns
    ]
    return df.drop_duplicates().sort_values(sort_cols).reset_index(drop=True)


def save_frame(df: pd.DataFrame, path: Path) -> str | None:
    if df.empty:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    return str(path)


def member_manifest(df: pd.DataFrame, path: Path, root: Path) -> dict:
    rows = int(len(df))
    result = {
        "status": "ok" if rows else "empty",
        "rows": rows,
    }
    if rows:
        result.update(
            {
                "file": str(path.relative_to(root)),
                "timestamp_min": str(df["timestamp_utc"].min()),
                "timestamp_max": str(df["timestamp_utc"].max()),
                "unique_timestamps": int(df["timestamp_utc"].nunique()),
                "resolutions": sorted(str(x) for x in df["resolution"].dropna().unique()),
                "timeseries": int(df["mRID"].nunique(dropna=True)),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    settings = load_settings()
    if not settings.entsoe_token:
        raise RuntimeError("ENTSOE_SECURITY_TOKEN is not available")

    year = args.year
    today_utc = datetime.now(timezone.utc).date()
    start_date = date(year, 1, 1)
    end_date_exclusive = min(date(year + 1, 1, 1), today_utc)
    if end_date_exclusive <= start_date:
        raise RuntimeError(f"No completed UTC days available for {year}")

    start = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(end_date_exclusive, datetime.min.time(), tzinfo=timezone.utc)

    cfg = settings.config
    extractor = EntsoeExtractor(
        base_url=settings.entsoe_base_url,
        security_token=settings.entsoe_token,
        http=HttpClient(timeout=120),
    )

    root = ROOT / "data" / "entsoe_backfill" / str(year)
    root.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "year": year,
        "start_utc": start.isoformat(),
        "end_utc_exclusive": end.isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "ENTSO-E Transparency Platform Web API",
        "datasets": {},
    }

    required_member_failures: list[str] = []

    for dataset_name, ds in cfg["entsoe"]["datasets"].items():
        print(f"\n=== {dataset_name} ===", flush=True)
        required = bool(ds.get("required", False))
        dataset_manifest = {
            "scope": ds["scope"],
            "required": required,
            "members": {},
            "total_rows": 0,
        }

        if ds["scope"] == "zone":
            members = [
                (zone_name, zone_cfg["eic"])
                for zone_name, zone_cfg in cfg["zones"].items()
            ]
            for zone_name, eic in members:
                print(f"Fetching {dataset_name} / {zone_name} / {year}", flush=True)
                try:
                    df = adaptive_fetch(
                        lambda a, b, ds=ds, eic=eic: extractor.fetch_zone(ds, eic, a, b),
                        start,
                        end,
                    )
                    if not df.empty:
                        df["zone"] = zone_name
                    df = clean_frame(df)
                    path = root / dataset_name / f"{zone_name}.parquet"
                    save_frame(df, path)
                    info = member_manifest(df, path, root)
                    dataset_manifest["members"][zone_name] = info
                    dataset_manifest["total_rows"] += info["rows"]
                    if required and info["status"] != "ok":
                        required_member_failures.append(f"{dataset_name}/{zone_name}: empty")
                except Exception as exc:
                    dataset_manifest["members"][zone_name] = {
                        "status": "error",
                        "rows": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"ERROR {dataset_name}/{zone_name}: {exc}", flush=True)
                    if required:
                        required_member_failures.append(
                            f"{dataset_name}/{zone_name}: {type(exc).__name__}"
                        )

        elif ds["scope"] == "border":
            for zone_a, zone_b in cfg["borders"]:
                for src, dst in ((zone_a, zone_b), (zone_b, zone_a)):
                    key = f"{src}_to_{dst}"
                    print(f"Fetching {dataset_name} / {src}->{dst} / {year}", flush=True)
                    try:
                        df = adaptive_fetch(
                            lambda a, b, ds=ds, src=src, dst=dst: extractor.fetch_border(
                                ds,
                                cfg["zones"][src]["eic"],
                                cfg["zones"][dst]["eic"],
                                a,
                                b,
                            ),
                            start,
                            end,
                        )
                        if not df.empty:
                            df["from_zone"] = src
                            df["to_zone"] = dst
                        df = clean_frame(df)
                        path = root / dataset_name / f"{key}.parquet"
                        save_frame(df, path)
                        info = member_manifest(df, path, root)
                        dataset_manifest["members"][key] = info
                        dataset_manifest["total_rows"] += info["rows"]
                        if required and info["status"] != "ok":
                            required_member_failures.append(f"{dataset_name}/{key}: empty")
                    except Exception as exc:
                        dataset_manifest["members"][key] = {
                            "status": "error",
                            "rows": 0,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        print(f"ERROR {dataset_name}/{src}->{dst}: {exc}", flush=True)
                        if required:
                            required_member_failures.append(
                                f"{dataset_name}/{key}: {type(exc).__name__}"
                            )
        else:
            raise ValueError(f"Unknown ENTSO-E scope: {ds['scope']}")

        manifest["datasets"][dataset_name] = dataset_manifest

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    attribution = root / "ATTRIBUTION.txt"
    attribution.write_text(
        "Source: ENTSO-E Transparency Platform (https://transparency.entsoe.eu/)\n"
        "Retrieved through the ENTSO-E Transparency Platform Web API.\n"
        "Use and redistribution must comply with the ENTSO-E Transparency Platform\n"
        "terms and the applicable re-use licence; retain source attribution.\n",
        encoding="utf-8",
    )

    print(f"\nManifest: {manifest_path}")
    for name, info in manifest["datasets"].items():
        print(f"{name}: {info['total_rows']} rows")

    if required_member_failures:
        raise RuntimeError(
            "Required ENTSO-E member downloads incomplete: "
            + " | ".join(required_member_failures)
        )

    print(f"ENTSO-E {year} backfill: COMPLETE")


if __name__ == "__main__":
    main()
