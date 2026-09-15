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

UA_EIC = "10Y1001C--00003F"
PL_EIC = "10YPL-AREA-----S"

NO_DATA_MARKERS = (
    "no matching data", "no data matching", "no data found", "no data available",
    "there is no data", "no content",
)


def is_no_data(exc: requests.HTTPError) -> bool:
    if exc.response is None:
        return False
    text = (exc.response.text or "").lower()
    return any(marker in text for marker in NO_DATA_MARKERS)


def iter_chunks(start: datetime, end: datetime, days: int = 31):
    cursor = start
    step = timedelta(days=days)
    while cursor < end:
        nxt = min(cursor + step, end)
        yield cursor, nxt
        cursor = nxt


def fetch_direction(extractor, ds, out_eic, in_eic, start, end):
    frames = []
    errors = []
    for a, b in iter_chunks(start, end, 31):
        try:
            part = extractor.fetch_border(ds, out_eic, in_eic, a, b)
            if not part.empty:
                frames.append(part)
        except requests.HTTPError as exc:
            if is_no_data(exc):
                continue
            errors.append(f"{a.isoformat()}->{b.isoformat()}: {exc}")
        except Exception as exc:
            errors.append(f"{a.isoformat()}->{b.isoformat()}: {type(exc).__name__}: {exc}")
    if not frames:
        return pd.DataFrame(), errors
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    return df.sort_values(["timestamp_utc", "mRID", "position"]).reset_index(drop=True), errors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--dataset", choices=["physical_flows", "scheduled_exchanges"], required=True)
    args = p.parse_args()

    settings = load_settings()
    if not settings.entsoe_token:
        raise RuntimeError("ENTSOE_SECURITY_TOKEN is not available")

    ds = settings.config["entsoe"]["datasets"][args.dataset]
    extractor = EntsoeExtractor(settings.entsoe_base_url, settings.entsoe_token, HttpClient(timeout=90))

    today = datetime.now(timezone.utc).date()
    start_date = date(args.year, 1, 1)
    end_date = min(date(args.year + 1, 1, 1), today)
    start = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(end_date, datetime.min.time(), tzinfo=timezone.utc)

    root = ROOT / "data" / "entsoe_ukraine_border" / str(args.year) / args.dataset
    root.mkdir(parents=True, exist_ok=True)

    directions = [
        ("PL", PL_EIC, "UA", UA_EIC),
        ("UA", UA_EIC, "PL", PL_EIC),
    ]
    manifest = {
        "year": args.year,
        "dataset": args.dataset,
        "source": "ENTSO-E Transparency Platform Web API",
        "members": {},
        "total_rows": 0,
    }

    for src, src_eic, dst, dst_eic in directions:
        key = f"{src}_to_{dst}"
        df, errors = fetch_direction(extractor, ds, src_eic, dst_eic, start, end)
        if not df.empty:
            df["from_zone"] = src
            df["to_zone"] = dst
            df.to_parquet(root / f"{key}.parquet", index=False, compression="zstd")
        manifest["members"][key] = {
            "rows": int(len(df)),
            "status": "ok" if len(df) else ("error" if errors else "empty"),
            "errors": errors,
            "min_timestamp_utc": str(df["timestamp_utc"].min()) if len(df) else None,
            "max_timestamp_utc": str(df["timestamp_utc"].max()) if len(df) else None,
        }
        manifest["total_rows"] += int(len(df))

    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
