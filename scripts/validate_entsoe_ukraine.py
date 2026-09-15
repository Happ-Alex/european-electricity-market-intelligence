from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.eem_intel.extractors.entsoe import EntsoeExtractor
from src.eem_intel.http import HttpClient
from src.eem_intel.settings import ROOT, load_settings


UA_EIC = "10Y1001C--00003F"
PL_EIC = "10YPL-AREA-----S"


def main() -> None:
    settings = load_settings()
    if not settings.entsoe_token:
        raise RuntimeError("ENTSOE_SECURITY_TOKEN is not available")

    cfg = settings.config
    datasets = cfg["entsoe"]["datasets"]
    extractor = EntsoeExtractor(
        base_url=settings.entsoe_base_url,
        security_token=settings.entsoe_token,
        http=HttpClient(timeout=90),
    )

    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    end = datetime(2024, 6, 4, tzinfo=timezone.utc)
    out_root = ROOT / "data" / "ukraine_validation"
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "ukraine_eic": UA_EIC,
        "poland_eic": PL_EIC,
        "period_start_utc": start.isoformat(),
        "period_end_utc": end.isoformat(),
        "source": "ENTSO-E Transparency Platform Web API",
        "zone_tests": {},
        "border_tests": {},
    }

    zone_datasets = [
        "day_ahead_prices",
        "actual_load",
        "day_ahead_load_forecast",
        "actual_generation_per_type",
        "day_ahead_generation_forecast",
        "wind_solar_forecast",
    ]

    for name in zone_datasets:
        ds = datasets[name]
        try:
            df = extractor.fetch_zone(ds, UA_EIC, start, end)
            status = "ok" if not df.empty else "empty"
            if not df.empty:
                df["zone"] = "UA"
                df.to_parquet(out_root / f"UA_{name}.parquet", index=False, compression="zstd")
            summary["zone_tests"][name] = {
                "status": status,
                "rows": int(len(df)),
                "min_timestamp_utc": str(df["timestamp_utc"].min()) if not df.empty else None,
                "max_timestamp_utc": str(df["timestamp_utc"].max()) if not df.empty else None,
            }
        except Exception as exc:
            summary["zone_tests"][name] = {
                "status": "error",
                "rows": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    for name in ["physical_flows", "scheduled_exchanges"]:
        ds = datasets[name]
        for src_name, src_eic, dst_name, dst_eic in [
            ("PL", PL_EIC, "UA", UA_EIC),
            ("UA", UA_EIC, "PL", PL_EIC),
        ]:
            key = f"{name}_{src_name}_to_{dst_name}"
            try:
                df = extractor.fetch_border(ds, src_eic, dst_eic, start, end)
                status = "ok" if not df.empty else "empty"
                if not df.empty:
                    df["from_zone"] = src_name
                    df["to_zone"] = dst_name
                    df.to_parquet(out_root / f"{key}.parquet", index=False, compression="zstd")
                summary["border_tests"][key] = {
                    "status": status,
                    "rows": int(len(df)),
                    "min_timestamp_utc": str(df["timestamp_utc"].min()) if not df.empty else None,
                    "max_timestamp_utc": str(df["timestamp_utc"].max()) if not df.empty else None,
                }
            except Exception as exc:
                summary["border_tests"][key] = {
                    "status": "error",
                    "rows": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    (out_root / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
