from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.eem_intel.extractors.entsoe import EntsoeExtractor
from src.eem_intel.http import HttpClient
from src.eem_intel.settings import ROOT, load_settings


SAMPLE_START = datetime(2026, 8, 15, tzinfo=timezone.utc)
SAMPLE_END = datetime(2026, 8, 16, tzinfo=timezone.utc)


def error_payload(exc: Exception) -> dict:
    payload = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        payload["http_status"] = exc.response.status_code
        text = (exc.response.text or "").replace("\n", " ").strip()
        payload["response_excerpt"] = text[:500]
    return payload


def summarize(df) -> dict:
    if df.empty:
        return {"status": "empty", "rows": 0}
    return {
        "status": "ok",
        "rows": int(len(df)),
        "timestamp_min": str(df["timestamp_utc"].min()),
        "timestamp_max": str(df["timestamp_utc"].max()),
        "resolutions": sorted(str(x) for x in df["resolution"].dropna().unique()),
        "non_null_values": int(df["value"].notna().sum()),
        "timeseries": int(df["mRID"].nunique(dropna=True)) if "mRID" in df.columns else None,
        "duplicate_timestamp_rows": int(df.duplicated(["timestamp_utc"], keep=False).sum()),
    }


def main() -> None:
    settings = load_settings()
    if not settings.entsoe_token:
        raise RuntimeError("ENTSOE_SECURITY_TOKEN is not available")

    cfg = settings.config
    extractor = EntsoeExtractor(
        base_url=settings.entsoe_base_url,
        security_token=settings.entsoe_token,
        http=HttpClient(timeout=90),
    )

    report: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_start": SAMPLE_START.isoformat(),
        "sample_end": SAMPLE_END.isoformat(),
        "datasets": {},
    }

    required_failures: list[str] = []

    for dataset_name, ds in cfg["entsoe"]["datasets"].items():
        scope = ds["scope"]
        required = bool(ds.get("required", False))
        dataset_report = {
            "scope": scope,
            "required": required,
            "members": {},
        }

        if scope == "zone":
            for zone_name, zone_cfg in cfg["zones"].items():
                key = zone_name
                print(f"Validating {dataset_name}: {key}", flush=True)
                try:
                    df = extractor.fetch_zone(ds, zone_cfg["eic"], SAMPLE_START, SAMPLE_END)
                    dataset_report["members"][key] = summarize(df)
                except Exception as exc:
                    dataset_report["members"][key] = error_payload(exc)

        elif scope == "border":
            for zone_a, zone_b in cfg["borders"]:
                for src, dst in ((zone_a, zone_b), (zone_b, zone_a)):
                    key = f"{src}->{dst}"
                    print(f"Validating {dataset_name}: {key}", flush=True)
                    try:
                        df = extractor.fetch_border(
                            ds,
                            cfg["zones"][src]["eic"],
                            cfg["zones"][dst]["eic"],
                            SAMPLE_START,
                            SAMPLE_END,
                        )
                        dataset_report["members"][key] = summarize(df)
                    except Exception as exc:
                        dataset_report["members"][key] = error_payload(exc)
        else:
            raise ValueError(f"Unknown scope {scope!r} for {dataset_name}")

        statuses = [m["status"] for m in dataset_report["members"].values()]
        ok_count = sum(s == "ok" for s in statuses)
        empty_count = sum(s == "empty" for s in statuses)
        error_count = sum(s == "error" for s in statuses)
        dataset_report["summary"] = {
            "ok": ok_count,
            "empty": empty_count,
            "error": error_count,
            "total": len(statuses),
        }

        # A required dataset is considered query-valid when at least one configured
        # zone/border returns real TimeSeries data. Missing member-level coverage is
        # retained as a warning because ENTSO-E publication completeness varies by area.
        if required and ok_count == 0:
            required_failures.append(dataset_name)

        report["datasets"][dataset_name] = dataset_report

    out_dir = ROOT / "data" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "entsoe_validation.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nENTSO-E validation summary")
    print("=" * 72)
    for name, ds_report in report["datasets"].items():
        s = ds_report["summary"]
        flag = "REQUIRED" if ds_report["required"] else "OPTIONAL"
        print(
            f"{name:34s} {flag:8s} "
            f"ok={s['ok']:2d} empty={s['empty']:2d} error={s['error']:2d} total={s['total']:2d}"
        )

    print(f"Validation report: {out_path}")

    if required_failures:
        raise RuntimeError(
            "Required ENTSO-E datasets with zero successful samples: "
            + ", ".join(required_failures)
        )

    print("ENTSO-E dataset validation: PASSED")


if __name__ == "__main__":
    main()
