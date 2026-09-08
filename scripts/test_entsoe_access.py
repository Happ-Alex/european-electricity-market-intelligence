from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from src.eem_intel.extractors.entsoe import EntsoeExtractor
from src.eem_intel.http import HttpClient
from src.eem_intel.settings import load_settings


START = datetime(2026, 8, 15, tzinfo=timezone.utc)
END = datetime(2026, 8, 16, tzinfo=timezone.utc)
HISTORICAL_START = datetime(2019, 6, 15, tzinfo=timezone.utc)
HISTORICAL_END = datetime(2019, 6, 16, tzinfo=timezone.utc)


def expected_daily_points(resolutions: list[str]) -> int | None:
    if len(resolutions) != 1:
        return None
    resolution = resolutions[0]
    if resolution == "PT15M":
        return 96
    if resolution == "PT30M":
        return 48
    if resolution == "PT60M" or resolution == "PT1H":
        return 24
    return None


def main() -> None:
    settings = load_settings()
    if not settings.entsoe_token:
        raise RuntimeError("ENTSOE_SECURITY_TOKEN is not available to the workflow")

    cfg = settings.config
    zone_name = "DE_LU"
    zone_eic = cfg["zones"][zone_name]["eic"]
    border_from = "DE_LU"
    border_to = "FR"
    border_from_eic = cfg["zones"][border_from]["eic"]
    border_to_eic = cfg["zones"][border_to]["eic"]

    def new_extractor() -> EntsoeExtractor:
        return EntsoeExtractor(
            base_url=settings.entsoe_base_url,
            security_token=settings.entsoe_token,
            http=HttpClient(timeout=90),
        )

    def run_one(name: str, ds: dict) -> tuple[str, bool, str]:
        required = bool(ds.get("required", False))
        extractor = new_extractor()
        try:
            if ds["scope"] == "zone":
                df = extractor.fetch_zone(ds, zone_eic, START, END)
                member = zone_name
            else:
                df = extractor.fetch_border(ds, border_from_eic, border_to_eic, START, END)
                member = f"{border_from}->{border_to}"

            resolutions = sorted(str(x) for x in df["resolution"].dropna().unique()) if not df.empty else []
            unique_timestamps = int(df["timestamp_utc"].nunique()) if not df.empty else 0
            duplicate_timestamps = int(df.duplicated(["timestamp_utc"], keep=False).sum()) if not df.empty else 0
            sequences = (
                sorted(str(x) for x in df["classification_sequence"].dropna().unique())
                if not df.empty and "classification_sequence" in df.columns
                else []
            )

            semantic_errors: list[str] = []
            if required and df.empty:
                semantic_errors.append("empty response")

            if name in {"day_ahead_prices", "physical_flows", "scheduled_exchanges"} and not df.empty:
                expected = expected_daily_points(resolutions)
                if expected is not None and unique_timestamps != expected:
                    semantic_errors.append(
                        f"expected {expected} unique timestamps, got {unique_timestamps}"
                    )
                if duplicate_timestamps:
                    semantic_errors.append(f"duplicate timestamp rows={duplicate_timestamps}")

            if name == "day_ahead_prices" and not df.empty:
                if sequences and sequences != ["1"]:
                    semantic_errors.append(f"expected DE-LU classification sequence 1, got {sequences}")

            line = (
                f"{name}: member={member}, rows={len(df)}, unique_ts={unique_timestamps}, "
                f"duplicate_ts={duplicate_timestamps}, resolutions={resolutions}, "
                f"sequences={sequences}, required={required}"
            )
            if semantic_errors:
                line += "; QA_ERROR=" + " | ".join(semantic_errors)
            return name, bool(semantic_errors), line
        except Exception as exc:
            line = f"{name}: ERROR {type(exc).__name__}: {exc}; required={required}"
            return name, required, line

    print("ENTSO-E API token: accepted")
    print(f"Modern sample period: {START.isoformat()} -> {END.isoformat()}")

    failures: list[str] = []
    results: dict[str, str] = {}
    datasets = cfg["entsoe"]["datasets"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_one, name, ds): name for name, ds in datasets.items()}
        for future in as_completed(futures):
            name, failed, line = future.result()
            results[name] = line
            if failed:
                failures.append(name)

    for name in datasets:
        print(results[name])

    # Historical guard for the start of the intended 2019-2026 research period.
    print(
        f"Historical price sample: {HISTORICAL_START.isoformat()} -> {HISTORICAL_END.isoformat()}"
    )
    try:
        price_ds = datasets["day_ahead_prices"]
        hist = new_extractor().fetch_zone(
            price_ds, zone_eic, HISTORICAL_START, HISTORICAL_END
        )
        resolutions = sorted(str(x) for x in hist["resolution"].dropna().unique()) if not hist.empty else []
        unique_timestamps = int(hist["timestamp_utc"].nunique()) if not hist.empty else 0
        duplicate_timestamps = int(hist.duplicated(["timestamp_utc"], keep=False).sum()) if not hist.empty else 0
        expected = expected_daily_points(resolutions)
        print(
            f"day_ahead_prices_2019: rows={len(hist)}, unique_ts={unique_timestamps}, "
            f"duplicate_ts={duplicate_timestamps}, resolutions={resolutions}"
        )
        if hist.empty or duplicate_timestamps or (expected is not None and unique_timestamps != expected):
            failures.append("day_ahead_prices_2019")
    except Exception as exc:
        print(f"day_ahead_prices_2019: ERROR {type(exc).__name__}: {exc}")
        failures.append("day_ahead_prices_2019")

    if failures:
        raise RuntimeError(
            "ENTSO-E dataset QA failed: " + ", ".join(sorted(set(failures)))
        )

    print("ENTSO-E representative dataset validation: PASSED")


if __name__ == "__main__":
    main()
