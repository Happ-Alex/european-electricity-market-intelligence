from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from src.eem_intel.extractors.entsoe import EntsoeExtractor
from src.eem_intel.http import HttpClient
from src.eem_intel.settings import load_settings


START = datetime(2026, 8, 15, tzinfo=timezone.utc)
END = datetime(2026, 8, 16, tzinfo=timezone.utc)


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

    def run_one(name: str, ds: dict) -> tuple[str, bool, str]:
        required = bool(ds.get("required", False))
        extractor = EntsoeExtractor(
            base_url=settings.entsoe_base_url,
            security_token=settings.entsoe_token,
            http=HttpClient(timeout=90),
        )
        try:
            if ds["scope"] == "zone":
                df = extractor.fetch_zone(ds, zone_eic, START, END)
                member = zone_name
            else:
                df = extractor.fetch_border(ds, border_from_eic, border_to_eic, START, END)
                member = f"{border_from}->{border_to}"

            resolutions = sorted(str(x) for x in df["resolution"].dropna().unique()) if not df.empty else []
            line = (
                f"{name}: member={member}, rows={len(df)}, "
                f"resolutions={resolutions}, required={required}"
            )
            failed = required and df.empty
            return name, failed, line
        except Exception as exc:
            line = f"{name}: ERROR {type(exc).__name__}: {exc}; required={required}"
            return name, required, line

    print("ENTSO-E API token: accepted")
    print(f"Sample period: {START.isoformat()} -> {END.isoformat()}")

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

    if failures:
        raise RuntimeError(
            "ENTSO-E required dataset smoke checks failed: " + ", ".join(sorted(failures))
        )

    print("ENTSO-E representative dataset validation: PASSED")


if __name__ == "__main__":
    main()
