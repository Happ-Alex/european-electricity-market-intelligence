from __future__ import annotations

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
    extractor = EntsoeExtractor(
        base_url=settings.entsoe_base_url,
        security_token=settings.entsoe_token,
        http=HttpClient(timeout=90),
    )

    zone_name = "DE_LU"
    zone_eic = cfg["zones"][zone_name]["eic"]
    border_from = "DE_LU"
    border_to = "FR"
    border_from_eic = cfg["zones"][border_from]["eic"]
    border_to_eic = cfg["zones"][border_to]["eic"]

    failures: list[str] = []

    print("ENTSO-E API token: accepted")
    print(f"Sample period: {START.isoformat()} -> {END.isoformat()}")

    for name, ds in cfg["entsoe"]["datasets"].items():
        required = bool(ds.get("required", False))
        try:
            if ds["scope"] == "zone":
                df = extractor.fetch_zone(ds, zone_eic, START, END)
                member = zone_name
            else:
                df = extractor.fetch_border(ds, border_from_eic, border_to_eic, START, END)
                member = f"{border_from}->{border_to}"

            resolutions = sorted(str(x) for x in df["resolution"].dropna().unique()) if not df.empty else []
            print(
                f"{name}: member={member}, rows={len(df)}, "
                f"resolutions={resolutions}, required={required}"
            )
            if required and df.empty:
                failures.append(f"{name}: empty response")
        except Exception as exc:
            print(f"{name}: ERROR {type(exc).__name__}: {exc}")
            if required:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")

    if failures:
        raise RuntimeError("ENTSO-E required dataset smoke checks failed: " + " | ".join(failures))

    print("ENTSO-E representative dataset validation: PASSED")


if __name__ == "__main__":
    main()
