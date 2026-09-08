from __future__ import annotations

from datetime import datetime, timezone

from src.eem_intel.extractors.entsoe import EntsoeExtractor
from src.eem_intel.http import HttpClient
from src.eem_intel.settings import load_settings


def main() -> None:
    settings = load_settings()
    if not settings.entsoe_token:
        raise RuntimeError("ENTSOE_SECURITY_TOKEN is not available to the workflow")

    cfg = settings.config
    zone_name = "DE_LU"
    domain = cfg["zones"][zone_name]["eic"]
    dataset = cfg["entsoe"]["datasets"]["day_ahead_prices"]

    extractor = EntsoeExtractor(
        base_url=settings.entsoe_base_url,
        security_token=settings.entsoe_token,
        http=HttpClient(timeout=60),
    )

    start = datetime(2025, 1, 2, tzinfo=timezone.utc)
    end = datetime(2025, 1, 3, tzinfo=timezone.utc)
    df = extractor.fetch_zone(dataset, domain, start, end)

    if df.empty:
        raise RuntimeError("ENTSO-E request succeeded but returned no day-ahead price rows")

    print("ENTSO-E API access: OK")
    print(f"Dataset: day_ahead_prices")
    print(f"Zone: {zone_name}")
    print(f"Rows: {len(df)}")
    print(f"Timestamp min: {df['timestamp_utc'].min()}")
    print(f"Timestamp max: {df['timestamp_utc'].max()}")
    print(f"Non-null values: {int(df['value'].notna().sum())}")


if __name__ == "__main__":
    main()
