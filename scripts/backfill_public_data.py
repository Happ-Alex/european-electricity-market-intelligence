from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# Public, redistributable sources only.
# Prices come from SMARD (CC BY 4.0). Energy-Charts price is intentionally not
# used here because some bidding-zone price series are restricted to private /
# internal use. Non-price Energy-Charts endpoints are used for generation,
# scheduled exchanges and physical flows.

ZONES = {
    "DE_LU": {"bzn": "DE-LU", "country": "de", "lat": 52.5200, "lon": 13.4050},
    "FR": {"bzn": "FR", "country": "fr", "lat": 48.8566, "lon": 2.3522},
    "BE": {"bzn": "BE", "country": "be", "lat": 50.8503, "lon": 4.3517},
    "NL": {"bzn": "NL", "country": "nl", "lat": 52.3676, "lon": 4.9041},
    "PL": {"bzn": "PL", "country": "pl", "lat": 52.2297, "lon": 21.0122},
    "CZ": {"bzn": "CZ", "country": "cz", "lat": 50.0755, "lon": 14.4378},
}

# SMARD market-price filters used by its chart-data download backend.
SMARD_FILTERS = {
    "DE_LU": 4169,
    "BE": 4996,
    "FR": 254,
    "NL": 256,
    "PL": 257,
    "CZ": 261,
}

BORDERS = [
    ("DE_LU", "FR"),
    ("DE_LU", "BE"),
    ("DE_LU", "NL"),
    ("DE_LU", "PL"),
    ("DE_LU", "CZ"),
    ("FR", "BE"),
    ("BE", "NL"),
]

ENERGY_CHARTS_BASE = "https://api.energy-charts.info/v2"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Happ-Alex/european-electricity-market-intelligence (academic research; public ETL)",
    "Accept": "application/json",
})


@dataclass
class Result:
    source: str
    zone: str
    rows: int
    file: str


def get_json(url: str, params: dict | None = None, *, attempts: int = 8, timeout: int = 120) -> dict:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = SESSION.get(url, params=params, timeout=timeout)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else min(90.0, 5.0 * (2 ** attempt))
                print(f"Retryable HTTP {response.status_code}; sleeping {wait:.1f}s: {response.url}")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            wait = min(90.0, 5.0 * (2 ** attempt))
            print(f"Network error; sleeping {wait:.1f}s: {exc}")
            time.sleep(wait)
        except requests.HTTPError:
            raise
    raise RuntimeError(f"Request failed after {attempts} attempts: {url}") from last_error


def parse_energy_charts_v2(payload: dict, prefix: str = "") -> pd.DataFrame:
    rows: list[dict] = []
    for item in payload.get("data", []) or []:
        rec = {"timestamp_utc": pd.to_datetime(item.get("timestamp"), utc=True)}
        values = item.get("values", {}) or {}
        for key, value in values.items():
            rec[f"{prefix}{key}"] = value
        rows.append(rec)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    return df


def quarter_ranges(start: date, end: date) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        q_month = ((cursor.month - 1) // 3) * 3 + 1
        q_start = date(cursor.year, q_month, 1)
        if q_month == 10:
            q_end = date(cursor.year, 12, 31)
        else:
            q_end = date(cursor.year, q_month + 3, 1) - timedelta(days=1)
        a = max(cursor, q_start)
        b = min(end, q_end)
        ranges.append((a, b))
        cursor = b + timedelta(days=1)
    return ranges


def fetch_energy_charts(endpoint: str, country: str, start: date, end: date, prefix: str) -> pd.DataFrame:
    url = f"{ENERGY_CHARTS_BASE}/{endpoint}"
    params = {"country": country, "start": start.isoformat(), "end": end.isoformat()}
    try:
        payload = get_json(url, params, attempts=8, timeout=180)
        return parse_energy_charts_v2(payload, prefix=prefix)
    except (requests.Timeout, requests.ConnectionError, RuntimeError):
        pass
    except requests.HTTPError as exc:
        # 413/5xx-like size issues are handled by splitting; other 4xx should surface.
        if exc.response is None or exc.response.status_code not in {408, 413, 429}:
            raise

    print(f"Falling back to quarterly chunks for {endpoint}/{country}/{start.year}")
    pieces: list[pd.DataFrame] = []
    for q_start, q_end in quarter_ranges(start, end):
        payload = get_json(
            url,
            {"country": country, "start": q_start.isoformat(), "end": q_end.isoformat()},
            attempts=8,
            timeout=180,
        )
        piece = parse_energy_charts_v2(payload, prefix=prefix)
        if not piece.empty:
            pieces.append(piece)
        time.sleep(2.0)
    if not pieces:
        return pd.DataFrame()
    return (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(subset=["timestamp_utc"])
        .sort_values("timestamp_utc")
    )


def fetch_smard_price(zone: str, start: date, end: date) -> pd.DataFrame:
    filter_id = SMARD_FILTERS[zone]
    index_url = f"https://www.smard.de/app/chart_data/{filter_id}/DE/index_hour.json"
    index = get_json(index_url, attempts=6, timeout=60)
    anchors = sorted(int(x) for x in index.get("timestamps", []))
    if not anchors:
        raise ValueError(f"No SMARD anchors for {zone}")

    start_ts = pd.Timestamp(start, tz="UTC")
    end_exclusive = pd.Timestamp(end + timedelta(days=1), tz="UTC")
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_exclusive.timestamp() * 1000)

    selected = [x for x in anchors if start_ms <= x < end_ms]
    predecessors = [x for x in anchors if x < start_ms]
    if predecessors:
        selected.insert(0, max(predecessors))
    selected = sorted(set(selected))

    pieces: list[pd.DataFrame] = []
    for i, anchor in enumerate(selected, start=1):
        url = f"https://www.smard.de/app/chart_data/{filter_id}/DE/{filter_id}_DE_hour_{anchor}.json"
        payload = get_json(url, attempts=5, timeout=60)
        series = payload.get("series", []) or []
        if series:
            piece = pd.DataFrame(series, columns=["timestamp_ms", "price_eur_mwh"])
            piece["timestamp_utc"] = pd.to_datetime(piece.pop("timestamp_ms"), unit="ms", utc=True)
            pieces.append(piece)
        if i % 10 == 0:
            print(f"SMARD {zone}: {i}/{len(selected)} weekly chunks")
        time.sleep(0.08)

    if not pieces:
        return pd.DataFrame(columns=["timestamp_utc", "price_eur_mwh"])

    df = pd.concat(pieces, ignore_index=True)
    df = df[(df["timestamp_utc"] >= start_ts) & (df["timestamp_utc"] < end_exclusive)]
    df = df.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")
    return df


def fetch_weather(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    payload = get_json(
        OPEN_METEO_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": "temperature_2m,wind_speed_10m,wind_speed_100m,cloud_cover,shortwave_radiation",
            "timezone": "UTC",
        },
        attempts=6,
        timeout=180,
    )
    hourly = payload.get("hourly", {}) or {}
    if not hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    df["timestamp_utc"] = pd.to_datetime(df.pop("time"), utc=True)
    return df.sort_values("timestamp_utc")


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True)
    numeric_cols = [c for c in work.columns if c != "timestamp_utc" and pd.api.types.is_numeric_dtype(work[c])]
    if not numeric_cols:
        return work[["timestamp_utc"]].drop_duplicates().sort_values("timestamp_utc")
    return (
        work.set_index("timestamp_utc")[numeric_cols]
        .resample("1h")
        .mean()
        .reset_index()
    )


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")


def save_csv_gz(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")


def build_zone_hourly(
    zone: str,
    price: pd.DataFrame,
    power: pd.DataFrame,
    cbet: pd.DataFrame,
    cbpf: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for df in (price, power, cbet, cbpf, weather):
        if not df.empty:
            frames.append(to_hourly(df))
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="timestamp_utc", how="outer")
    out.insert(0, "zone", zone)
    return out.sort_values("timestamp_utc")


def find_neighbor_col(df: pd.DataFrame, prefix: str, neighbor_country: str) -> str | None:
    candidates = [
        f"{prefix}{neighbor_country}",
        f"{prefix}{neighbor_country.lower()}",
        f"{prefix}{neighbor_country.upper()}",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    # Last-resort exact suffix match for stable snake_case series ids.
    for col in df.columns:
        if col.startswith(prefix) and col[len(prefix):].lower() == neighbor_country.lower():
            return col
    return None


def build_border_hourly(zone_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for a, b in BORDERS:
        if a not in zone_tables or b not in zone_tables:
            continue
        da = zone_tables[a]
        db = zone_tables[b]
        if da.empty or db.empty:
            continue
        aa = da[[c for c in ["timestamp_utc", "price_eur_mwh"] if c in da.columns]].copy()
        bb = db[[c for c in ["timestamp_utc", "price_eur_mwh"] if c in db.columns]].copy()
        if "price_eur_mwh" in aa.columns:
            aa = aa.rename(columns={"price_eur_mwh": "price_a_eur_mwh"})
        if "price_eur_mwh" in bb.columns:
            bb = bb.rename(columns={"price_eur_mwh": "price_b_eur_mwh"})
        border = aa.merge(bb, on="timestamp_utc", how="outer")

        neighbor = ZONES[b]["country"]
        cbet_col = find_neighbor_col(da, "cbet_", neighbor)
        cbpf_col = find_neighbor_col(da, "cbpf_", neighbor)
        if cbet_col:
            tmp = da[["timestamp_utc", cbet_col]].rename(columns={cbet_col: "scheduled_flow_a_to_b_gw"})
            tmp["scheduled_flow_a_to_b_gw"] = -pd.to_numeric(tmp["scheduled_flow_a_to_b_gw"], errors="coerce")
            border = border.merge(tmp, on="timestamp_utc", how="left")
        if cbpf_col:
            tmp = da[["timestamp_utc", cbpf_col]].rename(columns={cbpf_col: "physical_flow_a_to_b_gw"})
            tmp["physical_flow_a_to_b_gw"] = -pd.to_numeric(tmp["physical_flow_a_to_b_gw"], errors="coerce")
            border = border.merge(tmp, on="timestamp_utc", how="left")

        border.insert(1, "zone_a", a)
        border.insert(2, "zone_b", b)
        if "price_a_eur_mwh" in border.columns and "price_b_eur_mwh" in border.columns:
            border["price_spread_a_minus_b_eur_mwh"] = border["price_a_eur_mwh"] - border["price_b_eur_mwh"]
        pieces.append(border)

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True).sort_values(["zone_a", "zone_b", "timestamp_utc"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    year = args.year
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    start = date(year, 1, 1)
    end = min(date(year, 12, 31), yesterday)
    if end < start:
        raise SystemExit(f"Year {year} has no completed days yet")

    root = Path("data/backfill") / str(year)
    raw = root / "raw"
    processed = root / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "year": year,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "zones": list(ZONES),
        "sources": {
            "prices": "SMARD / Bundesnetzagentur (CC BY 4.0)",
            "generation_load": "Fraunhofer Energy-Charts v2 public_power",
            "scheduled_cross_border": "Fraunhofer Energy-Charts v2 cbet",
            "physical_cross_border": "Fraunhofer Energy-Charts v2 cbpf",
            "weather": "Open-Meteo Historical Weather API",
        },
        "success": [],
        "failed": [],
    }

    zone_tables: dict[str, pd.DataFrame] = {}

    for zone, cfg in ZONES.items():
        print(f"\n=== {year} / {zone} ===")
        datasets: dict[str, pd.DataFrame] = {}

        tasks = [
            ("price", lambda: fetch_smard_price(zone, start, end)),
            ("power", lambda: fetch_energy_charts("public_power", cfg["country"], start, end, "power_")),
            ("cbet", lambda: fetch_energy_charts("cbet", cfg["country"], start, end, "cbet_")),
            ("cbpf", lambda: fetch_energy_charts("cbpf", cfg["country"], start, end, "cbpf_")),
            ("weather", lambda: fetch_weather(cfg["lat"], cfg["lon"], start, end)),
        ]

        for name, fn in tasks:
            try:
                df = fn()
                datasets[name] = df
                out = raw / f"{name}_{zone}_{year}.parquet"
                save_parquet(df, out)
                manifest["success"].append({"zone": zone, "dataset": name, "rows": int(len(df)), "file": str(out)})
                print(f"OK {name}: {len(df):,} rows")
            except Exception as exc:
                datasets[name] = pd.DataFrame()
                manifest["failed"].append({
                    "zone": zone,
                    "dataset": name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            time.sleep(2.0 if name in {"power", "cbet", "cbpf"} else 0.3)

        market = build_zone_hourly(
            zone,
            datasets.get("price", pd.DataFrame()),
            datasets.get("power", pd.DataFrame()),
            datasets.get("cbet", pd.DataFrame()),
            datasets.get("cbpf", pd.DataFrame()),
            datasets.get("weather", pd.DataFrame()),
        )
        zone_tables[zone] = market
        if not market.empty:
            save_parquet(market, processed / f"market_hourly_{zone}_{year}.parquet")

    if zone_tables:
        all_market = pd.concat([df for df in zone_tables.values() if not df.empty], ignore_index=True, sort=False)
        all_market = all_market.sort_values(["zone", "timestamp_utc"])
        save_parquet(all_market, processed / f"market_hourly_all_zones_{year}.parquet")
        save_csv_gz(all_market, processed / f"market_hourly_all_zones_{year}.csv.gz")
        manifest["processed_market_rows"] = int(len(all_market))

        borders = build_border_hourly(zone_tables)
        if not borders.empty:
            save_parquet(borders, processed / f"border_hourly_{year}.parquet")
            save_csv_gz(borders, processed / f"border_hourly_{year}.csv.gz")
            manifest["processed_border_rows"] = int(len(borders))

    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "year": year,
        "success_count": len(manifest["success"]),
        "failure_count": len(manifest["failed"]),
        "market_rows": manifest.get("processed_market_rows", 0),
        "border_rows": manifest.get("processed_border_rows", 0),
    }, indent=2))


if __name__ == "__main__":
    main()
