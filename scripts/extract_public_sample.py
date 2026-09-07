from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

START = "2025-01-01"
END = "2025-01-03"
JAO_TEST_DATE = "2025-01-06T23:00:00.000Z"
OUT = Path("data/sample")
OUT.mkdir(parents=True, exist_ok=True)

ZONES = {
    "DE_LU": {"bzn": "DE-LU", "country": "de", "lat": 52.5200, "lon": 13.4050},
    "FR": {"bzn": "FR", "country": "fr", "lat": 48.8566, "lon": 2.3522},
    "BE": {"bzn": "BE", "country": "be", "lat": 50.8503, "lon": 4.3517},
    "NL": {"bzn": "NL", "country": "nl", "lat": 52.3676, "lon": 4.9041},
    "PL": {"bzn": "PL", "country": "pl", "lat": 52.2297, "lon": 21.0122},
    "CZ": {"bzn": "CZ", "country": "cz", "lat": 50.0755, "lon": 14.4378},
}

SMARD_FILTERS = {
    "DE_LU": 4169,
    "BE": 4996,
    "FR": 254,
    "NL": 256,
    "PL": 257,
    "CZ": 261,
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "european-electricity-market-intelligence/0.1"})
retry = Retry(
    total=6,
    connect=4,
    read=4,
    status=6,
    backoff_factor=1.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET"}),
    respect_retry_after_header=True,
    raise_on_status=False,
)
SESSION.mount("https://", HTTPAdapter(max_retries=retry))


def get_json(url: str, params: dict | None = None, timeout: int = 45):
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def parse_energy_charts_price(payload: dict, zone: str) -> pd.DataFrame:
    if "unix_seconds" in payload and "price" in payload:
        df = pd.DataFrame({
            "timestamp_utc": pd.to_datetime(payload["unix_seconds"], unit="s", utc=True),
            "price_eur_mwh": payload["price"],
        })
    elif "data" in payload:
        rows = []
        for row in payload.get("data", []):
            values = row.get("values", {}) or {}
            price = values.get("price")
            if price is None and len(values) == 1:
                price = next(iter(values.values()))
            rows.append({"timestamp_utc": row.get("timestamp"), "price_eur_mwh": price})
        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    else:
        raise ValueError(f"Unsupported Energy-Charts price schema: {payload.keys()}")
    df.insert(0, "zone", zone)
    df["source"] = "Fraunhofer Energy-Charts"
    return df


def parse_energy_charts_power(payload: dict, zone: str) -> pd.DataFrame:
    if "unix_seconds" in payload and "production_types" in payload:
        df = pd.DataFrame({"timestamp_utc": pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)})
        for series in payload.get("production_types", []):
            name = str(series.get("name", "unknown")).strip().lower().replace(" ", "_").replace("/", "_")
            data = series.get("data")
            if isinstance(data, list) and len(data) == len(df):
                df[name] = data
    elif "data" in payload:
        rows = []
        for row in payload.get("data", []):
            rec = {"timestamp_utc": row.get("timestamp")}
            rec.update(row.get("values", {}) or {})
            rows.append(rec)
        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    else:
        raise ValueError(f"Unsupported Energy-Charts power schema: {payload.keys()}")
    df.insert(0, "zone", zone)
    df["source"] = "Fraunhofer Energy-Charts"
    return df


def fetch_energy_charts_price(zone: str, bzn: str) -> pd.DataFrame:
    payload = get_json(
        "https://api.energy-charts.info/price",
        {"bzn": bzn, "start": START, "end": END},
    )
    return parse_energy_charts_price(payload, zone)


def fetch_energy_charts_power(zone: str, country: str) -> pd.DataFrame:
    payload = get_json(
        "https://api.energy-charts.info/public_power",
        {"country": country, "start": START, "end": END},
    )
    return parse_energy_charts_power(payload, zone)


def fetch_open_meteo(zone: str, lat: float, lon: float) -> pd.DataFrame:
    payload = get_json(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": START,
            "end_date": END,
            "hourly": "temperature_2m,wind_speed_10m,wind_speed_100m,cloud_cover,shortwave_radiation",
            "timezone": "UTC",
        },
    )
    hourly = payload.get("hourly", {})
    if not hourly:
        raise ValueError("Open-Meteo returned no hourly data")
    df = pd.DataFrame(hourly)
    df["timestamp_utc"] = pd.to_datetime(df.pop("time"), utc=True)
    df.insert(0, "zone", zone)
    df["latitude"] = payload.get("latitude")
    df["longitude"] = payload.get("longitude")
    df["source"] = "Open-Meteo Historical Weather API"
    return df


def fetch_smard_price(zone: str, filter_id: int) -> pd.DataFrame:
    # SMARD exposes market prices via region=DE even for neighbouring bidding zones.
    index_url = f"https://www.smard.de/app/chart_data/{filter_id}/DE/index_hour.json"
    idx = get_json(index_url)
    timestamps = idx.get("timestamps", [])
    if not timestamps:
        raise ValueError("SMARD returned no index timestamps")

    start_ms = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    candidates = [t for t in timestamps if int(t) <= start_ms]
    anchor = max(candidates) if candidates else min(timestamps)

    data_url = f"https://www.smard.de/app/chart_data/{filter_id}/DE/{filter_id}_DE_hour_{anchor}.json"
    payload = get_json(data_url)
    series = payload.get("series", [])
    df = pd.DataFrame(series, columns=["timestamp_ms", "price_eur_mwh"])
    if df.empty:
        return df
    df["timestamp_utc"] = pd.to_datetime(df.pop("timestamp_ms"), unit="ms", utc=True)
    end_exclusive = pd.Timestamp(END, tz="UTC") + pd.Timedelta(days=1)
    mask = (df["timestamp_utc"] >= pd.Timestamp(START, tz="UTC")) & (df["timestamp_utc"] < end_exclusive)
    df = df.loc[mask].copy()
    df.insert(0, "zone", zone)
    df["source"] = "SMARD / Bundesnetzagentur"
    return df


def fetch_jao_max_exchanges() -> pd.DataFrame:
    # Test on a normal business day; holiday dates can return a 400 on the public endpoint.
    payload = get_json(
        "https://publicationtool.jao.eu/core/api/core/maxExchanges/index",
        {"date": JAO_TEST_DATE},
    )
    rows = payload.get("maxExchanges", payload if isinstance(payload, list) else [])
    return pd.json_normalize(rows)


def save_csv(df: pd.DataFrame, name: str):
    path = OUT / name
    df.to_csv(path, index=False)
    return {"file": str(path), "rows": int(len(df)), "columns": list(df.columns)}


def main():
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "period": {"start": START, "end": END},
        "success": [],
        "failed": [],
    }

    for zone, cfg in ZONES.items():
        tasks = [
            (f"energy_charts_price_{zone}", lambda z=zone, b=cfg["bzn"]: fetch_energy_charts_price(z, b), f"energy_charts_price_{zone}.csv"),
            (f"energy_charts_power_{zone}", lambda z=zone, c=cfg["country"]: fetch_energy_charts_power(z, c), f"energy_charts_power_{zone}.csv"),
            (f"weather_{zone}", lambda z=zone, la=cfg["lat"], lo=cfg["lon"]: fetch_open_meteo(z, la, lo), f"weather_{zone}.csv"),
            (f"smard_price_{zone}", lambda z=zone, f=SMARD_FILTERS[zone]: fetch_smard_price(z, f), f"smard_price_{zone}.csv"),
        ]
        for task_name, fn, filename in tasks:
            try:
                info = save_csv(fn(), filename)
                manifest["success"].append({"task": task_name, **info})
                print(f"OK   {task_name}: {info['rows']} rows")
            except Exception as exc:
                manifest["failed"].append({"task": task_name, "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAIL {task_name}: {exc}")
            if task_name.startswith("energy_charts_"):
                # Fraunhofer's API is public but rate-limited. Be polite between calls.
                time.sleep(2.0)

    try:
        info = save_csv(fetch_jao_max_exchanges(), "jao_max_exchanges_2025-01-06.csv")
        manifest["success"].append({"task": "jao_max_exchanges", **info})
        print(f"OK   jao_max_exchanges: {info['rows']} rows")
    except Exception as exc:
        manifest["failed"].append({"task": "jao_max_exchanges", "error": f"{type(exc).__name__}: {exc}"})
        print(f"FAIL jao_max_exchanges: {exc}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
