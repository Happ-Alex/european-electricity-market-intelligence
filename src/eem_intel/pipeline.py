from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging

import pandas as pd

from .extractors import EntsoeExtractor, JaoExtractor, OpenMeteoExtractor
from .http import HttpClient
from .io import write_parquet
from .settings import Settings, ROOT


LOGGER = logging.getLogger(__name__)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _chunks(start: datetime, end: datetime, days: int):
    current = start
    while current < end:
        nxt = min(current + timedelta(days=days), end)
        yield current, nxt
        current = nxt


def run_entsoe(settings: Settings, start: datetime, end: datetime) -> list[Path]:
    if not settings.entsoe_token:
        raise RuntimeError("ENTSOE_SECURITY_TOKEN is not set. Copy .env.example to .env and add your token.")

    cfg = settings.config
    extractor = EntsoeExtractor(settings.entsoe_base_url, settings.entsoe_token, HttpClient())
    out: list[Path] = []
    default_chunk_days = int(cfg["entsoe"].get("chunk_days", 31))

    for name, ds in cfg["entsoe"]["datasets"].items():
        frames: list[pd.DataFrame] = []
        scope = ds["scope"]
        chunk_days = int(ds.get("chunk_days", default_chunk_days))

        if scope == "zone":
            for zone_name, zone_cfg in cfg["zones"].items():
                for chunk_start, chunk_end in _chunks(start, end, chunk_days):
                    LOGGER.info("ENTSO-E %s %s %s -> %s", name, zone_name, chunk_start, chunk_end)
                    df = extractor.fetch_zone(ds, zone_cfg["eic"], chunk_start, chunk_end)
                    if not df.empty:
                        df["zone"] = zone_name
                        frames.append(df)
        elif scope == "border":
            for from_zone, to_zone in cfg["borders"]:
                from_eic = cfg["zones"][from_zone]["eic"]
                to_eic = cfg["zones"][to_zone]["eic"]
                for direction in [(from_zone, to_zone, from_eic, to_eic), (to_zone, from_zone, to_eic, from_eic)]:
                    src, dst, out_eic, in_eic = direction
                    for chunk_start, chunk_end in _chunks(start, end, chunk_days):
                        LOGGER.info("ENTSO-E %s %s -> %s %s -> %s", name, src, dst, chunk_start, chunk_end)
                        df = extractor.fetch_border(ds, out_eic, in_eic, chunk_start, chunk_end)
                        if not df.empty:
                            df["from_zone"] = src
                            df["to_zone"] = dst
                            frames.append(df)
        else:
            raise ValueError(f"Unknown ENTSO-E dataset scope: {scope}")

        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not result.empty:
            path = ROOT / "data" / "raw" / "entsoe" / f"{name}_{start.date()}_{end.date()}.parquet"
            out.append(write_parquet(result, path))

    return out


def run_jao(settings: Settings, start: datetime, end: datetime) -> list[Path]:
    cfg = settings.config["jao"]
    extractor = JaoExtractor(settings.jao_base_url, HttpClient())
    out: list[Path] = []

    for name, endpoint in cfg["endpoints"].items():
        LOGGER.info("JAO %s %s -> %s", name, start, end)
        try:
            df = extractor.fetch(endpoint, start, end, cfg.get("request_params"))
        except Exception as exc:
            LOGGER.error("JAO endpoint %s failed: %s", endpoint, exc)
            continue
        if not df.empty:
            path = ROOT / "data" / "raw" / "jao" / f"{name}_{start.date()}_{end.date()}.parquet"
            out.append(write_parquet(df, path))

    return out


def run_weather(settings: Settings, start: datetime, end: datetime) -> list[Path]:
    cfg = settings.config
    weather_cfg = cfg["weather"]
    extractor = OpenMeteoExtractor(weather_cfg["base_url"], HttpClient())
    out: list[Path] = []

    for zone_name, zone_cfg in cfg["zones"].items():
        loc = zone_cfg["weather"]
        LOGGER.info("Open-Meteo %s %s -> %s", zone_name, start.date(), end.date())
        df = extractor.fetch_hourly(
            loc["latitude"],
            loc["longitude"],
            start.date(),
            (end - timedelta(days=1)).date() if end.time() == datetime.min.time() else end.date(),
            weather_cfg["hourly"],
        )
        if not df.empty:
            df["zone"] = zone_name
            path = ROOT / "data" / "raw" / "weather" / f"{zone_name}_{start.date()}_{end.date()}.parquet"
            out.append(write_parquet(df, path))

    return out


def run(settings: Settings, source: str = "all", start_override: str | None = None, end_override: str | None = None) -> list[Path]:
    project = settings.config["project"]
    start = _dt(start_override or project["start_date"])
    end = _dt(end_override or project["end_date"])
    if end <= start:
        raise ValueError("end date must be later than start date")

    outputs: list[Path] = []
    if source in {"all", "entsoe"}:
        outputs.extend(run_entsoe(settings, start, end))
    if source in {"all", "jao"}:
        outputs.extend(run_jao(settings, start, end))
    if source in {"all", "weather"}:
        outputs.extend(run_weather(settings, start, end))
    return outputs
