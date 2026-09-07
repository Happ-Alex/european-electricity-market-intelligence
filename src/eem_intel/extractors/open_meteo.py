from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import pandas as pd

from ..http import HttpClient


@dataclass
class OpenMeteoExtractor:
    base_url: str
    http: HttpClient

    def fetch_hourly(self, latitude: float, longitude: float, start: date, end: date, variables: list[str]) -> pd.DataFrame:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(variables),
            "timezone": "UTC",
        }
        payload = self.http.get(self.base_url, params=params).json()
        hourly = payload.get("hourly", {})
        if not hourly:
            return pd.DataFrame()
        df = pd.DataFrame(hourly)
        if "time" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df.pop("time"), utc=True)
        df["latitude"] = payload.get("latitude")
        df["longitude"] = payload.get("longitude")
        return df
