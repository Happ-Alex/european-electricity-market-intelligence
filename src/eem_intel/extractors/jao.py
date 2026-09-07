from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urljoin

import pandas as pd

from ..http import HttpClient


@dataclass
class JaoExtractor:
    base_url: str
    http: HttpClient

    @staticmethod
    def _iso(ts: datetime) -> str:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def fetch(self, endpoint: str, start: datetime, end: datetime, extra_params: dict | None = None) -> pd.DataFrame:
        url = urljoin(self.base_url.rstrip("/") + "/", endpoint.lstrip("/"))
        params = {
            "from": self._iso(start),
            "to": self._iso(end),
            **(extra_params or {}),
        }
        payload = self.http.get(url, params=params).json()

        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = (
                payload.get("content")
                or payload.get("data")
                or payload.get("results")
                or payload.get("items")
                or [payload]
            )
        else:
            raise TypeError(f"Unexpected JAO payload type: {type(payload)!r}")

        return pd.json_normalize(rows)
