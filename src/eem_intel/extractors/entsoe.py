from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import pandas as pd

from ..http import HttpClient


@dataclass
class EntsoeExtractor:
    base_url: str
    security_token: str
    http: HttpClient

    @staticmethod
    def _format_period(ts: datetime) -> str:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc).strftime("%Y%m%d%H%M")

    @staticmethod
    def _base_dataset_params(dataset_params: dict) -> dict:
        internal_keys = {"scope", "domain_params", "required", "chunk_days"}
        return {k: v for k, v in dataset_params.items() if k not in internal_keys}

    def _request(self, params: dict) -> str:
        query = {"securityToken": self.security_token, **params}
        return self.http.get(self.base_url, params=query).text

    def fetch_zone(self, dataset_params: dict, domain: str, start: datetime, end: datetime) -> pd.DataFrame:
        params = self._base_dataset_params(dataset_params)
        domain_params = dataset_params.get("domain_params", ["in_Domain", "out_Domain"])
        for param_name in domain_params:
            params[param_name] = domain
        params.update({
            "periodStart": self._format_period(start),
            "periodEnd": self._format_period(end),
        })
        return self.parse_timeseries(self._request(params))

    def fetch_border(self, dataset_params: dict, out_domain: str, in_domain: str, start: datetime, end: datetime) -> pd.DataFrame:
        params = self._base_dataset_params(dataset_params)
        params.update({
            "out_Domain": out_domain,
            "in_Domain": in_domain,
            "periodStart": self._format_period(start),
            "periodEnd": self._format_period(end),
        })
        return self.parse_timeseries(self._request(params))

    @staticmethod
    def _strip_ns(tag: str) -> str:
        return tag.split("}", 1)[-1]

    @classmethod
    def _child_text(cls, node: ET.Element, suffix: str) -> str | None:
        for child in node.iter():
            if cls._strip_ns(child.tag) == suffix and child.text is not None:
                return child.text.strip()
        return None

    @staticmethod
    def _iso_duration_to_minutes(value: str) -> int:
        if not value.startswith("PT"):
            raise ValueError(f"Unsupported ENTSO-E resolution: {value}")
        value = value[2:]
        if value.endswith("M"):
            return int(value[:-1])
        if value.endswith("H"):
            return int(value[:-1]) * 60
        raise ValueError(f"Unsupported ENTSO-E resolution: {value}")

    @classmethod
    def parse_timeseries(cls, xml_text: str) -> pd.DataFrame:
        root = ET.fromstring(xml_text)
        records: list[dict] = []

        for ts in root.iter():
            if cls._strip_ns(ts.tag) != "TimeSeries":
                continue

            series_meta = {
                "mRID": cls._child_text(ts, "mRID"),
                "business_type": cls._child_text(ts, "businessType"),
                "process_type": cls._child_text(ts, "process.processType"),
                "curve_type": cls._child_text(ts, "curveType"),
                "psr_type": cls._child_text(ts, "psrType"),
                "in_domain": cls._child_text(ts, "in_Domain.mRID"),
                "out_domain": cls._child_text(ts, "out_Domain.mRID"),
                "in_bidding_zone": cls._child_text(ts, "inBiddingZone_Domain.mRID"),
                "out_bidding_zone": cls._child_text(ts, "outBiddingZone_Domain.mRID"),
                "contract_market_agreement_type": cls._child_text(ts, "contract_MarketAgreement.type"),
                "currency": cls._child_text(ts, "currency_Unit.name"),
                "price_unit": cls._child_text(ts, "price_Measure_Unit.name"),
                "quantity_unit": cls._child_text(ts, "quantity_Measure_Unit.name"),
            }

            for period in ts.iter():
                if cls._strip_ns(period.tag) != "Period":
                    continue

                start_text = cls._child_text(period, "start")
                resolution = cls._child_text(period, "resolution")
                if not start_text or not resolution:
                    continue

                period_start = pd.Timestamp(start_text)
                step = pd.Timedelta(minutes=cls._iso_duration_to_minutes(resolution))

                for point in period:
                    if cls._strip_ns(point.tag) != "Point":
                        continue
                    position_text = cls._child_text(point, "position")
                    if not position_text:
                        continue
                    position = int(position_text)
                    value = (
                        cls._child_text(point, "price.amount")
                        or cls._child_text(point, "quantity")
                        or cls._child_text(point, "amount")
                    )
                    records.append({
                        **series_meta,
                        "timestamp_utc": period_start + (position - 1) * step,
                        "resolution": resolution,
                        "position": position,
                        "value": pd.to_numeric(value, errors="coerce"),
                    })

        return pd.DataFrame.from_records(records)
