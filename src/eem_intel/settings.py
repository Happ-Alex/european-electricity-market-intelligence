from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    config: dict
    entsoe_token: str | None
    entsoe_base_url: str
    jao_base_url: str


def load_settings(config_path: str | Path | None = None) -> Settings:
    load_dotenv(ROOT / ".env")
    path = Path(config_path) if config_path else ROOT / "config" / "config.yaml"
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    return Settings(
        config=config,
        entsoe_token=os.getenv("ENTSOE_SECURITY_TOKEN") or None,
        entsoe_base_url=os.getenv("ENTSOE_BASE_URL", "https://web-api.tp.entsoe.eu/api"),
        jao_base_url=os.getenv("JAO_BASE_URL", "https://publicationtool.jao.eu/core/api"),
    )
