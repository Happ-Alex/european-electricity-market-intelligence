from __future__ import annotations

import argparse
import logging

from src.eem_intel.pipeline import run
from src.eem_intel.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="European Electricity Market Intelligence ETL")
    parser.add_argument("--source", choices=["all", "entsoe", "jao", "weather"], default="all")
    parser.add_argument("--start", help="Override start date, YYYY-MM-DD")
    parser.add_argument("--end", help="Override end date, YYYY-MM-DD (exclusive)")
    parser.add_argument("--config", help="Path to config YAML")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    settings = load_settings(args.config)
    outputs = run(settings, source=args.source, start_override=args.start, end_override=args.end)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
