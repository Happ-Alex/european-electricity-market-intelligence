from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

PSR_MAP = {
    "B01": "biomass",
    "B02": "fossil_brown_coal_lignite",
    "B03": "fossil_coal_derived_gas",
    "B04": "fossil_gas",
    "B05": "fossil_hard_coal",
    "B06": "fossil_oil",
    "B09": "geothermal",
    "B10": "hydro_pumped_storage",
    "B11": "hydro_run_of_river",
    "B12": "hydro_water_reservoir",
    "B14": "nuclear",
    "B15": "other_renewable",
    "B16": "solar",
    "B17": "waste",
    "B18": "wind_offshore",
    "B19": "wind_onshore",
    "B20": "other",
}

RENEWABLE_PSR = {"B01", "B09", "B11", "B12", "B15", "B16", "B18", "B19"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    return p.parse_args()


def path(root: Path, dataset: str) -> str:
    return str(root / "datasets" / f"{dataset}_2019_2026.parquet")


def write_query(con: duckdb.DuckDBPyConnection, sql: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({sql}) TO '{str(out).replace(chr(39), chr(39)*2)}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    prices = path(args.input_root, "day_ahead_prices")
    actual_load = path(args.input_root, "actual_load")
    load_fc = path(args.input_root, "day_ahead_load_forecast")
    actual_gen = path(args.input_root, "actual_generation_per_type")
    gen_fc = path(args.input_root, "day_ahead_generation_forecast")
    wind_solar_fc = path(args.input_root, "wind_solar_forecast")
    physical = path(args.input_root, "physical_flows")
    scheduled = path(args.input_root, "scheduled_exchanges")

    for p in [prices, actual_load, load_fc, actual_gen, gen_fc, wind_solar_fc, physical, scheduled]:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    # Use the full price horizon as the canonical six-zone hourly grid.
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW price_hourly AS
        SELECT zone,
               date_trunc('hour', timestamp_utc) AS timestamp_utc,
               AVG(value) AS day_ahead_price_eur_mwh
        FROM read_parquet('{prices}')
        GROUP BY 1,2
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW actual_load_hourly AS
        SELECT zone, date_trunc('hour', timestamp_utc) AS timestamp_utc,
               AVG(value) AS actual_load_mw
        FROM read_parquet('{actual_load}') GROUP BY 1,2
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW load_fc_hourly AS
        SELECT zone, date_trunc('hour', timestamp_utc) AS timestamp_utc,
               AVG(value) AS day_ahead_load_forecast_mw
        FROM read_parquet('{load_fc}') GROUP BY 1,2
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW gen_fc_hourly AS
        SELECT zone, date_trunc('hour', timestamp_utc) AS timestamp_utc,
               AVG(value) AS day_ahead_generation_forecast_mw
        FROM read_parquet('{gen_fc}') GROUP BY 1,2
    """)

    # Generation: average sub-hourly power inside each PSR/hour, then pivot/sum.
    gen_case = []
    for code, name in PSR_MAP.items():
        gen_case.append(f"SUM(CASE WHEN psr_type = '{code}' THEN value_mw ELSE 0 END) AS generation_{name}_mw")
    renewable_case = ",".join(f"'{x}'" for x in sorted(RENEWABLE_PSR))
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW generation_hourly AS
        WITH x AS (
            SELECT zone, date_trunc('hour', timestamp_utc) AS timestamp_utc, psr_type,
                   AVG(value) AS value_mw
            FROM read_parquet('{actual_gen}')
            GROUP BY 1,2,3
        )
        SELECT zone, timestamp_utc,
               SUM(value_mw) AS total_generation_mw,
               SUM(CASE WHEN psr_type IN ({renewable_case}) THEN value_mw ELSE 0 END) AS renewable_generation_mw,
               {', '.join(gen_case)}
        FROM x
        GROUP BY 1,2
    """)

    # Forecast A69 commonly contains separate wind/solar PSR series; keep both mapped columns and total.
    ws_case = []
    for code in ["B16", "B18", "B19"]:
        ws_case.append(f"SUM(CASE WHEN psr_type = '{code}' THEN value_mw ELSE 0 END) AS forecast_{PSR_MAP[code]}_mw")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW wind_solar_fc_hourly AS
        WITH x AS (
            SELECT zone, date_trunc('hour', timestamp_utc) AS timestamp_utc, psr_type,
                   AVG(value) AS value_mw
            FROM read_parquet('{wind_solar_fc}')
            GROUP BY 1,2,3
        )
        SELECT zone, timestamp_utc,
               SUM(value_mw) AS wind_solar_forecast_total_mw,
               {', '.join(ws_case)}
        FROM x GROUP BY 1,2
    """)

    # Directed border power is averaged to hourly first, then converted into zone imports/exports.
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW physical_directed_hourly AS
        SELECT from_zone, to_zone, date_trunc('hour', timestamp_utc) AS timestamp_utc,
               AVG(value) AS flow_mw
        FROM read_parquet('{physical}') GROUP BY 1,2,3
    """)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW physical_zone_hourly AS
        WITH z AS (
            SELECT from_zone AS zone, timestamp_utc, 0.0 AS imports_mw, SUM(flow_mw) AS exports_mw
            FROM physical_directed_hourly GROUP BY 1,2
            UNION ALL
            SELECT to_zone AS zone, timestamp_utc, SUM(flow_mw) AS imports_mw, 0.0 AS exports_mw
            FROM physical_directed_hourly GROUP BY 1,2
        )
        SELECT zone, timestamp_utc, SUM(imports_mw) AS physical_imports_mw,
               SUM(exports_mw) AS physical_exports_mw,
               SUM(imports_mw) - SUM(exports_mw) AS physical_net_import_mw
        FROM z GROUP BY 1,2
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW scheduled_directed_hourly AS
        SELECT from_zone, to_zone, date_trunc('hour', timestamp_utc) AS timestamp_utc,
               AVG(value) AS exchange_mw
        FROM read_parquet('{scheduled}') GROUP BY 1,2,3
    """)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW scheduled_zone_hourly AS
        WITH z AS (
            SELECT from_zone AS zone, timestamp_utc, 0.0 AS imports_mw, SUM(exchange_mw) AS exports_mw
            FROM scheduled_directed_hourly GROUP BY 1,2
            UNION ALL
            SELECT to_zone AS zone, timestamp_utc, SUM(exchange_mw) AS imports_mw, 0.0 AS exports_mw
            FROM scheduled_directed_hourly GROUP BY 1,2
        )
        SELECT zone, timestamp_utc, SUM(imports_mw) AS scheduled_imports_mw,
               SUM(exports_mw) AS scheduled_exports_mw,
               SUM(imports_mw) - SUM(exports_mw) AS scheduled_net_import_mw
        FROM z GROUP BY 1,2
    """)

    generation_columns = ",\n               ".join(f"g.generation_{name}_mw" for name in PSR_MAP.values())
    master_sql = f"""
        SELECT p.zone,
               p.timestamp_utc,
               EXTRACT(year FROM p.timestamp_utc)::INTEGER AS year,
               EXTRACT(month FROM p.timestamp_utc)::INTEGER AS month,
               EXTRACT(dayofweek FROM p.timestamp_utc)::INTEGER AS day_of_week,
               EXTRACT(hour FROM p.timestamp_utc)::INTEGER AS hour_utc,
               p.day_ahead_price_eur_mwh,
               l.actual_load_mw,
               lf.day_ahead_load_forecast_mw,
               l.actual_load_mw - lf.day_ahead_load_forecast_mw AS load_forecast_error_mw,
               gf.day_ahead_generation_forecast_mw,
               g.total_generation_mw,
               g.renewable_generation_mw,
               CASE WHEN g.total_generation_mw <> 0 THEN g.renewable_generation_mw / g.total_generation_mw END AS renewable_share_of_generation,
               {generation_columns},
               ws.wind_solar_forecast_total_mw,
               ws.forecast_solar_mw,
               ws.forecast_wind_offshore_mw,
               ws.forecast_wind_onshore_mw,
               ph.physical_imports_mw,
               ph.physical_exports_mw,
               ph.physical_net_import_mw,
               se.scheduled_imports_mw,
               se.scheduled_exports_mw,
               se.scheduled_net_import_mw,
               l.actual_load_mw
                 - COALESCE(g.generation_solar_mw, 0)
                 - COALESCE(g.generation_wind_offshore_mw, 0)
                 - COALESCE(g.generation_wind_onshore_mw, 0) AS residual_load_mw
        FROM price_hourly p
        LEFT JOIN actual_load_hourly l USING(zone, timestamp_utc)
        LEFT JOIN load_fc_hourly lf USING(zone, timestamp_utc)
        LEFT JOIN gen_fc_hourly gf USING(zone, timestamp_utc)
        LEFT JOIN generation_hourly g USING(zone, timestamp_utc)
        LEFT JOIN wind_solar_fc_hourly ws USING(zone, timestamp_utc)
        LEFT JOIN physical_zone_hourly ph USING(zone, timestamp_utc)
        LEFT JOIN scheduled_zone_hourly se USING(zone, timestamp_utc)
        ORDER BY zone, timestamp_utc
    """

    master_parquet = args.output_root / "entsoe_master_hourly_2019_2026.parquet"
    write_query(con, master_sql, master_parquet)
    con.execute(
        f"COPY (SELECT * FROM read_parquet('{master_parquet}')) TO '{args.output_root / 'entsoe_master_hourly_2019_2026.csv.gz'}' "
        "(FORMAT CSV, HEADER, COMPRESSION GZIP)"
    )

    # Add model-friendly lag and rolling features without leaking future observations.
    feature_sql = f"""
        SELECT *,
               LAG(day_ahead_price_eur_mwh, 1) OVER w AS price_lag_1h,
               LAG(day_ahead_price_eur_mwh, 24) OVER w AS price_lag_24h,
               LAG(day_ahead_price_eur_mwh, 168) OVER w AS price_lag_168h,
               AVG(day_ahead_price_eur_mwh) OVER (PARTITION BY zone ORDER BY timestamp_utc ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS price_roll24_mean,
               STDDEV_SAMP(day_ahead_price_eur_mwh) OVER (PARTITION BY zone ORDER BY timestamp_utc ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS price_roll24_std,
               AVG(actual_load_mw) OVER (PARTITION BY zone ORDER BY timestamp_utc ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS load_roll24_mean,
               AVG(residual_load_mw) OVER (PARTITION BY zone ORDER BY timestamp_utc ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS residual_load_roll24_mean
        FROM read_parquet('{master_parquet}')
        WINDOW w AS (PARTITION BY zone ORDER BY timestamp_utc)
        ORDER BY zone, timestamp_utc
    """
    features_parquet = args.output_root / "entsoe_features_hourly_2019_2026.parquet"
    write_query(con, feature_sql, features_parquet)

    stats = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT zone), MIN(timestamp_utc), MAX(timestamp_utc), "
        "COUNT(*) - COUNT(day_ahead_price_eur_mwh), COUNT(*) - COUNT(actual_load_mw) "
        f"FROM read_parquet('{master_parquet}')"
    ).fetchone()
    summary = {
        "rows": int(stats[0]),
        "zones": int(stats[1]),
        "min_timestamp_utc": str(stats[2]),
        "max_timestamp_utc": str(stats[3]),
        "missing_price_rows": int(stats[4]),
        "missing_actual_load_rows": int(stats[5]),
        "master_parquet": master_parquet.name,
        "master_csv_gz": "entsoe_master_hourly_2019_2026.csv.gz",
        "features_parquet": features_parquet.name,
        "grain": "zone x UTC hour",
        "notes": [
            "Power measurements are averaged within each UTC hour, not summed.",
            "Generation per type is first averaged per PSR/hour and then summed across PSR types.",
            "Cross-border imports/exports cover only the configured borders in config/config.yaml.",
            "Lag/rolling features use prior observations only to avoid direct future leakage.",
        ],
    }
    (args.output_root / "master_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
