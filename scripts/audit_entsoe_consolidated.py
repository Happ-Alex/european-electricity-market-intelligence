from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

DATASETS = [
    "day_ahead_prices",
    "actual_load",
    "day_ahead_load_forecast",
    "actual_generation_per_type",
    "day_ahead_generation_forecast",
    "wind_solar_forecast",
    "physical_flows",
    "scheduled_exchanges",
]

RESOLUTION_MINUTES = {
    "PT1M": 1,
    "PT5M": 5,
    "PT10M": 10,
    "PT15M": 15,
    "PT30M": 30,
    "PT60M": 60,
    "PT1H": 60,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    return p.parse_args()


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def member_expr(columns: set[str]) -> str:
    if "zone" in columns:
        return "zone"
    if {"from_zone", "to_zone"}.issubset(columns):
        return "from_zone || '->' || to_zone"
    return "'ALL'"


def exact_duplicates(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    return int(
        con.execute(
            "SELECT (SELECT COUNT(*) FROM read_parquet(?)) - "
            "(SELECT COUNT(*) FROM (SELECT DISTINCT * FROM read_parquet(?)))",
            [str(path), str(path)],
        ).fetchone()[0]
    )


def compute_gap_coverage(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    path: Path,
    member: str,
) -> pd.DataFrame:
    """Calculate timestamp gaps without treating historical resolution changes as missing data.

    ENTSO-E can switch a member from PT60M to PT15M within a year. A single
    member-year resolution therefore overstates expected timestamps. We instead
    collapse to one row per observed timestamp, attach the finest advertised
    resolution at that timestamp, and compare every timestamp to the previous
    timestamp using the *previous* timestamp's resolution. That makes a normal
    60-minute -> 15-minute transition count as continuous while still detecting
    genuine holes inside either regime.
    """
    resolution_case = "CASE resolution " + " ".join(
        f"WHEN '{code}' THEN {minutes}" for code, minutes in RESOLUTION_MINUTES.items()
    ) + " ELSE NULL END"

    ts = con.execute(
        f"SELECT {member} AS member, EXTRACT(year FROM timestamp_utc)::INTEGER AS year, "
        "timestamp_utc, "
        f"MIN({resolution_case}) AS resolution_minutes "
        "FROM read_parquet(?) "
        "GROUP BY 1,2,3 ORDER BY 1,2,3",
        [str(path)],
    ).fetchdf()

    if ts.empty:
        return pd.DataFrame(columns=[
            "dataset", "member", "year", "observed_timestamps",
            "expected_from_observed_resolution", "missing_timestamps",
            "coverage_pct", "min_timestamp_utc", "max_timestamp_utc",
            "resolutions_minutes",
        ])

    ts["timestamp_utc"] = pd.to_datetime(ts["timestamp_utc"], utc=True)
    ts["resolution_minutes"] = pd.to_numeric(ts["resolution_minutes"], errors="coerce")

    rows: list[dict] = []
    for (member_value, year), g in ts.groupby(["member", "year"], sort=True, dropna=False):
        g = g.sort_values("timestamp_utc").reset_index(drop=True)
        observed = int(len(g))
        missing = 0

        if observed > 1:
            deltas = g["timestamp_utc"].diff().dt.total_seconds().div(60.0)
            prev_res = g["resolution_minutes"].shift(1)
            for delta, step in zip(deltas.iloc[1:], prev_res.iloc[1:]):
                if pd.isna(delta) or pd.isna(step) or step <= 0:
                    continue
                # Allow small floating-point/time conversion noise while counting
                # only whole expected intervals absent after the previous point.
                intervals = int(round(float(delta) / float(step)))
                if intervals > 1:
                    missing += intervals - 1

        expected = observed + missing
        coverage_pct = observed / expected * 100.0 if expected else None
        resolutions = sorted({int(x) for x in g["resolution_minutes"].dropna().tolist()})
        rows.append({
            "dataset": dataset,
            "member": member_value,
            "year": int(year),
            "observed_timestamps": observed,
            "expected_from_observed_resolution": int(expected),
            "missing_timestamps": int(missing),
            "coverage_pct": coverage_pct,
            "min_timestamp_utc": str(g["timestamp_utc"].min()),
            "max_timestamp_utc": str(g["timestamp_utc"].max()),
            "resolutions_minutes": ",".join(map(str, resolutions)),
        })

    return pd.DataFrame(rows)


def audit_dataset(con: duckdb.DuckDBPyConnection, dataset: str, path: Path, output_root: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    schema = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()
    columns = set(schema["column_name"].tolist())
    member = member_expr(columns)

    stats = con.execute(
        "SELECT COUNT(*) AS row_count, MIN(timestamp_utc) AS min_ts, MAX(timestamp_utc) AS max_ts, "
        "COUNT(DISTINCT timestamp_utc) AS distinct_timestamps "
        "FROM read_parquet(?)",
        [str(path)],
    ).fetchone()

    duplicate_rows = exact_duplicates(con, path)

    null_rows: list[dict] = []
    for col in schema["column_name"].tolist():
        row = con.execute(
            f"SELECT COUNT(*) AS row_count, COUNT(*) FILTER (WHERE {qident(col)} IS NULL) AS nulls "
            "FROM read_parquet(?)",
            [str(path)],
        ).fetchone()
        total, nulls = int(row[0]), int(row[1])
        null_rows.append({
            "dataset": dataset,
            "column": col,
            "rows": total,
            "nulls": nulls,
            "null_pct": (nulls / total * 100.0) if total else 0.0,
        })
    pd.DataFrame(null_rows).to_csv(output_root / f"nulls_{dataset}.csv", index=False)

    value_quality = {}
    if "value" in columns:
        v = con.execute(
            "SELECT COUNT(*) FILTER (WHERE value IS NULL) AS null_values, "
            "COUNT(*) FILTER (WHERE NOT isfinite(value)) AS nonfinite_values, "
            "COUNT(*) FILTER (WHERE value < 0) AS negative_values, "
            "MIN(value), MAX(value), AVG(value) FROM read_parquet(?)",
            [str(path)],
        ).fetchone()
        value_quality = {
            "null_values": int(v[0]),
            "nonfinite_values": int(v[1]),
            "negative_values": int(v[2]),
            "min_value": None if v[3] is None else float(v[3]),
            "max_value": None if v[4] is None else float(v[4]),
            "mean_value": None if v[5] is None else float(v[5]),
        }

    by_year = con.execute(
        f"SELECT EXTRACT(year FROM timestamp_utc)::INTEGER AS year, {member} AS member, COUNT(*) AS row_count, "
        "COUNT(DISTINCT timestamp_utc) AS distinct_timestamps, MIN(timestamp_utc) AS min_ts, MAX(timestamp_utc) AS max_ts "
        "FROM read_parquet(?) GROUP BY 1,2 ORDER BY 1,2",
        [str(path)],
    ).fetchdf()
    by_year = by_year.rename(columns={"row_count": "rows"})
    by_year.insert(0, "dataset", dataset)
    by_year.to_csv(output_root / f"coverage_{dataset}.csv", index=False)

    resolution_counts = pd.DataFrame()
    if "resolution" in columns:
        resolution_counts = con.execute(
            f"SELECT {member} AS member, EXTRACT(year FROM timestamp_utc)::INTEGER AS year, resolution, "
            "COUNT(*) AS row_count, COUNT(DISTINCT timestamp_utc) AS distinct_timestamps "
            "FROM read_parquet(?) GROUP BY 1,2,3 ORDER BY 2,1,row_count DESC",
            [str(path)],
        ).fetchdf()
        resolution_counts = resolution_counts.rename(columns={"row_count": "rows"})
        resolution_counts.insert(0, "dataset", dataset)
        resolution_counts.to_csv(output_root / f"resolutions_{dataset}.csv", index=False)

    gaps = pd.DataFrame()
    if "resolution" in columns and not resolution_counts.empty:
        gaps = compute_gap_coverage(con, dataset, path, member)
    gaps.to_csv(output_root / f"gaps_{dataset}.csv", index=False)

    return {
        "dataset": dataset,
        "file": str(path),
        "rows": int(stats[0]),
        "columns": int(len(schema)),
        "min_timestamp_utc": str(stats[1]) if stats[1] is not None else None,
        "max_timestamp_utc": str(stats[2]) if stats[2] is not None else None,
        "distinct_timestamps": int(stats[3]),
        "exact_duplicate_rows": duplicate_rows,
        "value_quality": value_quality,
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    datasets_root = args.input_root / "datasets"
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    summaries = []
    for dataset in DATASETS:
        print(f"Auditing {dataset}...", flush=True)
        summaries.append(
            audit_dataset(
                con,
                dataset,
                datasets_root / f"{dataset}_2019_2026.parquet",
                args.output_root,
            )
        )

    summary_df = pd.DataFrame([
        {
            "dataset": x["dataset"],
            "rows": x["rows"],
            "columns": x["columns"],
            "min_timestamp_utc": x["min_timestamp_utc"],
            "max_timestamp_utc": x["max_timestamp_utc"],
            "distinct_timestamps": x["distinct_timestamps"],
            "exact_duplicate_rows": x["exact_duplicate_rows"],
            "null_values": x["value_quality"].get("null_values"),
            "nonfinite_values": x["value_quality"].get("nonfinite_values"),
            "negative_values": x["value_quality"].get("negative_values"),
        }
        for x in summaries
    ])
    summary_df.to_csv(args.output_root / "dataset_summary.csv", index=False)

    all_gaps = []
    for dataset in DATASETS:
        p = args.output_root / f"gaps_{dataset}.csv"
        if p.exists() and p.stat().st_size > 0:
            d = pd.read_csv(p)
            if not d.empty:
                all_gaps.append(d)
    if all_gaps:
        gaps = pd.concat(all_gaps, ignore_index=True)
        gaps.to_csv(args.output_root / "coverage_summary.csv", index=False)
    else:
        gaps = pd.DataFrame()

    critical = []
    for s in summaries:
        if s["exact_duplicate_rows"] > 0:
            critical.append(f"{s['dataset']}: {s['exact_duplicate_rows']} exact duplicate rows")
        if s["value_quality"].get("null_values", 0) > 0:
            critical.append(f"{s['dataset']}: {s['value_quality']['null_values']} null values")
        if s["value_quality"].get("nonfinite_values", 0) > 0:
            critical.append(f"{s['dataset']}: {s['value_quality']['nonfinite_values']} non-finite values")

    gap_member_years = 0
    total_missing_timestamps = 0
    if not gaps.empty and "missing_timestamps" in gaps.columns:
        gap_member_years = int((gaps["missing_timestamps"].fillna(0) > 0).sum())
        total_missing_timestamps = int(gaps["missing_timestamps"].fillna(0).sum())

    report = {
        "dataset_count": len(summaries),
        "total_rows": int(sum(x["rows"] for x in summaries)),
        "datasets": summaries,
        "critical_findings": critical,
        "coverage_rows": int(len(gaps)),
        "member_years_with_timestamp_gaps": gap_member_years,
        "total_missing_timestamps_from_observed_resolution": total_missing_timestamps,
        "notes": [
            "Negative day-ahead prices are valid market observations and are not treated as errors.",
            "Coverage is timestamp-level; datasets with several PSR/time-series rows per timestamp are intentionally not expected to be unique by timestamp.",
            "Gap detection follows the advertised resolution at each observed timestamp, so PT60M/PT15M changes inside a member-year are not misclassified as missing data.",
            "Historical empty DE_LU<->BE scheduled exchange members in 2019 are handled upstream as valid empty members.",
        ],
    }
    (args.output_root / "audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
