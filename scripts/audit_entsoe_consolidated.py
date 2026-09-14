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
            f"SELECT {member} AS member, resolution, COUNT(*) AS row_count, COUNT(DISTINCT timestamp_utc) AS distinct_timestamps "
            "FROM read_parquet(?) GROUP BY 1,2 ORDER BY 1, row_count DESC",
            [str(path)],
        ).fetchdf()
        resolution_counts = resolution_counts.rename(columns={"row_count": "rows"})
        resolution_counts.insert(0, "dataset", dataset)
        resolution_counts.to_csv(output_root / f"resolutions_{dataset}.csv", index=False)

    gaps: list[dict] = []
    if "resolution" in columns and not resolution_counts.empty:
        coverage = con.execute(
            f"SELECT {member} AS member, EXTRACT(year FROM timestamp_utc)::INTEGER AS year, "
            "MIN(timestamp_utc) AS min_ts, MAX(timestamp_utc) AS max_ts, "
            "COUNT(DISTINCT timestamp_utc) AS observed_timestamps "
            "FROM read_parquet(?) GROUP BY 1,2 ORDER BY 2,1",
            [str(path)],
        ).fetchdf()
        res = resolution_counts.copy()
        res["minutes"] = res["resolution"].map(RESOLUTION_MINUTES)
        res = res.dropna(subset=["minutes"])
        finest = res.groupby("member", as_index=False)["minutes"].min()
        coverage = coverage.merge(finest, on="member", how="left")
        for _, r in coverage.iterrows():
            minutes = r["minutes"]
            expected = None
            missing = None
            coverage_pct = None
            if pd.notna(minutes) and pd.notna(r["min_ts"]) and pd.notna(r["max_ts"]):
                delta_minutes = (pd.Timestamp(r["max_ts"]) - pd.Timestamp(r["min_ts"])).total_seconds() / 60.0
                expected = int(round(delta_minutes / float(minutes))) + 1
                missing = max(0, expected - int(r["observed_timestamps"]))
                coverage_pct = min(100.0, int(r["observed_timestamps"]) / expected * 100.0) if expected else None
            gaps.append({
                "dataset": dataset,
                "member": r["member"],
                "year": int(r["year"]),
                "resolution_minutes": None if pd.isna(minutes) else int(minutes),
                "observed_timestamps": int(r["observed_timestamps"]),
                "expected_between_member_minmax": expected,
                "missing_between_member_minmax": missing,
                "coverage_pct": coverage_pct,
                "min_timestamp_utc": str(r["min_ts"]),
                "max_timestamp_utc": str(r["max_ts"]),
            })
    pd.DataFrame(gaps).to_csv(output_root / f"gaps_{dataset}.csv", index=False)

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

    report = {
        "dataset_count": len(summaries),
        "total_rows": int(sum(x["rows"] for x in summaries)),
        "datasets": summaries,
        "critical_findings": critical,
        "coverage_rows": int(len(gaps)),
        "notes": [
            "Negative day-ahead prices are valid market observations and are not treated as errors.",
            "Coverage is timestamp-level; datasets with several PSR/time-series rows per timestamp are intentionally not expected to be unique by timestamp.",
            "Historical empty DE_LU<->BE scheduled exchange members in 2019 are handled upstream as valid empty members.",
        ],
    }
    (args.output_root / "audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
