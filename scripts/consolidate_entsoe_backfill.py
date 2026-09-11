from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

YEARS = list(range(2019, 2027))
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
ARTIFACT_RE = re.compile(r"^entsoe-(?P<year>\d{4})-(?P<dataset>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def scan_manifests(input_root: Path) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    raw_manifests: list[dict] = []

    for artifact_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        match = ARTIFACT_RE.match(artifact_dir.name)
        if not match:
            continue
        year = int(match.group("year"))
        dataset = match.group("dataset")
        manifest_path = artifact_dir / "manifest.json"
        if not manifest_path.exists():
            rows.append({
                "artifact": artifact_dir.name,
                "year": year,
                "dataset": dataset,
                "member": None,
                "status": "missing_manifest",
                "rows": 0,
                "error_chunks": None,
                "file": None,
            })
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_manifests.append(manifest)
        for member, info in manifest.get("members", {}).items():
            rows.append({
                "artifact": artifact_dir.name,
                "year": year,
                "dataset": dataset,
                "member": member,
                "status": info.get("status"),
                "rows": int(info.get("rows", 0) or 0),
                "error_chunks": int(info.get("error_chunks", 0) or 0),
                "file": info.get("file"),
            })

    return pd.DataFrame(rows), raw_manifests


def validate_coverage(input_root: Path, qa: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    expected = {f"entsoe-{year}-{dataset}" for year in YEARS for dataset in DATASETS}
    actual = {p.name for p in input_root.iterdir() if p.is_dir() and ARTIFACT_RE.match(p.name)}

    missing_artifacts = sorted(expected - actual)
    unexpected_artifacts = sorted(actual - expected)
    if missing_artifacts:
        errors.append(f"Missing artifacts: {missing_artifacts}")
    if unexpected_artifacts:
        errors.append(f"Unexpected artifacts: {unexpected_artifacts}")

    if qa.empty:
        errors.append("No manifest members were found")
        return errors

    bad = qa[qa["status"].isin(["partial", "error", "missing_manifest"])]
    if not bad.empty:
        errors.append(
            "Partial/error manifest members detected: "
            + bad[["artifact", "member", "status", "error_chunks"]].to_json(orient="records")
        )

    errored_chunks = qa[qa["error_chunks"].fillna(0).astype(int) > 0]
    if not errored_chunks.empty:
        errors.append(
            "Manifest members with error chunks detected: "
            + errored_chunks[["artifact", "member", "error_chunks"]].to_json(orient="records")
        )

    for _, row in qa.iterrows():
        if row["status"] == "ok":
            if not row["file"]:
                errors.append(f"OK member without file: {row['artifact']} / {row['member']}")
                continue
            member_path = input_root / row["artifact"] / row["file"]
            if not member_path.exists():
                errors.append(f"Manifest references missing parquet: {member_path}")

    return errors


def parquet_files_for_dataset(input_root: Path, dataset: str) -> list[Path]:
    files: list[Path] = []
    for year in YEARS:
        artifact_dir = input_root / f"entsoe-{year}-{dataset}"
        files.extend(sorted(artifact_dir.glob("*.parquet")))
    return files


def sql_file_list(paths: list[Path]) -> str:
    return "[" + ",".join("'" + str(p).replace("'", "''") + "'" for p in paths) + "]"


def consolidate(input_root: Path, output_root: Path, qa: pd.DataFrame, raw_manifests: list[dict]) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    datasets_root = output_root / "datasets"
    datasets_root.mkdir(exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    dataset_outputs: dict[str, dict] = {}
    all_dataset_paths: list[Path] = []

    for dataset in DATASETS:
        files = parquet_files_for_dataset(input_root, dataset)
        if not files:
            raise RuntimeError(f"No parquet files found for {dataset}")

        parquet_out = datasets_root / f"{dataset}_2019_2026.parquet"
        csv_out = datasets_root / f"{dataset}_2019_2026.csv.gz"
        file_list = sql_file_list(files)

        con.execute(
            f"COPY (SELECT *, '{dataset}'::VARCHAR AS dataset, "
            "EXTRACT(year FROM CAST(timestamp_utc AS TIMESTAMP))::INTEGER AS source_year "
            f"FROM read_parquet({file_list}, union_by_name=true)) "
            f"TO '{parquet_out}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{parquet_out}')) "
            f"TO '{csv_out}' (FORMAT CSV, HEADER, COMPRESSION GZIP)"
        )

        stats = con.execute(
            f"SELECT COUNT(*) AS rows, MIN(timestamp_utc) AS min_ts, MAX(timestamp_utc) AS max_ts "
            f"FROM read_parquet('{parquet_out}')"
        ).fetchone()
        duplicate_count = con.execute(
            f"SELECT COALESCE(SUM(cnt - 1), 0) FROM ("
            f"SELECT COUNT(*) cnt FROM read_parquet('{parquet_out}') GROUP BY ALL HAVING COUNT(*) > 1)"
        ).fetchone()[0]

        dataset_outputs[dataset] = {
            "input_parquet_files": len(files),
            "rows": int(stats[0]),
            "min_timestamp_utc": str(stats[1]) if stats[1] is not None else None,
            "max_timestamp_utc": str(stats[2]) if stats[2] is not None else None,
            "exact_duplicate_rows": int(duplicate_count),
            "parquet": str(parquet_out.relative_to(output_root)),
            "csv_gz": str(csv_out.relative_to(output_root)),
        }
        all_dataset_paths.append(parquet_out)

    unified_out = output_root / "entsoe_all_2019_2026.parquet"
    unified_csv = output_root / "entsoe_all_2019_2026.csv.gz"
    union_list = sql_file_list(all_dataset_paths)
    con.execute(
        f"COPY (SELECT * FROM read_parquet({union_list}, union_by_name=true)) "
        f"TO '{unified_out}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute(
        f"COPY (SELECT * FROM read_parquet('{unified_out}')) "
        f"TO '{unified_csv}' (FORMAT CSV, HEADER, COMPRESSION GZIP)"
    )

    unified_stats = con.execute(
        f"SELECT COUNT(*), MIN(timestamp_utc), MAX(timestamp_utc), COUNT(DISTINCT dataset) "
        f"FROM read_parquet('{unified_out}')"
    ).fetchone()

    qa.to_csv(output_root / "qa_manifest_members.csv", index=False)
    (output_root / "source_manifests.json").write_text(
        json.dumps(raw_manifests, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "years": YEARS,
        "datasets": DATASETS,
        "expected_artifacts": len(YEARS) * len(DATASETS),
        "manifest_members": int(len(qa)),
        "manifest_status_counts": {str(k): int(v) for k, v in qa["status"].value_counts(dropna=False).items()},
        "manifest_error_chunks": int(qa["error_chunks"].fillna(0).sum()),
        "dataset_outputs": dataset_outputs,
        "unified": {
            "rows": int(unified_stats[0]),
            "min_timestamp_utc": str(unified_stats[1]) if unified_stats[1] is not None else None,
            "max_timestamp_utc": str(unified_stats[2]) if unified_stats[2] is not None else None,
            "dataset_count": int(unified_stats[3]),
            "parquet": unified_out.name,
            "csv_gz": unified_csv.name,
        },
    }
    (output_root / "qa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    qa, raw_manifests = scan_manifests(args.input_root)
    errors = validate_coverage(args.input_root, qa)

    args.output_root.mkdir(parents=True, exist_ok=True)
    qa.to_csv(args.output_root / "qa_manifest_members_premerge.csv", index=False)
    if errors:
        (args.output_root / "qa_failures.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
        raise RuntimeError("Strict QA failed:\n- " + "\n- ".join(errors))

    summary = consolidate(args.input_root, args.output_root, qa, raw_manifests)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
