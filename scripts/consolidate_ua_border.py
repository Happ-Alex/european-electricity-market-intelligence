from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    return p.parse_args()


def load_direction(path: Path, dataset: str, year: int, direction: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=['timestamp_utc','value','dataset','direction','year'])
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=['timestamp_utc','value','dataset','direction','year'])
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    out = df[['timestamp_utc','value']].copy()
    out['dataset'] = dataset
    out['direction'] = direction
    out['year'] = year
    return out


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for artifact_dir in sorted(args.input_root.iterdir()):
        if not artifact_dir.is_dir():
            continue
        m = re.fullmatch(r'entsoe-ua-(physical_flows|scheduled_exchanges)-(\d{4})', artifact_dir.name)
        if not m:
            continue
        dataset, year_s = m.groups()
        year = int(year_s)
        rows.append(load_direction(artifact_dir / 'PL_to_UA.parquet', dataset, year, 'PL_to_UA'))
        rows.append(load_direction(artifact_dir / 'UA_to_PL.parquet', dataset, year, 'UA_to_PL'))

    long = pd.concat(rows, ignore_index=True)
    long.to_csv(args.output_root / 'ua_pl_border_raw_long.csv.gz', index=False, compression='gzip')

    # Average sub-hourly power within each UTC hour; do not sum MW measurements.
    long['timestamp_hour_utc'] = long['timestamp_utc'].dt.floor('h')
    hourly = (long.groupby(['dataset','direction','timestamp_hour_utc'], as_index=False)['value']
                   .mean())
    wide = (hourly.pivot(index='timestamp_hour_utc', columns=['dataset','direction'], values='value')
                  .sort_index())
    wide.columns = [f'{d}_{direction}_mw' for d, direction in wide.columns]
    wide = wide.reset_index().rename(columns={'timestamp_hour_utc':'timestamp_utc'})

    expected = [
        'physical_flows_PL_to_UA_mw','physical_flows_UA_to_PL_mw',
        'scheduled_exchanges_PL_to_UA_mw','scheduled_exchanges_UA_to_PL_mw'
    ]
    for c in expected:
        if c not in wide.columns:
            wide[c] = pd.NA

    wide['physical_net_import_to_UA_mw'] = (
        wide['physical_flows_PL_to_UA_mw'].fillna(0) -
        wide['physical_flows_UA_to_PL_mw'].fillna(0)
    )
    # Keep net scheduled exchange null where the PL->UA series was not published,
    # rather than assuming a missing direction equals zero.
    both_sched = wide[['scheduled_exchanges_PL_to_UA_mw','scheduled_exchanges_UA_to_PL_mw']].notna().all(axis=1)
    wide['scheduled_net_import_to_UA_mw'] = pd.NA
    wide.loc[both_sched, 'scheduled_net_import_to_UA_mw'] = (
        wide.loc[both_sched, 'scheduled_exchanges_PL_to_UA_mw'] -
        wide.loc[both_sched, 'scheduled_exchanges_UA_to_PL_mw']
    )
    wide['year'] = wide['timestamp_utc'].dt.year
    wide.to_csv(args.output_root / 'ua_pl_border_hourly_2019_2026.csv.gz', index=False, compression='gzip')

    annual = (wide.groupby('year', as_index=False)
        .agg(
            physical_PL_to_UA_mean_mw=('physical_flows_PL_to_UA_mw','mean'),
            physical_UA_to_PL_mean_mw=('physical_flows_UA_to_PL_mw','mean'),
            physical_net_to_UA_mean_mw=('physical_net_import_to_UA_mw','mean'),
            scheduled_PL_to_UA_mean_mw=('scheduled_exchanges_PL_to_UA_mw','mean'),
            scheduled_UA_to_PL_mean_mw=('scheduled_exchanges_UA_to_PL_mw','mean'),
            scheduled_net_to_UA_mean_mw=('scheduled_net_import_to_UA_mw','mean'),
            physical_hours=('physical_net_import_to_UA_mw','count'),
            scheduled_net_hours=('scheduled_net_import_to_UA_mw','count'),
        ))
    annual.to_csv(args.output_root / 'ua_pl_border_annual_summary.csv', index=False)

    missing = pd.DataFrame({
        'column': wide.columns,
        'missing_n': [wide[c].isna().sum() for c in wide.columns],
        'missing_pct': [wide[c].isna().mean()*100 for c in wide.columns],
    })
    missing.to_csv(args.output_root / 'ua_pl_border_missingness.csv', index=False)

    print(annual.to_string(index=False))


if __name__ == '__main__':
    main()
