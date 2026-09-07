# European Electricity Market Intelligence

End-to-end Data Science / Data Engineering project for analysing and forecasting European day-ahead electricity prices and cross-border electricity flows.

The repository is designed to serve two purposes:

1. an academic data pipeline for a master's research project;
2. a portfolio-ready pet project that can later be extended with forecasting, congestion analysis, dashboards and model deployment.

## Research scope

Initial bidding zones:

- DE-LU — Germany / Luxembourg
- FR — France
- BE — Belgium
- NL — Netherlands
- PL — Poland
- CZ — Czechia

Primary targets planned for the modelling stage:

- `day_ahead_price_eur_mwh`
- `physical_cross_border_flow_mw`

The intended analytical chain is:

`generation + load + renewables + weather + fuel/CO2 + neighbouring prices + transmission constraints -> price -> price spread -> cross-border flows`

## Data sources

### ENTSO-E Transparency Platform

Main electricity-market source.

- API: https://web-api.tp.entsoe.eu/api
- Transparency Platform: https://transparency.entsoe.eu/
- Documentation: https://transparencyplatform.zendesk.com/

The ETL scaffold supports requests for:

- day-ahead prices (`A44`)
- actual load / day-ahead load forecast (`A65`)
- actual generation per type (`A75`)
- day-ahead generation forecast (`A71`)
- wind and solar forecast (`A69`)
- physical cross-border flows (`A11`)
- scheduled commercial exchanges (`A09`)
- estimated transfer capacity (`A61`)

ENTSO-E Web API access requires a personal security token.

### JAO Core Publication Tool

Flow-based market coupling and cross-zonal-capacity data.

- Tool: https://publicationtool.jao.eu/core/
- API UI: https://publicationtool.jao.eu/core/api
- Handbook: https://publicationtool.jao.eu/PublicationHandbook/Core_PublicationTool_Handbook_v2.2.pdf

The initial client contains configurable Core endpoints for:

- Max Exchanges (MaxBex)
- Min / Max Net Positions
- Final Computation / flow-based domain

JAO endpoint paths are kept in configuration so they can be changed without modifying the Python client if JAO changes its API routing.

### Open-Meteo

Historical/reanalysis weather source for prototype feature engineering.

- Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api

Initial variables:

- 2 m temperature
- 10 m / 100 m wind speed
- shortwave radiation
- cloud cover

> Important: historical realised weather must not be used as future information in a genuine day-ahead forecasting experiment. For the final forecasting setup, use historical weather forecasts or ENTSO-E day-ahead load / wind / solar forecasts to avoid leakage.

## Repository structure

```text
.
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/
│   └── eem_intel/
│       ├── extractors/
│       │   ├── entsoe.py
│       │   ├── jao.py
│       │   └── open_meteo.py
│       ├── http.py
│       ├── io.py
│       ├── settings.py
│       └── pipeline.py
├── .env.example
├── .gitignore
├── requirements.txt
└── run_etl.py
```

Raw API responses are normalised into tabular records and written as Parquet files. Raw data itself is ignored by Git and should be reproducible by running the ETL.

## Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create local environment variables:

```bash
copy .env.example .env
```

or on macOS/Linux:

```bash
cp .env.example .env
```

Set at least:

```text
ENTSOE_SECURITY_TOKEN=your_token_here
```

## Running the ETL

Run all configured sources for the smoke-test period:

```bash
python run_etl.py
```

Run only one source:

```bash
python run_etl.py --source entsoe
python run_etl.py --source jao
python run_etl.py --source weather
```

Override the configured date range:

```bash
python run_etl.py --start 2025-01-01 --end 2025-01-07
```

The default configuration intentionally uses a short date interval. Validate the schema and API behaviour before requesting multi-year hourly data.

## ETL principles

- UTC is the canonical timestamp for storage and joins.
- Source-specific identifiers are preserved in the raw layer.
- No API credentials are committed to Git.
- Large source data is reproducible and excluded from version control.
- Every extractor should be independently runnable and testable.
- Multi-year extraction should be chunked to respect source limits and reduce restart cost.

## Planned next stages

1. validate each API endpoint against a short known period;
2. add robust XML-to-timeseries parsing for all ENTSO-E document shapes;
3. build incremental extraction and checkpointing;
4. add EEX / carbon and fuel-price sources;
5. construct a canonical hourly market table;
6. engineer price-spread, lag and rolling features;
7. train time-series and ML baselines;
8. add SHAP / congestion diagnostics;
9. build a Power BI or Streamlit market-intelligence layer.

## Project status

**Phase 1 — ETL scaffold / data-source validation.**

The first goal is reproducible extraction, not model accuracy.
