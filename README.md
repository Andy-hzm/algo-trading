# algo-trading

A Python package for algorithmic trading research — price pattern detection, regime clustering, and backtesting.

## Structure

```
algotrading/
├── db/
│   ├── polygon_client.py   # fetch OHLCV bars from Polygon.io
│   └── s3_client.py        # read/write Parquet data on S3
└── backtest/
    ├── signal.py            # signal generation (BaseSignal + implementations)
    ├── portfolio.py         # walk-forward simulation
    └── evaluator.py         # Sharpe, drawdown, CAGR

scripts/
└── backfill.py             # ETL: fetch price history → S3

notebooks/
├── 01_polygon_exploration.ipynb
├── 02_universe_eda.ipynb
└── 03_price_eda.ipynb
```

## Setup

```bash
git clone https://github.com/Andy-hzm/algo-trading.git
cd algo-trading
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```
POLYGON_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-2
S3_BUCKET=...
```

## Data

### S3 layout

```
s3://{bucket}/
├── prod/
│   ├── dim/tickers.parquet                              # ticker universe
│   └── bars/hourly/ticker={T}/year={Y}/month={M}/data.parquet
└── dev/                                                 # experiments
```

### Backfill

```bash
# Write dim table + full 10yr history (prod)
python scripts/backfill.py --write-dim

# Custom date range
python scripts/backfill.py --start 2016-03-30 --end 2026-03-27

# Specific tickers only
python scripts/backfill.py --tickers AAPL MSFT GOOG

# Dev environment
python scripts/backfill.py --env dev --tickers AAPL MSFT --start 2024-01-01 --end 2024-06-01
```

## Data sources

- **Price bars**: [Polygon.io](https://polygon.io) Developer plan — 10yr hourly OHLCV, unlimited API calls
- **Storage**: AWS S3, Parquet
