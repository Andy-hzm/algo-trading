# algo-trading

A Python package for algorithmic trading research — fundamentals-based signal research, regime clustering, and backtesting.

## Structure

```
algotrading/
├── infra/
│   ├── polygon_client.py   # fetch OHLCV bars + financials from Polygon.io
│   ├── s3_client.py        # read/write Parquet data on S3
│   └── ec2_runner.py       # start/stop EC2, run jobs remotely via SSM

scripts/
├── backfill.py             # ETL: fetch hourly price history → S3
├── backfill_financials.py  # ETL: fetch quarterly/annual financials → S3
├── backfill_sectors.py     # ETL: fetch sector/industry from yfinance → S3
├── resample_daily.py       # engineer daily bars from hourly data
├── reindex_by_time.py      # copy hourly bars into cross-sectional partition layout
├── run_ec2.py              # run backfill or resample jobs on EC2
└── signals/
    └── common_signals.py   # build monthly panel (returns, ME) used by all signals

notebooks/
├── 01_polygon_exploration.ipynb
├── 02_universe_eda.ipynb
├── 03_price_eda.ipynb
├── 04_financials_eda.ipynb
└── signals/
    └── CCH_analysis.ipynb  # change-in-cash-holdings signal
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
AWS_REGION=us-east-1
S3_BUCKET=...
EC2_INSTANCE_ID=...
```

## Data

### S3 layout

```
s3://{bucket}/
└── algotrading/
    ├── prod/
    │   ├── dim/
    │   │   ├── tickers.parquet                              # ticker universe (~5000 tickers)
    │   │   └── sectors.parquet                              # sector/industry per ticker
    │   ├── bars/
    │   │   ├── hourly/ticker={T}/year={Y}/month={M}/data.parquet   # by-ticker layout
    │   │   ├── hourly_by_time/year={Y}/month={M}/ticker={T}/data.parquet  # cross-sectional layout
    │   │   └── daily/year={Y}/month={M}/data.parquet        # daily bars (all tickers per file)
    │   ├── financials/
    │   │   ├── by_ticker/ticker={T}/data.parquet            # per-ticker financials
    │   │   └── by_time/year={Y}/data.parquet                # cross-sectional financials
    │   └── signals/
    │       └── monthly_panel.parquet                        # ticker × month panel (ret, ME, ...)
    └── dev/                                                 # experiments
```

### ETL pipeline

**1. Price history (hourly)**
```bash
# Full backfill via EC2
python scripts/run_ec2.py --job backfill

# Local (small set)
python scripts/backfill.py --tickers AAPL MSFT --start 2024-01-01 --end 2024-06-01
```

**2. Daily bars** (engineered from hourly)
```bash
# Via EC2 (recommended — avoids egress)
python scripts/run_ec2.py --job resample-daily

# Local
python scripts/resample_daily.py
```

**3. Fundamentals**
```bash
python scripts/backfill_financials.py
```

**4. Sectors**
```bash
python scripts/backfill_sectors.py --workers 3
```

**5. Monthly panel** (base table for signal research)
```bash
python scripts/signals/common_signals.py
```

## Data sources

- **Price bars**: [Polygon.io](https://polygon.io) — 10yr hourly OHLCV for ~5000 US equities
- **Fundamentals**: [Polygon.io](https://polygon.io) — quarterly/annual financial statements
- **Sectors**: yfinance — GICS sector and industry classification
- **Storage**: AWS S3, Parquet (Snappy compressed)
