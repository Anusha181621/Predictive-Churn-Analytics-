# Fashion Churn Platform

An AI-powered customer churn prediction and retention platform for a fashion / e-commerce brand.
It uses three years of customer, transaction, return and product history to answer five business
questions: **who is likely to churn**, **why**, **how much revenue is at risk**, **what to do
about it**, and **who to contact first**.

> **Status: Task 1 complete — data foundation.**
> This repository currently contains the verified data layer: the CSV loader, configuration,
> logging, data-quality validation, an inspection script and tests. The churn model, the
> explainability layer, segmentation, the retention engine and the dashboard are **not
> implemented yet**. See [Next implementation step](#next-implementation-step).

---

## Data architecture: the CSV files are the source of truth

This is a deliberate, load-bearing constraint:

- The four CSV files in [`data/`](data/) **are** the database. They are read directly with Pandas.
- There is **no** data-ingestion pipeline, **no** PostgreSQL / DuckDB step, **no** ETL and **no**
  API layer. Nothing in this project writes to or mutates the source files.
- All file locations come from configuration, and **every path is relative** to the project root.
  Copy the project to another machine with the same CSV structure and it works unchanged.
- Exactly one module reads the CSVs — [`src/data/csv_loader.py`](src/data/csv_loader.py) — and
  every other module goes through it.

```
data/*.csv  ->  csv_loader  ->  validation  ->  features  ->  churn model  ->  SHAP
                                                    |             |
                                                    +-> segmentation, revenue at risk,
                                                        retention recommendations
                                                                  |
                                                                  +-> Streamlit dashboard
```

---

## Quickstart

Requires **Python 3.11+** (verified on CPython 3.14 / Windows).

```bash
# 1. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. (optional) configure — every setting already has a working default
copy .env.example .env          # Windows;  cp .env.example .env elsewhere

# 4. profile the data and run all quality checks
python scripts/inspect_data.py

# 5. run the tests
pytest
```

`scripts/inspect_data.py` prints a full profile, writes
[`outputs/data_profile.md`](outputs/) and `outputs/data_quality_report.json`, and exits non-zero
if any error-severity check fails.

### Loading the data in your own code

```python
from src.data.csv_loader import load_all

data = load_all()
data.customers      # 1,000 rows
data.transactions   # 20,000 rows
data.returns        # 5,048 rows
data.products       # 500 rows

data.transactions["order_id"].iloc[0]   # '000001'  <- still zero-padded
```

Columns are renamed to canonical `snake_case` on load (`Net Order Value` → `net_order_value`).
Pass `normalize_columns=False` to keep the original CSV headers for display.

---

## The dataset

Three years of history for a European fashion brand, denominated in **EUR**.

| Table | File | Rows | Grain |
|---|---|---|---|
| Customers | [`data/Customer.csv`](data/Customer.csv) | 1,000 | one row per customer |
| Products | [`data/Product.csv`](data/Product.csv) | 500 | one row per SKU |
| Transactions | [`data/Transaction.csv`](data/Transaction.csv) | 20,000 | one row per **order line** (one SKU within one order) |
| Returns | [`data/Return.csv`](data/Return.csv) | 5,048 | one row per returned order line |

```
Customer (1) --< Transaction (N)     Customer ID
Product  (1) --< Transaction (N)     SKU ID
Transaction (1) --< Return (0..1)    Order ID + SKU ID
```

Verified figures:

| Metric | Value |
|---|---|
| Unique customers / SKUs / orders | 1,000 / 500 / 6,726 |
| Transaction date range | **2023-01-02 → 2025-12-31** |
| Return date range | 2023-01-08 → 2026-01-29 |
| Units purchased / returned | 28,931 / 5,786 |
| Return rate (units) | **19.9993%** |
| Return rate (order lines) | 25.24% |
| Net sales | EUR 2,132,427.41 |
| Average orders per customer | 6.73 |
| One-order customers | 277 |

Column-level definitions live in
[`DATA_DICTIONARY_AND_VALIDATION.md`](DATA_DICTIONARY_AND_VALIDATION.md); the declarative
machine-readable version the code actually uses is [`src/data/schema.py`](src/data/schema.py).

---

## Data-quality findings

The data is exceptionally clean — zero nulls, zero duplicate keys, zero orphan foreign keys in
any direction, `Net Order Value` correct on all 20,000 rows, order grain intact. The findings
below are therefore **traps rather than defects**, and each one is encoded in the loader,
the validator or a test so it cannot be rediscovered as a bug.

| # | Finding | Where it is handled |
|---|---|---|
| 1 | **`Order ID` is a zero-padded 6-digit string** (`000001`). Pandas infers `int64` and silently destroys the padding, breaking every string join against `Return.csv`. | `schema.py` pins `dtype="string"`; `test_csv_loader.py` asserts the padding survives |
| 2 | **104 `Return Date`s fall after the last `Purchase Date`** (to 2026-01-29) — late-December orders returned in January. Features computed *as of* a date must clip returns, or a return that had not yet happened leaks in. | reported as `info` by `validation.py`; pinned by `test_data_integrity.py` |
| 3 | **5,048 return rows is *not* the ~20% return rate.** 20% is the **unit** rate (5,786/28,931); the line-level rate is 25.24%. Conflating them mis-states every return feature. | both rates reported side by side; both pinned by a test |
| 4 | **`Gender` exists in both `Customer.csv` and `Product.csv`** with different vocabularies (Female/Male/Other vs Men/Women/Unisex) → `Gender_x`/`Gender_y` collision on join. | renamed on load to `customer_gender` / `product_gender` |
| 5 | **Non-ASCII city names** (Düsseldorf, Liège); files are UTF-8 / CRLF, no BOM. Windows' default cp1252 would mangle them. | loader reads `utf-8-sig`; log and console output forced to UTF-8 |
| 6 | **`Registration Date` == first purchase date** for all 1,000 customers, so tenure and days-since-first-purchase are one quantity, not two features. | asserted by a test |
| 7 | **Every customer has ≥1 transaction**, so there is no cold-start cohort here to exercise a no-history code path. | asserted by a test |
| 8 | **Newest registration is 2025-12-21** — the newest customers have ≤10 days of history and are right-censored; they must be flagged or excluded when labelling churn. | reported as `info`; pinned by a test |
| 9 | **Volume grows steeply year over year** (1,092 → 2,446 → 3,188 orders in 2023/24/25), so a time-based split trains on materially less data than it tests on. | reported as `info` |
| 10 | **`Selling Price` deliberately drifts from `Product.Price`** (0.9699–1.0400×; only 13,965/20,000 match exactly). Not an error — do not "fix" it. | reported as `info`, never flagged |
| 11 | **Churn class balance is workable**: 40.2% positive at a 180-day threshold, 56.1% at 90 days. Heavy resampling is unnecessary. | informs the model step |

---

## Project layout

```
data/                     the four source CSVs — the source of truth
├── Customer.csv
├── Transaction.csv
├── Return.csv
└── Product.csv

src/
├── config/settings.py    environment-driven configuration, relative path resolution
├── data/
│   ├── schema.py         declarative columns, dtypes, allowed values
│   ├── csv_loader.py     THE reusable CSV utility (read-only, cached)
│   └── validation.py     read-only data-quality checks
├── features/             feature engineering            (not implemented yet)
├── models/               churn model train/eval/predict (not implemented yet)
├── explainability/       SHAP + readable explanations   (not implemented yet)
├── segmentation/         value / risk / behaviour       (not implemented yet)
├── retention/            revenue at risk, ROI, actions  (not implemented yet)
└── utils/
    ├── paths.py          the ONLY place path logic lives
    └── logging_config.py idempotent logging setup

app/                      Streamlit dashboard            (not implemented yet)
├── components/  charts/  pages/
models/                   serialised models (git-ignored)
outputs/                  generated reports (git-ignored)
logs/                     rotating log files (git-ignored)
scripts/inspect_data.py   reproducible data profile + quality report
tests/                    pytest suite
```

---

## Configuration

Settings are read from the environment, or from an optional `.env` in the project root (see
[`.env.example`](.env.example)). **Every setting has a working default, so no `.env` is
required.** Real environment variables take precedence over the file.

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `data` | Directory holding the CSVs, relative to the project root |
| `CUSTOMER_FILE` | `Customer.csv` | Customer file name |
| `TRANSACTION_FILE` | `Transaction.csv` | Transaction file name |
| `RETURN_FILE` | `Return.csv` | Return file name |
| `PRODUCT_FILE` | `Product.csv` | Product file name |
| `MODELS_DIR` | `models` | Where serialised models are written |
| `OUTPUTS_DIR` | `outputs` | Where reports and exports are written |
| `LOG_DIR` | `logs` | Where rotating log files are written |
| `LOG_LEVEL` | `INFO` | Root log level |
| `CHURN_INACTIVITY_DAYS` | `180` | Default inactivity window for the churn label |
| `AS_OF_DATE` | *(blank)* | Prediction date; blank means derive it from the data (2025-12-31) |
| `RISK_THRESHOLD_MEDIUM` | `0.30` | Low / Medium boundary |
| `RISK_THRESHOLD_HIGH` | `0.60` | Medium / High boundary |
| `RISK_THRESHOLD_CRITICAL` | `0.80` | High / Critical boundary |
| `RANDOM_SEED` | `42` | Reproducibility |
| `CURRENCY` | `EUR` | Display currency |

```python
from src.config.settings import get_settings

settings = get_settings()
settings.transaction_path    # absolute, resolved from the project root
settings.validate_files()    # raises ConfigError naming any missing CSV
```

Relative paths are resolved against the **project root**, not the working directory, so scripts
behave identically no matter where they are launched from. Absolute paths are still accepted if
you want to point `DATA_DIR` at a shared location.

---

## Testing

```bash
pytest              # whole suite
pytest -k loader    # loader tests only
```

The suite covers three things: that configuration resolves portably, that the loader preserves
the dtypes and encodings the data actually needs, and that every referential and business
invariant in the data still holds.

---

## Dependency notes

Pins in [`requirements.txt`](requirements.txt) that are deliberate rather than incidental:

- **`pandas>=2.2,<3.0`** — pandas 3.0 makes the string dtype and copy-on-write the default;
  those semantics are not yet well exercised against the scikit-learn / SHAP versions used here.
- **`shap>=0.51`** — 0.51.0 is the first SHAP release shipping a CPython 3.14 Windows wheel.
- `xgboost` and `lightgbm` publish version-independent `py3-none-win_amd64` wheels, so they are
  not tied to the interpreter's ABI tag.

---

## Next implementation step

**Churn label definition and the customer-level feature store.**

1. Derive the as-of date from the data (2025-12-31) rather than hard-coding it.
2. Compute **customer-specific expected purchase intervals** so that seasonal and
   low-frequency buyers are not mislabelled as churned merely for being inactive — the brief
   requires distinguishing true churn risk from normal, seasonal and new-customer inactivity.
3. Build the feature table strictly *as of* the prediction date: RFM, purchase-trend and gap
   features, lifecycle stage, seasonality scores, product affinity, discount/coupon behaviour,
   return behaviour and customer value.
4. Clip returns to the as-of date so finding 2 above cannot leak future information.
5. Flag the right-censored newest cohort (finding 8) so it is excluded from training labels.
