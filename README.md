# Fashion Churn Platform

An AI-powered customer churn prediction and retention platform for a fashion / e-commerce brand.
It uses three years of customer, transaction, return and product history to answer five business
questions: **who is likely to churn**, **why**, **how much revenue is at risk**, **what to do
about it**, and **who to contact first**.

> **Status: complete, end to end.**
> Data layer, per-file validation, the customer feature store (148 features as of a prediction
> date), the churn model with time-based validation and calibrated probabilities, per-customer SHAP
> explanations in plain English, the retention decision layer — revenue at risk, twelve segments,
> prioritisation and personalised recommendations — and the eight-page Streamlit dashboard.
> 456 tests pass. Run it with `streamlit run app/dashboard.py`; see [Dashboard](#dashboard).

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

The dashboard is the last box and nothing more: it reads the artefacts the stages before it wrote
and renders them. The one exception is the What-If simulator, which re-runs the real retention
layer rather than reimplementing its arithmetic in the browser.

### The constraint is enforced, not just documented

[`tests/test_architecture.py`](tests/test_architecture.py) turns each clause into a test, so the
architecture cannot drift the next time somebody adds a convenient dependency:

| Claim | How it is checked |
|---|---|
| No database is required | No driver or ORM is imported anywhere (parsed via AST, so a name in a comment cannot false-positive); no `CREATE TABLE`/`INSERT`/`create_engine`/`.execute(`; `requirements.txt` declares none |
| No ETL pipeline is required | No Airflow / Prefect / Dagster / Luigi / Celery / Spark import |
| No ingestion service exists | No web or ASGI framework; every `scripts/*.py` is a batch CLI with a `main()` that exits, and contains no `while True`, `serve_forever` or socket use |
| Original CSVs remain untouched | **The four files are hashed, the pipeline is run over them, and they are hashed again.** No write in the codebase targets a source-data path, and `csv_loader.py` contains no write call at all |
| CSV files are the source of truth | Only the loader reads them. The dashboard reads `outputs/*.csv`, which are generated artefacts the loader could not parse anyway — a different act, asserted separately |
| Dashboard works from CSV-based outputs | Every artefact it declares lives in `outputs/` or `models/` and names the script that produces it; the app imports no database or API client |

Portability is covered too: no absolute path is hardcoded, relative settings resolve against the
project root rather than the working directory, and an absolute override is honoured as the
documented escape hatch.

---

## The workflow

Five steps, from four CSV files to an interactive dashboard. Requires **Python 3.11+** (verified
on CPython 3.14 / Windows). There is nothing else to provision: no database to start, no service to
register, no API key, and no `.env` — every setting has a working default. Copy `.env.example` to
`.env` only if you want to change one.

### 0. Install

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1-5. Build the artefacts, then start the dashboard

```bash
# 1. place the four CSV files in data/
#    Customer.csv  Transaction.csv  Return.csv  Product.csv

# 2. generate features
python scripts/build_features.py

# 3. train the model
python scripts/train_model.py

# 4. generate predictions, explanations and retention actions
python scripts/predict.py
python scripts/explain.py
python scripts/retention.py

# 5. start the dashboard
streamlit run app/dashboard.py
```

Step 4 is three commands rather than one because the dashboard shows more than a probability: it
also shows *why* each customer is at risk and *what to do about it*, which are the explainability
and retention artefacts. Running only `predict.py` still works — the pages that need the other two
say which command produces what they are missing, instead of failing.

Steps 2-4 take a few minutes in total and each script is idempotent, so re-running one is safe.
They are batch CLIs that exit when done; only step 5 stays in the foreground.

### Starting the dashboard when the environment is already built

`.venv/`, `models/` and `outputs/` are git-ignored — they persist locally but never arrive with a
clone. When they are already in place, steps 0-4 are done and starting the dashboard is the whole
procedure:

```powershell
cd <project root>
.venv\Scripts\activate
streamlit run app\dashboard.py
```

Streamlit prints a local URL — **http://localhost:8501** by default — and holds the terminal until
`Ctrl+C`. Add `--server.port 8502` if that port is taken, or `--server.headless true` to keep it
from opening a browser.

**Launch it from the project root.** The data paths themselves do not care about the working
directory, because [`src/utils/paths.py`](src/utils/paths.py) anchors every relative setting to the
repo root via `Path(__file__).resolve().parents[2]` rather than to `cwd`. But `.streamlit/config.toml`
is only read from the directory you launch from, so starting elsewhere silently loses the theme.

### Optional, and useful before the first run

```bash
python scripts/validate_data.py     # 101 checks over the four CSVs -> outputs/data_quality_report.json
python scripts/inspect_data.py      # the above, plus a full data profile -> outputs/data_profile.md
pytest                              # 456 tests
```

Both scripts exit non-zero if any error-severity check fails, so either works as a pipeline or
CI gate — worth running first when the CSVs have been refreshed.

### If it does not start

| Symptom | Cause and fix |
|---|---|
| `streamlit: command not found` / not recognised | The virtual environment is not active. Activate it, or skip activation and call the interpreter directly: `.venv\Scripts\python.exe -m streamlit run app\dashboard.py` |
| A page reports a missing artefact | That step has not been run. The page names the file *and* the command that produces it — run that command. Nothing else on the page fails meanwhile. |
| `ConfigError` naming one or more CSVs | A source file is absent from `data/`. `Settings.validate_files()` lists each one by resolved path. |
| `joblib.load` warns or fails on the model | `models/churn_model.joblib` was written by a different scikit-learn version. Re-run `python scripts/train_model.py`. |
| The theme looks wrong (dark or unstyled) | Started from a different directory — see above. |
| Port 8501 already in use | `streamlit run app\dashboard.py --server.port 8502` |

The dashboard exposes a health endpoint while running, which is the quickest confirmation that the
server is actually up rather than merely launched:

```bash
curl http://localhost:8501/_stcore/health     # -> ok
```

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

## Validation layer

[`src/data/validation.py`](src/data/validation.py) is read-only by contract: a validator reads
DataFrames and *reports*. It never coerces a value, drops a row, repairs a field or writes to
`data/`. A failure means the data changed in a way the platform must know about — not that the
data should be quietly fixed.

One validator per source file, each usable on its own:

```python
from src.data.csv_loader import load_all
from src.data.validation import validate_customers, validate_datasets, compute_return_rate

data = load_all()

# one file at a time — cross-table checks are simply skipped when the other table is absent
report = validate_customers(data.customers)
report.ok                      # False if any error-severity check failed
report.metrics["unique_customers"]
report.check("Customer ID is unique").passed

# or everything at once
full = validate_datasets(data)
full.ok                        # True on the shipped data (101/101 checks)
full.save("outputs/data_quality_report.json")

compute_return_rate(data.transactions, data.returns)["unit_return_rate"]   # 0.199993
```

### What each validator checks

| Validator | Checks |
|---|---|
| `validate_customers` | key present / unique / non-null / non-blank; age numeric and in range; gender, country and channel domains; registration date parses, is not implausibly early, and does not fall after the last purchase |
| `validate_products` | key present / unique / non-null; price numeric and `>= 0`; category and subcategory populated; category, brand and gender domains |
| `validate_transactions` | keys present and non-blank; `Order ID` still a zero-padded string; purchase date valid; `quantity > 0`; `selling price >= 0`; `0 <= discount <= 100`; net order value matches its formula; coupon implies a discount; **FK** `Customer ID → Customer.csv` and `SKU ID → Product.csv` |
| `validate_returns` | keys present and non-blank; return date valid; `return quantity > 0`; one return per order line; every return corresponds to a real order line; return quantity never exceeds the purchased quantity; return date after purchase date; return attributed to the order's own customer |
| `validate_relationships` | order grain (one customer / date / payment method per order, unique SKU per order); registration equals first purchase; no purchase before registration; each city in one country; plus the intentional properties recorded as `info` |

Every validator also returns summary **metrics** — totals, unique keys, duplicates, missing
values per column, and date ranges — so a dashboard can render the numbers without re-deriving
them.

### Severities

| Severity | Meaning | Effect |
|---|---|---|
| `error` | A structural, referential or arithmetic invariant the platform relies on | fails the run |
| `warning` | A departure from the *documented* shape (unseen category, different row count, age outside 18–65, price of zero). Legitimate if the data was refreshed | reported, does not fail |
| `info` | A known, intentional property that is easy to misread as a bug | reported only |

### Measured return rate

The rate is computed, never assumed. On the shipped data:

| Definition | Value |
|---|---|
| **Unit** rate — `returned quantity / purchased quantity` (primary) | 5,786 / 28,931 = **19.9993%** |
| Line rate — returned lines / order lines | 5,048 / 20,000 = 25.24% |
| Order rate — orders with any return / orders | 3,233 / 6,726 = 48.07% |

All three are reported because they are different numbers and are easy to mistake for one
another. The brief's "approximately 20%" refers to the **unit** rate only.

### Report output

`outputs/data_quality_report.json` is written for the dashboard to display later. Timestamps are
ISO strings and all numpy scalars are converted, so the file is portable JSON:

```json
{
  "generated_at": "...", "ok": true,
  "summary":  {"total": 101, "passed": 101, "errors": 0, "warnings": 0},
  "dataset":  {"unit_return_rate": 0.199993, "purchased_units": 28931, "...": "..."},
  "tables": {
    "customers": {
      "ok": true,
      "metrics": {"total_customers": 1000, "unique_customers": 1000, "duplicate_customers": 0,
                  "missing_values": {}, "registration_date_min": "2023-01-02T00:00:00"},
      "checks":  [{"table": "customers", "check": "Customer ID is unique",
                   "passed": true, "severity": "error", "status": "PASS", "detail": "..."}]
    }
  }
}
```

The CLI can also validate a single file, and `--strict` promotes warnings to a non-zero exit:

```bash
python scripts/validate_data.py --table customers --table returns
python scripts/validate_data.py --strict --quiet
```

---

## Customer feature store

[`src/features/`](src/features/) turns the four CSVs into `customer_features`: **exactly one row
per Customer ID, 148 features**, computed strictly as of a prediction date.

```python
from src.features import build_customer_features

result = build_customer_features(as_of_date="2025-12-31")
result.features        # 1,000 rows x 149 columns (customer_id + 148 features)
result.feature_count   # 148
result.issues          # calculation caveats, reported rather than hidden
```

```bash
python scripts/build_features.py                     # -> outputs/customer_features.csv
python scripts/build_features.py --as-of 2024-06-30  # a historical as-of date
python scripts/build_features.py --list-features     # just the names
```

### The as-of date is the leakage guard

[`src/features/context.py`](src/features/context.py) is the **single choke point**: it clips the
data once, and every feature module reads only from the resulting `FeatureContext`. No module
receives the raw frames, so none can reach past the prediction date.

Two clipping rules, and the second is the one that is easy to get wrong:

| Rule | Why |
|---|---|
| `purchase_date <= as_of` | A transaction that has not happened cannot inform today's prediction. |
| `return_date <= as_of` | **The subtle half.** A return is a separate, later event from its purchase. Filtering only on purchase date would let a return that has not happened yet count against an order that has. On this dataset 104 returns are dated *after* the last purchase, the latest a month later. |

The consequence is deliberate: return features **understate** the eventual return rate, because
at any as-of date some returns are still in flight. At 2025-12-31 the features see 5,666 returned
units, not the 5,786 that eventually settle. A model trained on settled return rates would be
reading the future.

**This is proved, not asserted.** [`test_features.py`](tests/test_features.py) builds features at
date T from the full dataset and again from a dataset physically truncated at T, then asserts the
two are byte-identical — on the synthetic fixture *and* on the real 20,000-row data. If anything
leaked, those builds would differ.

### One row per Customer ID, always

Customers with no orders at the as-of date are **kept and flagged**, never dropped:
`has_purchase_history` and `registered_at_as_of` distinguish them. Dropping them would hide a
real cohort and make row counts incomparable between as-of dates. Verified across five dates:

| as-of | rows | with history | orders | net revenue | returned units |
|---|---|---|---|---|---|
| 2023-12-31 | 1,000 | 306 | 1,092 | 344,780 | 821 |
| 2024-06-30 | 1,000 | 457 | 2,070 | 632,305 | 1,643 |
| 2024-12-31 | 1,000 | 665 | 3,538 | 1,119,718 | 2,861 |
| 2025-06-30 | 1,000 | 842 | 4,989 | 1,567,312 | 4,128 |
| 2025-12-31 | 1,000 | 1,000 | 6,726 | 2,132,427 | 5,666 |

That stability is what lets Section 3 stack several historical as-of dates into a time-based
split.

### Feature groups (148 total)

| Group | n | Contents |
|---|---|---|
| identity | 9 | age, gender, city, country, channel, age band, cohort flags |
| rfm | 24 | recency, orders/units/revenue, AOV, and 30/90/180/365-day windows |
| gaps | 13 | mean/median/max gap, current gap, **`purchase_gap_ratio`**, regularity |
| trends | 16 | revenue/frequency/quantity/AOV growth, `spend_decline_pct`, recent-vs-historical |
| lifecycle | 17 | tenure, first/last purchase, active & inactive months, early-vs-recent |
| affinity | 16 | preferred category/subcategory/brand, breadth, diversity, `days_since_preferred_category_purchase` |
| discount | 12 | average/max discount, coupon rate, full-price rate, `discount_dependency_score` |
| returns | 11 | returned units/orders, `return_rate`, `recent_return_rate`, trend |
| seasonality | 12 | preferred month/quarter, concentration, **`seasonal_customer_score`**, in-season status |
| value | 8 | `annualized_revenue`, `customer_value_segment`, value percentile |
| segments | 8 | six behavioural flags, `behavioral_segment`, `lifecycle_stage`, `segment_reason` |

Windows are half-open — `(as_of - days, as_of]` — so `orders_30d` covers the 30 days *ending* at
the as-of date and an order exactly 30 days back falls outside it.

### Not classifying seasonal customers as churned

A flat "no purchase in 180 days" rule punishes a twice-a-year buyer and lets a weekly buyer go
three weeks unnoticed. Two features fix that:

**`purchase_gap_ratio`** = current gap ÷ the customer's *own* median gap. Personalised, so a
90-day silence is unremarkable for a quarterly buyer and alarming for a weekly one.

**`seasonal_customer_score`** uses **circular statistics**, not a month histogram. December and
January are adjacent in the year but maximally distant as bucket labels, so a customer who
reliably shops the Christmas-and-January-sales window would score as *unseasonal* under a
histogram — exactly backwards for a fashion retailer. Purchase dates are mapped to angles on a
circle instead, and the resultant length measures clustering with December and January correctly
adjacent. It is bias-corrected as `(n·R² − 1)/(n − 1)`, because raw `R` is 1.0 for a single
purchase; scores are withheld entirely below 3 orders across 2+ calendar years, since with one or
two orders any customer looks perfectly seasonal by accident.

Together they produce `seasonally_explained_inactivity` — "this customer is quiet, and quiet is
exactly what we should expect right now" — which `is_dormant_buyer` then respects. It has two
deliberate limits, both of which matter:

- A seasonal customer silent **during** their own season is genuinely worrying, so being in
  season removes the shield.
- A seasonal customer silent for **over a year** has already skipped a whole season, so
  `missed_full_season` removes the shield too. Without this, a customer two years absent would be
  excused indefinitely for being "out of season" — which is how a well-meant seasonality rule
  becomes a blind spot. On the real data this correctly moves 20 of 73 seasonal buyers into
  Dormant, while 53 mid-cycle ones keep their protection.

### Calculation notes on the real data

Reported by the builder rather than hidden, because each is a real limitation:

| Note | Count |
|---|---|
| Exactly one order, so no gap is measurable — `expected_purchase_interval_days` falls back to 90 days and `has_measurable_cadence` is `False` | 277 |
| No revenue in the previous 90-day window, so `*_growth` is **null rather than infinite** (tree models handle NaN natively; a fabricated number would invent a trend) | 634 |
| Below the evidence bar for a seasonality score | 487 |
| Tenure under 30 days, so `annualized_revenue` used a floored denominator and is an upper bound | 30 |

`outputs/customer_features.csv` is an **analytical artefact**, not a source dataset — the CSVs
under `data/` remain the source of truth. The in-memory table keeps full float precision for
modelling; rounding is applied only when writing the CSV.

---

## Churn model

```bash
python scripts/train_model.py          # -> models/churn_model.joblib + outputs/model_metrics.json
python scripts/predict.py              # -> outputs/customer_churn_predictions.csv
```

### The churn label is forward-looking

The brief defines churn as "no purchase within 180 days after the last purchase". Read literally
that is a *retrospective* rule applied to whoever looks quiet today — and it is the rule the brief
elsewhere warns against, because it cannot tell a churned customer from a seasonal one.

[`src/models/labels.py`](src/models/labels.py) turns it around: at a prediction date `as_of`, a
customer is churned if they made **no purchase in `(as_of, as_of + 180]`**. Same meaning, but:

- Features come from `<= as_of`, the label from `> as_of`. Leakage-free by construction.
- **It cannot mislabel a seasonal customer**, because it never infers churn from inactivity — it
  observes what the customer actually did next. The mislabelling problem is a property of
  retrospective rules; a forward-looking label removes it rather than patching it.

**Censoring.** A label needs its window to have finished. With data ending 2025-12-31 and a 180-day
horizon, the last labelable date is **2025-07-04**; later dates are right-censored and get `NA`,
never `0`. Treating an unfinished window as "did not churn" would teach the model that recent
customers never leave — the most damaging error available here, and the easiest to make by accident.

**Residual risk, measured not assumed.** A loyal annual buyer whose next purchase lands at day 200
is still called churned by a uniform window. `--label-mode adaptive` scales the horizon to each
customer's own cadence (`2 × expected interval`, floored at 90, capped at 365), and every run
reports the disagreement. At 2024-12-31: 92.8% agreement, **20 customers rescued** by the adaptive
horizon (the loyal-but-slow ones a uniform window mislabels) and 28 caught sooner.

### Time-based validation: a three-stage split

```
selection : inner_train ──[embargo]──▶ inner_validation     pick the model family
refit     : all fit dates                                   use the recent data too
calibrate : held-out calibration date
                          ──[embargo]──▶ test               report once
```

A row at `as_of = T` carries a label from `(T, T+180]`, so its *label* describes a period a later
row uses for its *features*. An embargo — a horizon-wide gap — closes that. The question is which
boundaries need one, and **"all of them" is the wrong answer**:

| Boundary | Embargo | Why |
|---|---|---|
| Before **test** | Mandatory | Everything fitted (model *and* calibrator) must resolve before the test date. This is what makes the reported number trustworthy. |
| Before **selection validation** | Mandatory, *inside* the training data | Buys nothing for the test estimate; it exists so model *selection* is unbiased. |

Applying it at both *outer* boundaries — the obvious first design — was a mistake I measured and
reverted. It consumed a year of the 24 usable months and confined training to the brand's growth
phase (churn 18–33% against 47% in test). The resulting model was **worse than a single feature**:
test ROC-AUC 0.68 versus 0.73 for `orders_365d` alone. Skipping the *inner* embargo instead is
equally wrong in the other direction: LightGBM then scored 0.88 validation PR-AUC against 0.66 on
test, and would have been picked over the linear model that generalised better.

### Results

Selected **LightGBM** (highest PR-AUC on the embargoed inner split), calibrated with isotonic
regression chosen **out-of-fold** on the held-out 2024-12-31 period.

| Test metric (2025-06-30, n=842, base rate 47.1%) | Value |
|---|---|
| ROC-AUC | 0.7056 |
| PR-AUC | 0.6380 |
| Precision / Recall / F1 | 0.640 / 0.442 / 0.442 |
| Brier / ECE | 0.2105 / 0.0834 |
| Calibration bias | +0.001 (mean predicted 0.472 vs observed 0.471) |
| Lift @ top decile | 1.77× random |
| **ROC-AUC among High Value customers** | **0.9135** |

Two honest caveats, both reported by the training script rather than buried:

1. **Ranking only matches the best single-feature heuristic** (`gap_vs_max_gap_ratio`, PR-AUC
   0.652 vs the model's 0.638). Every run scores eight one-line heuristics as a sanity floor and
   says so when the model fails to clear it. What the model adds is a *calibrated probability* —
   a raw feature cannot give one, and revenue at risk needs it — plus far stronger discrimination
   among high-value customers (ROC-AUC 0.91), which is where the money is.
2. **The base rate drifts** 24.9% → 31.3% → 45.0% → 47.1% across selection/fit/calibration/test.
   The brand was acquiring fast early on, so few customers had lapsed yet. Calibrating on a recent
   held-out period is what keeps the predicted *level* usable despite the shift.

Isotonic calibration collapses 1,000 customers onto 28 distinct probabilities. That is the price of
the best out-of-fold Brier score, and it matters less than it looks: revenue at risk multiplies by
customer value, which restores a fine-grained ranking. A near-tie on Brier is broken in favour of
the smooth sigmoid for exactly this reason.

### Predictions

`outputs/customer_churn_predictions.csv` carries the columns the brief asks for — Customer ID,
Prediction date, Churn probability, Risk level, Customer value, Lifetime revenue, Recent revenue,
Recency, Frequency, Revenue at risk — plus diagnostics the later sections need.

Risk bands are configurable (`RISK_THRESHOLD_*`), with the lower edge inclusive. **At the default
thresholds**: Low < 0.30 ≤ Medium < 0.60 ≤ High < 0.80 ≤ Critical, so `p = 0.60` is High. Change
those settings and the bands move — but the stored `Risk level` column is written at prediction
time, so re-run `python scripts/predict.py` (and `retention.py`, which segments on the same edges)
for the change to reach the artefacts and the dashboard.

**Revenue at risk** = `churn probability × lifetime revenue × horizon / max(tenure, horizon)`. The
`max(tenure, horizon)` denominator refuses to extrapolate past observed history. Annualising
instead — the obvious approach — ranked a customer with one €780 order and 24 days of history as
the most valuable account in the book, because 24 days scaled to a year implies €9,500 annually.

At 2025-12-31: 1,000 customers scored, **€162,302** total revenue at risk, €62,175 of it in the
High and Critical bands. Mean churn probability by segment orders sensibly — Frequent Buyer 0.041,
Dormant Buyer 0.596.

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

The requirements this platform is built against are recorded in [`PROMPTS.md`](PROMPTS.md),
Sections 0–7, with what has been delivered so far marked against each.

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
│   └── validation.py     reusable per-file validation layer (read-only)
├── features/
│   ├── context.py        THE as-of clipping choke point (leakage guard)
│   ├── params.py         every configurable threshold
│   ├── builder.py        orchestrator: one row per Customer ID
│   ├── rfm.py  gaps.py  trends.py  lifecycle.py
│   ├── affinity.py  discount.py  returns.py  seasonality.py
│   └── value.py  segments.py
├── models/
│   ├── labels.py         forward-looking churn label + censoring
│   ├── splits.py         three-stage time split with embargoes
│   ├── dataset.py        (customer, as-of date) panel builder
│   ├── preprocessing.py  column selection: drops period markers
│   ├── candidates.py     LogReg, RandomForest, LightGBM, XGBoost
│   ├── evaluate.py       ranking + calibration + business metrics
│   ├── train.py          select -> refit -> calibrate -> test once
│   ├── predict.py        scoring and the predictions CSV
│   ├── risk.py           risk bands and revenue at risk
│   └── registry.py       model persistence with a feature contract
├── explainability/
│   ├── shap_values.py    TreeSHAP, unwrapping + one-hot folding
│   ├── narratives.py     phrase grammar + driver concept groups
│   ├── global_explanations.py  importance, direction, dependence
│   ├── customer_explanations.py  per-customer top-k drivers
│   └── pipeline.py       CSVs -> features -> model -> SHAP -> sentences
├── segmentation/         empty: this work lives in retention/segments.py
├── retention/
│   ├── params.py         assumptions, kept separate from policy inputs
│   ├── value.py          expected future revenue (frequency x value)
│   ├── segments.py       the twelve business segments
│   ├── scoring.py        revenue at risk, propensity, opportunity score
│   ├── recommendations.py  action/channel/category/SKU/offer/reason
│   └── pipeline.py       both output CSVs + the assumption manifest
└── utils/
    ├── paths.py          the ONLY place path logic lives
    └── logging_config.py idempotent logging setup

app/                      Streamlit dashboard
├── dashboard.py          entry point: st.navigation over the eight pages
├── data_access.py        cached readers + the joined customer master frame
├── theme.py              palette, Plotly template, card styling
├── formatting.py         one shape for every currency / percentage / count
├── components/           kpi.py  filters.py  tables.py  layout.py
├── charts/               distributions.py  breakdowns.py  model.py
└── views/                one module per page (NOT `pages/` — see below)
models/                   serialised models (git-ignored)
outputs/                  generated reports (git-ignored)
logs/                     rotating log files (git-ignored)
scripts/
├── validate_data.py      run the validation layer, write the JSON report
├── inspect_data.py       full data profile + validation report
├── build_features.py     build outputs/customer_features.csv
├── train_model.py        train, select, calibrate, save the model
├── predict.py            score customers -> predictions CSV
├── explain.py            SHAP -> explanations CSV + global artefacts
└── retention.py          segments, scores, recommendations
PROMPTS.md                the build prompts, Sections 0-7, with delivery status
tests/
├── test_architecture.py  no DB/ETL/API; the CSVs are provably unmodified
├── test_config.py        configuration and portable path resolution
├── test_csv_loader.py    dtypes, encoding, the zero-padding trap
├── test_validation.py    the validators CATCH synthetic bad data
├── test_data_integrity.py the real CSVs satisfy every invariant
├── test_features.py      feature arithmetic + the leakage proofs
├── test_labels.py        label semantics, censoring, leakage
├── test_models.py        split embargoes, risk bands, persistence
├── test_explainability.py  SHAP folding, sentence correctness, grouping
├── test_retention.py     projection caps, ROI guardrails, recommendations
├── test_recommendation_personas.py  the brief's nine customer types
├── test_end_to_end.py    the real chain: scoring, live SHAP, campaign economics
└── test_dashboard.py     every page rendered; artefact totals reconcile
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
pytest                                      # whole suite: 456 tests
pytest tests/test_architecture.py           # the CSV-first constraint, as executable checks
pytest tests/test_validation.py             # the validation logic only
pytest tests/test_features.py               # feature arithmetic and the leakage proofs
pytest tests/test_recommendation_personas.py  # the nine customer types
```

**456 tests, and 48 of them skip on a fresh clone.** `outputs/` and `models/` are git-ignored, so
the tests that need generated artefacts skip with the command that produces them rather than
failing. Nothing is silently not run: `pytest -q` reports the skips and why. Once steps 2-4 of
[the workflow](#the-workflow) have been run, those 48 have their artefacts and the full 456 execute
— a clean run reports `456 passed` with no skips, and takes about three minutes.

| Suite | What it pins |
|---|---|
| [`test_architecture.py`](tests/test_architecture.py) | No database, no ETL, no ingestion service, no API. **Hashes the source CSVs, runs the pipeline over them, and hashes them again** — the check that earns the read-only claim rather than asserting it. |
| [`test_config.py`](tests/test_config.py) | Configuration resolves portably: relative paths anchor to the project root, absolute ones are honoured as an explicit escape hatch. |
| [`test_csv_loader.py`](tests/test_csv_loader.py) | Dtypes, UTF-8 encoding, and the `Order ID` zero-padding that pandas destroys by default. |
| [`test_validation.py`](tests/test_validation.py) | The validators **catch** bad data: a synthetic 4-table dataset with exactly one thing broken, asserting the specific named check fails. Without this, a validator that always returned `PASS` would look healthy. |
| [`test_data_integrity.py`](tests/test_data_integrity.py) | The real CSVs satisfy every invariant. Complementary to the above; neither is sufficient alone. |
| [`test_features.py`](tests/test_features.py) | Feature arithmetic against hand-computed values, and **no feature sees the future** — a build at date T compared against a build from data physically truncated at T. |
| [`test_labels.py`](tests/test_labels.py) | Label semantics, censoring, and that an unfinished outcome window is `NA` rather than a silent `0`. |
| [`test_models.py`](tests/test_models.py) | Split embargoes, risk banding, and the persistence feature contract. |
| [`test_explainability.py`](tests/test_explainability.py) | SHAP folding, concept grouping, and that **every sentence quotes the value its own row reports**. |
| [`test_retention.py`](tests/test_retention.py) | Projection caps, ROI guardrails, and the assumption/policy separation. |
| [`test_recommendation_personas.py`](tests/test_recommendation_personas.py) | The nine customer types the brief names, each asserted on the business rule rather than an exact action string. |
| [`test_end_to_end.py`](tests/test_end_to_end.py) | The real chain over the shipped data: probability range, scoring reproducibility, live TreeSHAP against the calibrated model, and the campaign economics reconciling with the rows beneath them. |
| [`test_dashboard.py`](tests/test_dashboard.py) | Every page rendered in a real Streamlit runtime, the five-artefact join staying 1:1, and headline figures equal to the artefacts they came from. |

Three of these earn their place by catching what unit tests structurally cannot. The architecture
suite proves the CSVs are unmodified rather than assuming it. `test_end_to_end.py` runs TreeSHAP
against the actual calibrated pipeline — the unit tests explain a hand-built result, so only this
one proves the calibration wrapper is unwrappable at all. And `test_dashboard.py` renders each
page, because a page can import cleanly and still fail at render time on a duplicate element key.

### The pipeline is reproducible, and this was measured

The whole workflow was re-run from scratch into empty directories — features, training, scoring,
explanations, retention — and compared against the shipped artefacts:

- every test-period metric matched **to the last decimal** (ROC-AUC 0.705646, PR-AUC 0.637983,
  Brier 0.210472, ECE 0.083379);
- all four output CSVs were **byte-identical**, including the 5,000-row explanations file;
- the four source CSVs still hash to their original values, with their original timestamps.

`RANDOM_SEED` is honoured throughout, so a rerun is a check rather than a new opinion.

---

## Dependency notes

Pins in [`requirements.txt`](requirements.txt) that are deliberate rather than incidental:

- **`pandas>=2.2,<3.0`** — pandas 3.0 makes the string dtype and copy-on-write the default;
  those semantics are not yet well exercised against the scikit-learn / SHAP versions used here.
- **`shap>=0.51`** — 0.51.0 is the first SHAP release shipping a CPython 3.14 Windows wheel.
- `xgboost` and `lightgbm` publish version-independent `py3-none-win_amd64` wheels, so they are
  not tied to the interpreter's ABI tag.

---

## Explainability (SHAP)

```bash
python scripts/explain.py                          # -> explanations CSV + global artefacts
python scripts/explain.py --customer CUST0234      # one customer's driver block
python scripts/explain.py --top-k 3 --risk-level High --risk-level Critical
```

### What "do not hardcode explanations" actually requires

Natural language cannot appear from nowhere, so it is worth being precise. What is forbidden is an
explanation whose *content* is fixed — a generic "the model predicts churn", or a canned top-five
list that reads the same for everyone. What is required is that **which** drivers appear, **in what
order**, and **every number in the sentence** all come from that customer's own SHAP contributions
and feature values.

[`narratives.py`](src/explainability/narratives.py) gives each feature a *phrase grammar* — a
template, a formatter, and optional companion/context features — composed at runtime:

```
1. [^] Typically 41 days between orders — raising churn risk
2. [^] Ordered in 4 months of their observable months — raising churn risk
3. [v] Spreads spending across 4 categories (diversity 0.80 of 1) — lowering churn risk
4. [^] EUR 2,751.85 of their spend came on discounted orders, higher than 83% of customers — raising churn risk
5. [^] Has bought 24 items in total, higher than 67% of customers — raising churn risk
```

A test asserts that **every sentence quotes the value its own row reports**, so no sentence can have
been written independently of the customer. Across 5,000 driver rows there are 1,907 distinct
sentences. A feature with no vocabulary entry still gets a real composed sentence rather than a
placeholder, so the driver list is never silently truncated to whatever happens to have nice wording.

### Three problems solved before the numbers are usable

**One-hot fragmentation.** The preprocessor expands `preferred_category` into one column per
category, so raw SHAP would report five weak drivers instead of one real one. Contributions are
summed back onto the source feature — valid because SHAP values are additive. Longest-prefix
matching stops `category` from claiming `category_diversity`'s columns.

**The calibration layer.** SHAP explains the model *before* probability calibration, because that is
where the trees are. Calibration is monotone, so driver ranking and direction carry over exactly;
what does not carry over is the arithmetic — contributions sum to the uncalibrated margin, not to
the reported probability. Every artefact says so rather than implying a false additivity.

**Near-duplicate features.** `median_purchase_gap` and `expected_purchase_interval_days` are the
same number by construction, and an ungrouped top-five spent two slots on "typically 49 days between
orders" and "typically orders every 49 days". Features are mapped onto **12 concept groups** and only
each group's strongest contributor competes, so five slots buy five distinct reasons. An unmapped
feature forms its own group, so nothing is silently merged.

### Direction of impact is measured, not assumed

Per feature, the Spearman correlation between its value and its own contribution. Beyond ±0.15 it is
reported as directional; inside that band the model has learned a non-monotone relationship and the
column says `mixed / non-monotone` rather than inventing a direction. **11 of 126 features** are
non-monotone.

### Global artefacts

Under `outputs/explainability/`: `global_feature_importance.csv`, `shap_summary.csv` (the beeswarm as
data — importance, mean signed contribution, direction and spread), `shap_dependence.csv` (binned
value-versus-contribution curves for the top 12), `top_churn_drivers.md`, and
`explainability_metadata.json`. Emitted as **data rather than PNGs** so the Streamlit dashboard can
render them interactively with Plotly and filter them, which a static image cannot do — and so
matplotlib is not a dependency for one chart.

| Rank | Feature | Mean \|SHAP\| | Share | Direction |
|---|---|---|---|---|
| 1 | `category_diversity` | 0.1724 | 11.5% | higher values lower churn risk |
| 2 | `subcategory_count` | 0.0692 | 4.6% | higher values lower churn risk |
| 3 | `active_months` | 0.0687 | 4.6% | higher values raise churn risk |
| 4 | `total_lines` | 0.0595 | 4.0% | higher values lower churn risk |
| 5 | `revenue_from_discounted_orders` | 0.0534 | 3.5% | higher values raise churn risk |

### Per-customer output

`outputs/customer_churn_explanations.csv` — 5,000 rows (1,000 customers × 5 drivers), long format,
with the brief's columns first: Customer ID, Churn probability, Risk level, Driver rank, Feature,
Feature value, Contribution, Direction, Human-readable explanation. Long format on purpose:
"every customer whose top driver is a widening purchase gap" is a one-line query on this shape.

Drivers are ranked by **absolute** contribution, so the strongest *protective* factor is not hidden —
a retention manager needs to know that a weekly ordering habit is the one thing still holding a
customer. The `Direction` column carries the sign (2,827 increase risk, 2,173 reduce it).

**Honest limitation.** `category_diversity` is the top driver for 554 of 1,000 customers and 11.5% of
global importance — consistent with the permutation importance from Section 3, but not an intuitive
churn signal. Combined with the model's ranking only matching a single-feature heuristic, this
suggests part of its signal rests on a proxy rather than on the behavioural story. Worth revisiting
alongside the feature-count reduction noted below.

---

## Retention decision layer

```bash
python scripts/retention.py                    # -> scores + recommendations + assumption manifest
python scripts/retention.py --customer CUST0234 # one customer's full retention record
python scripts/retention.py --propensity 0.15   # test a different assumption
python scripts/retention.py --min-roi 0.5       # demand a 50% margin before contacting
```

### Retention propensity is an assumption, and stays labelled as one

Propensity is *the probability that contacting a customer changes their behaviour*. Measuring it
needs a campaign log and an untreated control group. **This dataset has neither**, so it cannot be
learned here — there is nothing in four CSVs of transactions that identifies a causal effect.

So it is stated openly and propagated visibly:

- the base rate is one configurable number (25%), not a fitted-looking artefact;
- the behavioural multipliers are *directional* judgements with a stated rationale, deliberately
  coarse (×0.6, ×1.4) so nobody mistakes them for measurements;
- `outputs/retention_assumptions.json` ships alongside the CSVs, and the columns are literally named
  `Retention propensity (ASSUMED)` and `Propensity basis (ASSUMED)`;
- `Propensity basis` names every multiplier that fired for that customer — *"base 25% (assumption);
  discount-responsive ×1.4; still active ×1.2"* — so the number can be taken apart, not trusted.

**`revenue_at_risk` is deliberately kept free of the assumption.** It depends only on the model's
probability and the observed-revenue projection, so a business that rejects the propensity figures
can still use the exposure number. Everything downstream of propensity is flagged. `params.py`
separates `assumptions()` from `policy_inputs()` structurally — a reader must be able to see which
numbers came from the business and which were invented in the absence of data.

### Expected future revenue

All five inputs the brief names, combined as **frequency × value** rather than one ratio:

```
expected orders  = 0.6 × orders in last 365d  +  0.4 × lifetime order rate
expected value   = 0.6 × recent AOV           +  0.4 × lifetime AOV
expected revenue = expected orders × expected value, pro-rated to the horizon
```

Recent behaviour leads because the next order resembles the last few, but does not dominate, so one
quiet quarter cannot erase three years. **Tenure governs how far the projection may reach** — the
lesson from Section 3, where annualising a customer with one €780 order and 24 days of history made
them the most valuable account in the book. Two guards: the rate denominator is floored at 180 days,
and every projection is capped at 2× observed lifetime revenue. Historical annual revenue is carried
as a **cross-check** with the ratio between the two, so a reviewer can see when the models disagree.

At 2025-12-31: **€585,966** expected future revenue over 180 days, **€125,129** at risk.

### The twelve segments, multi-label by design

| Segment | Primary for | Also flagged for |
|---|---|---|
| Discount-Driven At Risk | 241 | 274 |
| Frequent but Declining | 160 | 176 |
| Dormant Customers | 143 | 378 |
| Loyal Customers | 115 | 226 |
| High-Return Customers | 108 | 195 |
| Champions | 87 | 127 |
| One-Time Buyers | 41 | 199 |
| New Customers | 33 | 78 |
| High-Value At Risk | 17 | 17 |
| Lost Customers | 16 | 16 |
| Seasonal Customers | 5 | 73 |
| Low-Value At Risk | 3 | 21 |

Counts are at the **default** `RISK_THRESHOLD_*` values, since the at-risk segments band on the same
configured edges as the model's own risk levels. Raising `RISK_THRESHOLD_MEDIUM` to 0.50, for
instance, moves 207 customers out of Discount-Driven At Risk.

The gap between the columns *is* the point — the brief asks that customers carry several analytical
dimensions rather than one rigid label. A High-Return Customer who is also High-Value At Risk needs
both facts: one says act now, the other says be careful what you offer.

**Discount-Driven At Risk uses a cohort percentile, not an absolute score.** The first version used
absolute flags and labelled **526 of 1,000** customers — with 50.6% of this dataset's order lines
discounted, that is not a segment, it is a description of the brand. Ranking within the cohort keeps
it discriminating whatever the brand's promotional intensity.

### Recommendations, driven by behaviour

Nine distinct actions in use, **77 distinct SKUs**, 15 distinct offers, **413 distinct reasons**
across 1,000 customers. Every field is derived:

- **Offer** — the discount depth *they* have responded to, capped by policy: 10/15/20/25% actually
  in use. A house-standard "15% off" for everybody would be the hardcoding the brief forbids.
- **SKU** — the best-selling product in the recommended category, filtered to their target gender
  and price band, **excluding everything they already own**.
- **Category** — their preferred one, or for cross-sell the category their peers most often pair
  with it, measured from the transactions rather than asserted from fashion intuition.
- **Channel** — inferred from the channel that acquired them, with age as a tiebreak.
- **Reason** — composed at runtime, citing their numbers.

**The brief's two guardrails are structural, not hoped for.** A full-price buyer can never reach a
discount rule because `Organic Engagement` is checked first and claims them. And a negative expected
ROI overrides the chosen action — *after* the fact, so the reason says what was proposed and why it
was dropped.

Cheap levers come before expensive ones: a predictable buyer barely into their interval gets a free
reminder, not margin.

### Campaign economics

779 targeted, 221 suppressed. Cost **€6,935**, expected return **€37,265**, blended ROI **+437%**.

The suppression list is broken down by *reason*, because "already engaged" and "unrecoverable" call
for completely different follow-up: 144 already highly engaged, 38 uneconomic, 23 seasonal and out of
season, 16 unrecoverable.

**A modelling error worth naming.** I first costed discount incentives against the customer's *whole*
projected spend. That made 315 of 1,000 discounts look uneconomic and suppressed **six of the top ten
opportunities**, because cost scaled with `EFR × depth` while the benefit was only
`EFR × churn × propensity` — roughly three times smaller. A win-back coupon is redeemed on the order
the intervention produces, so its expected cost is `depth × expected retained revenue`. Known
simplification, stated rather than guessed at: this understates cost by whatever the discount
cannibalises from customers who would have bought anyway, which needs a control group to measure.

---

## Dashboard

```bash
streamlit run app/dashboard.py        # -> http://localhost:8501, Ctrl+C to stop
```

Run it from the project root, with the virtual environment active; see
[the workflow](#the-workflow) for the full procedure and the troubleshooting table.

Eight pages — Executive Overview, Churn Risk, Revenue at Risk, Retention Action Center,
Customer 360, Customer Segmentation, What-If Simulator, Model Performance — reading `data/*.csv`,
`outputs/*.csv` and `models/*`. No database, no new pipeline. If an artefact has not been generated
yet the page names the missing file and the command that produces it, rather than raising.

### One name for revenue at risk

Two figures exist and both are correct, so the dashboard commits to one:

| Column | Definition | Total |
|---|---|---|
| **Revenue at risk** (used everywhere) | churn probability × expected future revenue | **EUR 125,129** |
| Revenue at risk (model estimate) | churn × lifetime × horizon / max(tenure, horizon) | EUR 162,302 |

The brief defines revenue at risk as *churn probability × expected future revenue*, which is the
retention layer's figure, so that is the one every business page shows. The model's own estimate
appears only on Model Performance, under an explicit name. They are never added together.

Similarly, **expected retained revenue on the overview counts targeted customers only**
(EUR 37,265), so it reconciles with the campaign cost and ROI beside it. The all-customer figure
is EUR 39,040; pairing that with a targeted-only cost would report ROI as +463% instead of +437%.

### The What-If simulator re-runs the real decision layer

It builds a modified `RetentionParams` and calls `build_retention_layer` — the same code path as
`scripts/retention.py` — rather than rescaling the shipped columns. That matters because propensity
and contact cost both feed the ROI guardrail that decides *who gets contacted at all*: raise the
cost and customers drop out of the campaign, which no linear rescale reproduces. At its default
slider positions it reproduces the shipped plan exactly (779 targeted, EUR 6,935, +437%), and at
`--propensity 0.15` it reproduces that run too (745 targeted, +369%). Each new scenario costs about
five seconds and is cached; the propensity sweep is opt-in behind a button for that reason.

### Two Streamlit details that are load-bearing

**The page modules live in `app/views/`, not `app/pages/`.** `pages/` is a magic directory name:
a folder of that name beside the entry script makes Streamlit build *automatic* multi-page
navigation at start-up, before the script runs. That duplicates every page in the sidebar — and,
because the automatic scan builds a `Page` from the resolved path of the entry script, it fails
outright when the project sits on a Windows network share, since `st.Page` rejects UNC paths by
design so that resolving one cannot open an SMB connection and disclose the server's credentials.
Naming the folder `views` avoids the magic directory; navigation is exactly what `dashboard.py`
declares.

**The What-If page imports scikit-learn, LightGBM and SHAP lazily.** `dashboard.py` imports every
view module at start-up, so importing them at module level made all eight pages wait about twenty
seconds for libraries only one page uses. They are imported inside the functions that need them.

### Charts

One shared Plotly template, so the eight pages read as one system. Risk levels wear a reserved
status palette (good → critical); everything else takes a fixed categorical slot, assigned by
entity so filtering never repaints the survivors. The palette was checked with a validator rather
than by eye: all eight slots sit inside the lightness band and clear the chroma floor, and the
worst adjacent pair separates by ΔE 9.1 under protanopia against a target of 8. Three slots fall
below 3:1 contrast on the chart surface, which is why every chart also ships a sortable table and
a CSV export.

The SHAP artefacts were emitted as data rather than images precisely so Model Performance can
render and filter them interactively.

### Worth revisiting later

The model's *ranking* only matches a single-feature heuristic. Two avenues, neither of which
should be tuned against the test period: shorten the horizon (a 90-day label leaves far more
timeline for training, since each embargo gap costs one horizon), or reduce the feature space —
~130 features against a few hundred effective customers is the wrong ratio, and every candidate
overfits heavily on the inner split.
