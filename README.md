# Fashion Churn Platform

An AI-powered customer churn prediction and retention platform for a fashion / e-commerce brand.
It uses three years of customer, transaction, return and product history to answer five business
questions: **who is likely to churn**, **why**, **how much revenue is at risk**, **what to do
about it**, and **who to contact first**.

> **Status: data layer + validation + feature store + churn model + SHAP explainability complete.**
> This repository contains the verified data layer, a reusable per-file validation layer, the
> customer feature store (148 features as of a prediction date), the churn model with time-based
> validation and calibrated probabilities, and per-customer SHAP explanations in plain English.
> 312 tests pass. The retention engine and the dashboard are **not implemented yet**.
> See [Next implementation step](#next-implementation-step).

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

# 4. validate the CSV files
python scripts/validate_data.py

# 5. profile the data (schema, dtypes, distributions) as well
python scripts/inspect_data.py

# 6. build the customer feature table
python scripts/build_features.py

# 7. train the churn model and score every customer
python scripts/train_model.py
python scripts/predict.py

# 8. explain why each customer is at risk
python scripts/explain.py

# 9. run the tests
pytest
```

Both scripts exit non-zero if any error-severity check fails, so either works as a pipeline or
CI gate. `validate_data.py` writes `outputs/data_quality_report.json`; `inspect_data.py` writes
that plus [`outputs/data_profile.md`](outputs/).

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

Risk bands are configurable (`RISK_THRESHOLD_*`), with the lower edge inclusive so `p = 0.60` is
High: Low < 0.30 ≤ Medium < 0.60 ≤ High < 0.80 ≤ Critical.

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
scripts/
├── validate_data.py      run the validation layer, write the JSON report
├── inspect_data.py       full data profile + validation report
├── build_features.py     build outputs/customer_features.csv
├── train_model.py        train, select, calibrate, save the model
├── predict.py            score customers -> predictions CSV
└── explain.py            SHAP -> explanations CSV + global artefacts
PROMPTS.md                the build prompts, Sections 0-7, with delivery status
tests/
├── test_config.py        configuration and portable path resolution
├── test_csv_loader.py    dtypes, encoding, the zero-padding trap
├── test_validation.py    the validators CATCH synthetic bad data
├── test_data_integrity.py the real CSVs satisfy every invariant
├── test_features.py      feature arithmetic + the leakage proofs
├── test_labels.py        label semantics, censoring, leakage
├── test_models.py        split embargoes, risk bands, persistence
└── test_explainability.py  SHAP folding, sentence correctness, grouping
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
pytest                            # whole suite: 312 tests
pytest tests/test_validation.py   # the validation logic only
pytest tests/test_features.py     # feature arithmetic and the leakage proofs
pytest tests/test_labels.py       # label semantics and censoring
```

The suite covers five things:

1. **Configuration resolves portably** — no absolute path, no dependence on the working directory.
2. **The loader preserves what the data needs** — dtypes, UTF-8 encoding, the `Order ID` padding.
3. **The validators catch bad data** — [`test_validation.py`](tests/test_validation.py) builds a
   small synthetic 4-table dataset, breaks exactly one thing (a duplicate key, a zero quantity, a
   150% discount, an over-return, a return dated before its purchase, an order spanning two
   customers…) and asserts the *specific named check* fails. Without these, a validator that
   always returned `PASS` would look perfectly healthy.
4. **The real CSVs satisfy every invariant** — [`test_data_integrity.py`](tests/test_data_integrity.py).
5. **No feature sees the future** — [`test_features.py`](tests/test_features.py) proves it by
   comparing a build at date T against a build from data truncated at T, and pins the feature
   arithmetic (gaps, windows, growth, month counts) against hand-computed values.

Points 3 and 4 are complementary and neither is sufficient alone.

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

## Next implementation step

**Revenue at risk, segmentation and the retention recommendation engine (Section 5).**

1. Replace the interim revenue-at-risk estimate with a fuller expected-future-revenue model using
   order frequency, average order value, recent behaviour and tenure.
2. Build the 12 business segments (Champions, High-Value At Risk, Discount-Driven At Risk, …),
   allowing a customer to carry several analytical dimensions rather than one rigid label.
3. Retention opportunity score = churn probability × expected future revenue × retention propensity,
   with the propensity assumption configurable and **clearly labelled as an assumption**.
4. Personalised recommendations driven by the features already present: `is_full_price_buyer` for
   who *not* to discount, `preferred_category` and `days_since_preferred_category_purchase` for what
   to recommend, `seasonally_explained_inactivity` for who to leave alone.
5. Write `outputs/customer_retention_scores.csv` and `outputs/retention_recommendations.csv`.

### Worth revisiting later

The model's *ranking* only matches a single-feature heuristic. Two avenues, neither of which
should be tuned against the test period: shorten the horizon (a 90-day label leaves far more
timeline for training, since each embargo gap costs one horizon), or reduce the feature space —
~130 features against a few hundred effective customers is the wrong ratio, and every candidate
overfits heavily on the inner split.
