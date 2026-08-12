# Project Prompts

The build prompts for this platform, kept in the repository so the requirements travel with the
code and any reviewer can check what was actually asked for against what was delivered.

## Provenance and status

Sections 0–2 are reproduced **as they were given to Claude Code** in the build session; Sections
3–7 are transcribed from `Segmented Code Prompt.docx` at the workspace root. All seven sections have been built.
`code prompt.txt` and `code Prompt.docx` hold an earlier, unsegmented version of
the same brief — see [the conflict note](#note-the-earlier-brief-contradicts-the-csv-first-rule)
at the end, because it matters.

| Section | Scope | Status |
|---|---|---|
| [0](#section-0--project-setup-for-a-csv-based-churn-platform) | Project setup, CSV loader, config, logging | **Delivered** |
| [1](#section-1--csv-validation--relational-integrity) | Validation layer, `data_quality_report.json` | **Delivered** |
| [2](#section-2--feature-engineering-from-csv-files) | Customer feature store, `customer_features.csv` | **Delivered** |
| [3](#section-3--churn-prediction-model) | Churn model, time-based validation, predictions | **Delivered** |
| [4](#section-4--explainable-churn-prediction) | SHAP, per-customer churn drivers | **Delivered** |
| [5](#section-5--revenue-risk-segmentation--recommendations) | Revenue at risk, segments, retention actions | **Delivered** |
| [6](#section-6--streamlit-retention-dashboard) | Eight-page Streamlit dashboard | **Delivered** |
| [7](#section-7--testing--finalization) | Full test and quality review | **Delivered** |

**Formatting note.** Wording is unchanged. The only edits are mechanical: `•` bullets became
markdown `-` bullets, and indented blocks became fenced code blocks. Nothing has been summarised,
reordered or paraphrased.

## The standing architecture constraint

Repeated in every section and treated as authoritative throughout:

> The four CSV files are the source of truth. Read directly from CSV. **No** database, **no**
> ingestion pipeline, **no** ETL, **no** API. Do not modify the source CSV files.

---

# Section 0 — Project Setup for a CSV-Based Churn Platform

We are building an AI-Powered Customer Churn Prediction & Retention Platform for a
fashion/e-commerce brand.

IMPORTANT: The source data already exists as CSV files.

**CRITICAL DATA ARCHITECTURE REQUIREMENT**

Do NOT build a data-ingestion pipeline.
Do NOT create database ingestion.
Do NOT create PostgreSQL/DuckDB ingestion as a required application step.
Do NOT create ETL pipelines.

The application must work directly with the existing CSV files.
The CSV files are the source of truth.

Claude Code should inspect the repository and locate the existing CSV files.

Expected files are conceptually:

```
data/
├── Customer.csv
├── Transaction.csv
├── Return.csv
└── Product.csv
```

The actual filenames may differ. Inspect the repository and identify them based on their columns.

## Dataset

The CSV files contain:

- 1,000 customers
- 20,000 transaction records
- 500 SKUs
- Approximately 20% return rate
- 3 years of historical data

### Customer.csv

Columns:

- Customer ID
- Age
- Gender
- City
- Country
- Customer acquisition channel
- Registration date / First Purchase Date

### Transaction.csv

Columns:

- Customer ID
- Order ID
- SKU ID
- Purchase date
- Quantity
- Selling price
- Discount
- Coupon used
- Net order value
- Payment method

### Return.csv

Columns:

- Customer ID
- Order ID
- SKU ID
- Return date
- Return Quantity

### Product.csv

Columns:

- SKU ID
- Category
- Subcategory
- Brand
- Gender
- Price

## Technology

Use:

- Python 3.11+
- Pandas or Polars
- Scikit-learn
- XGBoost or LightGBM
- SHAP
- Streamlit
- Plotly
- pytest

Use Pandas unless there is a strong reason to use Polars.

## CSV Loading

Create one reusable CSV utility such as:

```
src/data/csv_loader.py
```

The application should load the CSV files directly using Pandas.

For example:

```python
customers = pd.read_csv(...)
transactions = pd.read_csv(...)
returns = pd.read_csv(...)
products = pd.read_csv(...)
```

Use configuration for file locations.

Example:

```
DATA_DIR=data
CUSTOMER_FILE=Customer.csv
TRANSACTION_FILE=Transaction.csv
RETURN_FILE=Return.csv
PRODUCT_FILE=Product.csv
```

Do not hardcode absolute paths.

## Important

The application must work when the project is moved to another machine, provided the same CSV
structure exists.

Use relative paths.

## Project Structure

Create:

```
fashion-churn-platform/
│
├── data/
│   ├── Customer.csv
│   ├── Transaction.csv
│   ├── Return.csv
│   └── Product.csv
│
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── explainability/
│   ├── segmentation/
│   ├── retention/
│   └── utils/
│
├── app/
│   ├── components/
│   ├── charts/
│   ├── pages/
│   └── dashboard.py
│
├── models/
├── outputs/
├── tests/
├── scripts/
├── requirements.txt
├── .env.example
└── README.md
```

Do not create unnecessary database infrastructure.

## First Task

1. Inspect the repository.
2. Find the four CSV files.
3. Inspect their columns and data types.
4. Confirm row counts.
5. Confirm unique customer count.
6. Confirm unique SKU count.
7. Identify the minimum and maximum transaction dates.
8. Identify data-quality issues.
9. Create the project structure.
10. Create the CSV loading utility.
11. Create configuration management.
12. Create logging.
13. Create a basic README.

Do NOT implement:

- Churn model
- Dashboard
- Database
- ETL pipeline
- API

yet.

At the end, report:

- CSV files found
- Row counts
- Columns
- Date range
- Data-quality observations
- Files created
- Next implementation step

---

# Section 1 — CSV Validation & Relational Integrity

Continue from the existing project.

The four CSV files are the source of truth.

IMPORTANT:

- Read directly from CSV.
- Do not create a database.
- Do not build an ingestion pipeline.
- Do not modify the source CSV files.

Implement a reusable CSV validation layer.

## Validate Customer.csv

Check:

- Customer ID exists
- Customer ID is unique
- Customer ID is not null
- Age is valid
- Gender values are valid
- Registration date is valid

Report:

- Total customers
- Unique customers
- Missing values
- Duplicate customers
- Date range

Expected: 1,000 unique customers

## Validate Product.csv

Check:

- SKU ID exists
- SKU ID is unique
- SKU ID is not null
- Price >= 0
- Category exists
- Subcategory exists

Expected: 500 unique SKUs

## Validate Transaction.csv

Check:

- Customer ID exists
- SKU ID exists
- Order ID exists
- Purchase date valid
- Quantity > 0
- Selling price >= 0
- Discount between 0 and 100%
- Net order value is mathematically correct

Verify:

```
Transaction.Customer ID
→ Customer.Customer ID
```

and:

```
Transaction.SKU ID
→ Product.SKU ID
```

## Validate Return.csv

Check:

- Customer ID exists
- Order ID exists
- SKU ID exists
- Return date valid
- Return quantity > 0

Every return must correspond to an actual transaction.
Return quantity must not exceed purchased quantity.
Return date must be after purchase date.

## Return Rate

Calculate:

```
Return Rate = Total Returned Quantity / Total Purchased Quantity
```

Report the actual return rate.
Do not assume it is exactly 20%.

## Important

Do not alter the CSV files.

Create a validation report in memory and optionally save a report to:

```
outputs/data_quality_report.json
```

The dashboard should later be able to display this information.

Create unit tests for the validation logic.

Do not implement machine learning yet.

---

# Section 2 — Feature Engineering from CSV Files

Continue from the existing project.

IMPORTANT:

All features must be generated directly from the four CSV files.
Do NOT create a database.
Do NOT create an ingestion pipeline.
Do NOT require a separate ETL process.

The application should load the CSVs and calculate features using Pandas.

## Objective

Create a reusable customer feature-engineering module.

Input:

```
Customer.csv
Transaction.csv
Return.csv
Product.csv
```

Output: `customer_features` with exactly one row per Customer ID.

The feature pipeline must accept:

```
as_of_date
```

This is essential for avoiding data leakage.
Only transactions and returns occurring on or before the as-of date may be used.

## Features

Calculate:

### RFM

- recency_days
- total_orders
- total_units
- lifetime_revenue
- average_order_value
- orders_30d
- orders_90d
- orders_180d
- orders_365d
- revenue_30d
- revenue_90d
- revenue_180d
- revenue_365d

### Purchase Gaps

- average_purchase_gap
- median_purchase_gap
- maximum_purchase_gap
- current_purchase_gap
- purchase_gap_ratio

### Trends

- revenue_growth
- order_frequency_growth
- quantity_growth
- AOV_growth
- recent_vs_historical_revenue
- recent_vs_historical_frequency

### Lifecycle

- customer_tenure_days
- active_months
- inactive_months
- first_purchase_date
- last_purchase_date

### Product Affinity

Join Product.csv and calculate:

- preferred_category
- preferred_subcategory
- preferred_brand
- category_count
- SKU_count
- brand_count
- category_diversity

### Discount Behavior

Calculate:

- average_discount
- discount_order_rate
- coupon_usage_rate
- full_price_order_rate
- discount_dependency_score

### Returns

Join Return.csv and calculate:

- returned_units
- returned_orders
- return_rate
- recent_return_rate
- return_frequency

### Seasonality

Calculate:

- preferred_purchase_month
- preferred_purchase_quarter
- seasonal_purchase_concentration
- seasonal_customer_score

### Customer Value

Calculate:

- lifetime_revenue
- annualized_revenue
- average_order_value
- customer_value_segment

### Customer Behavioral Segments

Identify:

- Frequent Buyers
- Occasional Buyers
- Seasonal Buyers
- New Buyers
- Declining Buyers
- Dormant Buyers

Do not classify seasonal customers as churned merely because of a long purchase gap.

## Output

Save the calculated feature dataset as:

```
outputs/customer_features.csv
```

This output is an analytical artifact, not a source dataset.
The application must still treat the original CSVs as the source of truth.

Create tests for important calculations.

At the end, report:

- Number of customers
- Number of generated features
- Feature names
- Any calculation issues

---

# Section 3 — Churn Prediction Model

> **Delivered.**

Continue from the existing project.

Build the churn prediction model using features calculated directly from the CSV files.

IMPORTANT:

The original CSV files remain the source of truth.
Do not introduce a database.
Do not introduce an ingestion pipeline.
Do not require manual preprocessing before running the model.

The application should be able to execute:

```
CSV files
→ feature engineering
→ churn model
→ predictions
```

## Churn Definition

Default:

```
Customer churn = No purchase within 180 days after the customer's last purchase.
```

Make the threshold configurable.

Also account for:

- Frequent customers
- Occasional customers
- Seasonal customers
- New customers

Avoid falsely labeling seasonal customers as churned.

## Time-Based Validation

Do not use a simple random train/test split as the primary methodology.

Use historical as-of dates.

Example:

```
Earlier period → Training
Later period   → Validation
Final period   → Test
```

Features must only use information available before each prediction date.

## Models

Evaluate:

- Logistic Regression
- Random Forest
- XGBoost or LightGBM

Metrics:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- Calibration

Select the most useful model.

## Risk Levels

Default:

```
Low       < 30%
Medium    30–60%
High      60–80%
Critical  > 80%
```

Make configurable.

## Output

Create:

```
outputs/customer_churn_predictions.csv
```

Columns:

- Customer ID
- Prediction date
- Churn probability
- Risk level
- Customer value
- Lifetime revenue
- Recent revenue
- Recency
- Frequency
- Revenue at risk

Save the trained model under:

```
models/
```

At the end report:

- Model selected
- Model metrics
- Number of customers scored
- Risk distribution
- Top predictive features

---

# Section 4 — Explainable Churn Prediction

> **Delivered.**

Continue from the existing project.

Implement explainable AI using SHAP.

The goal is to answer: Why is this customer likely to churn?

## Global Explanation

Generate:

- Global feature importance
- SHAP summary
- Top churn drivers
- Direction of impact

## Customer-Level Explanation

For each customer, calculate the top 3–5 churn drivers.

Example:

```
Customer: CUST0234
Churn Probability: 78%
Top Drivers:
1. Current purchase gap is 2.8× historical average
2. Purchase frequency declined significantly
3. Recent revenue declined
4. Customer has not purchased from preferred category recently
5. Customer activity is below historical levels
```

These explanations must be generated dynamically.
Do not hardcode explanations.

## Output

Create:

```
outputs/customer_churn_explanations.csv
```

Columns:

- Customer ID
- Churn probability
- Risk level
- Driver rank
- Feature
- Feature value
- Contribution
- Direction
- Human-readable explanation

The dashboard must be able to read this output.

Also create global model explanation outputs under:

```
outputs/explainability/
```

---

# Section 5 — Revenue Risk, Segmentation & Recommendations

> **Delivered.**

Continue from the existing project.

Build the business decision layer directly on top of the CSV-derived features and churn
predictions.

## Revenue at Risk

Calculate:

```
Revenue at Risk = Churn Probability × Expected Future Revenue
```

Estimate expected future revenue using:

- Historical order frequency
- Average order value
- Recent behavior
- Customer tenure
- Historical annual revenue

Do not use future transactions when generating historical predictions.

## Customer Segmentation

Create:

- Champions
- Loyal Customers
- High-Value At Risk
- Frequent but Declining
- Discount-Driven At Risk
- Seasonal Customers
- New Customers
- One-Time Buyers
- Dormant Customers
- Lost Customers
- High-Return Customers
- Low-Value At Risk

## Retention Opportunity Score

Calculate:

```
Retention Opportunity Score =
    Churn Probability
  × Expected Future Revenue
  × Retention Propensity
```

If there is no historical intervention data, use a configurable assumption for retention
propensity and clearly label it as an assumption.

## Personalized Recommendations

Generate:

- Recommended action
- Recommended channel
- Recommended category
- Recommended product/SKU
- Recommended offer
- Reason
- Priority

Possible actions:

- Win-back
- Personalized recommendation
- Discount
- Free shipping
- Loyalty reward
- Replenishment
- Cross-sell
- New collection
- Seasonal campaign
- Organic engagement
- Do not target

Use actual customer behavior to select the recommendation.

## Output

Create:

```
outputs/customer_retention_scores.csv
outputs/retention_recommendations.csv
```

Do not hardcode recommendations.

---

# Section 6 — Streamlit Retention Dashboard

> **Delivered.**

Continue from the existing project.

Build a polished Streamlit dashboard.

IMPORTANT:

The dashboard must work directly with the CSV files and generated analytical CSV outputs.
Do NOT introduce a database.

The application startup should be as simple as:

```
streamlit run app/dashboard.py
```

The application should load:

```
data/*.csv
outputs/*.csv
models/*
```

where required.

## Pages

Create:

- Executive Overview
- Churn Risk
- Revenue at Risk
- Retention Action Center
- Customer 360
- Customer Segmentation
- What-If Simulator
- Model Performance

### Executive Overview

Display:

- Total customers
- Active customers
- At-risk customers
- High-risk customers
- Critical-risk customers
- Predicted churn rate
- Revenue at risk
- Expected retained revenue
- Expected ROI

Add:

- Churn risk chart
- Revenue-at-risk chart
- Risk by segment
- Risk by acquisition channel

### Churn Risk

Charts:

- Churn probability distribution
- Risk-level distribution
- Risk by segment
- Risk by geography
- Risk by acquisition channel
- Risk by category

Filters:

- Country
- City
- Acquisition channel
- Customer segment
- Risk level
- Customer value
- Category

### Revenue at Risk

Show:

- Total revenue at risk
- Revenue at risk by segment
- Geography
- Acquisition channel
- Product category

Add exportable customer table.

### Retention Action Center

Show:

- Customer
- Risk
- Churn probability
- Revenue at risk
- Customer value
- Main churn driver
- Recommended action
- Channel
- Offer
- Expected ROI
- Priority

Sort by retention opportunity.

### Customer 360

When a customer is selected, display:

- Customer profile
- Purchase history summary
- Recency
- Frequency
- Revenue
- AOV
- Preferred category
- Preferred brand
- Return rate
- Churn probability
- Risk level
- Top churn drivers
- Recommended retention action
- Expected revenue retained
- Expected ROI

### What-If Simulator

Allow users to modify:

- Discount
- Campaign cost
- Intervention success rate
- Customer value threshold
- Risk threshold

Calculate:

- Customers targeted
- Campaign cost
- Expected customers retained
- Expected revenue retained
- ROI

### Model Performance

Show:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- Calibration
- Confusion matrix
- Feature importance
- SHAP summary

## UI

Create a modern business dashboard.

Use:

- Plotly
- KPI cards
- Sidebar navigation
- Filters
- Tables
- Download buttons
- Responsive layouts

Do not use placeholder metrics.
Every number must come from the CSV-based analytical pipeline.

---

# Section 7 — Testing & Finalization

> **Delivered.**

Perform a complete test and quality review of the CSV-based churn platform.

## Critical Architecture Check

Confirm:

- No database is required
- No ETL pipeline is required
- No ingestion service exists
- Original CSVs remain untouched
- CSV files are the source of truth
- Dashboard works directly with CSV-based outputs

## Test Data Layer

Verify:

- Customer count
- SKU count
- Transaction count
- Return rate
- Referential integrity
- Date validity
- Quantity validity
- Net order value
- Missing values

## Test Feature Engineering

Verify:

- Recency
- Frequency
- Monetary value
- Purchase gaps
- Trends
- Return rate
- Discount behavior
- Seasonality
- Customer value

## Test ML

Verify:

- Time-based splitting
- No data leakage
- Probability range
- Risk categories
- Model reproducibility
- Evaluation metrics

## Test Explainability

Verify:

- SHAP calculations
- Customer-level explanations
- Human-readable drivers
- No hardcoded explanations

## Test Financial Calculations

Verify:

- Revenue at risk
- Expected future revenue
- Expected retained revenue
- Campaign cost
- ROI
- Retention opportunity score

## Test Recommendations

Test:

- Frequent buyer
- Occasional buyer
- Seasonal buyer
- Declining customer
- High-value customer
- Discount-sensitive customer
- Premium customer
- High-return customer
- New customer

## Run

```
pytest
```

Fix all important failures.

## Final README

Document the simple workflow:

```
1. Place four CSV files in data/
2. Run feature generation
3. Train/update model
4. Generate predictions
5. Start Streamlit
```

Example:

```
python scripts/build_features.py
python scripts/train_model.py
python scripts/predict.py
streamlit run app/dashboard.py
```

Adapt the commands to the actual project.

## Final Requirement

The completed application must transform:

```
Customer.csv
Transaction.csv
Return.csv
Product.csv
        ↓
Feature Engineering
        ↓
Churn Prediction
        ↓
Explainable AI
        ↓
Revenue at Risk
        ↓
Customer Segmentation
        ↓
Retention Recommendation
        ↓
Retention ROI
        ↓
Interactive Dashboard
```

The solution must remain CSV-first, lightweight, local, and easy to run.

Do not add database infrastructure unless explicitly requested later.

---

# Note: the earlier brief contradicts the CSV-first rule

`code prompt.txt` and `code Prompt.docx` at the repository root are an **earlier, unsegmented**
version of this brief. Their "Technology Requirements" and "Deliverables" sections ask for things
the segmented prompts above explicitly forbid:

| The earlier brief asks for | The segmented prompts require |
|---|---|
| PostgreSQL / SQL Server | No database |
| A FastAPI layer | No API |
| A database schema | The CSVs are the schema |
| Data ingestion scripts | No ingestion pipeline |

**The segmented prompts win.** They are the later instruction and state the constraint as a
"CRITICAL DATA ARCHITECTURE REQUIREMENT" repeated in every section. The earlier documents remain
in the repository because their Sections 1–33 contain useful detail on the *business* requirements
— churn drivers, explanation wording, ROI assumptions, segment definitions — which is worth
reading. Just do not build the infrastructure they describe.

# Other requests made during the build

Beyond the numbered sections, one request shaped the repository:

- **"save all the prompts in a markdown file in the same project"** — produced this file.
