# Fashion Brand Synthetic Dataset — Data Dictionary & Validation

**Period:** 2023-01-01 to 2025-12-31 (3 full years) · **Currency:** EUR · **Files:** `Customer.csv`, `Product.csv`, `Transaction.csv`, `Return.csv`


## Data model

```
Customer (1) ──< Transaction (N)      Customer.[Customer ID] → Transaction.[Customer ID]
Product  (1) ──< Transaction (N)      Product.[SKU ID]       → Transaction.[SKU ID]
Transaction (1) ──< Return (0..1)     Transaction.[Order ID] + [SKU ID] → Return.[Order ID] + [SKU ID]
Customer (1) ──< Return (N)           Customer.[Customer ID] → Return.[Customer ID]
```

**Grain.** One `Transaction.csv` row = **one order line** (one SKU inside one order). An order
may contain 1–6 SKUs; the same SKU never appears twice in the same order. `Customer ID`,
`Purchase Date` and `Payment Method` are constant across all lines of an order.
One `Return.csv` row = the returned units of one order line; at most one return row per
(`Order ID`, `SKU ID`).

## Formulas & definitions

- **Selling Price** = unit list price **before** discount (Product `Price` ± small pricing drift).
- **Discount** = percentage off the selling price, integer, one of {0, 5, 10, 15, 20, 25, 30, 40, 50}.
- **Net Order Value** = `Quantity × Selling Price × (1 − Discount/100)`, rounded to 2 decimals. This is
  the net value of the **line**, not of the whole order; sum the lines of an `Order ID` for order value.
- **Gross sales** = `Quantity × Selling Price` (pre-discount).
- **Return rate** (primary definition) = `sum(Return Quantity) / sum(Transaction.Quantity)` = **20.00%**
  of purchased *units*. A secondary line-level rate (returned lines / total lines) is reported below.

## Customer.csv — 1,000 rows

| Column | Type | Meaning | Allowed values |
|---|---|---|---|
| Customer ID | string (PK) | Unique customer key | `CUST0001`–`CUST1000` |
| Age | integer | Age at extraction | 18–65, concentrated 25–40 |
| Gender | string | Self-reported gender | Female (~58%), Male (~38%), Other / Prefer not to say (~4%) |
| City | string | Billing city | Must belong to Country |
| Country | string | Market | Germany, Netherlands, Austria, Belgium |
| Customer Acquisition Channel | string | First-touch channel | Organic Search, Paid Search, Google Ads, Instagram, Facebook, Influencer, Referral, Email, Direct |
| Registration Date | date (YYYY-MM-DD) | Sign-up date = **first purchase date** | 2023-01-01 – 2025-12-31 |

## Product.csv — 500 rows

| Column | Type | Meaning | Allowed values |
|---|---|---|---|
| SKU ID | string (PK) | Unique product key | `P0001`–`P0500` |
| Category | string | Top-level category | Apparel, Footwear, Activewear, Outerwear, Accessories |
| Subcategory | string | Product type | e.g. T-Shirts, Jeans, Dresses, Sneakers, Boots, Leggings, Parkas, Bags |
| Brand | string | Fictional house brand | UrbanEdge, ModeStreet, TrendAura, NovaWear, LuxeLine (premium), ActiveCore |
| Gender | string | Target gender | Men, Women, Unisex |
| Price | decimal | Base list price, EUR | ~5.99 – ~400, varies by category & brand tier |

## Transaction.csv — 20,000 rows

| Column | Type | Meaning | Allowed values |
|---|---|---|---|
| Customer ID | string (FK) | → Customer | `CUST0001`–`CUST1000` |
| Order ID | string | Order key, repeats across lines of the same order | zero-padded 6-digit, `000001`+ |
| SKU ID | string (FK) | → Product | `P0001`–`P0500` |
| Purchase Date | date | Order date, same for all lines of the order | 2023-01-01 – 2025-12-31 |
| Quantity | integer | Units bought on the line | 1–5 (mostly 1–3) |
| Selling Price | decimal | Unit price before discount, EUR | > 0 |
| Discount | integer | % off selling price | 0,5,10,15,20,25,30,40,50 |
| Coupon Used | string | Coupon applied | Yes / No (Yes always implies Discount > 0) |
| Net Order Value | decimal | Qty × Selling Price × (1 − Discount/100) | > 0 |
| Payment Method | string | Constant within an order | Credit Card, Debit Card, PayPal, Buy Now Pay Later |

## Return.csv

| Column | Type | Meaning | Allowed values |
|---|---|---|---|
| Customer ID | string (FK) | → Customer, matches the order's customer | |
| Order ID | string (FK) | → Transaction | |
| SKU ID | string (FK) | → Transaction line within that order | |
| Return Date | date | Always after Purchase Date | +3 to +30 days |
| Return Quantity | integer | Units returned | 1 ≤ RQ ≤ purchased Quantity |

*Note: returns of late-December orders can fall in January of the following year, which is why a
small number of Return Dates sit just after 2025-12-31.*

## Behavioural design (what is baked into the data)

- **Segments:** Frequent (~17%), Occasional (~40%), Seasonal (~17%), New/Low-Frequency (~13%),
  Lapsing→Non-Buyer (~13%). Lapsing customers' last purchase is ≥ ~7 months before the period end.
- **Seasonality:** Black Friday (strengthening year over year), December gifting, January clearance,
  late-June/July summer sale, September autumn launch; monthly shape is jittered per year so no two
  years are identical, and 2023 < 2024 < 2025 in overall volume.
- **Product seasonality:** Outerwear/Boots/Scarves peak Oct–Feb; Sandals/Shorts/Sunglasses/Dresses
  peak May–Aug; Activewear peaks in January.
- **Channel effects:** Referral and Email customers repeat more; Paid Search / Google Ads / Facebook
  repeat less; Influencer and Instagram return more; Referral, Email and Direct return less.
- **Price personas:** Full-Price, Discount-Sensitive, Premium (higher ASP, low coupon use),
  Sale-Driven (buys mainly in sale windows).
- **Category affinity:** each customer has 1–2 preferred categories and a gender-consistent product
  bias, so repeat purchases concentrate in previously bought categories.
- **Return personas:** ~12% serial returners (~2.6× base), ~30% near-never returners (~0.2× base).
  Apparel and Footwear return far more than Accessories; deep-discount and multi-unit lines return more.
- **Geography:** payment mix differs by country (PayPal/BNPL heavy in DE, Debit heavy in NL/BE) and
  by age (BNPL skews young); mild country differences in seasonal timing.
- **Cohorts:** acquisition spread across all 3 years (early cohorts have longer observable life,
  New/Low-Frequency customers are acquired mostly in the final ~16 months), so no survivorship bias.

## Validation results

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | Exactly 1,000 unique Customer IDs | PASS | 1000 |
| 2 | Exactly 500 unique SKU IDs | PASS | 500 |
| 3 | Exactly 20,000 transaction records | PASS | 20000 |
| 4 | Return rate = 20% of purchased units | PASS | 19.9993% |
| 5 | Every transaction Customer ID exists in Customer | PASS |  |
| 6 | Every transaction SKU ID exists in Product | PASS |  |
| 7 | Every return Customer ID exists in Customer | PASS |  |
| 8 | Every return Order ID exists in Transaction | PASS |  |
| 9 | Every return SKU ID exists in the corresponding order | PASS |  |
| 10 | Return Quantity <= purchased Quantity | PASS |  |
| 11 | Return Date > Purchase Date | PASS |  |
| 12 | Registration Date == first purchase date | PASS |  |
| 13 | No transaction before Registration Date | PASS |  |
| 14 | Net Order Value mathematically correct | PASS | 0 mismatches |
| 15 | No negative / zero quantity | PASS |  |
| 16 | No negative selling price | PASS |  |
| 17 | Discount within {0,5,...,50} | PASS |  |
| 18 | All Order IDs valid (1 customer + 1 date per order, unique SKU per order) | PASS |  |
| 19 | All SKU IDs valid & format P0001-P0500 | PASS |  |
| 20 | No impossible city-country combinations | PASS |  |
| 21 | No duplicate Customer IDs | PASS |  |
| 22 | No duplicate SKU IDs | PASS |  |
| 23 | No orphan transactions | PASS |  |
| 24 | No orphan returns | PASS |  |

## Validation summary

| Metric | Value |
|---|---|
| Number of customers | 1,000 |
| Number of products / SKUs | 500 |
| Number of transactions (order lines) | 20,000 |
| Number of distinct orders | 6,726 |
| Total purchased units | 28,931 |
| Number of returned units | 5,786 |
| Return rate (returned units / purchased units) | 20.00% |
| Return rate (returned lines / transaction lines) | 25.24% |
| Total gross sales (before discount) | EUR 2,366,515.51 |
| Total discounts | EUR 234,088.10 |
| Total net sales | EUR 2,132,427.41 |
| Average discount rate | 9.89% |
| Active customers (purchase in last 180 days) | 598 |
| Inactive / non-buying customers | 402 |
| Average orders per customer | 6.73 |
| Average order value (net) | EUR 317.04 |
| Average line value (net) | EUR 106.62 |
| Repeat customers (2+ orders) | 723 (72.3%) |
| One-time customers | 277 |
| Max orders by a single customer | 64 |
| Coupon usage rate | 19.9% |
| Discounted line share | 50.6% |

### Customers by behavioural segment

| Segment | Customers |
|---|---|
| Frequent | 170 |
| Occasional | 400 |
| Seasonal | 170 |
| New/Low-Frequency | 130 |
| Lapsing/Non-Buyer | 130 |

### Orders per customer distribution

| Orders | Customers |
|---|---|
| 1 order | 277 |
| 2-5 orders | 374 |
| 6-15 orders | 232 |
| 16-39 orders | 105 |
| 40+ orders | 12 |
