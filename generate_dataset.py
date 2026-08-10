#!/usr/bin/env python3
"""
Synthetic 3-year fashion e-commerce dataset generator.
Period: 2023-01-01 .. 2025-12-31
Outputs: Customer.csv (1000), Product.csv (500), Transaction.csv (20000 order lines), Return.csv
"""
import csv, math, random, os
from datetime import date, timedelta
from collections import defaultdict

random.seed(20230101)

OUT = os.path.dirname(os.path.abspath(__file__))
START = date(2023, 1, 1)
END = date(2025, 12, 31)
NDAYS = (END - START).days + 1
N_CUST, N_SKU, N_LINES = 1000, 500, 20000
RETURN_RATE = 0.20


def day(i):
    return START + timedelta(days=i)


def idx(dt):
    return (dt - START).days


# ---------------------------------------------------------------- PRODUCTS
CAT_STRUCT = {
    "Apparel": {
        "share": 0.40,
        "subs": {
            "T-Shirts":  (12, 45,  "all",    ["Men", "Women", "Unisex"]),
            "Shirts":    (25, 79,  "all",    ["Men", "Women"]),
            "Jeans":     (39, 119, "all",    ["Men", "Women"]),
            "Trousers":  (35, 99,  "all",    ["Men", "Women"]),
            "Dresses":   (39, 149, "summer", ["Women"]),
            "Skirts":    (25, 85,  "summer", ["Women"]),
            "Tops":      (18, 65,  "summer", ["Women"]),
            "Jackets":   (59, 179, "winter", ["Men", "Women", "Unisex"]),
        },
    },
    "Footwear": {
        "share": 0.18,
        "subs": {
            "Sneakers":     (45, 169, "all",    ["Men", "Women", "Unisex"]),
            "Formal Shoes": (59, 199, "all",    ["Men", "Women"]),
            "Sandals":      (25, 85,  "summer", ["Men", "Women"]),
            "Boots":        (69, 229, "winter", ["Men", "Women"]),
            "Loafers":      (55, 159, "all",    ["Men", "Women"]),
        },
    },
    "Activewear": {
        "share": 0.14,
        "subs": {
            "Track Pants":     (29, 85, "all",    ["Men", "Women", "Unisex"]),
            "Sports T-Shirts": (18, 55, "all",    ["Men", "Women", "Unisex"]),
            "Leggings":        (22, 69, "all",    ["Women"]),
            "Shorts":          (18, 55, "summer", ["Men", "Women", "Unisex"]),
        },
    },
    "Outerwear": {
        "share": 0.12,
        "subs": {
            "Coats":          (89, 329,  "winter", ["Men", "Women"]),
            "Parkas":         (109, 359, "winter", ["Men", "Women", "Unisex"]),
            "Puffer Jackets": (95, 289,  "winter", ["Men", "Women", "Unisex"]),
            "Trench Coats":   (109, 299, "winter", ["Men", "Women"]),
        },
    },
    "Accessories": {
        "share": 0.16,
        "subs": {
            "Belts":      (15, 55,  "all",    ["Men", "Women"]),
            "Scarves":    (15, 65,  "winter", ["Men", "Women", "Unisex"]),
            "Caps":       (12, 39,  "summer", ["Unisex"]),
            "Bags":       (35, 169, "all",    ["Women", "Unisex"]),
            "Socks":      (6, 22,   "all",    ["Men", "Women", "Unisex"]),
            "Sunglasses": (25, 99,  "summer", ["Unisex"]),
            "Wallets":    (20, 85,  "all",    ["Men", "Women", "Unisex"]),
        },
    },
}

BRANDS = {
    # brand: (price multiplier, allowed categories, weight)
    "UrbanEdge":  (1.00, ["Apparel", "Footwear", "Accessories", "Outerwear"], 0.22),
    "ModeStreet": (0.85, ["Apparel", "Accessories", "Footwear"], 0.20),
    "TrendAura":  (1.05, ["Apparel", "Accessories", "Outerwear"], 0.16),
    "NovaWear":   (0.92, ["Apparel", "Activewear", "Footwear"], 0.16),
    "LuxeLine":   (1.55, ["Apparel", "Outerwear", "Footwear", "Accessories"], 0.12),
    "ActiveCore": (0.95, ["Activewear", "Footwear", "Apparel"], 0.14),
}


def price_round(p):
    base = math.floor(p)
    return float(base) + random.choice([0.99, 0.99, 0.95, 0.90, 0.49])


products = []          # dicts
sku_by_cat = defaultdict(list)

# allocate SKU counts per category
cat_names = list(CAT_STRUCT)
counts = {c: int(round(CAT_STRUCT[c]["share"] * N_SKU)) for c in cat_names}
counts[cat_names[0]] += N_SKU - sum(counts.values())

sku_n = 0
for cat in cat_names:
    subs = list(CAT_STRUCT[cat]["subs"])
    for k in range(counts[cat]):
        sub = subs[k % len(subs)] if k < len(subs) else random.choice(subs)
        lo, hi, season, genders = CAT_STRUCT[cat]["subs"][sub]
        ok_brands = [b for b, v in BRANDS.items() if cat in v[1]]
        bw = [BRANDS[b][2] for b in ok_brands]
        brand = random.choices(ok_brands, weights=bw)[0]
        mult = BRANDS[brand][0]
        # skewed price inside the band (more cheap/mid SKUs than expensive)
        u = random.betavariate(2.0, 2.6)
        raw = (lo + (hi - lo) * u) * mult
        raw = max(5.0, min(raw, hi * 1.7))
        sku_n += 1
        sid = "P%04d" % sku_n
        gender = random.choices(genders, weights=[1.0] * len(genders))[0]
        # popularity: long tail, a few bestsellers
        pop = math.exp(random.gauss(0, 0.85))
        p = {
            "SKU ID": sid, "Category": cat, "Subcategory": sub, "Brand": brand,
            "Gender": gender, "Price": round(price_round(raw), 2),
            "season": season, "pop": pop,
        }
        products.append(p)
        sku_by_cat[cat].append(p)

PROD = {p["SKU ID"]: p for p in products}

# price tier percentile within catalogue
sorted_prices = sorted(p["Price"] for p in products)
for p in products:
    r = sorted_prices.index(p["Price"]) / len(sorted_prices)
    p["ptile"] = r

# ---------------------------------------------------------------- CUSTOMERS
GEO = {
    "Germany": (0.54, ["Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt", "Stuttgart",
                       "Düsseldorf", "Leipzig", "Dortmund", "Bremen", "Hannover", "Nuremberg"]),
    "Netherlands": (0.20, ["Amsterdam", "Rotterdam", "The Hague", "Utrecht", "Eindhoven",
                           "Groningen", "Tilburg", "Haarlem"]),
    "Austria": (0.13, ["Vienna", "Graz", "Linz", "Salzburg", "Innsbruck", "Klagenfurt"]),
    "Belgium": (0.13, ["Brussels", "Antwerp", "Ghent", "Bruges", "Leuven", "Liège"]),
}
CHANNELS = {
    # channel: (acquisition weight, retention multiplier, coupon propensity, return multiplier)
    "Organic Search": (0.17, 1.10, 0.18, 0.95),
    "Paid Search":    (0.12, 0.80, 0.34, 1.10),
    "Google Ads":     (0.11, 0.78, 0.36, 1.12),
    "Instagram":      (0.15, 0.95, 0.28, 1.18),
    "Facebook":       (0.09, 0.85, 0.30, 1.05),
    "Influencer":     (0.08, 0.90, 0.33, 1.22),
    "Referral":       (0.09, 1.30, 0.24, 0.85),
    "Email":          (0.10, 1.25, 0.45, 0.90),
    "Direct":         (0.09, 1.12, 0.15, 0.88),
}
SEGMENTS = [("Frequent", 0.17), ("Occasional", 0.40), ("Seasonal", 0.17),
            ("New/Low-Frequency", 0.13), ("Lapsing/Non-Buyer", 0.13)]


def sample_age():
    # mixture: core 25-38, secondary 39-52, tail young / older
    r = random.random()
    if r < 0.52:
        a = random.gauss(30, 5.0)
    elif r < 0.82:
        a = random.gauss(43, 6.5)
    elif r < 0.93:
        a = random.gauss(21.5, 2.4)
    else:
        a = random.gauss(57, 4.5)
    return int(max(18, min(65, round(a))))


# ---- daily demand weights (seasonality) -------------------------------------
MONTH_BASE = {1: 1.15, 2: 0.82, 3: 0.90, 4: 0.95, 5: 1.00, 6: 1.05,
              7: 1.12, 8: 0.88, 9: 1.06, 10: 1.00, 11: 1.30, 12: 1.38}
YEAR_FACTOR = {2023: 0.86, 2024: 1.00, 2025: 1.14}
MONTH_JITTER = {(y, m): random.uniform(0.86, 1.16) for y in (2023, 2024, 2025) for m in range(1, 13)}
BF_STRENGTH = {2023: 2.4, 2024: 2.9, 2025: 3.2}
WINTER_SALE = {2023: 1.7, 2024: 1.9, 2025: 1.75}
SUMMER_SALE = {2023: 1.5, 2024: 1.7, 2025: 1.85}


def black_friday(year):
    dt = date(year, 11, 1)
    while dt.weekday() != 4:
        dt += timedelta(days=1)
    return dt + timedelta(days=21)   # 4th Friday


BF = {y: black_friday(y) for y in (2023, 2024, 2025)}
SALE_DAYS = set()


def base_day_weight(dt):
    y, m = dt.year, dt.month
    w = MONTH_BASE[m] * YEAR_FACTOR[y] * MONTH_JITTER[(y, m)]
    w *= [1.06, 1.05, 1.02, 1.00, 0.95, 0.90, 0.96][dt.weekday()]
    sale = False
    delta = (dt - BF[y]).days
    if -6 <= delta <= 3:                       # Black Friday / Cyber Monday
        w *= BF_STRENGTH[y]
        sale = True
    if m == 12 and 1 <= dt.day <= 20:          # Christmas shopping
        w *= 1.55
    if m == 12 and dt.day >= 24:               # post-Christmas lull then sale
        w *= 0.75 if dt.day < 27 else 1.5
        sale = dt.day >= 27
    if m == 1 and 2 <= dt.day <= 18:           # winter clearance
        w *= WINTER_SALE[y]
        sale = True
    if (m == 6 and dt.day >= 24) or (m == 7 and dt.day <= 16):   # summer sale
        w *= SUMMER_SALE[y]
        sale = True
    if m == 9 and dt.day <= 14:                # autumn collection launch
        w *= 1.18
    if m == 3 and dt.day >= 20:                # spring collection
        w *= 1.10
    if sale:
        SALE_DAYS.add(dt)
    return w


DAY_W = [base_day_weight(day(i)) for i in range(NDAYS)]

SEASON_MASK = {
    "holiday": lambda m: 3.2 if m in (11, 12) else (1.8 if m == 1 else 0.16),
    "summer":  lambda m: 3.0 if m in (6, 7, 8) else (1.5 if m in (5, 9) else 0.18),
    "spring":  lambda m: 2.8 if m in (3, 4, 5) else (1.4 if m in (2, 6) else 0.20),
}

CAT_MONTH = {   # category demand shape across the year
    "Apparel":     {1: .95, 2: .9, 3: 1.0, 4: 1.05, 5: 1.1, 6: 1.1, 7: 1.05, 8: .95, 9: 1.05, 10: 1.0, 11: 1.05, 12: 1.05},
    "Footwear":    {1: 1.0, 2: .95, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.05, 7: 1.05, 8: .95, 9: 1.1, 10: 1.05, 11: 1.1, 12: 1.0},
    "Activewear":  {1: 1.5, 2: 1.3, 3: 1.15, 4: 1.1, 5: 1.05, 6: 1.0, 7: .9, 8: .9, 9: 1.05, 10: 1.0, 11: .95, 12: .9},
    "Outerwear":   {1: 1.35, 2: 1.0, 3: .6, 4: .35, 5: .2, 6: .15, 7: .15, 8: .3, 9: .9, 10: 1.5, 11: 1.8, 12: 1.5},
    "Accessories": {1: .95, 2: .9, 3: .95, 4: .95, 5: 1.0, 6: 1.05, 7: 1.05, 8: .95, 9: 1.0, 10: 1.05, 11: 1.35, 12: 1.6},
}
SEASON_SUB = {
    "winter": {1: 1.7, 2: 1.4, 3: .8, 4: .4, 5: .25, 6: .2, 7: .2, 8: .35, 9: .9, 10: 1.5, 11: 1.8, 12: 1.6},
    "summer": {1: .3, 2: .35, 3: .6, 4: 1.0, 5: 1.5, 6: 1.8, 7: 1.9, 8: 1.6, 9: .9, 10: .5, 11: .35, 12: .3},
    "all":    {m: 1.0 for m in range(1, 13)},
}
COUNTRY_TWEAK = {   # mild geographic differences in season timing / intensity
    "Germany": {}, "Austria": {12: 1.08, 1: 1.05, 7: 0.95},
    "Netherlands": {6: 1.08, 7: 1.10, 12: 0.96}, "Belgium": {7: 1.06, 1: 1.06},
}

customers = []
seg_pool = []
for s, share in SEGMENTS:
    seg_pool += [s] * int(round(share * N_CUST))
while len(seg_pool) < N_CUST:
    seg_pool.append("Occasional")
seg_pool = seg_pool[:N_CUST]
random.shuffle(seg_pool)

cum_reg_w = []
acc = 0.0
for i in range(NDAYS):
    acc += DAY_W[i] ** 0.7
    cum_reg_w.append(acc)

for i in range(N_CUST):
    cid = "CUST%04d" % (i + 1)
    seg = seg_pool[i]
    country = random.choices(list(GEO), weights=[GEO[c][0] for c in GEO])[0]
    city = random.choice(GEO[country][1])
    ch = random.choices(list(CHANNELS), weights=[CHANNELS[c][0] for c in CHANNELS])[0]
    gender = random.choices(["Female", "Male", "Other / Prefer not to say"],
                            weights=[0.58, 0.38, 0.04])[0]
    age = sample_age()

    # cohort placement: new/low buyers skew late, frequent skew early
    if seg == "New/Low-Frequency":
        lo, hi = int(NDAYS * 0.55), NDAYS - 8
    elif seg == "Frequent":
        lo, hi = 0, int(NDAYS * 0.80)
    elif seg == "Lapsing/Non-Buyer":
        lo, hi = 0, int(NDAYS * 0.62)
    else:
        lo, hi = 0, NDAYS - 15
    span = list(range(lo, hi + 1))
    reg_i = random.choices(span, weights=[DAY_W[j] ** 0.8 for j in span])[0]
    reg = day(reg_i)

    ret_mult = CHANNELS[ch][1]
    # price / promo persona
    price_type = random.choices(
        ["Full-Price", "Discount-Sensitive", "Premium", "Sale-Driven"],
        weights=[0.30, 0.30, 0.16, 0.24])[0]
    if seg == "Frequent":
        price_type = random.choices(["Full-Price", "Discount-Sensitive", "Premium", "Sale-Driven"],
                                    weights=[0.34, 0.22, 0.28, 0.16])[0]
    if seg == "Seasonal":
        price_type = random.choices(["Full-Price", "Discount-Sensitive", "Premium", "Sale-Driven"],
                                    weights=[0.18, 0.28, 0.10, 0.44])[0]

    # return persona
    r = random.random()
    return_type = "High" if r < 0.12 else ("Low" if r < 0.42 else "Normal")
    return_factor = {"High": 2.6, "Normal": 1.0, "Low": 0.20}[return_type]
    return_factor *= CHANNELS[ch][3]

    # category affinity
    prim = random.choices(cat_names, weights=[.36, .20, .16, .12, .16])[0]
    aff = {c: random.uniform(0.05, 0.25) for c in cat_names}
    aff[prim] = random.uniform(1.4, 3.2)
    sec = random.choice([c for c in cat_names if c != prim])
    aff[sec] = max(aff[sec], random.uniform(0.5, 1.3))
    if seg == "Frequent":                       # frequent buyers shop broader
        for c in cat_names:
            aff[c] += random.uniform(0.25, 0.7)

    gender_pref = {"Female": "Women", "Male": "Men"}.get(gender, random.choice(["Women", "Men"]))
    season_pref = random.choices(["holiday", "summer", "spring"], weights=[0.55, 0.30, 0.15])[0]

    customers.append({
        "Customer ID": cid, "Age": age, "Gender": gender, "City": city, "Country": country,
        "Customer Acquisition Channel": ch, "Registration Date": reg,
        "segment": seg, "ret_mult": ret_mult, "price_type": price_type,
        "return_type": return_type, "return_factor": return_factor,
        "aff": aff, "gender_pref": gender_pref, "season_pref": season_pref,
        "coupon_base": CHANNELS[ch][2],
    })

CUST = {c["Customer ID"]: c for c in customers}

# ---------------------------------------------------------------- ORDER PLAN
orders = []          # (customer, date)
for c in customers:
    reg_i = idx(c["Registration Date"])
    tenure = NDAYS - reg_i
    seg = c["segment"]
    if seg == "Frequent":
        rate = random.uniform(7.0, 19.0)
        if random.random() < 0.08:
            rate *= 1.5                                  # VIP tail
        active_end = NDAYS - 1
        decay_tau = None
    elif seg == "Occasional":
        rate = random.uniform(1.6, 4.2)
        active_end = NDAYS - 1 if random.random() > 0.25 else reg_i + int(tenure * random.uniform(0.5, 0.9))
        decay_tau = None
    elif seg == "Seasonal":
        rate = random.uniform(1.5, 3.4)
        active_end = NDAYS - 1
        decay_tau = None
    elif seg == "New/Low-Frequency":
        rate = random.uniform(0.5, 1.6)
        active_end = NDAYS - 1
        decay_tau = None
    else:   # Lapsing / becomes non-buyer
        rate = random.uniform(4.0, 8.0)
        max_end = NDAYS - 1 - random.randint(210, 560)
        active_end = min(max_end, reg_i + int(tenure * random.uniform(0.30, 0.70)))
        active_end = max(active_end, reg_i + 40)
        decay_tau = max(90.0, (active_end - reg_i) / 2.0)

    rate *= c["ret_mult"]
    n_orders = max(1, int(round(rate * (active_end - reg_i + 1) / 365.0 *
                                random.uniform(0.75, 1.25))))
    if seg == "New/Low-Frequency":
        n_orders = min(n_orders, 3)

    orders.append((c["Customer ID"], reg_i))            # first purchase = registration
    if n_orders > 1:
        span = list(range(reg_i + 1, max(reg_i + 2, active_end + 1)))
        w = []
        for j in span:
            ww = DAY_W[j]
            m = day(j).month
            if seg == "Seasonal":
                ww *= SEASON_MASK[c["season_pref"]](m)
            if decay_tau:
                ww *= math.exp(-(j - reg_i) / decay_tau)
            else:
                ww *= math.exp(-(j - reg_i) / (NDAYS * random.uniform(1.6, 6.0)))
            w.append(ww)
        picks = random.choices(span, weights=w, k=min(n_orders - 1, len(span)))
        for j in sorted(set(picks)):
            orders.append((c["Customer ID"], j))

# lines per order (mean ~2.2)
order_lines = []
for (cid, di) in orders:
    n = random.choices([1, 2, 3, 4, 5, 6], weights=[.35, .30, .19, .10, .045, .015])[0]
    if CUST[cid]["segment"] == "Frequent":
        n = min(6, n + (1 if random.random() < 0.25 else 0))
    order_lines.append([cid, di, n])

# --- reconcile to exactly N_LINES ------------------------------------------
first_order_day = {}
for cid, di, _ in order_lines:
    if cid not in first_order_day or di < first_order_day[cid]:
        first_order_day[cid] = di
protected = {i for i, (cid, di, _) in enumerate(order_lines) if di == first_order_day[cid]}

total = sum(o[2] for o in order_lines)
guard = 0
while total > N_LINES and guard < 5_000_000:
    guard += 1
    k = random.randrange(len(order_lines))
    if order_lines[k][2] > 1:
        order_lines[k][2] -= 1
        total -= 1
    elif order_lines[k][2] == 1 and k not in protected:
        order_lines[k][2] = 0          # mark whole order for deletion
        total -= 1
if any(o[2] == 0 for o in order_lines):
    order_lines = [o for o in order_lines if o[2] > 0]

guard = 0
while total < N_LINES and guard < 5_000_000:
    guard += 1
    k = random.randrange(len(order_lines))
    if order_lines[k][2] < 6:
        order_lines[k][2] += 1
        total += 1

order_lines.sort(key=lambda x: (x[1], x[0]))
assert sum(o[2] for o in order_lines) == N_LINES, sum(o[2] for o in order_lines)
assert len({o[0] for o in order_lines}) == N_CUST, "every customer must have >=1 order"

# ---------------------------------------------------------------- TRANSACTIONS
PAY_BASE = {
    "Germany":     {"PayPal": .35, "Buy Now Pay Later": .25, "Credit Card": .22, "Debit Card": .18},
    "Austria":     {"Credit Card": .30, "PayPal": .29, "Debit Card": .23, "Buy Now Pay Later": .18},
    "Netherlands": {"Debit Card": .38, "PayPal": .27, "Credit Card": .20, "Buy Now Pay Later": .15},
    "Belgium":     {"Debit Card": .33, "Credit Card": .27, "PayPal": .25, "Buy Now Pay Later": .15},
}
DISCOUNT_LEVELS = [0, 5, 10, 15, 20, 25, 30, 40, 50]


def pick_payment(c):
    w = dict(PAY_BASE[c["Country"]])
    if c["Age"] < 30:
        w["Buy Now Pay Later"] *= 1.9
    elif c["Age"] > 50:
        w["Buy Now Pay Later"] *= 0.35
        w["Credit Card"] *= 1.35
    if c["price_type"] == "Premium":
        w["Credit Card"] *= 1.45
        w["Buy Now Pay Later"] *= 0.45
    return random.choices(list(w), weights=list(w.values()))[0]


def pick_sku(c, dt, used):
    m = dt.month
    tweak = COUNTRY_TWEAK[c["Country"]]
    cw = []
    for cat in cat_names:
        w = c["aff"][cat] * CAT_MONTH[cat][m] * tweak.get(m, 1.0)
        cw.append(max(w, 0.001))
    for _ in range(12):
        cat = random.choices(cat_names, weights=cw)[0]
        pool = sku_by_cat[cat]
        weights = []
        for p in pool:
            w = p["pop"] * SEASON_SUB[p["season"]][m]
            if p["Gender"] == c["gender_pref"]:
                w *= 2.6
            elif p["Gender"] == "Unisex":
                w *= 1.25
            else:
                w *= 0.22
            if c["price_type"] == "Premium":
                w *= 0.4 + 2.6 * p["ptile"] ** 1.5
            elif c["price_type"] in ("Discount-Sensitive", "Sale-Driven"):
                w *= 1.7 - 1.1 * p["ptile"]
            weights.append(w)
        sku = random.choices(pool, weights=weights)[0]["SKU ID"]
        if sku not in used:
            return sku
    return None


def pick_discount(c, dt, prod):
    on_sale = dt in SALE_DAYS
    p = {"Full-Price": 0.22, "Discount-Sensitive": 0.62, "Premium": 0.14, "Sale-Driven": 0.72}[c["price_type"]]
    if on_sale:
        p = min(0.95, p + 0.30)
    if prod["Category"] == "Outerwear" and dt.month in (1, 2, 3):
        p = min(0.95, p + 0.20)          # end-of-season markdown
    if prod["season"] == "summer" and dt.month in (8, 9):
        p = min(0.95, p + 0.18)
    if random.random() > p:
        return 0
    if c["price_type"] in ("Discount-Sensitive", "Sale-Driven"):
        w = [0, .10, .17, .18, .20, .14, .11, .07, .03]
    elif c["price_type"] == "Premium":
        w = [0, .30, .30, .18, .12, .06, .03, .01, .00]
    else:
        w = [0, .20, .24, .19, .16, .11, .07, .02, .01]
    if on_sale:
        w = [0, .05, .10, .14, .21, .18, .16, .11, .05]
    return random.choices(DISCOUNT_LEVELS, weights=w)[0]


tx = []
order_no = 0
for cid, di, nlines in order_lines:
    c = CUST[cid]
    dt = day(di)
    order_no += 1
    oid = "%06d" % order_no
    pay = pick_payment(c)
    used = set()
    for _ in range(nlines):
        sku = pick_sku(c, dt, used)
        if sku is None:
            sku = random.choice([s for s in PROD if s not in used])
        used.add(sku)
        p = PROD[sku]
        qty = random.choices([1, 2, 3, 4, 5], weights=[.70, .19, .075, .025, .01])[0]
        if p["Subcategory"] in ("Socks", "T-Shirts", "Sports T-Shirts"):
            qty = random.choices([1, 2, 3, 4, 5], weights=[.45, .28, .16, .07, .04])[0]
        elif p["Category"] == "Outerwear" or p["Price"] > 180:
            qty = random.choices([1, 2], weights=[.93, .07])[0]
        drift = 1.0
        if random.random() < 0.30:
            drift = random.uniform(0.97, 1.04)
        sell = round(p["Price"] * drift, 2)
        disc = pick_discount(c, dt, p)
        cp = c["coupon_base"] * {"Full-Price": 0.8, "Discount-Sensitive": 1.5,
                                 "Premium": 0.45, "Sale-Driven": 1.35}[c["price_type"]]
        if dt in SALE_DAYS:
            cp *= 1.4
        if disc == 0:
            cp *= 0.15
        coupon = "Yes" if random.random() < min(cp, 0.85) else "No"
        if coupon == "Yes" and disc == 0:
            disc = random.choice([5, 10, 10, 15])
        nov = round(qty * sell * (1 - disc / 100.0), 2)
        tx.append({
            "Customer ID": cid, "Order ID": oid, "SKU ID": sku, "Purchase Date": dt,
            "Quantity": qty, "Selling Price": sell, "Discount": disc,
            "Coupon Used": coupon, "Net Order Value": nov, "Payment Method": pay,
        })

assert len(tx) == N_LINES

# ---------------------------------------------------------------- RETURNS
CAT_RET = {"Apparel": 0.26, "Footwear": 0.30, "Outerwear": 0.21,
           "Activewear": 0.17, "Accessories": 0.06}
SUB_RET = {"Jeans": 1.35, "Dresses": 1.45, "Trousers": 1.25, "Shirts": 1.15,
           "Boots": 1.25, "Formal Shoes": 1.30, "Sneakers": 1.10, "Leggings": 1.20,
           "Socks": 0.35, "Caps": 0.5, "Wallets": 0.5, "Sunglasses": 0.8, "Belts": 0.7}

props = []
for i, t in enumerate(tx):
    p = PROD[t["SKU ID"]]
    c = CUST[t["Customer ID"]]
    base = CAT_RET[p["Category"]] * SUB_RET.get(p["Subcategory"], 1.0)
    base *= c["return_factor"]
    if c["segment"] == "Frequent":
        base *= 0.9
    if t["Discount"] >= 30:
        base *= 1.18                      # deep-discount impulse buys come back
    elif t["Discount"] == 0:
        base *= 0.92
    if t["Quantity"] >= 2:
        base *= 1.15                      # bought 2 sizes, keep one
    if p["Price"] > 150:
        base *= 1.12
    props.append(min(0.90, max(0.005, base)))


def exp_qty(q):
    if q == 1:
        return 1.0
    return 0.60 * 1 + 0.25 * q + 0.15 * max(1, q - 1)


total_units = sum(t["Quantity"] for t in tx)
target_units = int(round(RETURN_RATE * total_units))

lo, hi = 0.0, 8.0
for _ in range(60):
    a = (lo + hi) / 2
    e = sum(min(0.95, props[i] * a) * exp_qty(tx[i]["Quantity"]) for i in range(N_LINES))
    if e < target_units:
        lo = a
    else:
        hi = a
alpha = (lo + hi) / 2

returned = {}
for i, t in enumerate(tx):
    if random.random() < min(0.95, props[i] * alpha):
        q = t["Quantity"]
        if q == 1:
            rq = 1
        else:
            rq = random.choices([1, q, max(1, q - 1)], weights=[.60, .25, .15])[0]
        returned[i] = min(rq, q)

cur = sum(returned.values())
order_pool = list(range(N_LINES))
random.shuffle(order_pool)
# fine-tune upward
pi = 0
while cur < target_units and pi < len(order_pool) * 3:
    i = order_pool[pi % len(order_pool)]
    pi += 1
    q = tx[i]["Quantity"]
    have = returned.get(i, 0)
    if have < q and random.random() < min(0.95, props[i] * alpha * 1.5 + 0.05):
        returned[i] = have + 1
        cur += 1
# fine-tune downward
keys = list(returned)
random.shuffle(keys)
ki = 0
while cur > target_units and ki < len(keys):
    i = keys[ki]
    ki += 1
    if returned[i] > 1:
        returned[i] -= 1
        cur -= 1
    else:
        del returned[i]
        cur -= 1

rets = []
for i, rq in sorted(returned.items()):
    t = tx[i]
    r = random.random()
    if r < 0.55:
        lag = random.randint(3, 10)
    elif r < 0.85:
        lag = random.randint(11, 20)
    else:
        lag = random.randint(21, 30)
    rets.append({
        "Customer ID": t["Customer ID"], "Order ID": t["Order ID"], "SKU ID": t["SKU ID"],
        "Return Date": t["Purchase Date"] + timedelta(days=lag), "Return Quantity": rq,
    })
rets.sort(key=lambda r: (r["Return Date"], r["Order ID"]))

# ---------------------------------------------------------------- WRITE CSVs
def w(path, header, rows):
    with open(os.path.join(OUT, path), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


for c in customers:
    c["Registration Date"] = c["Registration Date"].isoformat()
for t in tx:
    t["Purchase Date"] = t["Purchase Date"].isoformat()
for r in rets:
    r["Return Date"] = r["Return Date"].isoformat()

w("Customer.csv", ["Customer ID", "Age", "Gender", "City", "Country",
                   "Customer Acquisition Channel", "Registration Date"], customers)
w("Product.csv", ["SKU ID", "Category", "Subcategory", "Brand", "Gender", "Price"], products)
w("Transaction.csv", ["Customer ID", "Order ID", "SKU ID", "Purchase Date", "Quantity",
                      "Selling Price", "Discount", "Coupon Used", "Net Order Value",
                      "Payment Method"], tx)
w("Return.csv", ["Customer ID", "Order ID", "SKU ID", "Return Date", "Return Quantity"], rets)

# ================================================================ VALIDATION
# Re-reads the written CSVs so the checks validate the delivered files, not memory.
def rd(name):
    with open(os.path.join(OUT, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


C = rd("Customer.csv")
P = rd("Product.csv")
T = rd("Transaction.csv")
R = rd("Return.csv")

cid_set = {r["Customer ID"] for r in C}
sku_set = {r["SKU ID"] for r in P}
reg = {r["Customer ID"]: r["Registration Date"] for r in C}
line_key = {}          # (order, sku) -> (customer, qty, date)
order_cust, order_date = {}, {}
for r in T:
    line_key[(r["Order ID"], r["SKU ID"])] = (r["Customer ID"], int(r["Quantity"]), r["Purchase Date"])
    order_cust.setdefault(r["Order ID"], set()).add(r["Customer ID"])
    order_date.setdefault(r["Order ID"], set()).add(r["Purchase Date"])

checks = []


def chk(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


chk("1. Exactly 1,000 unique Customer IDs", len(cid_set) == 1000 == len(C), f"{len(cid_set)}")
chk("2. Exactly 500 unique SKU IDs", len(sku_set) == 500 == len(P), f"{len(sku_set)}")
chk("3. Exactly 20,000 transaction records", len(T) == 20000, f"{len(T)}")
units = sum(int(r["Quantity"]) for r in T)
runits = sum(int(r["Return Quantity"]) for r in R)
rate = runits / units
chk("4. Return rate = 20% of purchased units", abs(rate - 0.20) < 0.0005, f"{rate:.4%}")
chk("5. Every transaction Customer ID exists in Customer",
    all(r["Customer ID"] in cid_set for r in T))
chk("6. Every transaction SKU ID exists in Product", all(r["SKU ID"] in sku_set for r in T))
chk("7. Every return Customer ID exists in Customer", all(r["Customer ID"] in cid_set for r in R))
chk("8. Every return Order ID exists in Transaction",
    all(r["Order ID"] in order_cust for r in R))
chk("9. Every return SKU ID exists in the corresponding order",
    all((r["Order ID"], r["SKU ID"]) in line_key for r in R))
chk("10. Return Quantity <= purchased Quantity",
    all(0 < int(r["Return Quantity"]) <= line_key[(r["Order ID"], r["SKU ID"])][1] for r in R))
chk("11. Return Date > Purchase Date",
    all(r["Return Date"] > line_key[(r["Order ID"], r["SKU ID"])][2] for r in R))
firstbuy = {}
for r in T:
    k = r["Customer ID"]
    if k not in firstbuy or r["Purchase Date"] < firstbuy[k]:
        firstbuy[k] = r["Purchase Date"]
chk("12. Registration Date == first purchase date", all(reg[k] == v for k, v in firstbuy.items()))
chk("13. No transaction before Registration Date",
    all(r["Purchase Date"] >= reg[r["Customer ID"]] for r in T))
bad_nov = [r for r in T if abs(round(int(r["Quantity"]) * float(r["Selling Price"]) *
                                     (1 - float(r["Discount"]) / 100), 2) - float(r["Net Order Value"])) > 0.011]
chk("14. Net Order Value mathematically correct", not bad_nov, f"{len(bad_nov)} mismatches")
chk("15. No negative / zero quantity", all(int(r["Quantity"]) > 0 for r in T))
chk("16. No negative selling price", all(float(r["Selling Price"]) > 0 for r in T) and
    all(float(r["Price"]) > 0 for r in P))
chk("17. Discount within {0,5,...,50}",
    all(float(r["Discount"]) in {0, 5, 10, 15, 20, 25, 30, 40, 50} for r in T))
chk("18. All Order IDs valid (1 customer + 1 date per order, unique SKU per order)",
    all(len(v) == 1 for v in order_cust.values()) and all(len(v) == 1 for v in order_date.values())
    and len(line_key) == len(T))
chk("19. All SKU IDs valid & format P0001-P0500",
    all(r["SKU ID"].startswith("P") and 1 <= int(r["SKU ID"][1:]) <= 500 for r in P))
CITY_OK = {c: set(v[1]) for c, v in GEO.items()}
chk("20. No impossible city-country combinations",
    all(r["City"] in CITY_OK.get(r["Country"], set()) for r in C))
chk("21. No duplicate Customer IDs", len(cid_set) == len(C))
chk("22. No duplicate SKU IDs", len(sku_set) == len(P))
chk("23. No orphan transactions",
    all(r["Customer ID"] in cid_set and r["SKU ID"] in sku_set for r in T))
chk("24. No orphan returns",
    all((r["Order ID"], r["SKU ID"]) in line_key and
        line_key[(r["Order ID"], r["SKU ID"])][0] == r["Customer ID"] for r in R))

# ---- summary metrics
gross = sum(int(r["Quantity"]) * float(r["Selling Price"]) for r in T)
net = sum(float(r["Net Order Value"]) for r in T)
n_orders = len(order_cust)
opc = defaultdict(int)
for o, cs in order_cust.items():
    opc[list(cs)[0]] += 1
last_buy = {}
for r in T:
    k = r["Customer ID"]
    if k not in last_buy or r["Purchase Date"] > last_buy[k]:
        last_buy[k] = r["Purchase Date"]
cutoff = (END - timedelta(days=180)).isoformat()
active = sum(1 for v in last_buy.values() if v >= cutoff)
repeat = sum(1 for v in opc.values() if v >= 2)
seg_counts = defaultdict(int)
for c in customers:
    seg_counts[c["segment"]] += 1
ret_lines = len(R)

summary = [
    ("Number of customers", f"{len(C):,}"),
    ("Number of products / SKUs", f"{len(P):,}"),
    ("Number of transactions (order lines)", f"{len(T):,}"),
    ("Number of distinct orders", f"{n_orders:,}"),
    ("Total purchased units", f"{units:,}"),
    ("Number of returned units", f"{runits:,}"),
    ("Return rate (returned units / purchased units)", f"{rate:.2%}"),
    ("Return rate (returned lines / transaction lines)", f"{ret_lines/len(T):.2%}"),
    ("Total gross sales (before discount)", f"EUR {gross:,.2f}"),
    ("Total discounts", f"EUR {gross-net:,.2f}"),
    ("Total net sales", f"EUR {net:,.2f}"),
    ("Average discount rate", f"{(gross-net)/gross:.2%}"),
    ("Active customers (purchase in last 180 days)", f"{active:,}"),
    ("Inactive / non-buying customers", f"{len(C)-active:,}"),
    ("Average orders per customer", f"{n_orders/len(C):.2f}"),
    ("Average order value (net)", f"EUR {net/n_orders:,.2f}"),
    ("Average line value (net)", f"EUR {net/len(T):,.2f}"),
    ("Repeat customers (2+ orders)", f"{repeat:,} ({repeat/len(C):.1%})"),
    ("One-time customers", f"{len(C)-repeat:,}"),
    ("Max orders by a single customer", f"{max(opc.values())}"),
    ("Coupon usage rate", f"{sum(1 for r in T if r['Coupon Used']=='Yes')/len(T):.1%}"),
    ("Discounted line share", f"{sum(1 for r in T if float(r['Discount'])>0)/len(T):.1%}"),
]

for n, ok, d in checks:
    print(("PASS " if ok else "FAIL ") + n + ("  [" + d + "]" if d else ""))
print()
for k, v in summary:
    print(f"{k:52s} {v}")

# ---- write markdown doc
md = []
md.append("# Fashion Brand Synthetic Dataset — Data Dictionary & Validation\n")
md.append(f"**Period:** {START} to {END} (3 full years) · **Currency:** EUR · "
          "**Files:** `Customer.csv`, `Product.csv`, `Transaction.csv`, `Return.csv`\n")
md.append("""
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
""")
md.append("| # | Check | Result | Detail |\n|---|---|---|---|")
for i, (n, ok, d) in enumerate(checks, 1):
    md.append(f"| {i} | {n.split('. ',1)[1]} | {'PASS' if ok else 'FAIL'} | {d} |")
md.append("\n## Validation summary\n")
md.append("| Metric | Value |\n|---|---|")
for k, v in summary:
    md.append(f"| {k} | {v} |")
md.append("\n### Customers by behavioural segment\n")
md.append("| Segment | Customers |\n|---|---|")
for k in ["Frequent", "Occasional", "Seasonal", "New/Low-Frequency", "Lapsing/Non-Buyer"]:
    md.append(f"| {k} | {seg_counts[k]} |")
md.append("\n### Orders per customer distribution\n")
buckets = [(1, 1), (2, 5), (6, 15), (16, 39), (40, 10**9)]
labels = ["1 order", "2-5 orders", "6-15 orders", "16-39 orders", "40+ orders"]
md.append("| Orders | Customers |\n|---|---|")
for (lo_b, hi_b), lb in zip(buckets, labels):
    md.append(f"| {lb} | {sum(1 for v in opc.values() if lo_b <= v <= hi_b)} |")

with open(os.path.join(OUT, "DATA_DICTIONARY_AND_VALIDATION.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

print("\nAll checks passed:", all(ok for _, ok, _ in checks))
