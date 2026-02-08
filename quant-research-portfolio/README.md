## 1) Natural Gas Price Estimation (Monthly → Any Date + 1Y Extrapolation)

**Goal (simulation):** use month-end natural gas prices to estimate the gas price for *any date* in the past, and provide an **indicative forward extrapolation** for 1 year.

**Model:** quadratic time trend + month-of-year dummies (seasonality).  
Implemented in: `src/quant_portfolio/natgas/price_forecast.py`

What it prints:
- training date range
- example predicted price for a future date
- example storage contract value

---

## 2) Storage Contract Valuation Prototype

**Goal (simulation):** price a storage contract given:
- injection dates / volumes
- withdrawal dates / volumes
- max storage capacity
- storage cost per month
- optional per-unit injection/withdrawal costs

Implemented in: `src/quant_portfolio/natgas/storage_contract.py`

Core idea:
- **value = sell proceeds − buy costs − storage costs − operational costs**
- assumes interest rates = 0 and no holiday/weekend logic (per prompt)

---

## 3) Personal Loans: PD Model → Expected Loss

**Goal (simulation):** train a function that predicts **Probability of Default (PD)** from borrower characteristics, then compute **Expected Loss** assuming a recovery rate.

Implemented in: `src/quant_portfolio/credit/pd_model.py`

---

## 4) Mortgages: FICO Bucketing (Quantization)

**Goal (simulation):** map FICO scores into a fixed number of categorical **ratings**, choosing bucket boundaries that best summarize defaults.

Approach:
- maximize bucket-wise default **log-likelihood**
- solve optimal segmentation via **dynamic programming**

Implemented in: `src/quant_portfolio/credit/fico_quantization.py`

---

## Data

Raw simulation datasets are stored at:
- `data/raw/Nat_Gas.csv`
- `data/raw/Loan_Data.csv`
---
