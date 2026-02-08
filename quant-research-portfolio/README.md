# Quant Research Job Simulation Portfolio

This repository packages several quantitative research job-simulation tasks into a **GitHub-ready portfolio**:
- **Commodities:** natural gas price estimation + **storage contract valuation** prototype
- **Credit risk:** **Probability of Default (PD)** model → expected loss + **FICO bucketing (quantization)** via dynamic programming

The focus is on:
- clear problem framing (what / why)
- reproducible code (library-style modules + scripts)
- explainable baselines (what you’d ship first before iterating)

---

## 1) Natural Gas Price Estimation (Monthly → Any Date + 1Y Extrapolation)

**Goal (simulation):** use month-end natural gas prices to estimate the gas price for *any date* in the past, and provide an **indicative forward extrapolation** for 1 year.

**Model:** quadratic time trend + month-of-year dummies (seasonality).  
Implemented in: `src/quant_portfolio/natgas/price_forecast.py`

### Run the demo
```bash
python scripts/demo_natgas.py
```

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

### Run the demo
```bash
python scripts/demo_credit.py
```

---

## 4) Mortgages: FICO Bucketing (Quantization)

**Goal (simulation):** map FICO scores into a fixed number of categorical **ratings**, choosing bucket boundaries that best summarize defaults.

Approach:
- maximize bucket-wise default **log-likelihood**
- solve optimal segmentation via **dynamic programming**

Implemented in: `src/quant_portfolio/credit/fico_quantization.py`

---

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

---

## Data

Raw simulation datasets are stored at:
- `data/raw/Nat_Gas.csv`
- `data/raw/Loan_Data.csv`

If you replace with your own data, keep raw files out of git and update file paths in scripts.

---

## Suggested next improvements (if you want to level this up)

**Nat gas**
- use a model that respects time-series structure (SARIMAX / Prophet / gradient boosting)
- quantify forecast uncertainty (prediction intervals)
- add robustness checks: rolling validation, parameter stability

**Storage contract**
- optimize injection/withdraw schedule (dynamic programming / linear programming)
- incorporate forward curve / basis risk assumptions

**Credit risk**
- add calibration plot and AUC
- compare models: tree-based vs logistic, include feature importance
- add simple monitoring metrics (drift checks)

---

## License

MIT (add your preferred license).
