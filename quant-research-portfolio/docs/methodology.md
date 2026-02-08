# Methodology Notes

## Natural gas pricing

- Data: month-end prices (Oct 2020 → Sep 2024).
- Goal: estimate price for any date (past) and extrapolate 1 year forward (for indicative long-dated pricing).
- Baseline model:
  - quadratic time trend + month-of-year dummies (seasonality)
  - chosen for interpretability and speed
- Known limitations:
  - does not model regime shifts, volatility clustering, or forward curve microstructure
  - purely statistical: not constrained by storage arbitrage bounds

## Storage contract valuation (prototype)

- Cashflows modeled:
  - purchase/sale at predicted spot price on chosen injection/withdrawal dates
  - storage rent charged per month between events
  - per-unit injection/withdrawal costs
- Assumptions (prompt):
  - zero interest rates
  - no delay / no holiday adjustment

## Credit risk (job simulation)

- PD model:
  - logistic regression with standardized numeric features
- Expected loss:
  - EL = PD * LGD * EAD, with LGD = (1 - recovery_rate)

## FICO quantization

- Buckets are learned by maximizing default log-likelihood under piecewise-constant PD in each bucket.
- Solved with dynamic programming over distinct FICO values (optimal segmentation).
