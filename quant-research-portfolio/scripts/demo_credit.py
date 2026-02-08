import pandas as pd

from quant_portfolio.credit.pd_model import PDModel
from quant_portfolio.credit.fico_quantization import FICOQuantizer


def main():
    df = pd.read_csv("data/raw/Loan_Data.csv")

    # Task: PD + expected loss
    model = PDModel.fit(df)
    ex = {
        "credit_lines_outstanding": 5,
        "loan_amt_outstanding": 2000.0,
        "total_debt_outstanding": 8000.0,
        "income": 35000.0,
        "years_employed": 2,
        "fico_score": 610,
    }
    pd_hat = model.predict_pd(**ex)
    el = model.expected_loss(**ex, recovery_rate=0.10)
    print("Example PD:", pd_hat)
    print("Example expected loss:", el)

    # Task: FICO quantization buckets
    q = FICOQuantizer(df["fico_score"].to_numpy(), df["default"].to_numpy())
    buckets = q.fit(num_buckets=10)
    print("First 3 buckets:")
    for b in buckets[:3]:
        print(b)

    print("Rating for FICO=720:", q.map_fico_to_rating(720, buckets))
    print("Bucket PD for FICO=720:", q.estimate_pd_from_fico(720, buckets))


if __name__ == "__main__":
    main()
