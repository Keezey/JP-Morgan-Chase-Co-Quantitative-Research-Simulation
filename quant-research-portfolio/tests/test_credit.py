import pandas as pd
from quant_portfolio.credit.pd_model import PDModel


def test_pd_in_range():
    df = pd.read_csv("data/raw/Loan_Data.csv")
    model = PDModel.fit(df)
    ex = {
        "credit_lines_outstanding": 3,
        "loan_amt_outstanding": 1500.0,
        "total_debt_outstanding": 4000.0,
        "income": 50000.0,
        "years_employed": 3,
        "fico_score": 680,
    }
    p = model.predict_pd(**ex)
    assert 0.0 <= p <= 1.0
