import pandas as pd
from quant_portfolio.natgas.price_forecast import NaturalGasPriceForecaster


def test_predict_runs():
    df = pd.read_csv("data/raw/Nat_Gas.csv")
    f = NaturalGasPriceForecaster.fit(df)
    px = f.predict("2024-09-30")
    assert isinstance(px, float)
