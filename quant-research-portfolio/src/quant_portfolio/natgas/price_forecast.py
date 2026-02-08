from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


DateLike = Union[str, pd.Timestamp]


@dataclass
class NaturalGasPriceForecaster:
    """Simple, explainable monthly natural-gas forecaster.

    Model:
      price ~ a0 + a1*t + a2*t^2 + month dummies

    Notes:
    - Built for a job-simulation: quick, interpretable baseline.
    - Explicitly avoids training at import-time (so it behaves well as a library).
    """

    model: LinearRegression
    dummy_cols: list[str]
    feature_cols: list[str]
    train_start: pd.Timestamp
    train_end: pd.Timestamp

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame, outlier_z: float = 3.0) -> pd.DataFrame:
        if not {"Dates", "Prices"}.issubset(df.columns):
            raise ValueError("Expected columns: Dates, Prices")

        d = df.copy()
        d["Prices"] = d["Prices"].astype(float)
        d["Dates"] = pd.to_datetime(d["Dates"])
        d = d.sort_values("Dates").reset_index(drop=True)

        # optional outlier filter (simple and transparent)
        z_scores = np.abs(stats.zscore(d["Prices"]))
        d = d.loc[z_scores <= outlier_z].copy()

        d["Month"] = d["Dates"].dt.month
        d["DateNum"] = d["Dates"].map(pd.Timestamp.toordinal)
        d["DateNum2"] = d["DateNum"] ** 2

        month_dummies = pd.get_dummies(d["Month"], prefix="Month", drop_first=True)
        d = pd.concat([d, month_dummies], axis=1)

        return d

    @classmethod
    def fit_from_csv(cls, csv_path: str, date_format: Optional[str] = None) -> "NaturalGasPriceForecaster":
        df = pd.read_csv(csv_path)
        if date_format:
            df["Dates"] = pd.to_datetime(df["Dates"], format=date_format)
        return cls.fit(df)

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "NaturalGasPriceForecaster":
        d = cls._prepare_dataframe(df)
        dummy_cols = [c for c in d.columns if c.startswith("Month_")]
        feature_cols = ["DateNum", "DateNum2"] + dummy_cols

        X = d[feature_cols]
        y = d["Prices"].to_numpy()

        model = LinearRegression()
        model.fit(X, y)

        return cls(
            model=model,
            dummy_cols=dummy_cols,
            feature_cols=feature_cols,
            train_start=pd.to_datetime(d["Dates"].min()),
            train_end=pd.to_datetime(d["Dates"].max()),
        )

    def _features_for_date(self, date: DateLike) -> pd.DataFrame:
        ts = pd.to_datetime(date)
        date_num = ts.toordinal()
        month = ts.month

        row = {"DateNum": date_num, "DateNum2": date_num**2}
        for col in self.dummy_cols:
            row[col] = 0

        # Month_2 ... Month_12 exist depending on drop_first=True
        month_col = f"Month_{month}"
        if month_col in self.dummy_cols:
            row[month_col] = 1

        return pd.DataFrame([row], columns=self.feature_cols)

    def predict(self, date: DateLike) -> float:
        """Predict a price for any date (past or +1y forward in the sim)."""
        X_new = self._features_for_date(date)
        return float(self.model.predict(X_new)[0])

    def predict_month_ends(self, start: DateLike, periods: int) -> pd.DataFrame:
        """Convenience helper: forecast month-end prices."""
        start_ts = pd.to_datetime(start)
        dates = pd.date_range(start=start_ts, periods=periods, freq="ME")
        preds = [self.predict(d) for d in dates]
        return pd.DataFrame({"date": dates, "predicted_price": preds})
