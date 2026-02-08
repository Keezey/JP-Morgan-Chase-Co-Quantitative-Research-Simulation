from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
]


@dataclass
class PDModel:
    """Probability of Default (PD) model + Expected Loss helper."""

    pipeline: Pipeline
    feature_cols: list[str]

    @classmethod
    def fit_from_csv(
        cls,
        csv_path: str,
        feature_cols: Optional[list[str]] = None,
        target_col: str = "default",
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> "PDModel":
        df = pd.read_csv(csv_path)
        return cls.fit(df, feature_cols=feature_cols, target_col=target_col,
                       test_size=test_size, random_state=random_state)

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        target_col: str = "default",
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> "PDModel":
        cols = feature_cols or DEFAULT_FEATURES
        X = df[cols]
        y = df[target_col].astype(int)

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ])
        pipe.fit(X_train, y_train)

        return cls(pipeline=pipe, feature_cols=list(cols))

    def predict_pd(self, **loan_features: float) -> float:
        row = pd.DataFrame([loan_features], columns=self.feature_cols)
        pd_hat = self.pipeline.predict_proba(row)[0, 1]
        return float(pd_hat)

    def expected_loss(
        self,
        recovery_rate: float = 0.10,
        ead_field: str = "loan_amt_outstanding",
        **loan_features: float,
    ) -> float:
        """Expected loss: EL = PD * LGD * EAD, LGD = 1 - recovery_rate."""
        pd_hat = self.predict_pd(**loan_features)
        lgd = 1.0 - float(recovery_rate)
        ead = float(loan_features[ead_field])
        return float(pd_hat * lgd * ead)
