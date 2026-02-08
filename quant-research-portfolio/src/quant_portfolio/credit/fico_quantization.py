from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FICOBucket:
    rating: int
    fico_min: int
    fico_max: int
    n: int
    k: int
    pd: float


class FICOQuantizer:
    """Bucket FICO scores into K ratings by maximizing default log-likelihood.

    Uses dynamic programming over distinct FICO score values:
      maximize sum_b [ k_b log(p_b) + (n_b-k_b) log(1-p_b) ]
    where p_b = k_b / n_b within each bucket.

    This matches the job-simulation prompt and produces stable, explainable bins.
    """

    def __init__(self, fico_scores: np.ndarray, defaults: np.ndarray):
        d = pd.DataFrame({"fico_score": fico_scores.astype(int), "default": defaults.astype(int)})
        agg = (
            d.groupby("fico_score")
             .agg(n=("default", "size"), k=("default", "sum"))
             .reset_index()
             .sort_values("fico_score")
             .reset_index(drop=True)
        )
        self.agg = agg

        cum_n = agg["n"].cumsum().to_numpy()
        cum_k = agg["k"].cumsum().to_numpy()
        self.cum_n = np.concatenate([[0], cum_n])
        self.cum_k = np.concatenate([[0], cum_k])

    def _bucket_totals(self, start_idx: int, end_idx: int) -> tuple[int, int]:
        n = int(self.cum_n[end_idx + 1] - self.cum_n[start_idx])
        k = int(self.cum_k[end_idx + 1] - self.cum_k[start_idx])
        return n, k

    def _bucket_ll(self, start_idx: int, end_idx: int, eps: float = 1e-8) -> float:
        n, k = self._bucket_totals(start_idx, end_idx)
        if n == 0:
            return 0.0
        p = k / n
        p = float(np.clip(p, eps, 1 - eps))
        return float(k * np.log(p) + (n - k) * np.log(1 - p))

    def fit(self, num_buckets: int) -> List[FICOBucket]:
        N = len(self.agg)
        B = int(num_buckets)
        if B <= 0:
            raise ValueError("num_buckets must be positive")
        if B > N:
            raise ValueError("num_buckets cannot exceed number of distinct FICO scores")

        dp = np.full((B + 1, N), -np.inf, dtype=float)
        prev = np.full((B + 1, N), -1, dtype=int)

        for j in range(N):
            dp[1, j] = self._bucket_ll(0, j)

        for b in range(2, B + 1):
            for j in range(b - 1, N):
                best_ll = -np.inf
                best_i = -1
                for i in range(b - 2, j):
                    cand = dp[b - 1, i] + self._bucket_ll(i + 1, j)
                    if cand > best_ll:
                        best_ll, best_i = cand, i
                dp[b, j] = best_ll
                prev[b, j] = best_i

        # reconstruct buckets
        ranges: List[Tuple[int, int]] = []
        j = N - 1
        for b in range(B, 0, -1):
            i = prev[b, j]
            start = 0 if i == -1 else i + 1
            ranges.append((start, j))
            j = i
        ranges.reverse()

        buckets: List[FICOBucket] = []
        for b_idx, (s, e) in enumerate(ranges, start=1):
            fico_min = int(self.agg.loc[s, "fico_score"])
            fico_max = int(self.agg.loc[e, "fico_score"])
            n, k = self._bucket_totals(s, e)
            pd_bucket = (k / n) if n > 0 else 0.0

            # lower rating = better score (prompt wording)
            rating = b_idx
            buckets.append(FICOBucket(
                rating=rating,
                fico_min=fico_min,
                fico_max=fico_max,
                n=n,
                k=k,
                pd=float(pd_bucket),
            ))

        return buckets

    @staticmethod
    def map_fico_to_rating(fico_score: int, buckets: List[FICOBucket]) -> int:
        x = int(fico_score)
        for b in buckets:
            if b.fico_min <= x <= b.fico_max:
                return b.rating
        if x < buckets[0].fico_min:
            return buckets[0].rating
        return buckets[-1].rating

    @staticmethod
    def estimate_pd_from_fico(fico_score: int, buckets: List[FICOBucket]) -> float:
        x = int(fico_score)
        for b in buckets:
            if b.fico_min <= x <= b.fico_max:
                return float(b.pd)
        if x < buckets[0].fico_min:
            return float(buckets[0].pd)
        return float(buckets[-1].pd)
