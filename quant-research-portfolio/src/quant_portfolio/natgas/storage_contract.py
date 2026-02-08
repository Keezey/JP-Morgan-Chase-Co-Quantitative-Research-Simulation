from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

import pandas as pd

from .price_forecast import NaturalGasPriceForecaster, DateLike


Schedule = Sequence[Tuple[DateLike, float]]  # (date, volume)


@dataclass
class StorageContractInputs:
    """Inputs for a simplified commodity storage contract valuation."""

    injection_schedule: Schedule
    withdrawal_schedule: Schedule
    max_volume: float
    storage_cost_per_month: float = 0.0
    injection_cost_per_unit: float = 0.0
    withdrawal_cost_per_unit: float = 0.0


def price_storage_contract(
    forecaster: NaturalGasPriceForecaster,
    inputs: StorageContractInputs,
) -> float:
    """Compute the NPV (no discounting) of a storage strategy.

    Assumptions (matching the prompt):
    - Interest rates = 0
    - No transport delay
    - No weekends/holidays adjustments
    - Volumes execute on the chosen dates up to inventory/capacity constraints

    Cashflows:
    - inject:  - price(date) * volume  - injection_cost_per_unit * volume
    - withdraw: + price(date) * volume - withdrawal_cost_per_unit * volume
    - storage costs accrue between events based on whole-month differences
    """

    events: List[Tuple[str, pd.Timestamp, float]] = []
    for d, v in inputs.injection_schedule:
        events.append(("inject", pd.to_datetime(d), float(v)))
    for d, v in inputs.withdrawal_schedule:
        events.append(("withdraw", pd.to_datetime(d), float(v)))

    if not events:
        return 0.0

    events.sort(key=lambda x: x[1])

    inventory = 0.0
    value = 0.0
    prev_date = events[0][1]

    for event_type, date, requested_volume in events:
        # storage cost for whole months elapsed since previous event
        months = (date.year - prev_date.year) * 12 + (date.month - prev_date.month)
        if months > 0:
            value -= months * inputs.storage_cost_per_month

        px = forecaster.predict(date)

        if event_type == "inject":
            volume = min(requested_volume, inputs.max_volume - inventory)
            value += -px * volume - inputs.injection_cost_per_unit * volume
            inventory += volume

        else:  # withdraw
            volume = min(requested_volume, inventory)
            value += px * volume - inputs.withdrawal_cost_per_unit * volume
            inventory -= volume

        prev_date = date

    return float(value)
