import pandas as pd

from quant_portfolio.natgas.price_forecast import NaturalGasPriceForecaster
from quant_portfolio.natgas.storage_contract import StorageContractInputs, price_storage_contract


def main():
    df = pd.read_csv("data/raw/Nat_Gas.csv")
    forecaster = NaturalGasPriceForecaster.fit(df)

    print("Train range:", forecaster.train_start.date(), "→", forecaster.train_end.date())
    print("Example prediction 2025-09-30:", forecaster.predict("2025-09-30"))

    inputs = StorageContractInputs(
        injection_schedule=[("2024-06-30", 1_000_000)],
        withdrawal_schedule=[("2024-12-31", 1_000_000)],
        max_volume=2_000_000,
        storage_cost_per_month=100_000,
        injection_cost_per_unit=0.0,
        withdrawal_cost_per_unit=0.0,
    )
    v = price_storage_contract(forecaster, inputs)
    print("Example contract value:", v)


if __name__ == "__main__":
    main()
