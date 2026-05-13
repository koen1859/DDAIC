import matplotlib.pyplot as plt
import os
from load_data import Article


def plot_demand(article: Article, forecasts: list[float]):
    # Plot full true demand
    plt.figure(figsize=(12, 6))
    plt.plot(article.dates, article.demand, label="True Demand", color="black")

    # Plot forecasted demand (aligned with test_dates)
    if article.slow_mover:
        label = "Forecasted demand (Croston)"
    else:
        label = "Forecasted demand (Exponential smoothing)"
    plt.plot(
        article.test_dates,
        forecasts,
        label=label,
        color="red",
        linestyle="--",
    )

    plt.title(f"{article.name}")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{article.id}_demand_forecast.png")
    plt.close()


def plot_inventory_strategy(
    article: Article,
    dates: list,
    on_hand: list[int],
    inv_position: list[int],
    R_series: list[int],
    Q_series: list[int],
):
    # Plot inventory levels and policy parameters
    plt.figure(figsize=(12, 6))

    # Inventory levels
    plt.plot(dates, on_hand, label="On-hand inventory", color="blue")
    plt.plot(
        dates, inv_position, label="Inventory position", color="purple", linestyle="--"
    )

    # Policy parameters
    plt.plot(dates, R_series, label="Reorder point (R)", color="green")
    plt.plot(dates, Q_series, label="Order quantity (Q)", color="orange")

    plt.title(f"{article.name} - Inventory Strategy Over Time")
    plt.xlabel("Date")
    plt.ylabel("Units")
    plt.legend()
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{article.id}_inventory_strategy.png")
    plt.close()
