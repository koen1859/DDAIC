import matplotlib.pyplot as plt
import os
from load_data import Article


def plot_demand(article: Article, forecasts: list[float], first_test_index: int):
    # Plot full true demand
    plt.figure(figsize=(12, 6))
    plt.plot(article.dates, article.true_demand, label="True Demand", color="black")

    # Plot forecasted demand (aligned with test_dates)
    if article.slow_mover:
        label = "Forecasted demand (Croston)"
    else:
        label = "Forecasted demand (Exponential smoothing)"
    plt.plot(
        article.dates,
        forecasts,
        label=label,
        color="red",
        linestyle="--",
    )

    plt.axvline(
        x=article.dates[first_test_index],
        color="gray",
        linestyle=":",
        label="Test Period Start",
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
    on_hand: list[int],
    inv_position: list[int],
    R: list[int],
    Q: list[int],
    first_test_index: int,
):
    # Plot inventory levels and policy parameters
    plt.figure(figsize=(12, 6))

    # Inventory levels
    plt.plot(article.dates, on_hand, label="On-hand inventory", color="blue")
    plt.plot(
        article.dates,
        inv_position,
        label="Inventory position",
        color="purple",
        linestyle="--",
    )

    # Policy parameters
    plt.plot(article.dates, R, label="Reorder point (R)", color="green")
    plt.plot(
        article.dates,
        [R[i] + Q[i] for i in range(len(R))],
        label="Order up to (R + Q)",
        color="orange",
    )

    plt.axvline(
        x=article.dates[first_test_index],
        color="gray",
        linestyle=":",
        label="Test Period Start",
    )

    plt.title(f"{article.name} - Inventory Strategy Over Time")
    plt.xlabel("Date")
    plt.ylabel("Units")
    plt.legend()
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(f"../figures/{article.id}_inventory_strategy.png")
    plt.close()
