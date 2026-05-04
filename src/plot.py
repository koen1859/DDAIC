import matplotlib.pyplot as plt
import os
from load_data import Article


def plot_demand(article: Article, forecasts: list[float]):
    # Plot full true demand
    plt.figure(figsize=(12, 6))
    plt.plot(article.dates, article.demand, label="True Demand", color="black")

    # Plot forecasted demand (aligned with test_dates)
    plt.plot(
        article.test_dates,
        forecasts,
        label="Forecasted Demand",
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
