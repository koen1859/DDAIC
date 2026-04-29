from load_data import load_data, Article
from fast_mover import (
    ExponentialSmoothing,
    ExponentialSmoothingWithTrend,
)
from plot import plot_demand


articles: list[Article] = load_data()
for i, article in enumerate(articles):
    print(f"Forecasting for article {article.name} at index {i}")
    model = ExponentialSmoothingWithTrend(article.train_demand)
    forecast_demand: list[float] = model.forecast(article.test_demand)
    plot_demand(article, forecast_demand)
