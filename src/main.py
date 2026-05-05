from load_data import load_data, Article
from forecasting import ExponentialSmoothing, Croston
from plot import plot_demand
import multiprocessing


def process_article(article: Article):
    print(f"Forecasting for article {article.name}")
    if article.slow_mover:
        model = Croston(article)
    else:
        model = ExponentialSmoothing(article)

    # Loop over the test periods' dates
    for i, current_date in enumerate(article.test_dates):
        # Forecast once per week
        if i % 7 == 0:
            model.update(current_date)
            model.forecast()

    forecasts: list[float] = model.forecasts
    plot_demand(article, forecasts)


articles: list[Article] = load_data()

with multiprocessing.Pool() as pool:
    pool.map(process_article, articles)
