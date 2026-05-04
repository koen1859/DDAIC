from load_data import Article
from datetime import date


class ExponentialSmoothing:
    def __init__(self, article: Article, alpha: float = 0.3) -> None:
        self.article: Article = article
        self.alpha: float = alpha

        # Track the index of the last data point we have seen and used for
        # calculation of the parameters
        self.current_idx: int = 0

        # Initialize a_{-1} by Exponentially Weighted Moving Average
        numerator: float = 0.0
        denominator: float = 0.0
        weight: float = 1.0
        for x in article.train_demand:
            numerator += x * weight
            denominator += weight
            weight *= 1 - alpha
        self.a: float = numerator / denominator

        days: int = len(self.article.test_demand)
        self.forecasts: list[float] = [self.a] * days

    def update(self, current_date: date):
        """
        Given a date, update the a parameter based on the data in the
        time period between the date of the last update and the given date.
        """
        while (
            self.current_idx < len(self.article.test_dates)
            and self.article.test_dates[self.current_idx] <= current_date
        ):
            # If earliest not already used data point is before the current date, use it to update param
            self.a: float = (
                1 - self.alpha
            ) * self.a + self.alpha * self.article.test_demand[self.current_idx]
            self.current_idx += 1

    def forecast(self) -> None:
        """
        Forecast the demand until the end of the time period (30-06-2018)
        Does not need the date as input since it can infer what days need
        forecasting from the length of test_demand list
        """
        remaining = len(self.article.test_demand) - self.current_idx
        self.forecasts[self.current_idx :] = [self.a] * remaining


class ExponentialSmoothingWithTrend:
    def __init__(self, article: Article, alpha: float = 0.3, beta: float = 0.3) -> None:
        self.article: Article = article
        self.alpha: float = alpha
        self.beta: float = beta

        # Track the index of the last data point we have seen and used for
        # calculation of the parameters
        self.current_idx: int = 0

        # Initialize a_{-1}, b_{-1} with OLS
        N: int = len(article.train_demand)
        self.b: float = (
            sum((k + 1) * article.train_demand[k] for k in range(N))
            - (N + 1) / 2 * sum(article.train_demand)
        ) / (N * (N**2 - 1) / 12)
        self.a: float = 1 / N * sum(article.train_demand) - (N + 1) / 2 * self.b

        # Initialize the forecast
        days: int = len(self.article.test_demand)
        self.forecasts: list[float] = [self.a + (i + 1) * self.b for i in range(days)]

    def update(self, current_date: date):
        """
        Given a date, update the a and b parameters based on the data in the
        time period between the date of the last update and the given date.
        """
        while (
            self.current_idx < len(self.article.test_dates)
            and self.article.test_dates[self.current_idx] <= current_date
        ):
            # If earliest not already used data point is before the current date, use it to update params
            a_new: float = (1 - self.alpha) * (
                self.a + self.b
            ) + self.alpha * self.article.test_demand[self.current_idx]
            b_new = (1 - self.beta) * self.b + self.beta * (a_new - self.a)

            self.a = a_new
            self.b = b_new
            self.current_idx += 1

    def forecast(self) -> None:
        """
        Forecast the demand until the end of the time period (30-06-2018)
        Does not need the date as input since it can infer what days need
        forecasting from the length of test_demand list
        """
        remaining = len(self.article.test_demand) - self.current_idx
        self.forecasts[self.current_idx :] = [
            self.a + (h + 1) * self.b for h in range(remaining)
        ]
