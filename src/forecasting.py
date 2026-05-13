from load_data import Article
from datetime import date


class ExponentialSmoothing:
    def __init__(self, article: Article, alpha: float = 0.3) -> None:
        self.article: Article = article
        self.alpha: float = alpha

        # Track the index of the last data point we have seen and used for
        # calculation of the parameters
        self.current_idx: int = 0

        # Keep track of the training residuals
        self.residuals: list[float] = []

        # Initialize a_{-1} by Exponentially Weighted Moving Average
        numerator: float = 0.0
        denominator: float = 0.0
        weight: float = 1.0
        for x in article.train_demand:
            numerator += x * weight
            denominator += weight
            weight *= 1 - alpha
        self.a: float = numerator / denominator

        # Loop over data again for exponential smoothing a and track residuals
        for x in article.train_demand:
            self.residuals.append(x - self.a)
            self.a = (1 - self.alpha) * self.a + self.alpha * x

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
            x = self.article.test_demand[self.current_idx]

            # Add the new residual
            self.residuals.append(x - self.a)

            # If earliest not already used data point is before the current date, use it to update param
            self.a = (1 - self.alpha) * self.a + self.alpha * x
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


class Croston:
    def __init__(
        self,
        article: Article,
        alpha: float = 0.3,
        beta: float = 0.3,
    ) -> None:
        self.article: Article = article
        self.alpha: float = alpha
        self.beta: float = beta

        # Track index of last observed test demand used
        self.current_idx: int = 0

        # Track training residuals
        self.residuals: list[float] = []

        # Initialize demand size and interval from training data
        positive_indices = [i for i, x in enumerate(article.train_demand) if x > 0]

        if len(positive_indices) == 0:
            # No positive demand in training data
            self.k_hat: float = 1.0
            self.d_hat: float = 0.0
        else:
            positive_demands = [article.train_demand[i] for i in positive_indices]

            intervals = [
                positive_indices[i] - positive_indices[i - 1]
                for i in range(1, len(positive_indices))
            ]

            self.k_hat = sum(intervals) / len(intervals)
            self.d_hat = sum(positive_demands) / len(positive_demands)

        self.a: float = self.d_hat / self.k_hat if self.k_hat > 0 else 0.0
        self.periods_since_demand: int = 0

        # Update the parameters with exponential smoothing
        for demand in article.train_demand:
            self.residuals.append(demand - self.a)
            if demand > 0:
                # Update interval estimate
                self.k_hat = (
                    1 - self.alpha
                ) * self.k_hat + self.alpha * self.periods_since_demand

                # Update demand size estimate
                self.d_hat = (1 - self.beta) * self.d_hat + self.beta * demand

                # Forecast demand per period
                self.a = self.d_hat / self.k_hat

                # Reset interval counter
                self.periods_since_demand = 0

            self.current_idx += 1

        # Update periods since last positive demand
        for i in range(len(self.article.train_demand) - 1, -1, -1):
            if self.article.train_demand[i] == 0:
                self.periods_since_demand += 1
            else:
                break

        # Initialize forecasts
        days: int = len(self.article.test_demand)
        self.forecasts: list[float] = [self.a] * days

    def update(self, current_date: date):
        """
        Given a date, update Croston parameters based on all observed
        demand up to current_date.
        """
        while (
            self.current_idx < len(self.article.test_dates)
            and self.article.test_dates[self.current_idx] <= current_date
        ):
            demand = self.article.test_demand[self.current_idx]
            self.periods_since_demand += 1

            if demand > 0:
                # Update interval estimate
                self.k_hat = (
                    1 - self.alpha
                ) * self.k_hat + self.alpha * self.periods_since_demand

                # Update demand size estimate
                self.d_hat = (1 - self.beta) * self.d_hat + self.beta * demand

                # Forecast demand per period
                self.a = self.d_hat / self.k_hat

                # Reset interval counter
                self.periods_since_demand = 0

            self.current_idx += 1

    def forecast(self) -> None:
        """
        Forecast demand until end of test period.
        """
        remaining = len(self.article.test_demand) - self.current_idx
        self.forecasts[self.current_idx :] = [self.a] * remaining
