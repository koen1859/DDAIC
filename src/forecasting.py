from load_data import Article
from datetime import date


class ExponentialSmoothing:
    def __init__(self, article: Article, alpha: float = 0.15) -> None:
        self.article: Article = article
        self.alpha: float = alpha

        # Track the index of the last data point we have seen and used for
        # calculation of the parameters
        self.current_idx: int = 0

        # Keep track of the residuals
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

    def update(self, current_date: date):
        """
        Given a date, update the a parameter based on the data in the
        time period between the date of the last update and the given date.
        """
        while (
            self.current_idx < len(self.article.dates)
            and self.article.dates[self.current_idx] <= current_date
        ):
            x = self.article.demand[self.current_idx]

            # Add the new residual
            self.residuals.append(x - self.a)

            # If earliest not already used data point is before the current date, use it to update param
            self.a = (1 - self.alpha) * self.a + self.alpha * x
            self.current_idx += 1

    def forecast(self) -> float:
        return self.a


class Croston:
    def __init__(
        self,
        article: Article,
        alpha: float = 0.15,
        beta: float = 0.15,
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
        positive_demands = [article.train_demand[i] for i in positive_indices]

        intervals = [
            positive_indices[i] - positive_indices[i - 1]
            for i in range(1, len(positive_indices))
        ]

        self.k_hat = sum(intervals) / len(intervals)
        self.d_hat = sum(positive_demands) / len(positive_demands)

        self.a: float = self.d_hat / self.k_hat if self.k_hat > 0 else 0.0
        self.periods_since_demand: int = 0

    def update(self, current_date: date):
        """
        Given a date, update Croston parameters based on all observed
        demand up to current_date.
        """
        while (
            self.current_idx < len(self.article.dates)
            and self.article.dates[self.current_idx] <= current_date
        ):
            x = self.article.demand[self.current_idx]
            self.periods_since_demand += 1
            self.residuals.append(x - self.a)

            if x > 0:
                # Update interval estimate
                self.k_hat = (
                    1 - self.alpha
                ) * self.k_hat + self.alpha * self.periods_since_demand

                # Update demand size estimate
                self.d_hat = (1 - self.beta) * self.d_hat + self.beta * x

                # Forecast demand per period
                self.a = self.d_hat / self.k_hat

                # Reset interval counter
                self.periods_since_demand = 0

            self.current_idx += 1

    def forecast(self) -> float:
        return self.a
