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

    def forecast(self, days_forward: int | None = None) -> float:
        if days_forward is None:
            days_forward = self.article.lead_time
        return self.a * days_forward


class WintersTrendSeasonal:
    def __init__(
        self,
        article: Article,
        alpha: float = 0.1,
        beta: float = 0.05,
        gamma: float = 0.1,
        periods_per_year: int = 52,
    ) -> None:
        self.article: Article = article
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma
        self.periods_per_year: int = periods_per_year  # Number of periods per year

        # Track the index of the last data point used for calculation
        self.current_idx: int = 0

        # Keep track of residuals
        self.residuals: list[float] = []

        # Initialize level and trend from training data
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize level, trend, and seasonal indices from training data."""
        if len(self.article.train_demand) < self.periods_per_year:
            # Not enough data for seasonal initialization
            self.a = (
                sum(self.article.train_demand) / len(self.article.train_demand)
                if self.article.train_demand
                else 1.0
            )
            self.b = 0.0
            self.F = [1.0] * self.periods_per_year
            return

        # Initialize level: average of first seasonal period
        first_period = self.article.train_demand[: self.periods_per_year]
        self.a = (
            sum(first_period) / self.periods_per_year if sum(first_period) > 0 else 1.0
        )

        # Initialize trend: average change between first and second period
        if len(self.article.train_demand) >= 2 * self.periods_per_year:
            second_period = self.article.train_demand[
                self.periods_per_year : 2 * self.periods_per_year
            ]
            second_avg = sum(second_period) / self.periods_per_year
            self.b = (second_avg - self.a) / self.periods_per_year
        else:
            self.b = 0.0

        # Initialize seasonal indices
        self.F = [1.0] * self.periods_per_year
        if self.a > 0:
            for i in range(self.periods_per_year):
                if i < len(self.article.train_demand):
                    self.F[i] = self.article.train_demand[i] / self.a

        # Normalize seasonal indices so they sum to T
        sum_F = sum(self.F)
        if sum_F > 0:
            self.F = [f * self.periods_per_year / sum_F for f in self.F]

    def update(self, current_date: date) -> None:
        """
        Update level, trend, and seasonal indices based on data up to current_date.
        Uses equations (2.14), (2.15), and (2.16)-(2.18) from book.
        """
        while (
            self.current_idx < len(self.article.dates)
            and self.article.dates[self.current_idx] <= current_date
        ):
            x = self.article.demand[self.current_idx]

            # Determine which seasonal index to use (based on position in cycle)
            seasonal_idx = self.current_idx % self.periods_per_year

            # Calculate deseasonalized demand: x / F_t
            if self.F[seasonal_idx] > 0:
                x_deseasonalized = x / self.F[seasonal_idx]
            else:
                x_deseasonalized = x

            # Store residual
            forecasted = (self.a + self.b) * self.F[seasonal_idx]
            self.residuals.append(x - forecasted)

            # Update level (equation 2.14)
            a_new = (1 - self.alpha) * (self.a + self.b) + self.alpha * x_deseasonalized

            # Update trend (equation 2.15)
            b_new = (1 - self.beta) * self.b + self.beta * (a_new - self.a)

            # Update seasonal index (equation 2.16)
            if a_new > 0:
                F_prime = (1 - self.gamma) * self.F[seasonal_idx] + self.gamma * (
                    x / a_new
                )
            else:
                F_prime = self.F[seasonal_idx]

            # Immediately update the seasonal index (equation 2.17)
            self.F[seasonal_idx] = F_prime

            # Update state
            self.a = a_new
            self.b = b_new

            # Normalize seasonal indices to sum to T (equation 2.18)
            # This prevents the indices from drifting
            # Only normalize at the end of each seasonal cycle
            if (self.current_idx + 1) % self.periods_per_year == 0:
                sum_F = sum(self.F)
                if sum_F > 0:
                    self.F = [f * self.periods_per_year / sum_F for f in self.F]

            self.current_idx += 1

    def forecast(self, days_forward: int | None = None) -> float:
        if days_forward is None:
            days_forward = self.article.lead_time

        total_forecast = 0.0

        for k in range(1, days_forward + 1):
            # Determine which seasonal index applies (cycles every T periods)
            seasonal_idx = (self.current_idx + k - 1) % self.periods_per_year

            # Point forecast (equation 2.20)
            forecast_value = (self.a + k * self.b) * self.F[seasonal_idx]

            # Ensure non-negative forecast
            forecast_value = max(0.0, forecast_value)

            total_forecast += forecast_value

        return total_forecast


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

    def forecast(self, days_forward: int | None = None) -> float:
        if days_forward is None:
            days_forward = self.article.lead_time
        return self.a * days_forward
