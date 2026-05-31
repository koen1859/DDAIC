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
        periods_per_year: int = 52,  # 52 = weekly buckets, 12 = monthly buckets
    ) -> None:
        self.article = article
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.periods_per_year = periods_per_year
        self.current_idx = 0
        self.residuals: list[float] = []

        # Must be set before _initialize_parameters so _get_seasonal_idx works
        self._n_train = len(article.train_demand)
        self._initialize_parameters()

    def _get_seasonal_idx(self, absolute_day: int) -> int:
        day_of_year = absolute_day % 365
        return day_of_year * self.periods_per_year // 365

    def _initialize_parameters(self) -> None:
        T = self.periods_per_year
        n = self._n_train

        if n < 365:
            self.a = sum(self.article.train_demand) / max(n, 1) or 1.0
            self.b = 0.0
            self.F = [1.0] * T
            return

        # Only use complete calendar years so every bucket is equally represented
        n_complete = (n // 365) * 365
        train = self.article.train_demand[:n_complete]

        # Level: overall mean of complete years
        overall_mean = sum(train) / n_complete
        self.a = overall_mean if overall_mean > 0 else 1.0

        # Trend: average daily change from first to last complete year
        if n_complete >= 2 * 365:
            first_avg = sum(train[:365]) / 365
            last_avg = sum(train[n_complete - 365 :]) / 365
            self.b = (last_avg - first_avg) / (n_complete - 365)
        else:
            self.b = 0.0

        # Seasonal factors: average (demand / level) for each bucket across all years
        sums = [0.0] * T
        counts = [0] * T
        for i, d in enumerate(train):
            idx = self._get_seasonal_idx(i)
            sums[idx] += d
            counts[idx] += 1

        self.F = [
            (sums[i] / counts[i]) / self.a if counts[i] > 0 and self.a > 0 else 1.0
            for i in range(T)
        ]

        # Normalize so F sums to T
        sum_F = sum(self.F)
        if sum_F > 0:
            self.F = [f * T / sum_F for f in self.F]

    def update(self, current_date: date) -> None:
        while (
            self.current_idx < len(self.article.dates)
            and self.article.dates[self.current_idx] <= current_date
        ):
            x = self.article.demand[self.current_idx]

            abs_day = self._n_train + self.current_idx
            seasonal_idx = self._get_seasonal_idx(abs_day)

            self.residuals.append(x - (self.a + self.b) * self.F[seasonal_idx])

            x_deseas = x / self.F[seasonal_idx] if self.F[seasonal_idx] > 0 else x
            a_new = (1 - self.alpha) * (self.a + self.b) + self.alpha * x_deseas
            b_new = (1 - self.beta) * self.b + self.beta * (a_new - self.a)
            F_new = (
                (1 - self.gamma) * self.F[seasonal_idx] + self.gamma * (x / a_new)
                if a_new > 0
                else self.F[seasonal_idx]
            )

            self.F[seasonal_idx] = F_new
            self.a = a_new
            self.b = b_new

            # Normalize once per complete year
            if (abs_day + 1) % 365 == 0:
                sum_F = sum(self.F)
                if sum_F > 0:
                    self.F = [f * self.periods_per_year / sum_F for f in self.F]

            self.current_idx += 1

    def forecast(self, days_forward: int | None = None) -> float:
        if days_forward is None:
            days_forward = self.article.lead_time

        total = 0.0
        for k in range(1, days_forward + 1):
            abs_day = self._n_train + self.current_idx + k - 1
            seasonal_idx = self._get_seasonal_idx(abs_day)
            total += max(0.0, (self.a + k * self.b) * self.F[seasonal_idx])
        return total


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
