from forecasting import ExponentialSmoothing, Croston
from load_data import Article

from statistics import NormalDist, mean
from math import sqrt, ceil

normal = NormalDist()


def phi(x: float) -> float:
    """
    Normal PDF
    """
    return normal.pdf(x)


def Phi(x: float) -> float:
    """
    Normal CDF
    """
    return normal.cdf(x)


def G(x: float) -> float:
    """
    Unit normal loss function
    """
    return phi(x) - x * (1.0 - Phi(x))


class InvStratNormal:
    def __init__(
        self,
        article: Article,
        model: ExponentialSmoothing | Croston,
        min_fill_rate: float,
    ) -> None:
        self.min_fill_rate: float = min_fill_rate
        self.article: Article = article
        self.model: ExponentialSmoothing | Croston = model
        self.R: int = 0

        # At least MOQ and at least avg daily demand
        self.Q: int = max(article.min_order_quantity, ceil(model.forecasts[0]))

    def lead_time_demand(self) -> tuple[float, float]:
        mu: float = sum(self.model.forecasts[i] for i in range(self.article.lead_time))
        sigma2: float = self.article.lead_time * mean(
            r * r for r in self.model.residuals
        )
        sigma: float = sqrt(sigma2)
        return mu, sigma

    def F(
        self,
        x: float,
        mu: float,
        sigma: float,
        R: float | None = None,
        Q: float | None = None,
    ) -> float:
        """
        CDF of lead time inventory
        """
        if not R:
            R = self.R
        if not Q:
            Q = self.Q

        return (sigma / Q) * (G((R - x - mu) / sigma) - G((R + Q - x - mu) / sigma))

    def f(
        self,
        x: float,
        mu: float,
        sigma: float,
        R: float | None = None,
        Q: float | None = None,
    ) -> float:
        """
        PDF of lead time inventory
        """
        if not R:
            R = self.R
        if not Q:
            Q = self.Q

        return (1 / Q) * (Phi((R + Q - x - mu) / sigma) - Phi((R - x - mu) / sigma))

    def fill_rate(
        self,
        mu: float,
        sigma: float,
        R: float | None = None,
        Q: float | None = None,
    ) -> float:
        """
        Fill rate S2 given params
        """
        if not R:
            R = self.R
        if not Q:
            Q = self.Q
        return 1 - (sigma / Q) * (G((R - mu) / sigma) - G((R + Q - mu) / sigma))

    def optimize(self, tol: float = 1e-6, max_iter: int = 10_000) -> None:
        """
        Calculate the reorder point R as on page 98 of the book.
        Use as order quantity the max of 1 day of demand and the MOQ
        Optimal R is as low as possible s.t. we have at least min_fill_rate
        """
        self.Q = max(self.article.min_order_quantity, int(self.model.forecasts[0]))
        mu, sigma = self.lead_time_demand()

        # Bounds as in the book, ensure upper bound is enough to achieve min_fill_rate
        lower = -self.Q
        upper = mu + 10.0 * sigma
        while self.fill_rate(mu, sigma, upper, self.Q) < self.min_fill_rate:
            upper += 10.0 * sigma

        # Bisection
        for _ in range(max_iter):
            mid = 0.5 * (lower + upper)

            service = self.fill_rate(mu, sigma, mid, self.Q)

            if service < self.min_fill_rate:
                lower = mid
            else:
                upper = mid

            if abs(upper - lower) < tol:
                break

        # Smallest R achieving target
        self.R = ceil(upper)
