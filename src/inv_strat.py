from forecasting import ExponentialSmoothing, Croston
from load_data import Article

from scipy.stats import nbinom
from statistics import NormalDist, mean
from math import sqrt, ceil
from functools import cache

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
        model: ExponentialSmoothing,
        min_fill_rate: float,
    ) -> None:
        self.min_fill_rate: float = min_fill_rate
        self.article: Article = article
        self.model: ExponentialSmoothing | Croston = model
        self.R: int = 0

        # At least MOQ and at least avg daily demand
        self.Q: int = max(article.min_order_quantity, ceil(model.forecast()))

    def lead_time_demand(self) -> tuple[float, float]:
        mu: float = self.article.lead_time * self.model.forecast()
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
        self.Q = max(self.article.min_order_quantity, int(self.model.forecast()))
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


# Question is whether the cache decorator works for functions that take float as argument,
# due to floating point precision
@cache
def Dt_pmf(k: int, mu: float, sigma2: float) -> float:
    """
    PMF of lead time demand
    Parameters for this are inverse of in the book due to difference in how distr is defined
    """
    p: float = mu / sigma2
    r: float = mu * p / (1 - p)
    return float(nbinom.pmf(k, r, p))


@cache
def IL_pmf(
    j: int,
    mu: float,
    sigma2: float,
    R: int,
    Q: int,
) -> float:
    """
    PMF of lead time inventory level
    """
    sum: float = 0
    for k in range(max(R + 1, j), R + Q + 1):
        sum += Dt_pmf(k - j, mu, sigma2)

    return 1 / Q * sum


@cache
def fill_rate(
    mu: float,
    sigma2: float,
    R: int,
    Q: int,
) -> float:
    """
    Fill rate S2 given params
    """
    max_demand: int = ceil(mu + 8 * sqrt(sigma2))
    max_IL: int = R + Q + 1

    numerator: float = 0.0
    denominator: float = 0.0

    dt_probs: list[float] = [Dt_pmf(k, mu, sigma2) for k in range(max_demand)]
    il_probs: list[float] = [IL_pmf(j, mu, sigma2, R, Q) for j in range(max_IL)]
    for k in range(max_demand):
        denominator += k * dt_probs[k]
        for j in range(max_IL):
            numerator += min(j, k) * dt_probs[k] * il_probs[j]
    return numerator / denominator


class InvStratCompPois:
    def __init__(
        self,
        article: Article,
        model: Croston,
        min_fill_rate: float,
    ) -> None:
        self.min_fill_rate: float = min_fill_rate
        self.article: Article = article
        self.model: ExponentialSmoothing | Croston = model
        self.R: int = 0

        # At least MOQ and at least avg daily demand
        self.Q: int = max(article.min_order_quantity, ceil(model.forecast()))

    def optimize(self) -> None:
        """
        Calculate the reorder point R as on page 98 of the book.
        Use as order quantity the max of 1 day of demand and the MOQ
        Optimal R is as low as possible s.t. we have at least min_fill_rate
        """
        mu: float = self.model.forecast() * self.article.lead_time
        sigma2: float = self.article.lead_time * mean(
            r * r for r in self.model.residuals
        )

        self.Q = max(self.article.min_order_quantity, ceil(mu))

        # Bounds as in the book, ensure upper bound is enough to achieve min_fill_rate
        lower: int = -self.Q
        upper: int = ceil(mu + 30 * sqrt(sigma2))

        # Bisection
        for _ in range(50):
            if upper - lower <= 1:
                break
            mid: int = ceil((lower + upper) / 2)

            service = fill_rate(mu, sigma2, mid, self.Q)

            if service < self.min_fill_rate:
                lower = mid
            else:
                upper = mid

        self.R = upper

        # Take into account the multiplier
        self.Q /= self.article.demand_multiplier
        self.R /= self.article.demand_multiplier
