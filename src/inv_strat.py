from forecasting import ExponentialSmoothing, Croston, WintersTrendSeasonal
from load_data import Article

from scipy.stats import nbinom
from statistics import NormalDist, mean
from math import sqrt, ceil
import numpy as np

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
        model: ExponentialSmoothing | WintersTrendSeasonal | Croston,
        min_fill_rate: float,
    ) -> None:
        self.min_fill_rate: float = min_fill_rate
        self.article: Article = article
        self.model: ExponentialSmoothing | Croston | WintersTrendSeasonal = model
        self.R: int = 0

        # At least MOQ and at least avg daily demand over the lead time
        self.Q: int = max(
            article.min_order_quantity,
            ceil(model.forecast(article.lead_time) / article.lead_time),
        )

    def lead_time_demand(self) -> tuple[float, float]:
        mu: float = self.model.forecast(self.article.lead_time)
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
        self.Q = max(
            self.article.min_order_quantity,
            int(self.model.forecast() / self.article.lead_time),
        )
        mu, sigma = self.lead_time_demand()
        sigma = max(sigma, 1e-6)

        # Bounds as in the book, ensure upper bound is enough to achieve min_fill_rate
        lower = -self.Q
        upper = mu + 30.0 * sigma

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


def Dt_pmf(k: int, mu: float, sigma2: float) -> float:
    """
    PMF of lead time demand
    Parameters for this are inverse of in the book due to difference in how distr is defined
    """
    p: float = mu / sigma2
    r: float = mu * p / (1 - p)
    return float(nbinom.pmf(k, r, p))


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
    k_arr = np.arange(max(R + 1, j), R + Q + 1)
    pmf = np.array([Dt_pmf(k - j, mu, sigma2) for k in k_arr])
    return 1 / Q * np.sum(pmf)


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

    dt_probs = np.array([Dt_pmf(k, mu, sigma2) for k in range(max_demand)])
    il_probs = []
    for j in range(max_IL):
        min_k = max(R + 1, j)
        max_k = R + Q + 1

        pmf_sum = np.sum(dt_probs[min_k - j : max_k - j])
        il_probs.append(pmf_sum / Q)
    il_probs = np.array(il_probs)

    k_arr = np.arange(max_demand)
    denominator = np.sum(k_arr * dt_probs)

    k_arr = k_arr[np.newaxis, :]
    j_arr = np.arange(max_IL)[:, np.newaxis]
    dt_arr = dt_probs[np.newaxis, :]
    il_arr = il_probs[:, np.newaxis]

    numerator = np.sum(np.minimum(j_arr, k_arr) * dt_arr * il_arr)

    return float(numerator / denominator)


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
        self.Q: int = max(
            article.min_order_quantity, ceil(model.forecast() / self.article.lead_time)
        )

    def optimize(self) -> None:
        """
        Calculate the reorder point R as on page 98 of the book.
        Use as order quantity the max of 1 day of demand and the MOQ
        Optimal R is as low as possible s.t. we have at least min_fill_rate
        """
        mu: float = self.model.forecast(self.article.lead_time)
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
