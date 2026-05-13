from forecasting import ExponentialSmoothing, Croston
from load_data import Article

from statistics import NormalDist, mean
from math import sqrt, ceil, inf

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
        self.R: float = 0.0
        self.Q: float = 0.0

        # Backorder cost per unit per day
        self.b1: float = self.article.sales_price / 2

        # Holding cost per unit per day
        self.h: float = self.article.sales_price * 0.2 * (1 / 365)

        # In optimal solution service levels are:
        self.S2 = self.S3 = max(self.b1 / (self.h + self.b1), self.min_fill_rate)

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

    def target_fill_rate(self) -> float:
        return max(self.S2, self.min_fill_rate)

    def expected_backorders(self, R: float, Q: float, mu: float, sigma: float) -> float:
        if sigma <= 0:
            return max(0.0, mu - R)
        return sigma * (G((R - mu) / sigma) - G((R + Q - mu) / sigma))

    def expected_on_hand(self, R: float, Q: float, mu: float, sigma: float) -> float:
        return Q / 2 + R - mu + self.expected_backorders(R, Q, mu, sigma)

    def solve_R_for_Q(self, Q: float, mu: float, sigma: float, target: float) -> float:
        if sigma <= 0:
            return max(mu, 0.0)

        lo: float = 0.0
        hi: float = mu + 10 * sigma + Q
        while self.fill_rate(mu, sigma, R=hi, Q=Q) < target:
            hi += 5 * sigma + Q
            if hi > mu + 100 * sigma + 10 * Q:
                break

        for _ in range(60):
            mid = (lo + hi) / 2
            if self.fill_rate(mu, sigma, R=mid, Q=Q) < target:
                lo = mid
            else:
                hi = mid
        return hi

    def optimize_RQ(self) -> tuple[int, int]:
        mu, sigma = self.lead_time_demand()
        target = self.target_fill_rate()

        Q_min = max(1, int(ceil(self.article.min_order_quantity)))
        Q_max = Q_min + max(50, int(6 * sigma + mu**0.5))

        best_cost = inf
        best_R = 0.0
        best_Q = Q_min

        for Q in range(Q_min, Q_max + 1):
            R = self.solve_R_for_Q(Q, mu, sigma, target)

            bo = self.expected_backorders(R, Q, mu, sigma)
            oh = self.expected_on_hand(R, Q, mu, sigma)
            cost = self.h * oh + self.b1 * bo

            if cost < best_cost:
                best_cost = cost
                best_R = R
                best_Q = Q

        self.R = int(ceil(best_R))
        self.Q = int(ceil(best_Q))
        return self.R, self.Q
