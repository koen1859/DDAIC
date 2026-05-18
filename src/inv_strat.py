from forecasting import ExponentialSmoothing, Croston
from load_data import Article


from scipy.stats import poisson
from statistics import NormalDist, mean
from math import sqrt, ceil, factorial, exp, floor, comb
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
        while self.fill_rate(mu, sigma, upper, self.R) < self.min_fill_rate:
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


# The book uses as Geom(beta) where beta is failure prob of Bernoulli trial if I am not mistaken
# Hence the inverse of the wikipedia of Geometric Distribution
@cache
def geom_pmf(k: int, beta: float) -> float:
    return beta ** (k - 1) * (1 - beta)


@cache
def geom_cdf(k: int, beta: float) -> float:
    return 1 - beta ** floor(k) if k >= 1 else 0


@cache
def geom_mean(beta: float) -> float:
    return 1 / (1 - beta)


def _f_jk(self, j: int, k: int, beta: float) -> float:
    """
    As in the book but the recursion is slow so not used
    """
    if k == 0:
        return 1.0 if j == 0 else 0.0
    sum: float = 0
    for i in range(k - 1, j):
        sum += self.f_jk(i, k - 1, beta) * geom_pmf(j - i, beta)

    return sum


@cache
def f_jk(j: int, k: int, beta: float) -> float:
    """
    Sum of geometric is actually negative binomial
    """
    if k == 0:
        return 1.0 if j == 0 else 0.0
    if j < k:
        return 0.0

    return comb(j - 1, k - 1) * (1 - beta) ** k * beta ** (j - k)


@cache
def Dt_pmf(
    j: int,
    t: int,
    lamda: float,
    beta: float,
) -> float:
    """
    PMF of demand in t periods (where in our case we will take t=lead_time) but i made the function general
    """
    sum: float = 0
    for k in range(
        int(geom_mean(beta) * lamda * t + 8 * sqrt(geom_mean(beta) * lamda * t)) + 1
    ):
        sum += poisson.pmf(k, lamda * t) * f_jk(j, k, beta)
    return sum


@cache
def IL_pmf(
    j: int,
    t: int,
    lamda: float,
    beta: float,
    R: int,
    Q: int,
) -> float:
    """
    PMF of inventory level in t periods
    """
    sum: float = 0
    for k in range(max(R + 1, j), R + Q + 1):
        sum += Dt_pmf(k - j, t, lamda, beta)

    return 1 / Q * sum


@cache
def fill_rate(
    t: int,
    lamda: float,
    beta: float,
    R: int,
    Q: int,
) -> float:
    """
    Fill rate S2 given params
    """
    max_demand: int = int(lamda * t + 8 * sqrt(lamda * t)) + 1
    max_IL: int = R + Q + 1

    numerator: float = 0.0
    for k in range(max_demand):
        dt: float = Dt_pmf(k, t, lamda, beta)
        for j in range(max_IL):
            numerator += min(j, k) * dt * IL_pmf(j, t, lamda, beta, R, Q)
    return numerator / (lamda * t * geom_mean(beta))


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
        mu: float = self.model.forecast()
        sigma: float = sqrt(mean(r * r for r in self.model.residuals))
        beta: float = 1 - (2 / (1 + sigma**2 / mu))
        lamda: float = mu * (1 - beta)

        self.Q = max(self.article.min_order_quantity, ceil(mu))

        # Bounds as in the book, ensure upper bound is enough to achieve min_fill_rate
        lower: int = -self.Q
        upper: int = ceil(mu + 10 * sigma)

        while (
            fill_rate(self.article.lead_time, lamda, beta, upper, self.Q)
            < self.min_fill_rate
        ):
            upper += ceil(10 * sigma)

        # Bisection
        for i in range(50):
            if upper - lower <= 1:
                break
            mid: int = ceil((lower + upper) // 2)

            service = fill_rate(self.article.lead_time, lamda, beta, mid, self.Q)

            if service < self.min_fill_rate:
                lower = mid
            else:
                upper = mid

        self.R = upper
