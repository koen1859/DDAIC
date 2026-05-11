"""
It uses the existing Article object and any forecasting model that exposes:
    - update(current_date)
    - forecast()
    - forecasts: list[float]
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import ceil, exp, isfinite, pi, sqrt
from statistics import NormalDist, mean, pstdev
from typing import Callable, Iterable

from load_data import Article


_NORMAL = NormalDist()


@dataclass
class RQPolicy:
    article_id: int
    article_name: str
    target_fill_rate: float
    R0: int
    Q: int
    lead_time: int
    sigma_daily: float


@dataclass
class RQResult:
    article_id: int
    article_name: str
    target_fill_rate: float
    realized_fill_rate: float
    total_demand: int
    filled_demand: int
    lost_sales: int
    avg_on_hand: float
    orders_placed: int
    units_ordered: int
    avg_R: float
    Q: int

    def as_dict(self) -> dict:
        return asdict(self)


def _normal_pdf(z: float) -> float:
    return exp(-0.5 * z * z) / sqrt(2.0 * pi)


def _unit_normal_loss(z: float) -> float:
    """Standard normal loss function L(z)=phi(z)-z(1-Phi(z))."""
    return _normal_pdf(z) - z * (1.0 - _NORMAL.cdf(z))


def safety_factor_for_fill_rate(fill_rate: float, q: float, sigma_lead_time: float) -> float:
    """Return z for beta fill rate under an (R,Q) normal lead-time demand approximation.

    beta = 1 - E[shortage per cycle] / Q
    E[shortage per cycle] = sigma_L * L(z)
    R = mu_L + z * sigma_L
    """
    fill_rate = min(max(fill_rate, 0.0001), 0.9999)
    q = max(float(q), 1.0)
    sigma_lead_time = max(float(sigma_lead_time), 0.0)

    if sigma_lead_time == 0:
        return 0.0

    target_loss = (1.0 - fill_rate) * q / sigma_lead_time

    # L(z) is decreasing in z. Search wide enough to allow low service levels too.
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _unit_normal_loss(mid) > target_loss:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def estimate_sigma_daily(article: Article, fitted_daily_mean: float | None = None) -> float:
    """Estimate daily demand uncertainty from 2015-2017 training demand.

    A simple robust fallback is used because many SKUs are intermittent.
    """
    x = [float(v) for v in article.train_demand]
    if not x:
        return 0.0
    if fitted_daily_mean is None:
        fitted_daily_mean = mean(x)
    residuals = [v - fitted_daily_mean for v in x]
    sigma = pstdev(residuals) if len(residuals) > 1 else 0.0
    return sigma if isfinite(sigma) else 0.0


def order_quantity(article: Article, multiplier: int = 1, min_days_of_supply: int = 0) -> int:
    """Choose Q. Default is MOQ; optional days-of-supply prevents tiny Q for high-volume SKUs."""
    moq = max(1, int(ceil(article.min_order_quantity)))
    if min_days_of_supply <= 0:
        return moq * max(1, multiplier)
    avg_daily = mean(article.train_demand) if article.train_demand else 0.0
    dos_q = int(ceil(avg_daily * min_days_of_supply))
    return max(moq, dos_q) * max(1, multiplier)


def lead_time_forecast_sum(forecasts: list[float], start_idx: int, lead_time: int) -> float:
    """Expected demand during lead time using the currently available daily forecasts."""
    if not forecasts:
        return 0.0
    lead_time = max(1, int(lead_time))
    end = min(len(forecasts), start_idx + lead_time)
    total = sum(max(0.0, f) for f in forecasts[start_idx:end])

    # If lead time extends beyond the horizon, pad with the final forecast.
    missing = lead_time - (end - start_idx)
    if missing > 0:
        total += missing * max(0.0, forecasts[-1])
    return total


def reorder_point(
    forecasts: list[float],
    day_idx: int,
    lead_time: int,
    target_fill_rate: float,
    q: int,
    sigma_daily: float,
) -> int:
    """Dynamic R. Change target_fill_rate, lead_time, Q, sigma, or forecasts to get another R."""
    mu_lt = lead_time_forecast_sum(forecasts, day_idx, lead_time)
    sigma_lt = max(0.0, sigma_daily) * sqrt(max(1, int(lead_time)))
    z = safety_factor_for_fill_rate(target_fill_rate, q, sigma_lt)
    return max(0, int(ceil(mu_lt + z * sigma_lt)))


def make_forecasts(article: Article, model_factory: Callable[[Article], object], update_every: int = 7) -> list[float]:
    """Produce rolling forecasts for the 2018 test period using your existing model classes."""
    model = model_factory(article)
    for i, current_date in enumerate(article.test_dates):
        if i % update_every == 0:
            model.update(current_date)
            model.forecast()
    return [max(0.0, float(f)) for f in model.forecasts]


def simulate_rq(
    article: Article,
    forecasts: list[float],
    target_fill_rate: float,
    q: int | None = None,
    sigma_daily: float | None = None,
    review_every: int = 1,
    initial_on_hand: int | None = None,
) -> RQResult:
    """Evaluate a lost-sales (R,Q) policy over the first six months of 2018.

    The assignment asks for the first six months of 2018, so this function cuts the
    test set at 2018-06-30 even if Article.test_demand contains the full year.
    """
    horizon = [i for i, d in enumerate(article.test_dates) if d.year == 2018 and d.month <= 6]
    if not horizon:
        return RQResult(article.id, article.name, target_fill_rate, 1.0, 0, 0, 0, 0.0, 0, 0, 0.0, 0)

    end_idx = horizon[-1] + 1
    actual = article.test_demand[:end_idx]
    fc = forecasts[:end_idx]

    q = int(q if q is not None else order_quantity(article))
    q = max(1, q)
    sigma_daily = estimate_sigma_daily(article) if sigma_daily is None else max(0.0, sigma_daily)
    lead_time = max(1, int(article.lead_time))

    r0 = reorder_point(fc, 0, lead_time, target_fill_rate, q, sigma_daily)
    on_hand = int(initial_on_hand if initial_on_hand is not None else r0 + q)
    outstanding: list[tuple[int, int]] = []  # (arrival_day_idx, quantity)

    total_demand = filled_demand = lost_sales = 0
    on_hand_sum = 0.0
    orders_placed = units_ordered = 0
    r_values: list[int] = []

    for t, demand in enumerate(actual):
        # Receive orders due today.
        arrivals = [qty for arrival_t, qty in outstanding if arrival_t <= t]
        if arrivals:
            on_hand += sum(arrivals)
            outstanding = [(arrival_t, qty) for arrival_t, qty in outstanding if arrival_t > t]

        # Demand occurs; unfilled units are treated as lost sales for fill-rate measurement.
        d = max(0, int(demand))
        filled = min(on_hand, d)
        on_hand -= filled
        total_demand += d
        filled_demand += filled
        lost_sales += d - filled
        on_hand_sum += on_hand

        # Continuous-review approximation checked daily. Use review_every=7 for weekly review sensitivity.
        if t % max(1, review_every) == 0:
            r_t = reorder_point(fc, t, lead_time, target_fill_rate, q, sigma_daily)
            r_values.append(r_t)
            inventory_position = on_hand + sum(qty for _, qty in outstanding)
            while inventory_position <= r_t:
                outstanding.append((t + lead_time, q))
                inventory_position += q
                orders_placed += 1
                units_ordered += q

    realized = filled_demand / total_demand if total_demand > 0 else 1.0
    return RQResult(
        article_id=article.id,
        article_name=article.name,
        target_fill_rate=target_fill_rate,
        realized_fill_rate=realized,
        total_demand=total_demand,
        filled_demand=filled_demand,
        lost_sales=lost_sales,
        avg_on_hand=on_hand_sum / max(1, len(actual)),
        orders_placed=orders_placed,
        units_ordered=units_ordered,
        avg_R=(sum(r_values) / len(r_values)) if r_values else float(r0),
        Q=q,
    )


def aggregate_fill_rate(results: Iterable[RQResult]) -> float:
    results = list(results)
    demand = sum(r.total_demand for r in results)
    filled = sum(r.filled_demand for r in results)
    return filled / demand if demand > 0 else 1.0
