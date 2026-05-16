from math import ceil
from plot import plot_demand, plot_inventory_strategy
from load_data import load_data, Article
from forecasting import ExponentialSmoothing, Croston
from inv_strat import InvStratNormal
import multiprocessing
import polars as pl

GLOBAL_TARGET: float = 0.98


def process_article(args: tuple[Article, float]) -> dict:
    article: Article = args[0]
    min_fill_rate: float = args[1]
    print(f"Simulating for article {article.name}")
    if article.slow_mover:
        model = Croston(article)
    else:
        model = ExponentialSmoothing(article)
    inv_strat: InvStratNormal = InvStratNormal(article, model, min_fill_rate)

    backorders: int = 0
    on_hand: int = 0
    order: list[int] = [0] * len(article.dates)  # (arrival_index, quantity)

    on_hand_hist: list[int] = []
    inv_pos_hist: list[int] = []
    R_hist: list[int] = []
    Q_hist: list[int] = []
    total_demand: int = 0
    demand_satisfied_from_stock: int = 0

    # Loop over the test periods' dates
    for i, current_date in enumerate(article.dates):
        # Receive orders arriving today
        on_hand += order[i]

        # Fullfill backorders
        if backorders > 0 and on_hand > 0:
            fullfilled = min(on_hand, backorders)
            on_hand -= fullfilled
            backorders -= fullfilled

        # Update models
        model.update(current_date)
        model.forecast()
        inv_strat.optimize()

        demand = article.demand[i]

        # If in test period, log
        if current_date in article.test_dates:
            total_demand += demand

            satisfied_now = min(on_hand, demand)
            demand_satisfied_from_stock += satisfied_now

        if on_hand >= demand:
            on_hand -= demand
        else:
            backorders += demand - on_hand
            on_hand = 0

        on_order = sum(order[(i + 1) :])  # orders arriving later
        inventory_pos = on_hand + on_order - backorders

        if inventory_pos <= inv_strat.R:
            order_qty = ceil(inv_strat.R + inv_strat.Q - inventory_pos)
            arrival_t = i + article.lead_time
            if arrival_t < len(order):
                order[arrival_t] = order_qty

        # If in test period, log
        if current_date in article.test_dates:
            on_hand_hist.append(on_hand)
            inv_pos_hist.append(inventory_pos)
            R_hist.append(inv_strat.R)
            Q_hist.append(inv_strat.Q)

    plot_demand(article, model.forecasts)
    plot_inventory_strategy(
        article, article.test_dates, on_hand_hist, inv_pos_hist, R_hist, Q_hist
    )

    return {
        "article_id": article.id,
        "article_name": article.name,
        "total_demand": total_demand,
        "demand_satisfied_from_stock": demand_satisfied_from_stock,
        "achieved_fill_rate": demand_satisfied_from_stock / total_demand,
        "target_fill_rate": min_fill_rate,
        "slow_mover": article.slow_mover,
    }


def main():
    articles: list[Article] = load_data()

    with multiprocessing.Pool() as pool:
        min_fill_rate: float = GLOBAL_TARGET
        args = [(a, min_fill_rate) for a in articles]
        results: pl.DataFrame = pl.DataFrame(pool.map(process_article, args))
        results.write_csv("../results/results.csv")
        print(
            f"Global fill rate achieved: {
                sum(results['demand_satisfied_from_stock'])
                / sum(results['total_demand'])
            }"
        )


if __name__ == "__main__":
    main()
