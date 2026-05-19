from math import ceil
from plot import plot_demand, plot_inventory_strategy
from load_data import load_data, Article
from forecasting import ExponentialSmoothing, Croston
from inv_strat import InvStratNormal, InvStratCompPois
import multiprocessing
import polars as pl
import sys

GLOBAL_TARGET: float = 0.98


def process_article(args: tuple[Article, float]) -> dict:
    article: Article = args[0]
    min_fill_rate: float = args[1]
    if article.slow_mover:
        model = Croston(article)
        inv_strat: InvStratCompPois = InvStratCompPois(article, model, min_fill_rate)
    else:
        model = ExponentialSmoothing(article)
        inv_strat: InvStratNormal = InvStratNormal(article, model, min_fill_rate)

    backorders: int = 0
    on_hand: int = 0
    order: list[int] = [0] * len(article.dates)  # (arrival_index, quantity)

    forecasts: list[float] = []
    on_hand_list: list[int] = []
    inv_pos_list: list[int] = []
    R_list: list[int] = []
    Q_list: list[int] = []
    total_demand: int = 0
    demand_satisfied_from_stock: int = 0

    in_test_period: bool = False
    first_test_index: int = -1

    for i, current_date in enumerate(article.dates):
        if not in_test_period:
            if current_date in article.test_dates:
                in_test_period = True
                first_test_index = i

        # Receive orders arriving today
        on_hand += order[i]

        # Fullfill backorders
        if backorders > 0 and on_hand > 0:
            fullfilled = min(on_hand, backorders)
            on_hand -= fullfilled
            backorders -= fullfilled

        demand = article.demand[i]

        if demand > 0:
            # Update models only on days with a positive demand,
            # since we never order inventory on a day with 0 demand,
            # since we can not drop below R when there is no demand
            model.update(current_date)
            inv_strat.optimize()

        forecasts.append(model.forecast())

        # Log results from test period
        if in_test_period:
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

        on_hand_list.append(on_hand)
        inv_pos_list.append(inventory_pos)
        R_list.append(inv_strat.R)
        Q_list.append(inv_strat.Q)

    plot_demand(article, forecasts, first_test_index)
    plot_inventory_strategy(
        article,
        on_hand_list,
        inv_pos_list,
        R_list,
        Q_list,
        first_test_index,
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

        raw_results = []
        total_articles = len(args)

        # Use imap_unordered to catch results as they finish
        for i, result in enumerate(pool.imap_unordered(process_article, args), 1):
            raw_results.append(result)

            # Progress bar
            percent = (i / total_articles) * 100
            bar_length = 40
            filled_length = int(bar_length * i // total_articles)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)
            sys.stdout.write(
                f"\rProgress: |{bar}| {percent:.1f}% ({i}/{total_articles})"
            )
            sys.stdout.flush()

        # Convert the collected list of dicts to a df
        results: pl.DataFrame = pl.DataFrame(raw_results)
        results.write_csv("../results/results.csv")

        print(
            f"Global fill rate achieved: {
                sum(results['demand_satisfied_from_stock'])
                / sum(results['total_demand'])
            }"
        )


if __name__ == "__main__":
    main()
