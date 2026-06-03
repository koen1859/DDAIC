from load_data import load_data, Article
from forecasting import ExponentialSmoothing, Croston, ExponentialSmoothingWithTrend
from inv_strat import InvStratNormal, InvStratCompPois
from plot import plot_demand, plot_inventory_strategy

from statistics import mean, stdev
from math import ceil
import polars as pl
from multiprocessing import Pool, Manager, Queue
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich.console import Group
import time

GLOBAL_TARGET: float = 0.98
INITIAL_TARGET: float = 0.5
MIN_FILL_RATE: float = 0.5
MAX_TARGET: float = 0.99999999999999999999


def process_article(article: Article, min_fill_rate: float, queue: Queue) -> dict:
    if article.slow_mover:
        model = Croston(article)
        model_name = "Croston"
        inv_strat: InvStratCompPois = InvStratCompPois(article, model, min_fill_rate)
    else:
        if (
            article.seasonal and len(article.train_demand) > 365 * 2
        ):  # Requires at least two cycles
            model = ExponentialSmoothingWithTrend(article, periods_per_year=12)
            model_name = "Exponential Smoothing with Trend"
        else:
            model = ExponentialSmoothing(article)
            model_name = "Exponential Smoothing"
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

    holding_cost: float = 0.0

    in_test_period: bool = False
    first_test_index: int = -1
    days_since_update: int = 0

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

        demand = article.true_demand[i]

        if demand > 0:
            # Update models only on days with a positive demand,
            # since we never order inventory on a day with 0 demand,
            # since we can not drop below R when there is no demand
            model.update(current_date)
            if article.slow_mover:
                # only update R,Q for slow movers at most once a month since otherwise very slow
                if days_since_update > 7:
                    inv_strat.optimize()
                    days_since_update = 0
                days_since_update += 1
            else:
                inv_strat.optimize()

        forecasts.append(model.forecast(1) / article.demand_multiplier)

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

        if in_test_period:
            holding_cost += on_hand * article.holding_price

        if i % 10 == 0:
            queue.put({"id": article.id, "progress": (i / len(article.dates)) * 100})

    plot_demand(article, forecasts, first_test_index, model_name)
    plot_inventory_strategy(
        article,
        on_hand_list,
        inv_pos_list,
        R_list,
        Q_list,
        first_test_index,
        demand_satisfied_from_stock / total_demand,
    )

    queue.put({"id": article.id, "done": True})

    return {
        "article_id": article.id,
        "article_name": article.name,
        "total_demand": total_demand,
        "demand_satisfied_from_stock": demand_satisfied_from_stock,
        "achieved_fill_rate": demand_satisfied_from_stock / total_demand,
        "target_fill_rate": min_fill_rate,
        "slow_mover": article.slow_mover,
        "holding_cost": holding_cost,
    }


def article_fill_rate(row) -> float:
    return row["demand_satisfied_from_stock"] / row["total_demand"]


def global_fill_rate(results: pl.DataFrame) -> float:
    return sum(results["demand_satisfied_from_stock"]) / sum(results["total_demand"])


def unmet_demand(row) -> float:
    return row["total_demand"] - row["demand_satisfied_from_stock"]


def initial_optimization(articles: list[Article], article_targets: dict):
    manager = Manager()
    queue = manager.Queue()

    active_tasks = {}

    main_progress = Progress(
        TextColumn(
            f"[bold blue]Initial Optimization with target fill rate of {INITIAL_TARGET}"
        ),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
    )
    task_id = main_progress.add_task("Total", total=len(articles))

    with Live(main_progress, refresh_per_second=10) as live:
        with Pool() as pool:
            raw_results = [
                pool.apply_async(
                    process_article, (article, article_targets[article.id], queue)
                )
                for article in articles
            ]

            completed: int = 0
            while completed < len(articles):
                while not queue.empty():
                    msg = queue.get()
                    if "done" in msg:
                        completed += 1
                        main_progress.update(task_id, advance=1)
                        if msg["id"] in active_tasks:
                            del active_tasks[msg["id"]]
                    else:
                        active_tasks[msg["id"]] = msg["progress"]

                table = Table(title="Active Workers")
                table.add_column("Article ID")
                table.add_column("Progress")
                for art_id, prog in active_tasks.items():
                    table.add_row(str(art_id), f"{prog:.1f}%")

                live.update(Group(main_progress, table))
                time.sleep(0.1)

    return raw_results


def iterative_optimization(
    articles: list[Article],
    article_targets: dict,
    results_dict: dict,
    current_global: float,
):
    manager = Manager()
    queue = manager.Queue()
    active_tasks = {}

    with Live(refresh_per_second=10) as live:
        with Pool() as pool:
            iteration: int = 0
            while current_global < GLOBAL_TARGET:
                iteration += 1

                candidate_rows = [
                    row
                    for row in results_dict.values()
                    if abs(row["target_fill_rate"] - MAX_TARGET) > 1e-9
                    and row["achieved_fill_rate"] < 1.0
                ]

                if not candidate_rows:
                    print("No more articles can be improved.")
                    break

                # Find 160 worst articles to reprocess
                worst_rows = sorted(candidate_rows, key=unmet_demand, reverse=True)[
                    :160
                ]
                worst_articles: list[Article] = []
                for row in worst_rows:
                    id = row["article_id"]

                    # Increase target for this article
                    article_targets[id] = min(
                        (article_targets[id] + MAX_TARGET) / 2,
                        MAX_TARGET,
                    )

                    # Find original article object
                    article = next(a for a in articles if a.id == id)
                    worst_articles.append(article)

                raw_results = [
                    pool.apply_async(
                        process_article, (article, article_targets[article.id], queue)
                    )
                    for article in worst_articles
                ]

                completed: int = 0
                while completed < len(worst_articles):
                    while not queue.empty():
                        msg = queue.get()
                        if "done" in msg:
                            completed += 1
                            if msg["id"] in active_tasks:
                                del active_tasks[msg["id"]]
                        else:
                            active_tasks[msg["id"]] = msg["progress"]

                    main_progress = Progress(
                        TextColumn(
                            f"[bold blue]Iterative Improvement - Iteration {iteration} - "
                            f"Global Fill Rate: {current_global:.4f} / {GLOBAL_TARGET:.4f}"
                        ),
                    )
                    main_progress.add_task("", total=1)

                    table = Table(title="Active Workers")
                    table.add_column("Article ID")
                    table.add_column("Progress")
                    for art_id, prog in active_tasks.items():
                        table.add_row(str(art_id), f"{prog:.1f}%")

                    live.update(Group(main_progress, table))
                    time.sleep(0.1)

                for res in raw_results:
                    result = res.get()
                    results_dict[result["article_id"]] = result

                current_global = global_fill_rate(
                    pl.DataFrame(list(results_dict.values()))
                )

    return current_global, results_dict


def our_iterative_method(articles: list[Article]) -> pl.DataFrame:
    article_targets = {article.id: INITIAL_TARGET for article in articles}

    raw_results = initial_optimization(articles, article_targets)

    results_dict = {res.get()["article_id"]: res.get() for res in raw_results}
    results: pl.DataFrame = pl.DataFrame([res.get() for res in raw_results])
    current_global: float = global_fill_rate(results)

    print(
        f"\nGlobal fill rate achieved after initial optimiztion: {current_global:.4f}"
    )

    current_global, results_dict = iterative_optimization(
        articles, article_targets, results_dict, current_global
    )
    results: pl.DataFrame = pl.DataFrame(list(results_dict.values()))

    print(f"\nReached global target: {current_global:.4f}")
    total_row: pl.DataFrame = pl.DataFrame(
        {
            "article_id": 0,
            "article_name": "total",
            "total_demand": sum(results["total_demand"]),
            "demand_satisfied_from_stock": sum(results["demand_satisfied_from_stock"]),
            "achieved_fill_rate": global_fill_rate(results),
            "target_fill_rate": mean(results["target_fill_rate"]),
            "slow_mover": False,
            "holding_cost": sum(results["holding_cost"]),
        }
    )
    results = pl.concat([results, total_row])

    return results


def fill_rate_differentiation(articles: list[Article]) -> dict[int, float]:
    stats = [
        (
            article,
            mean(article.train_demand),
            stdev(article.train_demand),
        )
        for article in articles
    ]

    stats.sort(
        key=lambda x: x[2] * x[0].holding_price / x[1],
        reverse=True,
    )

    weights = [sigma * article.holding_price for article, mu, sigma in stats]

    suffix = [0.0] * (len(weights) + 1)

    for i in range(len(weights) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + weights[i]

    A = (1 - GLOBAL_TARGET) * sum(mu for _, mu, _ in stats)

    article_targets = {}

    for k, (article, mu, sigma) in enumerate(stats):
        S2 = max(
            1 - (A * sigma * article.holding_price) / (mu * suffix[k]),
            MIN_FILL_RATE,
        )

        article_targets[article.id] = S2

        A -= mu * (1 - S2)

    return article_targets


def run_with_fill_rate_differentiation(articles: list[Article]) -> pl.DataFrame:
    article_targets = fill_rate_differentiation(articles)
    raw_results = initial_optimization(articles, article_targets)

    results: pl.DataFrame = pl.DataFrame([res.get() for res in raw_results])
    total_row: pl.DataFrame = pl.DataFrame(
        {
            "article_id": 0,
            "article_name": "total",
            "total_demand": sum(results["total_demand"]),
            "demand_satisfied_from_stock": sum(results["demand_satisfied_from_stock"]),
            "achieved_fill_rate": global_fill_rate(results),
            "target_fill_rate": mean(results["target_fill_rate"]),
            "slow_mover": False,
            "holding_cost": sum(results["holding_cost"]),
        }
    )
    results = pl.concat([results, total_row])
    return results


def main():
    articles: list[Article] = load_data()

    # results_our_method: pl.DataFrame = our_iterative_method(articles)
    # results_our_method.write_csv("../results/results.csv")

    results_teunter_method: pl.DataFrame = run_with_fill_rate_differentiation(articles)
    results_teunter_method.write_csv("../results/results_fr_diff.csv")


if __name__ == "__main__":
    main()
