from multiprocessing import Manager, Queue
from load_data import load_data, Article
from forecasting import ExponentialSmoothing, Croston
from inv_strat import InvStratNormal, InvStratCompPois
from plot import plot_demand, plot_inventory_strategy
from math import ceil

GLOBAL_TARGET: float = 0.98


def process_article(article: Article, min_fill_rate: float, queue: Queue) -> dict:
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
                if days_since_update > 30:
                    inv_strat.optimize()
                    days_since_update = 0
            else:
                inv_strat.optimize()
        days_since_update += 1

        forecasts.append(model.forecast() / article.demand_multiplier)

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

        if i % 10 == 0:
            queue.put({"id": article.id, "progress": (i / len(article.dates)) * 100})

    plot_demand(article, forecasts, first_test_index)
    plot_inventory_strategy(
        article,
        on_hand_list,
        inv_pos_list,
        R_list,
        Q_list,
        first_test_index,
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
    }


articles: list[Article] = load_data()
slow_movers: list[Article] = [article for article in articles if article.slow_mover]

article_2613: Article | None = None
for article in articles:
    if article.id == 2613:
        article_2613 = article

if article_2613 is not None:
    manager = Manager()
    queue = manager.Queue()
    process_article(article_2613, 0.8, queue)
