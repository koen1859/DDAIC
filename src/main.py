from plot import plot_demand, plot_inventory_strategy
from load_data import load_data, Article
from forecasting import ExponentialSmoothing, Croston
from inv_strat import InvStratNormal
import multiprocessing

GLOBAL_TARGET: float = 0.98


def process_article(args: tuple[Article, float]):
    article: Article = args[0]
    min_fill_rate: float = args[1]
    print(f"Forecasting for article {article.name}")
    if article.slow_mover:
        model = Croston(article)
    else:
        model = ExponentialSmoothing(article)
    inv_strat: InvStratNormal = InvStratNormal(article, model, min_fill_rate)

    on_hand: int = 0
    backorders: int = 0
    order: list[int] = [0] * len(article.test_dates)  # (arrival_index, quantity)

    on_hand_hist: list[int] = []
    inv_pos_hist: list[int] = []
    R_hist: list[int] = []
    Q_hist: list[int] = []

    # Loop over the test periods' dates
    for i, current_date in enumerate(article.test_dates):
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
        R, Q = inv_strat.optimize_RQ()

        demand = article.test_demand[i]

        if on_hand >= demand:
            on_hand -= demand
        else:
            backorders += demand - on_hand
            on_hand = 0

        on_order = sum(order[i:])  # orders arriving later
        inventory_pos = on_hand + on_order - backorders

        if inventory_pos <= R:
            order_qty = max(Q, int(article.min_order_quantity))
            arrival_t = i + article.lead_time
            if arrival_t < len(order):
                order[arrival_t] = order_qty

        on_hand_hist.append(on_hand)
        inv_pos_hist.append(inventory_pos)
        R_hist.append(R)
        Q_hist.append(Q)

    plot_demand(article, model.forecasts)
    plot_inventory_strategy(
        article, article.test_dates, on_hand_hist, inv_pos_hist, R_hist, Q_hist
    )


articles: list[Article] = load_data()

with multiprocessing.Pool() as pool:
    min_fill_rate: float = 0.80
    args = [(a, min_fill_rate) for a in articles]
    pool.map(process_article, args)
