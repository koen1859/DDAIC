"""Run forecasting + (R,Q) inventory experiments.

Drop this file in src/ together with inventory.py. Run from src/:
    python run_rq_experiment.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from load_data import load_data, Article
from forecasting import Croston, ExponentialSmoothing, ExponentialSmoothingWithTrend
from inventory import aggregate_fill_rate, make_forecasts, order_quantity, simulate_rq


def choose_model(article: Article):
    return Croston if article.slow_mover else ExponentialSmoothing


def evaluate_for_minimum(min_fill_rate: float, global_target: float = 0.98):
    articles = load_data()

    # First pass: all SKUs get at least the selected minimum.
    # For a clean baseline, use the same target for all SKUs. If total fill is too low,
    # raise all targets by binary search until the aggregate unit fill rate reaches 98%.
    low, high = min_fill_rate, 0.999
    best_results = None
    best_target = high

    for _ in range(12):
        sku_target = (low + high) / 2.0
        results = []
        for article in articles:
            forecasts = make_forecasts(article, choose_model(article), update_every=7)
            q = order_quantity(article, min_days_of_supply=0)  # default: MOQ
            results.append(simulate_rq(article, forecasts, target_fill_rate=sku_target, q=q))

        total_fr = aggregate_fill_rate(results)
        if total_fr >= global_target:
            best_results = results
            best_target = sku_target
            high = sku_target
        else:
            low = sku_target

    return best_target, best_results or results


def main():
    out_dir = Path("../results")
    out_dir.mkdir(exist_ok=True)

    summary_rows = []
    for min_fr in [0.50, 0.60, 0.70, 0.80]:
        sku_target, results = evaluate_for_minimum(min_fr)
        total_fr = aggregate_fill_rate(results)
        avg_stock = sum(r.avg_on_hand for r in results)
        orders = sum(r.orders_placed for r in results)
        units_ordered = sum(r.units_ordered for r in results)

        summary_rows.append({
            "minimum_fill_rate": min_fr,
            "common_sku_target_used": sku_target,
            "realized_total_fill_rate": total_fr,
            "sum_avg_on_hand": avg_stock,
            "orders_placed": orders,
            "units_ordered": units_ordered,
        })

        with open(out_dir / f"rq_sku_results_min_{int(min_fr*100)}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].as_dict().keys()))
            writer.writeheader()
            writer.writerows(r.as_dict() for r in results)

    with open(out_dir / "rq_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
