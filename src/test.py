from plot import plot_demand_no_fc
from multiprocessing import Manager, Queue
from load_data import load_data, Article
from main import process_article, global_fill_rate
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

results = pl.read_csv("../results/results.csv")

outliers = results.filter(pl.col("achieved_fill_rate") < 0.80)
global_fill_rate(outliers)

results_without_outliers = results.filter(pl.col("achieved_fill_rate") >= 0.80)
results_without_outliers = results_without_outliers.filter(
    pl.col("article_name") != "total"
)
global_fill_rate(results_without_outliers)

# fill_rates = [fr for fr in results["achieved_fill_rate"] if fr < 0.99]
fill_rates = results["achieved_fill_rate"]
plt.figure(figsize=(10, 6))

plt.hist(
    fill_rates,
    bins=30,
    alpha=0.8,
    edgecolor="black",
)

plt.axvline(
    np.mean(fill_rates),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {np.mean(fill_rates):.3f}",
)

plt.xlabel("Achieved Fill Rate")
plt.ylabel("Number of Articles")
plt.title("Distribution of Achieved Fill Rates")
plt.xlim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("../figures/fill_rate_hist.png")
plt.close()

articles: list[Article] = load_data()
slow_movers: list[Article] = [article for article in articles if article.slow_mover]

manager = Manager()
queue = manager.Queue()
article: Article
for a in articles:
    if a.id == 6524:
        article = a
process_article(article, 0.5, queue)
plot_demand_no_fc(article)

for article in reversed(articles):
    process_article(article, 0.5, queue)
