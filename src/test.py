from plot import plot_demand_no_fc
from multiprocessing import Manager, Queue
from load_data import load_data, Article
from main import process_article, global_fill_rate
import polars as pl
import matplotlib.pyplot as plt

results = pl.read_csv("../results/results.csv")

outliers = results.filter(pl.col("achieved_fill_rate") < 0.80)
global_fill_rate(outliers)

results_without_outliers = results.filter(pl.col("achieved_fill_rate") >= 0.80)
results_without_outliers = results_without_outliers.filter(
    pl.col("article_name") != "total"
)
global_fill_rate(results_without_outliers)

plt.hist(results["achieved_fill_rate"])
plt.show()


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
