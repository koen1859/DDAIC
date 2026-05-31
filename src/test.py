from plot import plot_demand_no_fc
from multiprocessing import Manager, Queue
from load_data import load_data, Article
from main import process_article

GLOBAL_TARGET: float = 0.98


articles: list[Article] = load_data()
slow_movers: list[Article] = [article for article in articles if article.slow_mover]

manager = Manager()
queue = manager.Queue()
article: Article
for a in articles:
    if a.id == 7096:
        article = a
plot_demand_no_fc(article)
