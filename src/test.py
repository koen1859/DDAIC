from multiprocessing import Manager, Queue
from load_data import load_data, Article
from main import process_article

GLOBAL_TARGET: float = 0.98


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
