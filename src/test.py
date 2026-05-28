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
    if a.id == 6564:
        article = a
process_article(article, 0.9, queue)

for article in reversed(articles):
    print(f"Processing for article {article.id}")
    result = process_article(article, 0.9, queue)
    print(f"Achieved fill rate: {result['achieved_fill_rate']}")
