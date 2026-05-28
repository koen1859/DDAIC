from multiprocessing import Manager, Queue
from load_data import load_data, Article
from main import process_article

GLOBAL_TARGET: float = 0.98


articles: list[Article] = load_data()
slow_movers: list[Article] = [article for article in articles if article.slow_mover]

manager = Manager()
queue = manager.Queue()
process_article(articles[0], 0.8, queue)
