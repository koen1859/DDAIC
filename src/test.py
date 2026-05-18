from load_data import Article, load_data
from main import process_article

articles: list[Article] = load_data()
slow_movers: list[Article] = [article for article in articles if article.slow_mover]

process_article((slow_movers[0], 0.8))
