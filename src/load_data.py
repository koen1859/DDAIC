import polars as pl
from datetime import date, datetime
import multiprocessing


def sanitize_name(name: str) -> str:
    invalid_chars = r'\/:*?"<>| '
    for c in invalid_chars:
        name = name.replace(c, "_")
    return name


class Article:
    def __init__(self, info_row: list, demand_row: list, dates: list) -> None:
        # Check if rows correspond
        if int(info_row[0]) != int(demand_row[0]) or str(info_row[1]) != str(
            demand_row[1]
        ):
            print("Article ids or names do not match")
            return

        # Load the basic article information
        self.id: int = int(info_row[0])
        self.name: str = str(info_row[2])  # English name
        self.target_sl_given: float = float(info_row[3])
        self.min_order_quantity: float = float(info_row[4])
        self.sales_price: float = float(info_row[5])
        self.lead_time: int = int(info_row[6])

        # Load the daily demands
        self.dates: list[date] = [
            datetime.strptime(d, "%Y%m%d").date() for d in dates[2:]
        ]
        self.demand: list[int] = [int(d) for d in demand_row[2:]]

        # Remove leading zeros for products only in assortment from later on
        first_nonzero = next(
            (i for i, d in enumerate(self.demand) if d != 0), len(self.demand)
        )
        self.demand = self.demand[first_nonzero:]
        self.dates = self.dates[first_nonzero:]

        # We initialize methods on 2015-2017 and test on 2018
        self.train_dates = [d for d in self.dates if d.year < 2018]
        self.test_dates = [d for d in self.dates if d.year == 2018]
        self.train_demand = self.demand[: len(self.train_dates)]
        self.test_demand = self.demand[len(self.train_dates) :]

        # Find if slow mover or not (0 demand more than 50% of time)
        # This is just a random metric I thought of maybe it works
        self.slow_mover: bool = (
            (sum(1 for d in self.train_demand if d == 0) / len(self.train_demand)) > 0.5
            if self.demand
            else True
        )

    # Method to print class for debugging
    def __str__(self):
        return (
            f"Article(id={self.id}, name='{self.name}', target_sl_given={self.target_sl_given}, "
            f"min_order_quantity={self.min_order_quantity}, sales_price={self.sales_price}, "
            f"lead_time={self.lead_time}, slow_mover={self.slow_mover})"
        )


def process_article(args):
    info_row, demand_row, dates = args
    any_demand = any(int(d) != 0 for d in demand_row[2:])
    if not any_demand:
        return None
    return Article(info_row, demand_row, dates)


def load_data() -> list[Article]:
    article_data = pl.read_excel(
        "../data/DDAIC_ass_data_2026.xlsx", sheet_name="ArticleData"
    )
    demand_data = pl.read_excel(
        "../data/DDAIC_ass_data_2026.xlsx", sheet_name="Demand data"
    )
    demand_data = demand_data.rename(
        {
            demand_data.columns[i]: str(demand_data[0, i])
            for i in range(2, demand_data.width)
        }
    ).slice(1)

    dates = list(demand_data.columns)
    args_list = [
        (list(article_data.row(i)), list(demand_data.row(i)), dates)
        for i in range(demand_data.shape[0])
    ]

    # Store the data in list of Article
    with multiprocessing.Pool() as pool:
        results = list(pool.map(process_article, args_list))

    articles = [a for a in results if a is not None]
    return articles
