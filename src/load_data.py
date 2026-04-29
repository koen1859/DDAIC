import polars as pl
from datetime import date, datetime


def sanitize_name(name: str) -> str:
    invalid_chars = r'\/:*?"<>|'
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
        self.name: str = sanitize_name(str(info_row[2]))  # English name
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

        # Find if slow mover or not (0 demand more than 50% of time)
        # This is just a random metric I thought of maybe it works
        self.slow_mover: bool = (
            (sum(1 for d in self.demand if d == 0) / len(self.demand)) > 0.5
            if self.demand
            else True
        )

        # We initialize methods on 2015-2017 and test on 2018
        self.train_dates = [d for d in self.dates if d.year < 2018]
        self.test_dates = [d for d in self.dates if d.year == 2018]
        self.train_demand = self.demand[: len(self.train_dates)]
        self.test_demand = self.demand[len(self.train_dates) :]

    # Method to print class for debugging
    def __str__(self):
        return (
            f"Article(id={self.id}, name='{self.name}', target_sl_given={self.target_sl_given}, "
            f"min_order_quantity={self.min_order_quantity}, sales_price={self.sales_price}, "
            f"lead_time={self.lead_time}, slow_mover={self.slow_mover})"
        )


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

    # Store the data in list of Article
    articles: list[Article] = []
    for row_id in range(demand_data.shape[0]):
        info_row = list(article_data.row(row_id))
        demand_row = list(demand_data.row(row_id))

        # Filter out any articles that never get any demand
        any_demand: bool = False
        for d in demand_row[2:]:
            # If any is nonzero set to true and stop loop
            if int(d) != 0:
                any_demand = True
                break
        # If no demand over entire period, just drop the article
        if not any_demand:
            continue

        articles.append(Article(info_row, demand_row, list(demand_data.columns)))
    return articles
