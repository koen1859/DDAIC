import polars as pl

article_data: pl.DataFrame = pl.read_excel(
    "data/DDAIC_ass_data_2026.xlsx", sheet_name="ArticleData"
)
demand_data: pl.DataFrame = pl.read_excel(
    "data/DDAIC_ass_data_2026.xlsx", sheet_name="Demand data", has_header=True
)
demand_data.columns = [demand_data.columns[0], demand_data.columns[1]] + [
    str(demand_data[0, i]) for i in range(2, demand_data.width)
]
demand_data.slice(1)
