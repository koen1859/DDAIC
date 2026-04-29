import polars as pl

article_data = pl.read_excel("data/DDAIC_ass_data_2026.xlsx", sheet_name="ArticleData")
demand_data = pl.read_excel("data/DDAIC_ass_data_2026.xlsx", sheet_name="Demand data")
demand_data = demand_data.rename(
    {
        demand_data.columns[i]: str(demand_data[0, i])
        for i in range(2, demand_data.width)
    }
).slice(1)

print(article_data)
print(demand_data)
