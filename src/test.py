from main import global_fill_rate
import polars as pl
import matplotlib.pyplot as plt

results = pl.read_csv("../results/results_fr_diff_0999.csv")

outliers = results.filter(pl.col("achieved_fill_rate") < 0.85)
global_fill_rate(outliers)

results_without_outliers = results.filter(pl.col("achieved_fill_rate") >= 0.85)
results_without_outliers = results_without_outliers.filter(
    pl.col("article_name") != "total"
)
global_fill_rate(results_without_outliers)
len(outliers) / (len(results) - 1)

fill_rates = [fr for fr in results["achieved_fill_rate"] if fr < 0.98]
# fill_rates = results["achieved_fill_rate"]
plt.figure(figsize=(10, 6))
plt.hist(
    fill_rates,
    bins=30,
    alpha=0.8,
    edgecolor="black",
)
plt.xlabel("Achieved Fill Rate")
plt.ylabel("Number of Articles")
plt.title("Distribution of Achieved Fill Rates")
plt.xlim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("../figures/fill_rate_hist_fr_diff_098.png")
plt.close()
