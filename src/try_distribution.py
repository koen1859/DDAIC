from load_data import load_data, Article
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

articles: list[Article] = load_data()


def positive_fraction(article):
    if not article.demand:
        return 0
    return sum(d > 0 for d in article.demand) / len(article.demand)


lowest_20 = sorted(articles, key=positive_fraction)[:20]

for i, article in enumerate(lowest_20, start=1):
    frac = positive_fraction(article)
    print(f"{i}: {article.id} fraction={frac:.4f}")

# I think that article with id 2505 looks good to try to fit a distribution of demand size per customer
# This is the one with the 10th lowest fraction of days with positive demand
article_2505: Article = lowest_20[-1]

# Only use positive demands
data = np.array([d for d in article.demand if d > 0])

print(f"Article id: {article.id}")
print(f"Number of positive observations: {len(data)}")
print(f"Mean: {np.mean(data):.3f}")
print(f"Variance: {np.var(data):.3f}")

# Candidate distributions: Poisson, Geometric, Negative Binomial

x = np.arange(0, data.max() + 1)

# Empirical histogram
plt.figure(figsize=(10, 6))
plt.hist(
    data,
    bins=np.arange(data.max() + 2) - 0.5,
    density=True,
    alpha=0.5,
    label="Empirical",
)

# Poisson fit
poisson_lambda = np.mean(data)
poisson_pmf = stats.poisson.pmf(x, poisson_lambda)

plt.plot(
    x,
    poisson_pmf,
    "o-",
    label=f"Poisson λ={poisson_lambda:.2f}",
)

# -------------------
# Geometric fit
# scipy geometric starts at 1
# mean = 1/p
# -------------------
geom_p = 1 / np.mean(data)
geom_pmf = stats.geom.pmf(x, geom_p)

plt.plot(
    x,
    geom_pmf,
    "o-",
    label=f"Geometric p={geom_p:.2f}",
)

# -------------------
# Negative Binomial fit
# Method of moments
# variance = mu + mu^2 / r
# => r = mu^2 / (var - mu)
# p = r / (r + mu)
# -------------------
mu = np.mean(data)
var = np.var(data)

if var > mu:
    r = mu**2 / (var - mu)
    p = r / (r + mu)

    nbinom_pmf = stats.nbinom.pmf(x, r, p)

    plt.plot(
        x,
        nbinom_pmf,
        "o-",
        label=f"NegBin r={r:.2f}, p={p:.2f}",
    )
else:
    print("Variance <= mean, Negative Binomial not appropriate")

plt.xlabel("Demand size")
plt.ylabel("Probability")
plt.title(f"Demand distribution fit for article {article.id}")
plt.legend()
plt.grid()
plt.show()


# From this we conclude that the geometric distribution is the best fit
