from load_data import load_data, Article
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

articles: list[Article] = load_data()


def positive_fraction(article):
    if not article.demand:
        return 0
    return sum(d > 0 for d in article.demand) / len(article.demand)


lowest_20 = sorted(articles, key=positive_fraction)[:20]

for i, article in enumerate(lowest_20, start=1):
    frac = positive_fraction(article)
    print(f"{i}: {article.id} fraction={frac:.4f}")

    # Only use positive demands
    data = np.array([d for d in article.demand if d > 0])

    print(f"Article id: {article.id}")
    print(f"Number of positive observations: {len(data)}")
    print(f"Mean: {np.mean(data):.3f}")
    print(f"Variance: {np.var(data):.3f}")

    # Candidate distributions: Poisson, Geometric, Negative Binomial, Logarithmic

    x = np.arange(1, data.max() + 1)

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

    # Geometric fit
    # scipy geometric starts at 1
    # mean = 1/p
    geom_p = 1 / np.mean(data)
    geom_pmf = stats.geom.pmf(x, geom_p)

    plt.plot(
        x,
        geom_pmf,
        "o-",
        label=f"Geometric p={geom_p:.2f}",
    )

    # Negative Binomial fit
    # Method of moments
    # variance = mu + mu^2 / r
    # => r = mu^2 / (var - mu)
    # p = r / (r + mu)
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

    # Logarithmic (log-series) fit
    # Support starts at 1

    # Maximum likelihood estimate of p
    # Negative log-likelihood
    def neg_ll(p):
        if p <= 0 or p >= 1:
            return np.inf
        return -np.sum(stats.logser.logpmf(data, p))

    result = minimize_scalar(
        neg_ll,
        bounds=(1e-6, 0.999999),
        method="bounded",
    )

    logser_p = result.x

    logser_pmf = stats.logser.pmf(x, logser_p)

    plt.plot(
        x,
        logser_pmf,
        "o-",
        label=f"Logarithmic p={logser_p:.2f}",
    )
    plt.xlabel("Demand size")
    plt.ylabel("Probability")
    plt.title(f"Demand distribution fit for article {article.id}")
    plt.legend()
    plt.grid()
    plt.savefig(f"../figures/distr_test_{article.id}.png")
    plt.close()

    # From this we conclude that the geometric or logartithmic distribution are the best fits
    # Logarithmic is much easier to work with so use this one in analysis.
