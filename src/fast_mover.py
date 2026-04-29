import numpy as np


class ExponentialSmoothing:
    def __init__(self, train_demand: list[int]) -> None:
        self.a: float = float(np.mean(train_demand))

    def forecast(self, test_demand: list[int], alpha: float = 0.3) -> list[float]:
        forecast: list[float] = []
        forecast.append(self.a)
        for i in range(1, len(test_demand)):
            forecast.append(alpha * test_demand[i - 1] + (1 - alpha) * forecast[i - 1])
        return forecast


class ExponentialSmoothingWithTrend:
    def __init__(self, demand: list[int]) -> None:
        self.a: float = float(np.mean(demand))
        self.b: float = (demand[-1] - demand[0]) / (len(demand) - 1)

    def forecast(
        self, test_demand: list[int], alpha: float = 0.3, beta: float = 0.3
    ) -> list[float]:
        forecast: list[float] = []
        forecast.append(self.a + self.b)
        for i in range(1, len(test_demand)):
            a_new: float = alpha * test_demand[i - 1] + (1 - alpha) * (
                forecast[i - 1] - self.b
            )
            b_new: float = beta * (a_new - forecast[i - 1]) + (1 - beta) * self.b
            forecast.append(a_new + b_new)
            self.a, self.b = a_new, b_new
        return forecast
