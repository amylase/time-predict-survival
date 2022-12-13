import math
import random
from typing import List, Tuple


def normal_pdf_grad(x: float, mu: float = 0, sigma: float = 1) -> float:
    # d pdf / d mu
    d = x - mu
    t = d / (math.sqrt(2) * sigma)
    return (d * math.exp(-t**2)) / (math.sqrt(2 * math.pi) * sigma**3)


def normal_sf_grad(x: float, mu: float = 0, sigma: float = 1) -> float:
    # d sf / d mu 
    t = (x - mu) / (math.sqrt(2) * sigma)
    return math.exp(-t**2) / (math.sqrt(2 * math.pi) * sigma)


def single_regression(x: List[float], y: List[float]) -> Tuple[float, float]:
    n = len(x)
    x_sum = sum(x)
    y_sum = sum(y)
    xy_sum = sum(x * y for x, y in zip(x, y))
    sqx_sum = sum(x ** 2 for x in x)
    slope = (n * xy_sum - x_sum * y_sum) / (n * sqx_sum - x_sum ** 2)
    intercept = (sqx_sum * y_sum - xy_sum * x_sum) / (n * sqx_sum - x_sum ** 2)
    return slope, intercept


def fit(ratings: List[float], times: List[float], censoreds: List[bool], rng: random.Random) -> Tuple[float, float]:
    n_items = len(ratings)

    # use uncensored data to determine initial estimate
    uncensored_ratings, uncensored_logtimes = [], []
    for rating, time, censored in zip(ratings, times, censoreds):
        if not censored:
            uncensored_ratings.append(rating)
            uncensored_logtimes.append(math.log(time))
    slope, intercept = single_regression(uncensored_ratings, uncensored_logtimes)
    # todo: estimate variance or use known value first.

    lr_slope, lr_intercept = 0.0000001, 0.001
    for _iter in range(100):
        orders = list(range(n_items))
        rng.shuffle(orders)
        for order in orders:
            rating = ratings[order]
            logtime = math.log(times[order])
            censored = censoreds[order]
            g_slope, g_intercept = 0, 0
            mu = slope * rating + intercept
            if censored:
                grad = normal_sf_grad(logtime, mu)
            else:
                grad = normal_pdf_grad(logtime, mu)
            g_slope += grad * rating
            g_intercept += grad
            slope += g_slope * lr_slope
            intercept += g_intercept * lr_intercept

            slope = min(-1e-6, slope)
        lr_slope *= 0.9
        lr_intercept *= 0.9
    return slope, intercept


def estimate_difficulty(slope, intercept):
    expected_solve_time = 3600
    # solve rating * slope + intercept = log(expected_solve_time)
    return (math.log(expected_solve_time) - intercept) / slope


def main():
    rng = random.Random(1)
    n_contestants = 3000

    # generate contestant ratings
    ratings = [rng.randrange(0, 4000) for _ in range(n_contestants)]

    # generate actual solve time
    slope, intercept = -0.001, 10
    times = [rng.lognormvariate(rating * slope + intercept, 1) for rating in ratings]
    # for rating, time in zip(ratings, times):
    #     print(rating, time)

    for threshold in range(600, 7201, 600):
        censored_times = [min(time, threshold) for time in times]
        censoreds = [time >= threshold for time in times]
        e_slope, e_intercept = fit(ratings, censored_times, censoreds, rng)
        print(f"threshold = {threshold} (censored {sum(censoreds)} items): slope = {e_slope}, intercept = {e_intercept}, estimated difficulty={estimate_difficulty(e_slope, e_intercept)}")
    print(f"actually difficulty: {estimate_difficulty(slope, intercept)}")


if __name__ == '__main__':
    main()