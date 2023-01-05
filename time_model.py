import math
import random
from typing import List, Tuple
import statistics


def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    d = x - mu
    t = d / (math.sqrt(2) * sigma)
    return math.exp(-t**2) / (math.sqrt(2 * math.pi) * sigma)


def normal_pdf_grad_mu(x: float, mu: float = 0, sigma: float = 1) -> float:
    # d pdf / d mu
    d = x - mu
    t = d / (math.sqrt(2) * sigma)
    return (d * math.exp(-t**2)) / (math.sqrt(2 * math.pi) * sigma**3)


def normal_pdf_grad_sigma(x: float, mu: float = 0, sigma: float = 1) -> float:
    # d pdf / d sigma
    pdf = normal_pdf(x, mu, sigma)
    return (x-mu)**2 * pdf / sigma**3 - pdf / sigma


def normal_sf(x: float, mu: float = 0, sigma: float = 1) -> float:
    t = (x - mu) / (math.sqrt(2) * sigma)
    return (math.erfc(t)) / 2


def normal_sf_grad_mu(x: float, mu: float = 0, sigma: float = 1) -> float:
    # d sf / d mu 
    t = (x - mu) / (math.sqrt(2) * sigma)
    return math.exp(-t**2) / (math.sqrt(2 * math.pi) * sigma)


def normal_sf_grad_sigma(x: float, mu: float = 0, sigma: float = 1) -> float:
    # d sf / d sigma
    t = (x - mu) / (math.sqrt(2) * sigma)
    return math.exp(-t**2) * math.sqrt(2) * (x - mu) / (math.sqrt(math.pi) * sigma**2)


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

    uncensored_ratings, uncensored_logtimes = [], []
    for rating, time, censored in zip(ratings, times, censoreds):
        if not censored:
            uncensored_ratings.append(rating)
            uncensored_logtimes.append(math.log(time))
    slope, intercept = single_regression(uncensored_ratings, uncensored_logtimes)

    uncensored_predict_errors = []
    for rating, logtime in zip(uncensored_ratings, uncensored_logtimes):
        uncensored_predict_errors.append(rating * slope + intercept - logtime)
    sigma = statistics.stdev(uncensored_predict_errors)

    # todo: may need to optimize hyperparameters
    lr_slope, lr_intercept, lr_sigma = 0.01, 1, 0.3
    eps = 10 ** -10
    r_slope, r_intercept, r_sigma = eps, eps, eps
    for _iter in range(100):
        orders = list(range(n_items))
        rng.shuffle(orders)
        for order in orders:
            rating = ratings[order]
            logtime = math.log(times[order])
            censored = censoreds[order]
            mu = slope * rating + intercept
            if censored:
                sf = normal_sf(logtime, mu, sigma)
                gm = normal_sf_grad_mu(logtime, mu, sigma)
                grad = 0 if gm == 0 else gm / sf

                gs = normal_sf_grad_sigma(logtime, mu, sigma)
                g_sigma = 0 if gs == 0 else gs / sf
            else:
                pdf = normal_pdf(logtime, mu, sigma)
                gm = normal_pdf_grad_mu(logtime, mu, sigma)
                grad = 0 if gm == 0 else gm / pdf

                gs = normal_pdf_grad_sigma(logtime, mu, sigma)
                g_sigma = 0 if gs == 0 else gs / pdf
            g_slope = grad * rating
            g_intercept = grad
            r_slope += g_slope ** 2
            r_intercept += g_intercept ** 2
            r_sigma += g_sigma ** 2
            slope += g_slope * lr_slope * (r_slope ** -0.5)
            intercept += g_intercept * lr_intercept * (r_intercept ** -0.5)
            sigma += g_sigma * lr_sigma * (r_sigma ** -0.5)

            slope = min(-1e-6, slope)
            sigma = max(eps, sigma)
    return slope, intercept