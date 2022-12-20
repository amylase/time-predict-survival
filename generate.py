import math
import random
from time_model import fit


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