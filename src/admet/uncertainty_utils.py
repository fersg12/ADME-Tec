import numpy as np


def uncertainty_gaussian(x, mu, sigma):
    return ((x - mu)**2 / sigma**4) * np.exp(-((x - mu)**2) / (2 * sigma**2))


def uncertainty_trapezoidal(x, a, b, c, d):
    inc = np.zeros_like(x, dtype=float)
    inc += ((x >= a) & (x < b)) * (1 / (b - a))
    inc += ((x > c) & (x <= d)) * (1 / (d - c))
    inc += ((x < a) | (x > d)) * 1.0
    return inc


def uncertainty_sigmoidal(x, a, b):
    k = 10 / (b - a)
    x0 = (a + b) / 2
    fx = 1 / (1 + np.exp(k * (x - x0)))
    return fx * (1 - fx)

UNCERTAINTY_FUNCTIONS = {
    "ugaussian": uncertainty_gaussian,
    "utrapezoidal": uncertainty_trapezoidal,
    "usigmoidal": uncertainty_sigmoidal
}


