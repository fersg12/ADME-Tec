import numpy as np

def gaussian_desirability(x, mu, sigma):
    """
    Computes the Gaussian desirability function.
    
    Parameters:
        x (float or np.ndarray): Input value(s).
        mu (float): Mean of the Gaussian function.
        sigma (float): Standard deviation of the Gaussian function.
        
    Returns:
        float or np.ndarray: Gaussian desirability score(s).
    """
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def trapezoidal_desirability_with_floor(x, a, b, c, d, floor=0.05):
    """
    Computes a trapezoidal desirability function with a minimum floor value.
    
    Parameters:
        x (float or np.ndarray): Input value(s).
        a, b, c, d (float): Shape parameters defining the trapezoid.
        floor (float): Minimum desirability value to avoid returning zero.
        
    Returns:
        float or np.ndarray: Trapezoidal desirability score(s).
    """
    value = np.where(
        x < a, 0,
        np.where(x < b, (x - a) / (b - a),
        np.where(x <= c, 1,
        np.where(x < d, (d - x) / (d - c), 0))))
    
    return np.maximum(value, floor)


def compute_trapezoidal_ranges(series, margin=0.05):
    """
    Computes the trapezoidal shape parameters (a, b, c, d) from a data series.
    
    Parameters:
        series (pd.Series): Data series from which to derive the quantiles.
        margin (float): Extra margin to extend the 'a' and 'd' values.
        
    Returns:
        tuple: (a, b, c, d) with adjusted margins.
    """
    a, b, c, d = series.quantile([0.05, 0.25, 0.75, 0.95])
    range_ = d - a
    return a - margin * range_, b, c, d + margin * range_


def desirability_increasing(x, a, b, probabilistic=False):
    """
    Computes an increasing desirability function.
    It can be linear or logistic depending on the `probabilistic` flag.
    
    Parameters:
        x (float or np.ndarray): Input value(s).
        a (float): Minimum threshold (lower bound).
        b (float): Maximum threshold (upper bound).
        probabilistic (bool): If True, use logistic curve; otherwise, use linear scale.
        
    Returns:
        float or np.ndarray: Increasing desirability score(s).
    """
    if probabilistic:
        k = 10 / (b - a)  # Slope of the logistic function
        x0 = (a + b) / 2  # Midpoint
        return 1 / (1 + np.exp(-k * (x - x0)))
    else:
        return np.clip((x - a) / (b - a), 0, 1)


def desirability_decreasing(x, a, b, probabilistic=False):
    """
    Computes a decreasing desirability function.
    It can be linear or logistic depending on the `probabilistic` flag.
    
    Parameters:
        x (float or np.ndarray): Input value(s).
        a (float): Minimum threshold (lower bound).
        b (float): Maximum threshold (upper bound).
        probabilistic (bool): If True, use logistic curve; otherwise, use linear scale.
        
    Returns:
        float or np.ndarray: Decreasing desirability score(s).
    """
    if probabilistic:
        k = 10 / (b - a)
        x0 = (a + b) / 2
        return 1 / (1 + np.exp(k * (x - x0)))
    else:
        return np.clip((b - x) / (b - a), 0, 1)



