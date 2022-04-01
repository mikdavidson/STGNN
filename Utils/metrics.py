import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math


def smape(actual, predicted):
    """
    Calculates the SMAPE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        smape - returns smape metric
    """

    return np.mean(abs(predicted - actual) / ((abs(predicted) + abs(actual)) / 2)) * 100

def mse(target, pred):
    """
    Calculates the MSE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        mse - returns MSE metric
    """

    return mean_squared_error(target, pred, squared=True)



def rmse(target, pred):
    """
    Calculates the RMSE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        root - returns RMSE metric
    """

    root = math.sqrt(mse(target, pred))
    return root


def mae(target, pred):
    """
    Calculates the MAE metric
    Parameters:
        actual - target values
        predicted - output values predicted by model
    Returns:
        mae - returns MAE metric
    """
    return mean_absolute_error(target, pred)

