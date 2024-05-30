import matplotlib.pyplot as plt
import numpy as np

def running_mean(data, window_size):
    
    """
    Compute the running mean of a 1D array.

    Args:
        data (list): Input data.
        window_size (int): Window size for the running mean.

    Returns:
        list: Running mean of the input data.
    """
    
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
