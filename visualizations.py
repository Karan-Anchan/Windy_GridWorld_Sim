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

def plot_running_average(q_learning_rewards, sarsa_rewards, expected_sarsa_rewards, window_size=100):
    
    """
    Plot the running average of cumulative rewards for Q-learning, SARSA, and Expected SARSA.

    Args:
        q_learning_rewards (list): Cumulative rewards for Q-learning.
        sarsa_rewards (list): Cumulative rewards for SARSA.
        expected_sarsa_rewards (list): Cumulative rewards for Expected SARSA.
        window_size (int): Window size for computing the running average.
    """
    
    q_learning_running_avg = running_mean(q_learning_rewards, window_size)
    sarsa_running_avg = running_mean(sarsa_rewards, window_size)
    expected_sarsa_running_avg = running_mean(expected_sarsa_rewards, window_size)

    plt.figure(figsize=(12, 6))
    plt.plot(q_learning_running_avg, label='Q-learning')
    plt.plot(sarsa_running_avg, label='SARSA')
    plt.plot(expected_sarsa_running_avg, label='Expected SARSA')
    plt.xlabel('Episode')
    plt.ylabel('Running Average Cumulative Reward')
    plt.title(f'Q-learning vs SARSA vs Expected SARSA: Running Average (Window Size = {window_size})')
    plt.legend()
    plt.grid(True)
    plt.savefig('q_learning_vs_sarsa_vs_expected_sarsa_running_avg.png')
    plt.show()