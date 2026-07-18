import os

from windy_gridworld import WindyGridworld
from algorithms import q_learning, sarsa, expected_sarsa, train_q_table
from visualizations import plot_learning_curves, animate_greedy_path


def main():
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    env = WindyGridworld(10, 7, wind, start_state=(0, 3), goal_state=(7, 3))

    episodes, trials = 500, 300
    results = {
        'Q-learning': q_learning(env, episodes=episodes, trials=trials),
        'SARSA': sarsa(env, episodes=episodes, trials=trials),
        'Expected SARSA': expected_sarsa(env, episodes=episodes, trials=trials),
    }

    os.makedirs("images", exist_ok=True)
    plot_learning_curves(results, filename="images/learning_curves.png")
    steps = animate_greedy_path(env, train_q_table(env), filename="images/agent_path.gif")
    print(f"greedy path reaches the goal in {steps} steps")


if __name__ == "__main__":
    main()
