import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def running_mean(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_learning_curves(results, window_size=10, filename=None):
    """Plot running-average cumulative reward per episode for each control method."""
    plt.figure(figsize=(10, 5))
    for label, rewards in results.items():
        plt.plot(running_mean(rewards, window_size), label=label, lw=1.5)
    plt.xlabel('Episode')
    plt.ylabel(f'Cumulative reward ({window_size}-episode running avg)')
    plt.title('Windy Gridworld — TD control methods')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=110)
    plt.show()


def animate_greedy_path(env, Q, filename=None, max_steps=60):
    """Follow the greedy policy from start to goal and animate the path over the wind field."""
    path, state = [env.start_state], env.start_state
    for _ in range(max_steps):
        if state == env.goal_state:
            break
        state, _ = env.step(state, max(Q[state], key=Q[state].get))
        path.append(state)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(np.tile(env.wind, (env.height, 1)).astype(float), cmap='Blues', alpha=0.7)
    ax.text(*env.start_state, 'S', ha='center', va='center', color='k', fontweight='bold')
    ax.text(*env.goal_state, 'G', ha='center', va='center', color='green', fontweight='bold')
    for x in range(env.width):
        ax.text(x, env.height - 0.35, str(env.wind[x]), ha='center', va='center', color='navy', fontsize=8)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_title("Learned greedy path (wind strength shaded / labelled)")
    dot, = ax.plot([], [], 'o', color='crimson', ms=14)
    trail, = ax.plot([], [], '-', color='crimson', lw=2, alpha=0.6)

    def update(i):
        dot.set_data([path[i][0]], [path[i][1]])
        trail.set_data([p[0] for p in path[:i + 1]], [p[1] for p in path[:i + 1]])
        return dot, trail

    anim = animation.FuncAnimation(fig, update, frames=len(path), interval=300)
    if filename:
        anim.save(filename, writer=animation.PillowWriter(fps=3))
    plt.close()
    return len(path) - 1
