import numpy as np


class WindyGridworld:
    """Gridworld where per-column wind pushes the agent upward; -1 per step, 0 at the goal."""

    def __init__(self, width, height, wind, start_state, goal_state):
        self.width = width
        self.height = height
        self.wind = wind
        self.start_state = start_state
        self.goal_state = goal_state
        self.actions = ['up', 'down', 'left', 'right']
        self.action_effects = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}

    def step(self, state, action):
        x, y = state
        dx, dy = self.action_effects[action]
        x = min(max(x + dx, 0), self.width - 1)
        y = min(max(y + dy, 0), self.height - 1)
        y = min(max(y - self.wind[x], 0), self.height - 1)  # wind blows up
        reward = 0 if (x, y) == self.goal_state else -1
        return (x, y), reward

    def reset(self):
        return self.start_state
