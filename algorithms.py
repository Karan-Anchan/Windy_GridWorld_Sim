import numpy as np


def _fresh_Q(env):
    return {(x, y): {a: 0 for a in env.actions} for x in range(env.width) for y in range(env.height)}


def _greedy(Q, state):
    return max(Q[state], key=Q[state].get)


def q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500, trials=10):
    """Off-policy TD control; returns mean cumulative reward per episode over trials."""
    all_rewards = np.zeros((trials, episodes))
    for t in range(trials):
        Q = _fresh_Q(env)
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            while state != env.goal_state:
                action = np.random.choice(env.actions) if np.random.rand() < epsilon else _greedy(Q, state)
                next_state, reward = env.step(state, action)
                total_reward += reward
                Q[state][action] += alpha * (reward + gamma * Q[next_state][_greedy(Q, next_state)] - Q[state][action])
                state = next_state
            all_rewards[t, e] = total_reward
    return np.mean(all_rewards, axis=0)


def sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500, trials=10):
    """On-policy TD control; returns mean cumulative reward per episode over trials."""
    all_rewards = np.zeros((trials, episodes))
    for t in range(trials):
        Q = _fresh_Q(env)
        for e in range(episodes):
            state = env.reset()
            action = np.random.choice(env.actions) if np.random.rand() < epsilon else _greedy(Q, state)
            total_reward = 0
            while state != env.goal_state:
                next_state, reward = env.step(state, action)
                total_reward += reward
                next_action = np.random.choice(env.actions) if np.random.rand() < epsilon else _greedy(Q, next_state)
                Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
                state, action = next_state, next_action
            all_rewards[t, e] = total_reward
    return np.mean(all_rewards, axis=0)


def expected_sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500, trials=10):
    """TD control backing up the ε-greedy expectation; returns mean reward per episode over trials."""
    n = len(env.actions)
    all_rewards = np.zeros((trials, episodes))
    for t in range(trials):
        Q = _fresh_Q(env)
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            while state != env.goal_state:
                action = np.random.choice(env.actions) if np.random.rand() < epsilon else _greedy(Q, state)
                next_state, reward = env.step(state, action)
                total_reward += reward
                best = _greedy(Q, next_state)
                expected = sum(Q[next_state][a] * (epsilon / n + (1 - epsilon) * (a == best)) for a in env.actions)
                Q[state][action] += alpha * (reward + gamma * expected - Q[state][action])
                state = next_state
            all_rewards[t, e] = total_reward
    return np.mean(all_rewards, axis=0)


def train_q_table(env, alpha=0.5, gamma=1.0, epsilon=0.1, episodes=400):
    """Train a single Q-learning agent and return its Q-table (for greedy-path extraction)."""
    Q = _fresh_Q(env)
    for _ in range(episodes):
        state = env.reset()
        while state != env.goal_state:
            action = np.random.choice(env.actions) if np.random.rand() < epsilon else _greedy(Q, state)
            next_state, reward = env.step(state, action)
            Q[state][action] += alpha * (reward + gamma * max(Q[next_state].values()) - Q[state][action])
            state = next_state
    return Q
