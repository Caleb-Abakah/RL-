"""
agent.py


A tabular Q-Learning agent, implemented from first principles.

All of the agent's knowledge lives in ``self.Q``, a (n_states x n_actions)
array of action-value estimates.

Update rule
-----------
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
"""

import numpy as np


class QLearningAgent:
    """Tabular Q-Learning agent with epsilon-greedy exploration."""

    def __init__(self, n_states, n_actions,
                 alpha=0.1,            # learning rate
                 gamma=0.99,           # discount factor
                 epsilon=1.0,          # starting exploration rate
                 epsilon_min=0.01,     # minimum exploration rate
                 epsilon_decay=0.9995, # multiplicative decay per episode
                 seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(seed)

        # Q-table: every state-action value starts at 0.
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def choose_action(self, state):
        """Epsilon-greedy action selection (used during training)."""
        if self.rng.random() < self.epsilon:
            # explore: random action
            return int(self.rng.integers(self.n_actions))
        # exploit: best known action, breaking ties at random
        return self._argmax_random(self.Q[state])

    def greedy_action(self, state):
        """Pure exploitation, no exploration (used during evaluation)."""
        return self._argmax_random(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        """Apply the Q-Learning update for a single transition."""
        best_next = 0.0 if done else self.Q[next_state].max()
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        """Reduce exploration over time, never below epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        """Return the greedy policy: best action for every state."""
        return np.argmax(self.Q, axis=1)

    def _argmax_random(self, q_values):
        """argmax that breaks ties uniformly at random."""
        best = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best))
