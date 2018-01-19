import numpy as np


class QLearning:
    def __init__(self, gamma, epsilon, learning_rate, env):
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env
        self.learning_rate = learning_rate

        self.states = range(self.env.n_states)
        self.Q = [[0.]*4 for _ in self.states]

    def __repr__(self):
        return 'Q-Learning, learning rate = %s' % self.learning_rate

    def epsilon_greedy(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        actions = self.env.state_actions(state)

        if np.random.random() < self.epsilon:
            return np.random.choice(actions)
        else:
            return max(actions, key=lambda action: self.Q[state][action]) # pythonic argmax

    def update(self, state, action, step_reward, next_state, term):
        alpha = self.learning_rate
        if term:
            self.Q[state][action] = (1-alpha) * self.Q[state][action] + alpha * step_reward
        else:
            self.Q[state][action] = (1-alpha) * self.Q[state][action] + alpha * (step_reward + self.gamma * max(self.Q[next_state]))
