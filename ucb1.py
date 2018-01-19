import numpy as np


class UCB1:
    def __init__(self, MAB, xi=1.):
        self.MAB_arms = [index for index, _ in enumerate(MAB)]
        self.xi = xi

        self.N_t = [0 for _ in MAB]
        self.S_t = [0 for _ in MAB]

        self.n_draws = 0

    def upper_bound(self, arm, t):
        bound = self.S_t[arm] / self.N_t[arm] + np.sqrt(self.xi * np.log(self.n_draws) / (self.N_t[arm]))
        return bound

    def choose_arm(self):
        self.n_draws += 1
        if self.n_draws-1 < len(self.MAB_arms):
            return self.n_draws-1

        return max(self.MAB_arms, key=lambda arm: self.upper_bound(arm, self.n_draws))

    def update(self, arm, reward):
        self.S_t[arm] += reward
        self.N_t[arm] += 1
