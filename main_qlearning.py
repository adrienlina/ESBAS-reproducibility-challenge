from gridworld2 import GridWorld2
import numpy as np
import pandas as pd
from q_learning import QLearning

import matplotlib.pyplot as plt

env = GridWorld2

############# Q-Learning #################################################

states = range(env.n_states)
t_max = 100
N_iterations = 10**4
epsilon = 0.2

episode_rewards = [0.]*N_iterations

q_learning = QLearning(gamma=0.9, epsilon=epsilon, learning_rate=0.01, env=env)

for n in range(N_iterations):
    x_t = env.reset()

    t = 0
    term = False
    while (t < t_max and not term):
        a_t = q_learning.epsilon_greedy(x_t)
        next_state, step_reward, term = env.step(x_t, a_t)
        episode_rewards[n] += step_reward

        q_learning.update(x_t, a_t, step_reward, next_state, term)

        t += 1
        x_t = next_state

reward_rolling_mean = pd.rolling_mean(pd.Series(episode_rewards), N_iterations//10)

plt.plot(reward_rolling_mean)
plt.show()

############# Display world with the policy #########################
# env.reset()
# policy = [q_learning.epsilon_greedy(state) for state in range(env.n_grid_states)]
# gui.render_policy(env, policy)
