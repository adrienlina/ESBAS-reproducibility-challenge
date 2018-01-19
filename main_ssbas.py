from gridworld2 import GridWorld2
import numpy as np
import pandas as pd
from q_learning import QLearning
from sliding_ucb1 import SlidingUCB1

import matplotlib.pyplot as plt

env = GridWorld2

############# UCB1 AS #################################################
states = range(env.n_states)
t_max = 100
epsilon = 0.6
gamma = 0.9
N = 100000

arms = [
    QLearning(gamma=gamma, epsilon=epsilon, learning_rate=0.001, env=env),
    QLearning(gamma=gamma, epsilon=epsilon, learning_rate=0.01, env=env),
    QLearning(gamma=gamma, epsilon=epsilon, learning_rate=0.1, env=env),
    QLearning(gamma=gamma, epsilon=epsilon, learning_rate=0.5, env=env),
]

episode_rewards = []
choosen_arm = []
n_episodes = 0
sliding_ucb1 = SlidingUCB1(arms, xi=1/4)
for tau in range(N):
    arm = sliding_ucb1.choose_arm()
    choosen_arm.append(arm)
    algo = arms[arm]

    x_t = env.reset()

    t = 0
    term = False
    while (t < t_max and not term):
        a_t = algo.epsilon_greedy(x_t, epsilon=1/(1+0.01*tau))
        next_state, step_reward, term = env.step(x_t, a_t)

        for algo2 in arms:
            algo2.update(x_t, a_t, step_reward, next_state, term)

        t += 1
        x_t = next_state

    if t < t_max:
        sliding_ucb1.update(arm, ((t_max-t)/t_max+1)/2)
    else:
        sliding_ucb1.update(arm, 0)
    episode_rewards.append(-t)
    n_episodes += 1

n = 3000
######### Reward plotting #################
reward_rolling_mean = pd.rolling_mean(-pd.Series(episode_rewards), n)

plt.plot(reward_rolling_mean)
plt.show()

########## Arm plotting ###################
choosen_arm = np.asarray(choosen_arm)
arms_average = [[] for _ in arms]
for t in range(0, N-n):
    _sum = 0
    for arm in range(len(arms)):
        _sum += np.sum(choosen_arm[t:t+n] == arm) / n
        arms_average[arm].append(_sum)

fig, ax = plt.subplots()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']

x = [t for t in range(N-n)]
for arm, algo in enumerate(arms):
    if arm == 0:
        ax.fill_between(x, 0, arms_average[0], color=colors[0], label='%s'%algo)
    else:
        ax.fill_between(x, arms_average[arm-1], arms_average[arm], color=colors[arm], label='%s'%algo)

ax.legend()
plt.show()
