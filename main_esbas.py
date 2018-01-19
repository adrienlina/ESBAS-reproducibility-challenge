from gridworld2 import GridWorld2
import numpy as np
import pandas as pd
from q_learning import QLearning
from ucb1 import UCB1

import matplotlib.pyplot as plt

env = GridWorld2

############# UCB1 AS #################################################
states = range(env.n_states)
beta_max = 14
t_max = 100
epsilon = 0.8
gamma = 0.9

arms = [
    QLearning(gamma=gamma, epsilon=epsilon, learning_rate=0.001, env=env),
    QLearning(gamma=gamma, epsilon=epsilon, learning_rate=0.01, env=env),
    QLearning(gamma=gamma, epsilon=epsilon, learning_rate=0.1, env=env),
    QLearning(gamma=gamma, epsilon=epsilon, learning_rate=0.5, env=env),
]

episode_rewards = []
choosen_arm = []
n_episode_per_epochs = 20
n_episodes = 0
for beta in range(beta_max):
    if beta > 2:
        n_episode_per_epochs *= 2

    trajectories = []
    ucb1 = UCB1(arms, xi=1/4)

    # QLearning do not learn during an epoch, UCB1 determines "best" algorithm
    for tau in range(n_episode_per_epochs):
        arm = ucb1.choose_arm()
        choosen_arm.append(arm)
        algo = arms[arm]

        x_t = env.reset()

        trajectory_reward = 0
        t = 0
        term = False
        while (t < t_max and not term):
            a_t = algo.epsilon_greedy(x_t, epsilon=epsilon**beta)
            next_state, step_reward, term = env.step(x_t, a_t)

            trajectories.append((x_t, a_t, step_reward, next_state, term,))
            trajectory_reward += step_reward

            t += 1
            x_t = next_state

        if t < t_max:
            ucb1.update(arm, ((t_max-t)/t_max+1)/2)
        else:
            ucb1.update(arm, 0)
        episode_rewards.append(-t)
        n_episodes += 1

    # End of an epoch, we update the algorithms with the new set of trajectories
    for state, action, step_reward, next_state, term in trajectories:
        for algo in arms:
            algo.update(state, action, step_reward, next_state, term)

    print('end epoch %s, total episodes %s' % (beta, n_episodes))

######### Reward plotting #################
reward_rolling_mean = pd.rolling_mean(-pd.Series(episode_rewards), 200)

plt.plot(reward_rolling_mean)
plt.show()

########## Arm plotting ###################
choosen_arm = np.asarray(choosen_arm)
arms_average = [[] for _ in arms]
N = 100
for t in range(0, (2**beta_max)-N-1, N):
    _sum = 0
    for arm in range(len(arms)):
        _sum += np.sum(choosen_arm[t:t+N] == arm) / N
        arms_average[arm].append(_sum)

fig, ax = plt.subplots()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']

x = [t for t in range(0, (2**beta_max)-N-1, N)]
for arm, algo in enumerate(arms):
    if arm == 0:
        ax.fill_between(x, 0, arms_average[0], color=colors[0], label='%s'%algo)
    else:
        ax.fill_between(x, arms_average[arm-1], arms_average[arm], color=colors[arm], label='%s'%algo)

ax.legend()
plt.show()
