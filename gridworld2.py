import numpy as np
from collections import namedtuple
import numbers

MDP = namedtuple('MDP', 'S,A,P,R,gamma,d0')


class GridWorld:
    def __init__(self, gamma=0.95, ksi=1., grid=None, render=False):
        self.grid = grid

        self.action_names = np.array(['right', 'down', 'left', 'up'])

        self.n_rows, self.n_cols = len(self.grid), max(map(len, self.grid))

        # Create a map to translate coordinates [r,c] to scalar index
        # (i.e., state) and vice-versa
        self._coord2state = np.empty_like(self.grid, dtype=np.int)
        self.n_grid_states = 0
        self._state2coord = []
        self._fruit_coords = []
        for i in range(self.n_rows):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] != 'x':
                    self._coord2state[i, j] = self.n_grid_states
                    self.n_grid_states += 1
                    self._state2coord.append([i, j])
                else:
                    self._coord2state[i, j] = -1

                if isinstance(self.grid[i][j], numbers.Number):
                    self._fruit_coords.append((i,j))

        self.n_states = self.n_grid_states * (2**len(self._fruit_coords)-1)

        # compute the actions available in each state
        self.compute_available_actions()
        self.gamma = gamma
        self.ksi = ksi
        self.render = render

    def coord2state(self, r, c):
        return self._coord2state[r,c] + self.current_state_offset

    def state2coord(self, state):
        return self._state2coord[state - self.current_state_offset]

    def state_actions(self, state):
        return self._state_actions[state - self.current_state_offset]

    def remove_fruit(self, r, c):
        self.tmp_grid[r,c] = ''
        self.remaining_fruits -= 1

        fruit_offset = 0
        fruits_left = 0
        for index, fruit_coord in enumerate(self._fruit_coords):
            r, c = fruit_coord
            if self.tmp_grid[r,c] == '':
                fruit_offset += self.n_grid_states * (2**index)
            else:
                fruits_left += 1

        assert fruits_left == self.remaining_fruits
        self.current_state_offset = fruit_offset

    def reset(self):
        """
        Returns:
            An initial state randomly drawn from
            the initial distribution
        """
        self.remaining_fruits = 4
        self.tmp_grid = np.copy(self.grid)
        self.current_state_offset = 0

        # x_0 = np.random.randint(0, self.n_grid_states)
        # while self.state2coord(x_0) in self._fruit_coords:
        #     x_0 = np.random.randint(0, self.n_grid_states)
        x_0 = 5

        return x_0

    def step(self, state, action):
        """
        Args:
            state (int): the amount of good
            action (int): the action to be executed

        Returns:
            next_state (int): the state reached by performing the action
            reward (float): a scalar value representing the immediate reward
            absorb (boolean): True if the next_state is absorsing, False otherwise
        """
        r, c = self.state2coord(state)
        assert action in self.state_actions(state)


        if action == 0:
            c = min(self.n_cols - 1, c + 1)
        elif action == 1:
            r = min(self.n_rows - 1, r + 1)
        elif action == 2:
            c = max(0, c - 1)
        elif action == 3:
            r = max(0, r - 1)

        if self.tmp_grid[r][c] == 'x':
            next_state = state
            r, c = self.state2coord(next_state)

        try:
            reward = float(self.tmp_grid[r][c])
        except ValueError:
            reward = 0.
        if reward:
            self.remove_fruit(r, c)
        reward += np.random.normal(0, self.ksi)

        next_state = self.coord2state(r, c)
        term = self.remaining_fruits == 0

        return next_state, reward, term

    def compute_available_actions(self):
        # define available actions in each state
        # actions are indexed by: 0=right, 1=down, 2=left, 3=up
        self._state_actions = []
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.grid[i][j] != 'x':
                    actions = [0, 1, 2, 3]
                    if i == 0:
                        actions.remove(3)
                    if j == self.n_cols - 1:
                        actions.remove(0)
                    if i == self.n_rows - 1:
                        actions.remove(1)
                    if j == 0:
                        actions.remove(2)

                    for a in actions.copy():
                        r, c = i, j
                        if a == 0:
                            c = min(self.n_cols - 1, c + 1)
                        elif a == 1:
                            r = min(self.n_rows - 1, r + 1)
                        elif a == 2:
                            c = max(0, c - 1)
                        else:
                            r = max(0, r - 1)
                        if self.grid[r][c] == 'x':
                            actions.remove(a)

                    self._state_actions.append(actions)


grid2 = [
    [1., 'x', 'x', '', 1.],
    ['', '', '', '', 'x'],
    ['x', '', 'x', '', ''],
    ['', '', '', 'x', ''],
    [1., 'x', '', '', 1.],
]
GridWorld2 = GridWorld(gamma=0.95, grid=grid2, ksi=1.)
