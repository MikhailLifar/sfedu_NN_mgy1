from itertools import cycle

import numpy as np
import gymnasium as gym
from gymnasium.core import RenderFrame


class TicTacToe:
    def __init__(self, size=13, win_size=5):
        # the field is numpy 2d-array
        # the empty cell is 0
        # the x is 1
        # the O is -1
        self.size = size
        self.win_size = win_size
        self.field = np.zeros((self.size, self.size))
        self.turn = 1
        self.opponent = -1

    def check_win(self, x, y):
        # the trick is to check only lines involving the last move
        lines = [
            self.field[y, max(x - self.win_size, 0):min(x + self.win_size, self.size)],  # horizontal
            self.field[max(y - self.win_size, 0):min(y + self.win_size, self.size), x],  # vertical
        ]

        # (left, top) -> (right, bottom) diagonal
        lt_to_rb_diag = [self.field[y, x]]
        x_, y_ = x - 1, y - 1
        while min(x_, y_) >= 0:
            lt_to_rb_diag = [self.field[y_, x_], ] + lt_to_rb_diag
            x_ -= 1
            y_ -= 1
        x_, y_ = x + 1, y + 1
        while max(x_, y_) < self.size:
            lt_to_rb_diag = lt_to_rb_diag + [self.field[y_, x_], ]
            x_ += 1
            y_ += 1
        lines.append(np.array(lt_to_rb_diag, dtype=np.uint8))

        # (left, bottom) -> (right, top) diagonal
        lb_to_rt_diag = [self.field[y, x]]
        x_, y_ = x - 1, y + 1
        while x_ >= 0 and y_ < self.size:
            lb_to_rt_diag = [self.field[y_, x_], ] + lb_to_rt_diag
            x_ -= 1
            y_ += 1
        x_, y_ = x + 1, y - 1
        while x_ < self.size and y_ >= 0:
            lb_to_rt_diag = lb_to_rt_diag + [self.field[y_, x_], ]
            x_ += 1
            y_ -= 1
        lines.append(np.array(lb_to_rt_diag, dtype=np.uint8))

        def check_line(line):
            s = 0
            for el in line:
                if el == self.turn:
                    s += 1
                else:
                    s = 0
                if s == self.win_size:
                    return True

        return np.any([check_line(l) for l in lines])

    def invert_turn(self):
        self.turn, self.opponent = self.opponent, self.turn

    def move(self, x, y):
        if min(x, y) < 0 or max(x, y) > self.size:
            return -2
        if self.field[y, x] != 0:
            return -1
        self.field[y, x] = self.turn
        if self.check_win(x, y):
            return 1
        self.invert_turn()
        return 0

    def reset(self):
        self.field[:, :] = 0
        self.turn = 1
        self.opponent = -1


class CLIRenderer:
    def __init__(self, ticTacToe: TicTacToe):
        self.ticTacToe = ticTacToe

    def render(self):
        field = self.ticTacToe.field
        size = field.shape[0]
        str_repr = {0: ' ', 1: 'X', -1: '0'}
        for i in range(size):
            print('-' * (2 * size + 1))
            print('|' + '|'.join(str_repr[el] for el in field[i]) + '|')
        print('-' * (2 * size + 1))


class TicTacToeEnv(gym.Env):
    def __init__(self, tictactoe, opponents: list, renderer,
                 agent_turn=1, agent_turn_fixed=False):
        self.tictactoe = tictactoe
        self.size = self.tictactoe.size

        self.opponents_li = opponents
        self.opponents = cycle(opponents)
        self.opponent = None

        self.renderer = renderer
        self.agent_turn = agent_turn
        if not agent_turn_fixed:
            self.agent_turn *= -1  # since reset will be called
        self.agent_turn_fixed = agent_turn_fixed

    def step(self, action):
        assert self.agent_turn == self.tictactoe.turn

        # move
        x, y = action % self.size, action // self.size
        code = self.tictactoe.move(x, y)
        # assert code >= 0, 'Incorrect action'
        if code == -1:
            return self.agent_turn * np.copy(self.tictactoe.field.flatten()), -1., True, False, {}

        # has the agent won?
        if code == 1:
            return self.agent_turn * np.copy(self.tictactoe.field.flatten()), 1., True, False, {}
        obs = self.agent_turn * np.copy(self.tictactoe.field.flatten())
        action_mask = obs == 0
        is_draw = not np.any(action_mask)

        if not is_draw:
            opponent_action = self.opponent.act(-obs, {'action_mask': action_mask})
            x, y = opponent_action % self.size, opponent_action // self.size
            code = self.tictactoe.move(x, y)
            assert code >= 0

            # has the opponent won?
            if code == 1:
                return self.agent_turn * np.copy(self.tictactoe.field.flatten()), -1., True, False, {}
            obs = self.agent_turn * np.copy(self.tictactoe.field.flatten())
            is_draw = not np.any(action_mask)

        return obs, 0., is_draw, False, {}

    def move(self, action):
        x, y = action % self.size, action // self.size
        code = self.tictactoe.move(x, y)
        obs = self.tictactoe.turn * np.copy(self.tictactoe.field.flatten())
        action_mask = obs == 0
        return obs, code, {'action_mask': action_mask}

    def add_opponent(self, opponent):
        self.opponents_li.append(opponent)
        self.opponents = cycle(self.opponents_li)

    def reset(self, seed=None, options=None):
        self.tictactoe.reset()

        if not self.agent_turn_fixed:
            self.agent_turn *= -1
        if self.agent_turn == -1:
            obs = self.agent_turn * np.copy(self.tictactoe.field.flatten())
            action_mask = obs == 0
            opponent_action = self.opponent.act(-obs, {'action_mask': action_mask})
            x, y = opponent_action % self.size, opponent_action // self.size
            code = self.tictactoe.move(x, y)
            assert code == 0

        self.opponent = next(self.opponents)

        obs = self.agent_turn * np.copy(self.tictactoe.field.flatten())
        return obs, {}

    def render(self):
        self.renderer.render()
