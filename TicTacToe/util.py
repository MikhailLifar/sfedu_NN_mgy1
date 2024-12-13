import numpy as np


class RandomPlayer:
    def __init__(self, nActions):
        self.nActions = nActions

    def act(self, obs, info=None):
        if info is not None and 'action_mask' in info:
            return np.random.choice(np.arange(self.nActions)[info['action_mask']])
        else:
            return np.random.choice(np.arange(self.nActions))


class UnconditionalPlayer:
    def __init__(self, program):
        self.program = program
        self.idx = 0

    def act(self, obs, info=None):
        action = self.program[self.idx]
        self.idx += 1
        return action

    def reset(self):
        self.idx = 0


def moving_average(arr, n: int, mode='valid'):
    avg = np.convolve(arr, np.ones(n), mode=mode) / n
    return avg
