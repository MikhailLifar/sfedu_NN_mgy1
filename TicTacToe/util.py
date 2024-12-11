import numpy as np


class RandomOpponent:
    def __init__(self, nActions):
        self.nActions = nActions

    def act(self, obs, info=None):
        if info is not None and 'action_mask' in info:
            return np.random.choice(np.arange(self.nActions)[info['action_mask']])
        else:
            return np.random.choice(np.arange(self.nActions))


def moving_average(arr, n: int, mode='valid'):
    avg = np.convolve(arr, np.ones(n), mode=mode) / n
    return avg
