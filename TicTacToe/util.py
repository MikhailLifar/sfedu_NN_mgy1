import numpy as np


class RandomOpponent:
    def __init__(self, nActions):
        self.nActions = nActions

    def act(self, obs, info):
        assert 'action_mask' in info
        return np.random.choice(np.arange(self.nActions)[info['action_mask']])


def moving_average(arr, n: int, mode='valid'):
    avg = np.convolve(arr, np.ones(n), mode=mode) / n
    return avg
