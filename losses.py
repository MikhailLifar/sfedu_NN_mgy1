import numpy as np


class Loss:
    def forward(self, y_true, y_pred):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class CrossEntropy(Loss):
    def __init__(self):
        self.store_for_grad = {'y_true': None, 'y_pred': None}
        self.can_backward = False

    def forward(self, y_true, y_pred, require_grad=True):
        y_true_oh = np.fromfunction(lambda i, j: i == y_true, shape=y_pred.shape).astype(float)
        if require_grad:
            self.store_for_grad['y_true'] = np.copy(y_true_oh)
            self.store_for_grad['y_pred'] = np.copy(y_pred)
            self.can_backward = True
        return -np.sum(y_true_oh * np.log(y_pred))

    def backward(self):
        assert self.can_backward, 'need forward before backward'
        y_true_oh = self.store_for_grad['y_true']
        y_pred = self.store_for_grad['y_pred']

        self.can_backward = False

        return -y_true_oh * (y_pred ** -1)
