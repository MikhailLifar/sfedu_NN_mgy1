import copy
import numpy as np


class Layer:

    def forward(self, x, require_grad=True):
        raise NotImplementedError

    def backward(self, error):
        raise NotImplementedError

    def get_parameters(self):
        return None


class Linear(Layer):
    def __init__(self, dim_in, dim_out):
        self.w = np.random.normal(size=(dim_out, dim_in))
        self.b = np.random.normal(size=(dim_out, 1))
        self.dim_in, self.dim_out = dim_in, dim_out
        self.store_for_grad = {'x': None}
        self.can_backward = False

    def forward(self, x, require_grad=True):
        if require_grad:
            self.store_for_grad['x'] = np.copy(x)
            self.can_backward = True
        # out = self.w @ x
        # out = out + self.b
        return self.w @ x + self.b

    def backward(self, error):
        assert self.can_backward, 'need forward before backward'
        grad_w = error @ self.store_for_grad['x'].transpose()
        grad_b = np.copy(error)
        # error = np.array([np.dot(self.w[j, :], error) for j in range(self.dim_in)])
        error = self.w.transpose() @ error

        self.can_backward = False

        return error, (grad_w, grad_b)

    def get_parameters(self):
        return self.w, self.b


class ReLU(Layer):
    def __init__(self):
        self.store_for_grad = {'x': None}
        self.can_backward = False

    def forward(self, x, require_grad=True):
        if require_grad:
            self.store_for_grad['x'] = np.copy(x)
            self.can_backward = True
        return np.clip(x, a_min=0., a_max=None)

    def backward(self, error):
        assert self.can_backward, 'need forward before backward'
        x = self.store_for_grad['x']

        self.can_backward = False

        return error * (x >= 0).astype(float), None


class Softmax(Layer):
    def __init__(self):
        self.store_for_grad = {'out': None}
        self.can_backward = False

    def forward(self, x, require_grad=True):
        out = np.exp(x) / np.sum(np.exp(x))
        if require_grad:
            self.store_for_grad['out'] = np.copy(out)
            self.can_backward = True
        return out

    def backward(self, error):
        assert self.can_backward, 'need forward before backward'
        out = self.store_for_grad['out'].flatten()

        dim = len(error)
        error = error.transpose() @ np.diag(out) @ (np.eye(dim) - np.tile(out, (dim, 1)))

        self.can_backward = False

        return error.transpose(), None

