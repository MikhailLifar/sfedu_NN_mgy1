import numpy as np

from layers import *
from losses import *


class Model:
    def __init__(self, layers, loss, batch_size=16, lr=3.e-4):
        self.layers = layers
        self.loss = loss
        self.y_pred = []
        self.can_backward = False

        self.parameters = []
        for l in self.layers:
            params = l.get_parameters()
            if params is not None:
                self.parameters.extend(params)
        self.batch_size = batch_size
        self.lr = lr

    def _forward(self, x, require_grad=True):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        for l in self.layers:
            x = l.forward(x, require_grad)
        if require_grad:
            self.y_pred = np.copy(x)
            self.can_backward = True
        return x

    def _get_gradients(self, y_true):
        assert self.can_backward, 'need forward before backward'
        self.loss.forward(y_true, self.y_pred)
        error = self.loss.backward()

        gradients = []
        for l in self.layers[::-1]:
            error, grads = l.backward(error)
            if grads is not None:
                gradients.extend(grads[::-1])
        gradients = gradients[::-1]

        return gradients

    def batch_training(self, X, Y, shuffle=True):
        N = len(X)
        if shuffle:
            idx = np.random.permutation(N)
            X = X[idx]
            Y = Y[idx]

        for b_start in range(0, N, self.batch_size):
            X_batch = X[b_start:b_start + min(N - b_start, self.batch_size)]
            Y_batch = Y[b_start:b_start + min(N - b_start, self.batch_size)]
            gradients = []
            for x, y in zip(X_batch, Y_batch):
                self._forward(x)
                gradients.append(self._get_gradients(y))

            mean_grad = []
            for i, p in enumerate(self.parameters):
                mean_grad.append(np.mean(np.stack([g[i] for g in gradients]), axis=0))

            for i, p, g in zip(range(len(self.parameters)), self.parameters, mean_grad):
                p -= self.lr * g


class MLPClassifier(Model):
    def __init__(self, dim_in, num_classes,
                 inner_dims,
                 batch_size=16,
                 lr=3.e-4):
        self.dims = (dim_in, *inner_dims, num_classes)
        self.dim_in = dim_in
        self.num_classes = num_classes
        layers = []
        for dim1, dim2 in zip(self.dims[:-2], self.dims[1:-1]):
            layers.extend([Linear(dim1, dim2), ReLU()])
        layers.extend([
            Linear(self.dims[-2], self.dims[-1]),
            Softmax(),
        ])
        Model.__init__(
            self,
            layers=layers,
            loss=CrossEntropy(),
            batch_size=batch_size,
            lr=lr,
        )

    def predict(self, x, require_grad=True):
        return np.argmax(self._forward(x, require_grad))
