import numpy as np

from layers import *
from losses import *


class Model:
    def __init__(self, layers, loss, lr=3.e-4):
        self.layers = layers
        self.loss = loss
        self.y_pred = None
        self.can_backward = False

        self.parameters = []
        for l in self.layers:
            params = l.get_parameters()
            if params is not None:
                self.parameters.extend(params)
        self.lr = lr

    def forward(self, x, require_grad=True):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        for l in self.layers:
            x = l.forward(x, require_grad)
        if require_grad:
            self.y_pred = np.copy(x)
            self.can_backward = True
        return x

    def backward(self, y_true):
        assert self.can_backward, 'need forward before backward'
        self.loss.forward(y_true, self.y_pred)
        error = self.loss.backward()

        gradients = []
        for l in self.layers[::-1]:
            error, grads = l.backward(error)
            if grads is not None:
                gradients.extend(grads[::-1])
        gradients = gradients[::-1]

        for i, p, g in zip(range(len(self.parameters)), self.parameters, gradients):
            p -= self.lr * g


class MLPClassifier(Model):
    def __init__(self, dim_in, num_classes,
                 inner_dims,
                 lr=3.e-4):
        self.dims = (dim_in, *inner_dims, num_classes)
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
            lr=lr,
        )

    def predict(self, x, require_grad=True):
        return np.argmax(self.forward(x, require_grad))
