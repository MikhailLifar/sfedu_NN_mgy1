from layers import *
from losses import *
from models import *
from metrics import *

from sklearn import datasets

SEED = 1


def perceptron_try0():
    X, Y = datasets.load_iris(return_X_y=True, as_frame=True)
    num_classes = len(Y.unique())
    X, Y = X.to_numpy(), Y.to_numpy()
    # print(X)
    # print(Y)

    np.random.seed(SEED)
    model = MLPClassifier(dim_in=X.shape[1], num_classes=num_classes,
                          inner_dims=(), lr=1.e-3)

    Y_pred = np.array([model.predict(x) for x in X])
    print(Y_pred)
    print(f'Accuracy prior training: {accuracy(Y, Y_pred)}')

    # training cycle
    num_epochs = 300
    for i in range(num_epochs):
        for x, y in zip(X, Y):
            model.forward(x)
            model.backward(y)
        Y_pred = np.array([model.predict(x) for x in X])
        print(f'Accuracy after epoch {i}: {accuracy(Y, Y_pred)}')


def main():
    perceptron_try0()


if __name__ == '__main__':
    main()
