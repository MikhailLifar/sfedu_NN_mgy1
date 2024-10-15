import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.datasets

from layers import *
from losses import *
from models import *
from metrics import *

from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import r2_score

SEED = 1


LABEL_TO_COLOR = {0: 'r', 1: 'b', 2: 'purple',
                  3: 'y', 4: 'black'}


def plot_linear_boundaries(clsfr_, axis, xlim, ylim,
                           grid_resolution=400, alpha=0.5):
    assert clsfr_.dim_in == 2

    X1, X2 = np.meshgrid(np.linspace(*xlim, grid_resolution),
                         np.linspace(*ylim, grid_resolution))
    data = np.hstack((X1.ravel().reshape(-1, 1), X2.ravel().reshape(-1, 1)))
    Y_pred = np.apply_along_axis(lambda x: clsfr_.predict(x), 1, data)

    levels = np.arange(-0.5, clsfr_.num_classes, 1.)
    colors = [LABEL_TO_COLOR[i] for i in range(clsfr_.num_classes)]
    axis.contourf(X1, X2,
                  Y_pred.reshape(grid_resolution, grid_resolution),
                  levels=levels, colors=colors, alpha=alpha)


def basic_scatter(x, y, ax=None, filePath=None, color=None, xlim=None, ylim=None,
                  xlabel=None, ylabel=None):
    if ax is None:
        assert filePath is not None
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        assert filePath is None
    ax.scatter(x, y, color=color)
    if xlim is None:
        xlim = (None, ) * 2
    if ylim is None:
        ylim = (None, ) * 2
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if filePath is not None:
        fig.savefig(filePath, bbox_inches='tight', dpi=300)


def prediction_plot_classif(model, X, target_true, filePath):
    assert X.shape[1] == 2
    name0, name1 = X.columns
    X = X.to_numpy()

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_linear_boundaries(model, ax, (np.min(X[:, 0]), np.max(X[:, 0])), (np.min(X[:, 1]), np.max(X[:, 1])))
    basic_scatter(X[:, 0], X[:, 1], ax=ax, color=[LABEL_TO_COLOR[y] for y in target_true])
    ax.set_xlabel(name0)
    ax.set_ylabel(name1)
    fig.savefig(filePath, bbox_inches='tight', dpi=300)


def pairwise_scatters_classif(X, Y, dirPath):
    for i, col0 in enumerate(X.columns[:-1]):
        for col1 in X.columns[i+1:]:
            fig, ax = plt.subplots(figsize=(12, 8))
            basic_scatter(X[col0], X[col1], ax=ax, color=[LABEL_TO_COLOR[y] for y in Y])

            Y_pred = cross_val_predict(ExtraTreesClassifier(random_state=0), X[[col0, col1]], Y, cv=5)
            r2 = r2_score(Y, Y_pred)

            ax.set_xlabel(col0)
            ax.set_ylabel(col1)
            fig.savefig(f'{dirPath}/{r2:.5f}_{col0}_vs_{col1}.png', bbox_inches='tight', dpi=300)


def perceptron_try0():
    X, Y = datasets.load_iris(return_X_y=True, as_frame=True)
    num_classes = len(Y.unique())
    X_df = copy.deepcopy(X)
    X, Y = X.to_numpy(), Y.to_numpy().reshape(-1, 1)
    X = X[:, 2:]
    # print(X)
    # print(Y)
    # print(X.shape)
    # print(Y.shape)

    np.random.seed(SEED)
    model = MLPClassifier(dim_in=X.shape[1], num_classes=num_classes,
                          inner_dims=(), batch_size=16, lr=1.e-2)
    # print(model.parameters[0])  # debug

    Y_pred = np.array([model.predict(x) for x in X])
    print(Y_pred)
    print(f'Accuracy prior training: {accuracy(Y.flatten(), Y_pred)}')

    # training cycle
    num_epochs = 300
    for i in range(num_epochs):
        model.batch_training(X, Y)
        Y_pred = np.array([model.predict(x) for x in X])
        print(f'Accuracy after epoch {i}: {accuracy(Y.flatten(), Y_pred)}')

    # print(model.parameters[0])  # debug
    prediction_plot_classif(model, X_df.iloc[:, 2:], Y.flatten(), '../plots/iris_perceptron_pred_plot.png')


def mlp_on_nonlinear():
    # X, Y = datasets.load_wine(return_X_y=True, as_frame=True)
    # X.columns = X.columns.map(lambda s: s.replace('/', '|'))
    # num_classes = len(Y.unique())
    # data = X.join(Y)
    # pairwise_scatters_classif(X, Y, '../plots/wine_pairwise')

    X, Y = sklearn.datasets.make_moons(n_samples=200, noise=0.05, random_state=SEED)
    Y = Y.reshape(-1, 1)
    basic_scatter(X[:, 0], X[:, 1], color=[LABEL_TO_COLOR[y] for y in Y.flatten()],
                  xlabel='x1', ylabel='x2',
                  filePath='../plots/mlp_moons/scatter.png')

    np.random.seed(SEED)
    model = MLPClassifier(dim_in=X.shape[1], num_classes=2,
                          inner_dims=(7, 7, 7), batch_size=16, lr=1.e-2)
    # print(model.parameters[0])  # debug

    Y_pred = np.array([model.predict(x) for x in X])
    # print(Y)
    # print(Y_pred)
    # print(np.sum(Y.flatten() == Y_pred))
    print(f'Accuracy prior training: {accuracy(Y.flatten(), Y_pred)}')

    # # training cycle
    num_epochs = 300
    for i in range(num_epochs):
        model.batch_training(X, Y)
        Y_pred = np.array([model.predict(x) for x in X])
        print(f'Accuracy after epoch {i}: {accuracy(Y.flatten(), Y_pred)}')

    # print(model.parameters[0])  # debug

    Xdf = pd.DataFrame(columns=['x1', 'x2'], data=X)
    prediction_plot_classif(model, Xdf, Y.flatten(), '../plots/mlp_moons/pred_plot.png')


def main():
    # perceptron_try0()
    mlp_on_nonlinear()


if __name__ == '__main__':
    main()
