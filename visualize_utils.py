import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier


def make_meshgrid(X_train, X_test, y_train, y_test, h=.02):
    x_min = min(X_train[:, 0].min(), X_test[:, 0].min()) - .5
    x_max = max(X_train[:, 0].max(), X_test[:, 0].max()) + .5

    y_min = min(X_train[:, 1].min(), X_test[:, 1].min()) - .5
    y_max = max(X_train[:, 1].max(), X_test[:, 1].max()) + .5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    return xx, yy


def predict_proba_om_mesh(clf, xx, yy):
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    return Z


def plot_predictions(xx, yy, Z, plot_name="1.png", X_train=None, X_test=None,
                     y_train=None, y_test=None, figsize=(10, 10),
                     title="predictions",
                     cm=plt.cm.RdBu,
                     cm_bright=ListedColormap(['#FF0000', '#0000FF'])):
    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    if X_train is not None:
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, alpha=0.2)

    if X_test is not None:
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                    edgecolors='k')

    plt.xlim((xx.min(), xx.max()))
    plt.ylim((yy.min(), yy.max()))
    plt.title(plot_name)
    plt.tight_layout()
    # plt.show()
    plt.savefig(plot_name)
