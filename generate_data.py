from sklearn.datasets import make_moons, make_circles, make_blobs
import matplotlib.pyplot as plt


def get_dataset(random_state, n_samples, generate_type='moons'):
    if type == 'moons':
        X, y = make_moons(noise=0.09, random_state=random_state, n_samples=n_samples)
    if type == 'circles':
        X, y = make_circles(noise=0.09, random_state=random_state, n_samples=n_samples, factor=0.5)
    if type == 'blobs':
        X, y = make_blobs(random_state=random_state, n_samples=n_samples, centers=2)
    return X, y


def draw_dataset(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
