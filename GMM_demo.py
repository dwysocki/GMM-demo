#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

from matplotlib.widgets import Slider, Button

from functools import reduce
from itertools import product

from pprint import pprint




def _get_dimn(data, params):
    N, d = np.shape(data)
    K = n_components(params)

    return N, K, d


def self_outer(X):
    return np.outer(X, X)


def n_components(params):
    return np.size(params["weight"])


def multinorm_pdf(x, params):
    return reduce(
        np.add,
        (params["weight"][k]*multivariate_normal.pdf(x,
                                                      mean=params["mean"][k],
                                                      cov=params["cov"][k])
                   for k in range(n_components(params))))


def multinorm_rvs(size, params):
    counts = np.random.multinomial(size, params["weight"])
    return np.concatenate([
        multivariate_normal.rvs(size=counts[k],
                                mean=params["mean"][k],
                                cov=params["cov"][k])
        for k in range(n_components(params))
    ])


def index_product(*indices):
    """
    Return an iterable of indices, for all combinations.

    for i, j in index_product(I, J):
        ...

    is equivalent to

    for i in range(I):
        for j in range(J):
            ...
    """
    return product(*map(range, indices))


def responsibility(data, params,
                   dtype=float):
    """
    Computes the responsibility matrix r_{ik}, as given by Murphy 11.27.
    """

    N, K, _ = _get_dimn(data, params)

    Rmat = np.empty((N, K), dtype=dtype)

    for i in range(N):
        for k in range(K):
            Rmat[i, k] = \
              multivariate_normal.pdf(data[i],
                                      mean=params["mean"][k],
                                      cov=params["cov"][k])

        Rmat[i] *= params["weight"]
        Rmat[i] /= np.sum(Rmat[i])

    return Rmat


def iterate(data, params,
            dtype=float):
    N, K, d = _get_dimn(data, params)

    mean = params["mean"]

    Rmat = responsibility(data, params, dtype=dtype)
    Rvec = np.sum(Rmat, axis=0)

    weight_new = Rvec / N
    mean_new = np.empty((K, d), dtype=dtype)
    cov_new    = np.empty((K, d, d), dtype=dtype)
    for k in range(K):
        mean_new[k] = \
            sum(Rmat[i, k] * data[i]
                for i in range(N)) / Rvec[k]
        cov_new[k] = \
            sum(Rmat[i, k] * self_outer(data[i] - mean_new[k])
                for i in range(N)) / Rvec[k]

    return dict(weight=weight_new,
                mean=mean_new,
                cov=cov_new)


def learn(data, params_guess, iterations, dtype=float):
    params = [params_guess]
    params_next = params_guess

    for i in range(iterations):
        params_next = iterate(data, params_next, dtype=dtype)
        params.append(params_next)

    return params


def main():
    np.random.seed(1)

    xmin, xmax = 0, 10
    ymin, ymax = 0, 10
    x, y = np.mgrid[0:10:0.01, 0:10:0.01]
    pos = np.dstack((x,y))

    shape = 11, 5
    height, width = shape

    fig, _ = plt.subplots(height, width)

    ax_T_label = plt.subplot2grid(shape, (0, 0))
    ax_G_label = plt.subplot2grid(shape, (0, 1))

    ax_T_mean_1x = plt.subplot2grid(shape, (1, 0))
    ax_T_mean_1y = plt.subplot2grid(shape, (2, 0))
    ax_T_mean_2x = plt.subplot2grid(shape, (3, 0))
    ax_T_mean_2y = plt.subplot2grid(shape, (4, 0))
    ax_G_mean_1x = plt.subplot2grid(shape, (1, 1))
    ax_G_mean_1y = plt.subplot2grid(shape, (2, 1))
    ax_G_mean_2x = plt.subplot2grid(shape, (3, 1))
    ax_G_mean_2y = plt.subplot2grid(shape, (4, 1))

    ax_T_cov_11 = plt.subplot2grid(shape, (5, 0))
    ax_T_cov_22 = plt.subplot2grid(shape, (6, 0))
    ax_T_cov_12 = plt.subplot2grid(shape, (7, 0))
    ax_G_cov_11 = plt.subplot2grid(shape, (5, 1))
    ax_G_cov_22 = plt.subplot2grid(shape, (6, 1))
    ax_G_cov_12 = plt.subplot2grid(shape, (7, 1))

    ax_T_weight = plt.subplot2grid(shape, (8, 0))
    ax_G_weight = plt.subplot2grid(shape, (8, 1))

    ax_iteration = plt.subplot2grid(shape, (9, 0),
                                    colspan=2)
    ax_blank = plt.subplot2grid(shape, (10, 0))
    ax_blank.axis("off")
    ax_refit = plt.subplot2grid(shape, (10, 1))

    ax_plot = plt.subplot2grid(shape, (0, 2),
                               colspan=width-2,
                               rowspan=height)

    ax_T_label.get_xaxis().set_visible(False)
    ax_T_label.get_yaxis().set_visible(False)
    ax_G_label.get_xaxis().set_visible(False)
    ax_G_label.get_yaxis().set_visible(False)

    ax_T_label.text(0.5, 0.5, "True Dist", ha="center", va="center")
    ax_G_label.text(0.5, 0.5, "Init Guess", ha="center", va="center")

    sl_T_mean_1x = Slider(ax_T_mean_1x, r"$\mu_{1x}$",
                          xmin, xmax, valinit=xmin)
    sl_T_mean_1y = Slider(ax_T_mean_1y, r"$\mu_{1y}$",
                          xmin, ymax, valinit=ymin)
    sl_T_mean_2x = Slider(ax_T_mean_2x, r"$\mu_{2x}$",
                          xmin, xmax, valinit=xmin)
    sl_T_mean_2y = Slider(ax_T_mean_2y, r"$\mu_{2y}$",
                          xmin, ymax, valinit=ymin)
    sl_G_mean_1x = Slider(ax_G_mean_1x, r"$\mu_{1x}$",
                          xmin, xmax, valinit=xmin)
    sl_G_mean_1y = Slider(ax_G_mean_1y, r"$\mu_{1y}$",
                          xmin, ymax, valinit=ymin)
    sl_G_mean_2x = Slider(ax_G_mean_2x, r"$\mu_{2x}$",
                          xmin, xmax, valinit=xmin)
    sl_G_mean_2y = Slider(ax_G_mean_2y, r"$\mu_{2y}$",
                          xmin, ymax, valinit=ymin)

    sl_T_cov_11 = Slider(ax_T_cov_11, r"$\Sigma_{11}$",
                          0.01, 2, valinit=1)
    sl_T_cov_22 = Slider(ax_T_cov_22, r"$\Sigma_{22}$",
                          0.01, 2, valinit=1)
    sl_T_cov_12 = Slider(ax_T_cov_12, r"$\Sigma_{12}$",
                          0.00, 2, valinit=0)
    sl_G_cov_11 = Slider(ax_G_cov_11, r"$\Sigma_{11}$",
                          0.01, 2, valinit=1)
    sl_G_cov_22 = Slider(ax_G_cov_22, r"$\Sigma_{22}$",
                          0.01, 2, valinit=1)
    sl_G_cov_12 = Slider(ax_G_cov_12, r"$\Sigma_{12}$",
                          0.00, 2, valinit=0)

    sl_T_weight = Slider(ax_T_weight, r"$\pi_1$",
                         0.00, 1.00, valinit=0.5)
    sl_G_weight = Slider(ax_G_weight, r"$\pi_1$",
                         0.00, 1.00, valinit=0.5)

    sl_iteration = Slider(ax_iteration, r"Iter",
                          0, 100, valinit=0, valfmt="%d")

    bu_refit = Button(ax_refit, "Refit")


    def update_true(*args, **kwargs):
        fig.canvas.draw_idle()

    sl_T_mean_1x.on_changed(update_true)

    # Parameters of true distribution.
    weight_true = np.array([0.2, 0.8])
    mean_true = np.array([[7, 7],
                          [2, 2]])
    cov_true = np.array([[[1, 0],
                          [0, 1]],

                         [[2, 1],
                          [1, 2]]])
    params_true = dict(weight=weight_true,
                       mean=mean_true,
                       cov=cov_true)

    data = multinorm_rvs(100, params_true)

    # Initial guess for parameters
    weight_guess = np.array([0.5, 0.5])
    mean_guess = np.array([[2, 2],
                           [7, 7]])
    cov_guess = np.array([[[1, 0],
                           [0, 1]],

                          [[1, 0],
                           [0, 1]]])
    params_guess = dict(weight=weight_guess,
                        mean=mean_guess,
                        cov=cov_guess)

    ax_plot.scatter(*(data.T))
    ax_plot.contour(x, y, multinorm_pdf(pos, params_guess))

    params_learned = learn(data, params_guess, 100)[-1]

    print("True Distribution:")
    pprint(params_true)
    print("Learned Distribution:")
    pprint(params_learned)

    fig.tight_layout(h_pad=0.5, w_pad=0.2)

    plt.show(block=True)



if __name__ == "__main__":
    main()
