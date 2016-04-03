#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

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
    """
    Produce the pdf of a multi-modal normal distribution, with modes
    defined by *modes*. *modes* is a sequence of (loc, scale, weight)
    triplets.
    """
    return reduce(
        np.add,
        (params["weight"][k]*multivariate_normal.pdf(x,
                                                      mean=params["mean"][k],
                                                      cov=params["cov"][k])
                   for k in range(n_components(params))))


def multinorm_rvs(size, params):
    """
    Draw samples from a multi-modal normal distribution, with modes
    defined by *modes*. *modes* is a sequence of (loc, scale, weight)
    triplets.
    """
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

    fig, (ax1, ax2) = plt.subplots(1, 2)
    x, y = np.mgrid[-5:15:0.01, -5:15:0.01]
    pos = np.dstack((x,y))

    # Parameters of true distribution.
    weight_true = np.array([0.2, 0.8])
    mean_true = np.array([[5, 5],
                          [0, 0]])
    cov_true = np.array([[[1, 0],
                          [0, 1]],

                         [[3, 2],
                          [2, 3]]])
    params_true = dict(weight=weight_true,
                       mean=mean_true,
                       cov=cov_true)

    data = multinorm_rvs(100, params_true)

    # Initial guess for parameters
    weight_guess = np.array([0.5, 0.5])
    mean_guess = np.array([[0, 10],
                           [10, 0]])
    cov_guess = np.array([[[1, 0],
                           [0, 1]],

                          [[1, 0],
                           [0, 1]]])
    params_guess = dict(weight=weight_guess,
                        mean=mean_guess,
                        cov=cov_guess)

    ax1.scatter(*(data.T))
    ax1.contour(x, y, multinorm_pdf(pos, params_guess))

    params_learned = learn(data, params_guess, 100)[-1]

    ax2.scatter(*(data.T))
    ax2.contour(x, y, multinorm_pdf(pos, params_learned))

    print("True Distribution:")
    pprint(params_true)
    print("Learned Distribution:")
    pprint(params_learned)

    fig.show()


if __name__ == "__main__":
    main()
