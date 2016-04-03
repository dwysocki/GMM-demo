import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.stats import multivariate_normal

from functools import reduce


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


def clear_contours(ax, contours):
    if contours is None:
        return

    for coll in contours.collections:
        ax.collections.remove(coll)


class PlotContainer(object):
    contour = None
    scatter = None


def main():
    np.random.seed(1)

    plt_container = PlotContainer()
    plt_container.samples = 100

    ## Limits ##

    max_iter = 20

    var_min, var_max = 0.01, 2.00
    cov_min, cov_max = 0.00, 2.00

    weight_min, weight_max = 0.00, 1.00

    xmin, xmax = 0, 20
    ymin, ymax = 0, 20

    mean_min, mean_max = 5, 15

    ## Grid points for contours ##

    x, y = np.mgrid[0:20:0.01, 0:20:0.01]
    pos = np.dstack((x,y))

    ## Initial Slider Values ##

    init_T_weight = 0.8
    init_G_weight = 0.5

    init_T_mean_1x, init_T_mean_1y = 7, 7
    init_T_mean_2x, init_T_mean_2y = 12, 12
    init_G_mean_1x, init_G_mean_1y = mean_min, mean_min
    init_G_mean_2x, init_G_mean_2y = mean_max, mean_max

    init_T_cov_1_11, init_T_cov_1_22, init_T_cov_1_12 = 1.25, 1.25, 0.00
    init_T_cov_2_11, init_T_cov_2_22, init_T_cov_2_12 = 0.50, 0.50, 0.20
    init_G_cov_1_11, init_G_cov_1_22, init_G_cov_1_12 = 1.00, 1.00, 0.00
    init_G_cov_2_11, init_G_cov_2_22, init_G_cov_2_12 = 1.00, 1.00, 0.00

    ## Layout ##
    shape = 14, 5
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

    ax_T_cov_1_11 = plt.subplot2grid(shape, (5, 0))
    ax_T_cov_1_22 = plt.subplot2grid(shape, (6, 0))
    ax_T_cov_1_12 = plt.subplot2grid(shape, (7, 0))
    ax_T_cov_2_11 = plt.subplot2grid(shape, (8, 0))
    ax_T_cov_2_22 = plt.subplot2grid(shape, (9, 0))
    ax_T_cov_2_12 = plt.subplot2grid(shape, (10, 0))
    ax_G_cov_1_11 = plt.subplot2grid(shape, (5, 1))
    ax_G_cov_1_22 = plt.subplot2grid(shape, (6, 1))
    ax_G_cov_1_12 = plt.subplot2grid(shape, (7, 1))
    ax_G_cov_2_11 = plt.subplot2grid(shape, (8, 1))
    ax_G_cov_2_22 = plt.subplot2grid(shape, (9, 1))
    ax_G_cov_2_12 = plt.subplot2grid(shape, (10, 1))

    ax_T_weight = plt.subplot2grid(shape, (11, 0))
    ax_G_weight = plt.subplot2grid(shape, (11, 1))

    ax_iteration = plt.subplot2grid(shape, (12, 0),
                                    colspan=2)
    ax_blank = plt.subplot2grid(shape, (13, 0))
    ax_refit = plt.subplot2grid(shape, (13, 1))

    ax_plot = plt.subplot2grid(shape, (0, 2),
                               colspan=width-2,
                               rowspan=height)

    ## Labels ##
    ax_T_label.get_xaxis().set_visible(False)
    ax_T_label.get_yaxis().set_visible(False)
    ax_G_label.get_xaxis().set_visible(False)
    ax_G_label.get_yaxis().set_visible(False)

    ax_T_label.text(0.5, 0.5, "True Dist",
                    ha="center", va="center")
    ax_G_label.text(0.5, 0.5, "Init Guess",
                    ha="center", va="center")

    ## Mean Sliders ##

    sl_T_mean_1x = Slider(ax_T_mean_1x, r"$\mu_{1x}$",
                          mean_min, mean_max, valinit=init_T_mean_1x)
    sl_T_mean_1y = Slider(ax_T_mean_1y, r"$\mu_{1y}$",
                          mean_min, mean_max, valinit=init_T_mean_1y)
    sl_T_mean_2x = Slider(ax_T_mean_2x, r"$\mu_{2x}$",
                          mean_min, mean_max, valinit=init_T_mean_2x)
    sl_T_mean_2y = Slider(ax_T_mean_2y, r"$\mu_{2y}$",
                          mean_min, mean_max, valinit=init_T_mean_2y)
    sl_G_mean_1x = Slider(ax_G_mean_1x, r"$\mu_{1x}$",
                          mean_min, mean_max, valinit=init_G_mean_1x)
    sl_G_mean_1y = Slider(ax_G_mean_1y, r"$\mu_{1y}$",
                          mean_min, mean_max, valinit=init_G_mean_1y)
    sl_G_mean_2x = Slider(ax_G_mean_2x, r"$\mu_{2x}$",
                          mean_min, mean_max, valinit=init_G_mean_2x)
    sl_G_mean_2y = Slider(ax_G_mean_2y, r"$\mu_{2y}$",
                          mean_min, mean_max, valinit=init_G_mean_2y)

    ## Covariance Sliders ##

    sl_T_cov_1_11 = Slider(ax_T_cov_1_11, r"$\Sigma^1_{11}$",
                           var_min, var_max, valinit=init_T_cov_1_11)
    sl_T_cov_1_22 = Slider(ax_T_cov_1_22, r"$\Sigma^1_{22}$",
                           var_min, var_max, valinit=init_T_cov_1_22)
    sl_T_cov_1_12 = Slider(ax_T_cov_1_12, r"$\Sigma^1_{12}$",
                           cov_min, cov_max, valinit=init_T_cov_1_12)
    sl_T_cov_2_11 = Slider(ax_T_cov_2_11, r"$\Sigma^2_{11}$",
                           var_min, var_max, valinit=init_T_cov_2_11)
    sl_T_cov_2_22 = Slider(ax_T_cov_2_22, r"$\Sigma^2_{22}$",
                           var_min, var_max, valinit=init_T_cov_2_22)
    sl_T_cov_2_12 = Slider(ax_T_cov_2_12, r"$\Sigma^2_{12}$",
                           cov_min, cov_max, valinit=init_T_cov_2_12)
    sl_G_cov_1_11 = Slider(ax_G_cov_1_11, r"$\Sigma^1_{11}$",
                           var_min, var_max, valinit=init_G_cov_1_11)
    sl_G_cov_1_22 = Slider(ax_G_cov_1_22, r"$\Sigma^1_{22}$",
                           var_min, var_max, valinit=init_G_cov_1_22)
    sl_G_cov_1_12 = Slider(ax_G_cov_1_12, r"$\Sigma^1_{12}$",
                           cov_min, cov_max, valinit=init_G_cov_1_12)
    sl_G_cov_2_11 = Slider(ax_G_cov_2_11, r"$\Sigma^2_{11}$",
                           var_min, var_max, valinit=init_G_cov_2_11)
    sl_G_cov_2_22 = Slider(ax_G_cov_2_22, r"$\Sigma^2_{22}$",
                           var_min, var_max, valinit=init_G_cov_2_22)
    sl_G_cov_2_12 = Slider(ax_G_cov_2_12, r"$\Sigma^2_{12}$",
                           cov_min, cov_max, valinit=init_G_cov_2_12)

    ## Weight Sliders ##

    sl_T_weight = Slider(ax_T_weight, r"$\pi_1$",
                         weight_min, weight_max, valinit=init_T_weight)
    sl_G_weight = Slider(ax_G_weight, r"$\pi_1$",
                         weight_min, weight_max, valinit=init_G_weight)

    ## Iteration Slider ##

    sl_iteration = Slider(ax_iteration, r"Iter",
                          0, max_iter, valinit=0, valfmt="%d")

    ## Refit Button ##

    ax_blank.axis("off")
    bu_refit = Button(ax_refit, "Refit")


    def parse_params():
        """
        Set distribution parameters from slider values.
        """
        ## Weights ##
        T_weight = sl_T_weight.val
        G_weight = sl_G_weight.val
        weight_true  = np.array([T_weight, 1-T_weight])
        weight_guess = np.array([G_weight, 1-G_weight])
        ## Means ##
        mean_true  = np.array([
            [sl_T_mean_1x.val, sl_T_mean_1y.val],
            [sl_T_mean_2x.val, sl_T_mean_2y.val]
        ])
        mean_guess = np.array([
            [sl_G_mean_1x.val, sl_G_mean_1y.val],
            [sl_G_mean_2x.val, sl_G_mean_2y.val]
        ])
        ## Cov's ##
        cov_true  = np.array([
            [[sl_T_cov_1_11.val, sl_T_cov_1_12.val],
             [sl_T_cov_1_12.val, sl_T_cov_1_22.val]],

            [[sl_T_cov_2_11.val, sl_T_cov_2_12.val],
             [sl_T_cov_2_12.val, sl_T_cov_2_22.val]]
        ])
        cov_guess = np.array([
            [[sl_G_cov_1_11.val, sl_G_cov_1_12.val],
             [sl_G_cov_1_12.val, sl_G_cov_1_22.val]],

            [[sl_G_cov_2_11.val, sl_G_cov_2_12.val],
             [sl_G_cov_2_12.val, sl_G_cov_2_22.val]]
        ])
        ## Store values in container ##
        plt_container.params_true  = dict(
            weight=weight_true,
            mean=mean_true,
            cov=cov_true
        )
        plt_container.params_guess = dict(
            weight=weight_guess,
            mean=mean_guess,
            cov=cov_guess
        )

    def plot_contours():
        i = int(sl_iteration.val)
        plt_container.contour = \
          ax_plot.contour(x, y,
                          multinorm_pdf(pos, plt_container.params_learned[i]))

    def update_true(*args, **kwargs):
        parse_params()
        data = multinorm_rvs(plt_container.samples, plt_container.params_true)
        plt_container.params_learned = learn(data,
                                             plt_container.params_guess,
                                             max_iter)

        if plt_container.scatter is not None:
            plt_container.scatter.set_offsets(data)
        else:
            plt_container.scatter = ax_plot.scatter(*(data.T))

        update_fit() # calls fig.canvas.draw_idle(), so no need to call here

    def update_fit(*args, **kwargs):
        clear_contours(ax_plot, plt_container.contour)
        plot_contours()
        fig.canvas.draw_idle()

    bu_refit.on_clicked(update_true)
    sl_iteration.on_changed(update_fit)

    update_true()

    ax_plot.set_xlim([xmin, xmax])
    ax_plot.set_ylim([ymin, ymax])
    fig.tight_layout(h_pad=0.5, w_pad=0.5)


    plt.show(block=True)



if __name__ == "__main__":
    main()
