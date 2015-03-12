"""Test snippets to try out stuff from the book "Gaussian Processes for Machine
Learning" by Rasmussen & Williams (a.k.a. R&W in the docstrings below)."""

import sys

from math import pi, log
import numpy as np
from numpy import dot, identity, transpose
import scipy
import scipy.stats as stats
from scipy.linalg import cholesky
from scipy.stats import norm
from scipy.spatial.distance import sqeuclidean
from numpy.linalg import inv
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

def axes_maker(rows, cols):
    """Returns a closure that, when called, will return the next subplot in a
    figure. 'rows' and 'cols' indicate the number of subplots."""
    fig = plt.figure()
    current_subplot = [1] # Use list in order to modify
    def next_axes(**kwargs):
        current_subplot[0] += 1
        axes = fig.add_subplot(rows, cols, current_subplot[0] - 1, **kwargs)
        return axes
    return next_axes

def squared_exp_cov(x_p, x_q):
    """Calculates the squared exponential covariance function between the
    outputs f(x_p) and f(x_q) given the input vectors x_p and x_q, as per Eq. 2.16 of
    R&W. 

    NOTE: In contrast to sqeuclidean used below, the sq_dist function from the
    code accompanying the book calculates ALL pairwise distances between column
    vectors of two matrices."""
    return np.exp(-0.5 * sqeuclidean(x_p, x_q))

def my_multivariate_sample(mean, cov, cholesky_epsilon=1e-8):
    """Generate a multivariate sample, as per Sec. A.2 of R&W. Add
    cholesky_epsilon times the identity to ensure numerical stability."""
    n = mean.size
    sample_indep = norm.rvs(0, 1, size=(n, 1))
    joint = dot(cholesky(cov + np.identity(n) * cholesky_epsilon, lower=True),
                sample_indep)
    return mean + joint

def multivariate_sample(mean, cov, cholesky_epsilon=1e-8):
    '''Use scipy to create multivariate sample. Fall back on homemade algorithm.'''
    if scipy.__version__ < '0.14.0':
        print 'You have an old Scipy (version < 0.14.0). Falling back on '\
              'homemade algorithm to sample from multivariate normal.'
        return my_multivariate_sample(mean, cov, cholesky_epsilon=1e-8)
    else:
        rv = scipy.stats.multivariate_normal(mean, cov)
        return rv.rvs()

def make_cov_array(X_p, X_q, cov_fun):
    """Create a covariance matrix (actually a 2-D array), given input matrices
    X_p and X_q. These are D x n matrices, where D is the dimension of the
    input space and n is the number of input (e.g. training) cases."""
    n_p = X_p.shape[1]
    n_q = X_q.shape[1]
    K = np.array(np.zeros((n_p, n_q)))
    for i in range(n_p):
        for j in range(n_q):
            K[i, j] = cov_fun(X_p[:,i], X_q[:,j])
    return K

def make_se_cov_array(X_p, X_q):
    return make_cov_array(X_p, X_q, cov_fun=squared_exp_cov)

def unconditioned_sample(x_star=np.arange(-5, 5, 0.2),
                         cov_mtx_calculator=make_se_cov_array):
    """Create an unconditioned sample for one-dimensional test inputs x_star."""
    if len(x_star.shape) == 1:
        x_star.shape = (1, x_star.shape[0])
    K = cov_mtx_calculator(x_star, x_star)
    return (x_star, multivariate_sample(np.zeros(x_star.shape).T, K))

def conditioned_mean_cov_old(X, y, 
                       x_star=np.arange(-5, 5, 0.2),
                       noise_var=0,
                       cov_mtx_calculator=make_se_cov_array):
    '''Estimate mean and covariance for test inputs x_star, conditioned on
    the observations in X. X should be a D x n matrix and y a column(!)
    vector of length n. x_star is a column vector of test inputs.

    '''
    if len(x_star.shape) == 1:
        x_star.shape = (1, x_star.shape[0])
    K_x_xs = cov_mtx_calculator(X, x_star)
    K_x_x = cov_mtx_calculator(X, X)
    K_xs_x = cov_mtx_calculator(x_star, X)
    K_xs_xs = cov_mtx_calculator(x_star, x_star)
    K_x_x_inv = inv(K_x_x + noise_var * np.identity(K_x_x.shape[0]))
    mean = dot(dot(K_xs_x, K_x_x_inv), y)
    cov = K_xs_xs - dot(dot(K_xs_x, K_x_x_inv), K_x_xs)
    return mean, cov

def conditioned_mean_cov(X, y, x_star, noise_var=0,
                         cov_mtx_calculator=make_se_cov_array):
    '''Estimate mean and covariance for test inputs x_star, conditioned on
    the observations in X. X should be a D x n matrix and y a column(!)
    vector of length n. x_star is a column vector of test inputs.

    Based on Algorithm 2.1 of R&W.
    '''
    K = cov_mtx_calculator(X, X)
    K_star = cov_mtx_calculator(X, x_star)
    K_xs_xs = cov_mtx_calculator(x_star, x_star)
    L = cholesky(K + noise_var * identity(K.shape[0]), lower=True)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mean = dot(K_star.T, alpha)
    v = np.linalg.solve(L, K_star)
    cov = K_xs_xs - dot(v.T, v)
    return mean, cov

def conditioned_sample(X, y, 
                       x_star=np.arange(-5, 5, 0.2),
                       noise_var=0,
                       cov_mtx_calculator=make_se_cov_array):
    '''Create a sample conditioned on the observations. X should be a D x n
    matrix and y a column(!) vector of length n. x_star is a column
    vector of test inputs.

    '''
    mean, cov = conditioned_mean_cov(X, y, x_star, noise_var, cov_mtx_calculator)
    return (x_star, multivariate_sample(mean, cov), cov)

def gaussian_process_mean_pred(X, y, noise_var, x_star,
                               cov_mtx_calculator=make_se_cov_array):
    """Implementation of Algorithm 2.1 of R&W. X is a D x n matrix of
    observed driver variables, and y is the corresponding n-element
    column vector of observed dependent variables. x_star are the test
    inputs.

    """
    K = cov_mtx_calculator(X, X)
    L = cholesky(K + noise_var * identity(K.shape[0]), lower=True)
    L_inv = inv(L)
    a = dot(dot(inv(L.T), L_inv), y)
    def single_input_regression(x_in):
        x_in.shape = (x_in.shape[0], 1)
        k_in = cov_mtx_calculator(X, x_in)
        mean = dot(k_in.T, a)
        v = dot(L_inv, k_in)
        var = cov_mtx_calculator(x_in, x_in) - dot(v.T, v)
        return mean[0], var[0]
    mean_var = [single_input_regression(x_star[:,i]) \
                for i in range(x_star.shape[1])]
    mean, var = zip(*mean_var)
    n = X.shape[1]
    log_marg_lik = -0.5 * dot(y.T, a) - L.diagonal().sum() - n/2. * log(2*pi)
    return np.array(mean), np.array(var), log_marg_lik

if __name__ == '__main__':
    next_axes = axes_maker(2, 3)

    # # Testing the multivariate sampler.
    # cov = np.array([[1, .9], [.9, 1]])
    # mean = np.array([0, 0])
    # samples = [multivariate_sample(mean, cov) for i in range(10000)]
    # x, y = zip(*samples)
    # axes = next_axes()
    # axes.plot(x, y, '.')
    # axes.set_title("Multivariate sample,\ncov = [%2.1f %2.1f; %2.1f %2.1f]." % \
    #                tuple(np.array(cov).flatten()))

    # Plot the covariance matrix for the above observations and a subset of the
    # predictions
    x = np.arange(-5, 5, 0.25)
    x.shape = (1, x.shape[0])
    cov = make_cov_array(x, x, squared_exp_cov)
    i = np.arange(cov.shape[0])
    j = np.arange(cov.shape[1])
    i, j = np.meshgrid(i, j)
    axes = next_axes(projection="3d")
    surf = axes.plot_surface(i, j, cov, rstride=1, cstride=1,
                             linewidth=1, antialiased=True)
    axes.set_title("Covariance matrix for unconditioned GP")

    # Unconditioned samples from GP with SE covariance function
    axes = next_axes()
    for i in range(4):
        x, y = unconditioned_sample()
        print "y.shape:", y.shape
        axes.plot(x.T, y)
    axes.set_xlim([-5, 5])
    axes.set_title("Samples from unconditioned GP,\nref Fig. 2.2 (a) of R&W")

    # Adding observations, sample from the posterior
    observations = np.array(((-4, -2), (-3, 0), (-1, 1), (0, 2), (1, -1)))
    X = observations[:,0]
    X.shape = (1, X.shape[0])
    y = observations[:,1]
    y.shape = (y.shape[0], 1)
    axes = next_axes()
    axes.plot(X.T, y, '+', ms=10)
    for i in range(4):
        x_sampled, y_sampled, cov = conditioned_sample(X, y)
        print "y_sampled.shape:", y_sampled.shape
        axes.plot(x_sampled.T, y_sampled)
    axes.set_xlim([-5, 5])
    axes.set_title("Samples from conditioned GP,\nref Fig. 2.2 (b) of R&W")

    # Plot the covariance matrix for the above observations and a subset of the
    # predictions
    axes = next_axes(projection="3d")
    step=0.25
    cov = conditioned_sample(X, y, x_star=np.arange(-5, 5, step))[2]
    i = np.arange(cov.shape[0])
    j = np.arange(cov.shape[1])
    i, j = np.meshgrid(i, j)
    surf = axes.plot_surface(i, j, cov, rstride=1, cstride=1,
                             linewidth=1, antialiased=True)
    axes.set_title("Covariance matrix for conditioned GP\n (showing regressions " \
                   "with step length %f)" % step)

    # GP mean prediction on the observations
    axes = next_axes()
    axes.plot(X.T, y, '+', ms=10)
    x_star = np.arange(-5, 5, 0.1)
    x_star.shape = (1, x_star.shape[0])
    noise_var=0.01
    mean, var, log_lik = gaussian_process_mean_pred(X, y, noise_var=noise_var,
                                                    x_star=x_star)
    axes.plot(x_star.T, mean)
    axes.plot(x_star.T, mean + var, '--')
    axes.plot(x_star.T, mean - var, '--')
    axes.set_xlim([-5, 5])
    axes.set_title("GP mean prediction with noise %f +/- variance" % noise_var)

    # "Predicting" a periodic signal with a non-periodic covariance function
    X = np.linspace(-25, 25, 50 * 4)
    X.shape = (1, X.shape[0])
    y = np.sin(X.T)
    x_star=np.arange(0, 40, 0.4)
    axes = next_axes()
    axes.plot(X.T, y, '+')
    for i in range(3):
        # Regression only on the last part of the observations due to numerical
        # instability of Cholesky decomposition.
        x_s, y_s, cov = conditioned_sample(X, y, x_star, noise_var=0.1)
        axes.plot(x_s.T, y_s)
    axes.set_xlim([-25, 40])
    axes.set_title("\"Predicting\" a sine using SE covariance func")

    plt.show()
