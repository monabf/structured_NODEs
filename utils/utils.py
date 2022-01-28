import copy
import logging
import sys
from typing import Callable

import numpy as np
import torch
from pandas.api.types import is_numeric_dtype
from scipy.interpolate import interp1d
from torchinterp1d import Interp1d

# Some useful functions

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


# Root mean square of a matrix
def RMS(x):
    if torch.is_tensor(x):
        return torch.sqrt(torch.sum(torch.mean(torch.square(x), dim=0)))
    else:
        return np.sqrt(np.sum(np.mean(np.square(x), axis=0)))

# Various MSE/RMSE/MS errors averaged over different dimensions
def MS(x, dim=None):
    if dim is None:
        return torch.mean(torch.square(x))
    else:
        return torch.mean(torch.square(x), dim=dim)


def MSE(x, y, dim=None):
    """
    Compute the mean squared error between x and y along dimension dim.

    Parameters
    ----------
    x: torch.tensor
    y: torch.tensor
    dim: int
        Dimension along which to compute the mean.

    Returns
    -------
    error: torch.tensor
        Computed RMSE.
    """
    error = torch.nn.functional.mse_loss(x, y, reduction='none')
    if dim is None:
        return torch.mean(error)
    else:
        return torch.mean(error, dim=dim)


def RMSE(x, y, dim=None):
    """
    Compute the root mean squared error between x and y along dimension dim.

    Parameters
    ----------
    x: torch.tensor
    y: torch.tensor
    dim: int
        Dimension along which to compute the mean.

    Returns
    -------
    error: torch.tensor
        Computed RMSE.
    """
    return torch.sqrt(MSE(x=x, y=y, dim=dim))


# Log likelihood of a matrix given a mean and variance of same shape
def log_multivariate_normal_likelihood(x, mean, var):
    # TODO use apply to vectorialize over matrix rather than for loop!
    device = x.device
    log_likelihood_array = torch.zeros((x.shape[0], 1), device=device)
    for idx, xi in enumerate(x):
        covar = reshape_pt1_tonormal(var[idx])
        if len(torch.nonzero(covar, as_tuple=False)) == 0:
            covar = 1e-8 * torch.ones_like(covar, device=device)
        if len(covar.shape) <= 1:
            distrib = torch.distributions.MultivariateNormal(
                mean[idx], torch.eye(mean[idx].shape[0], device=device) * covar)
        else:
            distrib = torch.distributions.MultivariateNormal(
                mean[idx], torch.diag(reshape_pt1_tonormal(covar)))
        log_likelihood_array[idx] = reshape_pt1(distrib.log_prob(xi))
    log_likelihood = torch.mean(log_likelihood_array, dim=0)
    return log_likelihood


# Create new numpy nested list from 2d torch tensor
def list_torch_to_numpy(t):
    l = []
    for i in range(len(t)):
        row = []
        for j in range(len(t[0])):
            row.append(copy.deepcopy(t[i][j].detach().cpu().numpy()))
        l.append(row)
    return l


# Create new 2d torch tensor from numpy nested list
def list_numpy_to_torch(l, device):
    t = torch.zeros((len(l), len(l[0])), device=device)
    for i in range(len(list)):
        for j in range(len(l[0])):
            t[i][j] = torch.tensor(l[i][j], device)  # copy
    return t


# Concatenate 2 nested lists along axis=1 as numpy would do it. Must be same
# length along axis=0
def concatenate_lists(l1, l2):
    l = copy.deepcopy(l1)
    for i in range(len(l1)):
        l[i] = l1[i] + l2[i]
    return l


# Reshape any vector of (length,) object to (length, 1) (possibly several
# points but of dimension 1)
def reshape_dim1(x):
    if torch.is_tensor(x):
        if len(x.shape) == 0:
            x = x.view(1, 1)
        elif len(x.shape) == 1:
            x = x.view(x.shape[0], 1)
    else:
        if np.isscalar(x) or np.array(x).ndim == 0:
            x = np.reshape(x, (1, 1))
        else:
            x = np.array(x)
        if len(x.shape) == 1:
            x = np.reshape(x, (x.shape[0], 1))
    return x


# Same as reshape_dim1 but for difftraj when the first 2 dimensions stay
def reshape_dim1_difftraj(x):
    if torch.is_tensor(x):
        if len(x.shape) == 0:
            x = x.veiw(1, 1, 1)
        elif len(x.shape) == 1:
            x = x.view(1, x.shape[0], 1)
        elif len(x.shape) == 2:
            x = x.view(x.shape[0], x.shape[1], 1)
    else:
        if np.isscalar(x) or np.array(x).ndim == 0:
            x = np.reshape(x, (1, 1, 1))
        else:
            x = np.array(x)
        if len(x.shape) == 1:
            x = np.reshape(x, (1, x.shape[0], 1))
        elif len(x.shape) == 2:
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x


# Reshape any vector of (length,) object to (1, length) (single point of
# certain dimension)
def reshape_pt1(x):
    if torch.is_tensor(x):
        if len(x.shape) == 0:
            x = x.view(1, 1)
        elif len(x.shape) == 1:
            x = x.view(1, x.shape[0])
    else:
        if np.isscalar(x) or np.array(x).ndim == 0:
            x = np.reshape(x, (1, 1))
        else:
            x = np.array(x)
        if len(x.shape) == 1:
            x = np.reshape(x, (1, x.shape[0]))
    return x


# Same as reshape_pt1 but for difftraj when the first and last dimensions stay
def reshape_pt1_difftraj(x):
    if torch.is_tensor(x):
        if len(x.shape) == 0:
            x = x.view(1, 1, 1)
        elif len(x.shape) == 1:
            x = x.view(1, 1, x.shape[0])
        elif len(x.shape) == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
    else:
        if np.isscalar(x) or np.array(x).ndim == 0:
            x = np.reshape(x, (1, 1, 1))
        else:
            x = np.array(x)
        if len(x.shape) == 1:
            x = np.reshape(x, (1, 1, x.shape[0]))
        elif len(x.shape) == 2:
            x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    return x


# Reshape any point of type (1, length) to (length,)
def reshape_pt1_tonormal(x):
    if torch.is_tensor(x):
        if len(x.shape) == 0:
            x = x.view(1, )
        elif len(x.shape) == 1:
            x = x.view(x.shape[0], )
        elif x.shape[0] == 1:
            x = torch.squeeze(x, 0)
    else:
        if np.isscalar(x) or np.array(x).ndim == 0:
            x = np.reshape(x, (1,))
        elif len(x.shape) == 1:
            x = np.reshape(x, (x.shape[0],))
        elif x.shape[0] == 1:
            x = np.reshape(x, (x.shape[1],))
    return x


# Reshape any vector of type (length, 1) to (length,)
def reshape_dim1_tonormal(x):
    if torch.is_tensor(x):
        if len(x.shape) == 0:
            x = x.view(1, )
        elif len(x.shape) == 1:
            x = x.view(x.shape[0], )
        elif x.shape[1] == 1:
            x = x.view(x.shape[0], )
    else:
        if np.isscalar(x) or np.array(x).ndim == 0:
            x = np.reshape(x, (1,))
        elif len(x.shape) == 1:
            x = np.reshape(x, (x.shape[0],))
        elif x.shape[1] == 1:
            x = np.reshape(x, (x.shape[0],))
    return x


# Functions returning the value of the information criterion to optimize at a
# certain point, given a trained GP model
def posterior_variance(x, model):
    x = reshape_pt1(x)
    (mean, var) = model.predict(x)
    return var


def entropy(x, model):
    x = reshape_pt1(x)
    (mean, var) = model.predict(x)
    return 1 / 2 * np.log(2 * np.pi * np.exp(0) * var ** 2)


# Remove outliers from a pandas dataframe
def remove_outlier(df):
    # https://gist.github.com/ariffyasri/70f1e9139da770cb8514998124560281
    low = .001
    high = .999
    quant_df = df.quantile([low, high])
    mask = [True]
    for name in list(df.columns):
        if is_numeric_dtype(df[name]):
            mask = (df[name] >= quant_df.loc[low, name]) & (
                    df[name] <= quant_df.loc[high, name])
    return mask


# Vector x = (t, x(t)) of time steps t at which x is known is interpolated at
# given time t, imposing initial value, and interpolating along each output
# dimension independently if there are more than one
# https://github.com/aliutkus/torchinterp1d
# https://stackoverflow.com/questions/44634158/scipy-parallel-interpolation-of-multiple-arrays
def interpolate(t, x, t0, init_value, method='linear', impose_init=False):
    x = reshape_pt1(x)
    if torch.is_tensor(x):
        with torch.no_grad():  # not building computational graph!
            if method != 'linear':
                raise NotImplementedError(
                    'Only linear regular grid interpolator available in '
                    'pytorch!')
            if x.device != t.device:
                logging.error('Data and time to interpolate should be on same '
                              'device!')
            points, values = reshape_dim1(x[:, 0].contiguous()).t(), \
                             reshape_dim1(x[:, 1:].contiguous()).t()
            if len(t.shape) == 0:
                t = t.view(1, 1)
            else:
                t = reshape_dim1(t.contiguous()).t()
            if len(x) == 1:
                # If only one value of x available, assume constant
                interpolate_x = x[0, 1:].repeat(len(t), 1)
            else:
                # Interpolate data t_x at array of times wanted; if several out
                # dims, interpolate all input dims for each output dim
                interpolate_x = Interp1d()(points.expand(
                    values.shape[0], -1), values, t).t()
                t = reshape_dim1_tonormal(t)
            if torch.isnan(torch.min(interpolate_x)):
                print(t, x)
                logging.error('NaNs in interpolation: values need to be '
                              'interpolated outside of range!')

    else:
        points, values = x[:, 0], x[:, 1:]
        if np.isscalar(t):
            t = np.array([t])
        else:
            t = reshape_dim1_tonormal(t)
        if len(x) == 1:
            # If only one value of x available, assume constant
            interpolate_x = np.tile(reshape_pt1(x[0, 1:]), (len(t), 1))
        else:
            # Interpolate data t_x at array of times wanted; if several output
            # dims, interpolate all input dims for each output dim
            interpolate_x = interp1d(x=points, y=values.T, kind=method,
                                     fill_value="extrapolate")
        if np.isnan(np.min(interpolate_x)):
            print(t, x)
            logging.error('NaNs in interpolation: values need to be '
                          'interpolated outside of range!')

    if t[0] == t0 and impose_init:
        # Impose initial value
        interpolate_x[0] = reshape_pt1(init_value)
    tf = x[-1, 0]

    # Interpolation slightly outside of range
    if len(x) >= 2:
        tol = 100 * (tf - x[-2, 0])
        if tf < t[-1] <= tf + tol:
            # If t[-1] less than tol over last available t, return x[-1]
            interpolate_x[-1] = reshape_pt1(x[-1, 1:])
        elif t0 > t[0] >= t0 - tol:
            # If t[0] lass than tol before first available t, return x[0]
            if impose_init:
                interpolate_x[0] = reshape_pt1(init_value)
            else:
                interpolate_x[0] = reshape_pt1(x[0, 1:])

    return reshape_pt1(interpolate_x)


# Vector x = (t_i, x(t_i)) of time steps t_i at which x is known is
# interpolated at given time t, interpolating along each output dimension
# independently if there are more than one. Returns a function interp(t,
# any other args) which interpolates x at times t
def interpolate_func(x, t0, init_value, method='linear', impose_init=False) -> \
        Callable:
    """
    Takes a vector of times and values, returns a callable function which
    interpolates the given vector (along each output dimension independently).

    :param x: vector of (t_i, x(t_i)) to interpolate
    :type x: torch.tensor
    param t0: initial time at which to impose initial condition
    :type t0: torch.tensor
    param init_value: initial condition to impose
    :type init_value: torch.tensor
    param impose_init: whether to impose initial condition
    :type impose_init: bool

    :returns: function interp(t, other args) which interpolates x at t
    :rtype:  Callable[[List[float]], np.ndarray]
    """
    x = reshape_pt1(x)
    if torch.is_tensor(x):  # not building computational graph!
        with torch.no_grad():
            if method != 'linear':
                raise NotImplementedError(
                    'Only linear interpolator available in pytorch!')
            points, values = reshape_dim1(x[:, 0].contiguous()).t(), \
                             reshape_dim1(x[:, 1:].contiguous()).t()
            interp_function = Interp1d()

            def interp(t, *args, **kwargs):
                if len(t.shape) == 0:
                    t = t.view(1, 1)
                else:
                    t = reshape_dim1(t.contiguous()).t()
                if len(x) == 1:
                    # If only one value of x available, assume constant
                    interpolate_x = x[0, 1:].repeat(len(t[0]), 1)
                else:
                    interpolate_x = interp_function(points.expand(
                        values.shape[0], -1), values, t).t()
                if t[0, 0] == t0 and impose_init:
                    # Impose initial value
                    interpolate_x[0] = reshape_pt1(init_value)
                return interpolate_x

    else:
        points, values = x[:, 0], x[:, 1:].T
        interp_function = interp1d(x=points, y=values, kind=method,
                                   fill_value="extrapolate")

        def interp(t, *args, **kwargs):
            if np.isscalar(t):
                t = np.array([t])
            else:
                t = reshape_dim1_tonormal(t)
            if len(x) == 1:
                # If only one value of x available, assume constant
                interpolate_x = np.tile(reshape_pt1(x[0, 1:]), (len(t), 1))
            else:
                interpolate_x = interp_function(t).T
            if t[0] == t0 and impose_init:
                # Impose initial value
                interpolate_x[0] = reshape_pt1(init_value)
            return interpolate_x

    return interp

# Same as previous function, but as a class to enable pickling
# https://stackoverflow.com/questions/32883491/pickling-scipy-interp1d-spline
class Interpolate_func:
    def __init__(self, x, t0, init_value, method='linear', impose_init=False):
        """
        Takes a vector of times and values, returns a callable function which
        interpolates the given vector (along each output dimension
        independently).

        :param x: vector of (t_i, x(t_i)) to interpolate
        :type x: torch.tensor
        param t0: initial time at which to impose initial condition
        :type t0: torch.tensor
        param init_value: initial condition to impose
        :type init_value: torch.tensor
        param impose_init: whether to impose initial condition
        :type impose_init: bool

        :returns: function interp(t, other args) which interpolates x at t
        :rtype:  Callable[[List[float]], np.ndarray]
        """
        self.x = x
        self.t0 = t0
        self.init_value = init_value
        self.method = method
        self.impose_init = impose_init
        self.interp = self.create_interp_func(
            self.x, self.t0, self.init_value, self.method, self.impose_init)

    def __call__(self, *args, **kwargs):
        if self.interp is None:
            # recreate interp function (can be deleted when pickling...)
            self.interp = self.create_interp_func(
                self.x, self.t0, self.init_value, self.method, self.impose_init)
        return self.interp(*args, **kwargs)

    def __getstate__(self):
        return self.x, self.t0, self.init_value, self.method, \
               self.impose_init

    def __setstate__(self, state):
        self.interp = self.__init__(state[0], state[1], state[2], state[3],
                                    state[4])

    def create_interp_func(self, x, t0, init_value, method, impose_init):
        x = reshape_pt1(x)
        if torch.is_tensor(x):  # not building computational graph!
            with torch.no_grad():
                if method != 'linear':
                    raise NotImplementedError(
                        'Only linear interpolator available in pytorch!')
                points, values = reshape_dim1(x[:, 0].contiguous()).t(), \
                                 reshape_dim1(x[:, 1:].contiguous()).t()
                interp_function = Interp1d()

                def interp(t, *args, **kwargs):
                    if len(t.shape) == 0:
                        t = t.view(1, 1)
                    else:
                        t = reshape_dim1(t.contiguous()).t()
                    if len(x) == 1:
                        # If only one value of x available, assume constant
                        interpolate_x = x[0, 1:].repeat(len(t), 1)
                    else:
                        interpolate_x = interp_function(points.expand(
                            values.shape[0], -1), values, t).t()
                    if t[0, 0] == t0 and impose_init:
                        # Impose initial value
                        interpolate_x[0] = reshape_pt1(init_value)
                    return interpolate_x

        else:
            points, values = x[:, 0], x[:, 1:].T
            interp_function = interp1d(x=points, y=values, kind=method,
                                       fill_value="extrapolate")

            def interp(t, *args, **kwargs):
                if np.isscalar(t):
                    t = np.array([t])
                else:
                    t = reshape_dim1_tonormal(t)
                if len(x) == 1:
                    # If only one value of x available, assume constant
                    interpolate_x = np.tile(reshape_pt1(x[0, 1:]), (len(t), 1))
                else:
                    interpolate_x = interp_function(t).T
                if t[0] == t0 and impose_init:
                    # Impose initial value
                    interpolate_x[0] = reshape_pt1(init_value)
                return interpolate_x

        return interp

# General method for solving ODE of dynamics fx with RK4
# https://www.codeproject.com/Tips/792927/Fourth-Order-Runge-Kutta-Method-in-Python
def rk4(x, f, deltat, accelerate=False, accelerate_deltat=0.01, *args):
    x = reshape_pt1(x)
    if not accelerate:
        k1 = f(x, *args) * deltat
        xk = x + k1 / 2
        k2 = f(xk, *args) * deltat
        xk = x + k2 / 2
        k3 = f(xk, *args) * deltat
        xk = x + k3
        k4 = f(xk, *args) * deltat
        xnext = reshape_pt1(x + (k1 + 2 * (k2 + k3) + k4) / 6)
        return xnext
    else:
        nb_iter = int(np.ceil(deltat / accelerate_deltat))
        # Perform simulation with RK4 of accelerate_deltat given, but perform
        # nb_iter steps of that simulation to return x(t + deltat)
        for i in range(nb_iter):
            k1 = f(x, *args) * accelerate_deltat
            xk = x + k1 / 2
            k2 = f(xk, *args) * accelerate_deltat
            xk = x + k2 / 2
            k3 = f(xk, *args) * accelerate_deltat
            xk = x + k3
            k4 = f(xk, *args) * accelerate_deltat
            xnext = reshape_pt1(x + (k1 + 2 * (k2 + k3) + k4) / 6)
            x = xnext
        return xnext

# General method for solving ODE of dynamics fx with explicit Euler
def euler(x, f, deltat, accelerate=False, accelerate_deltat=0.01, *args):
    x = reshape_pt1(x)
    if not accelerate:
        xnext = reshape_pt1(x + deltat * f(x, *args))
        return xnext
    else:
        nb_iter = int(np.ceil(deltat / accelerate_deltat))
        # Perform simulation with RK4 of accelerate_deltat given, but perform
        # nb_iter steps of that simulation to return x(t + deltat)
        for i in range(nb_iter):
            xnext = reshape_pt1(x + accelerate_deltat * f(x, *args))
            x = xnext
        return xnext


# Real cubic root of negative numbers for pytorch
def torch_cbrt(x):
    # https://discuss.pytorch.org/t/incorrect-pow-function/62735/4
    return torch.sign(x) * torch.abs(x).pow(1. / 3)


# Kronecker product between two matrices for pytorch, in dev currently
def kronecker(matrix1, matrix2):
    return torch.ger(matrix1.view(-1), matrix2.view(-1)).view(*(
            matrix1.size() + matrix2.size())).permute([0, 2, 1, 3]).view(
        matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))


# Log functions
def start_log():
    logging.INFO
    logging.FileHandler("{0}/{1}.log".format(
        '../Figures/Logs', 'log' + str(sys.argv[1])))
    logging.StreamHandler(sys.stdout)


def stop_log():
    logging._handlers.clear()
    logging.shutdown()


def save_log(results_folder):
    logging.INFO
    logging.FileHandler(
        "{0}/{1}.log".format(results_folder, 'log' + str(sys.argv[1])))
    logging.basicConfig(level=logging.INFO)
