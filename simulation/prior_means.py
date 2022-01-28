import torch
import sys
import time

import dill as pkl

sys.path.append('.')

from .dynamics import dynamics_traj
from utils.utils import reshape_pt1, reshape_pt1_tonormal, reshape_dim1

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


# Prior mean objects for several systems, of form f0(x, u, prior_kwargs) when
# called.
# These are called a lot so they should be as efficient as possible:
# class of functions like dynamics_functions and observer_functions,
# not recreating objects all the time, using a simple solver (like Euler) if
# one step of simple dynamical system is necessary, and None if only zeros as
# prior!

# Priors for the Duffing osccilator
class Duffing_prior:

    def __init__(self, device, prior_kwargs):
        self.device = device
        self.alpha = prior_kwargs.get('alpha')
        self.beta = prior_kwargs.get('beta')
        self.delta = prior_kwargs.get('delta')
        self.dt = prior_kwargs.get('dt')
        self.A = torch.tensor([[0., 1.], [-self.alpha, -self.delta]],
                              device=self.device)

    def continuous_prior_mean(self, x, u, prior_kwargs):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        F = torch.tensor([
            torch.zeros_like(x[:, 0]),
            - self.beta * torch.pow(x[:, 0], 3) + reshape_pt1_tonormal(u)],
            device=device)
        xdot = torch.matmul(self.A, x.t()).t() + F
        return xdot

    def continuous_justvelocity(self, x, u, prior_kwargs):
        xdot = self.continuous_prior_mean(x, u, prior_kwargs)
        return reshape_dim1(xdot[:, -1])

    # Prior mean for continuous time Duffing equation extended for
    # Michelangelo's extended high gain observer. Gives Phihat(xhat)
    def continuous_Michelangelo_u(self, x, u, prior_kwargs):
        x = reshape_pt1(x)
        phi = reshape_dim1(- self.beta * ((x[:, 0]) ** 3) - self.alpha *
                           x[:, 0] - self.delta * x[:, 1]) + reshape_pt1(u)
        return phi

    def continuous_Michelangelo_deriv(self, x, u, prior_kwargs):
        x = reshape_pt1(x)
        phi_deriv = torch.tensor([
            - 3 * self.beta * ((x[:, 0]) ** 2) - self.alpha,
            - self.delta * torch.ones_like(x[:, 0])])
        return reshape_pt1(phi_deriv)

    def continuous_Michelangelo_deriv_u(self, x, u, prior_kwargs):
        deriv = self.continuous_Michelangelo_deriv(
            x, u, prior_kwargs)
        phi_deriv = torch.cat(
            (deriv, torch.ones((deriv.shape[0], u.shape[1]))), dim=1)
        return phi_deriv

    # Prior mean only velocity for discrete given continuous prior mean
    # Gives x_n(t) + dt * f_prior(t)
    def discrete_justvelocity(self, x, u, prior_kwargs):
        x = reshape_pt1(x)
        u = reshape_pt1(u)

        def version(t, xl, ul, t0, init_control, process_noise_var, kwargs,
                    impose_init_control):
            return self.continuous_prior_mean(xl, ul, kwargs)

        vnext = torch.tensor([reshape_pt1_tonormal(
            dynamics_traj(x0=x[i, :], u=u[i, :], t0=0, dt=self.dt,
                          init_control=u[i, :],
                          discrete=False, version=version, meas_noise_var=0,
                          process_noise_var=0, method='dopri5',
                          t_eval=[self.dt], kwargs=prior_kwargs)[:, -1])
            for i in range(len(x))])
        return reshape_dim1(vnext)


# Priors for a simple chain of intergators, without further information
class Chain_integ_prior:

    def __init__(self, device, prior_kwargs):
        self.device = device
        self.dt = prior_kwargs.get('dt')

    # Without prior mean, the prior on x_n(t+dt) is just x_n(t)
    def discrete_justvelocity(self, x, u, prior_kwargs):
        return reshape_dim1(x[:, -1])


# Read previous GP model stored with pickle in init, make predictions with
# GP model contained in class or in prior
class read_discrete_GP_prior:

    def __init__(self, prior_GP_model_file, kernel=None):
        prior_GP_model = pkl.load(open(prior_GP_model_file, 'rb'))
        if kernel:
            prior_GP_model.kernel = kernel
        self.prior_GP_model = prior_GP_model

    def predict(self, x, u, scale=True, only_prior=False):
        xnext, varnext, lowconf, uppconf = self.prior_GP_model.predict(
            reshape_pt1(x), reshape_pt1(u), scale=scale, only_prior=only_prior)
        return xnext, varnext, lowconf, uppconf

    def predict_deriv(self, x, u, scale=True, only_x=False, only_prior=False):
        xnext, varnext, lowconf, uppconf = self.prior_GP_model.predict_deriv(
            reshape_pt1(x), reshape_pt1(u), scale=scale, only_x=only_x,
            only_prior=only_prior)
        return xnext, varnext, lowconf, uppconf

    def predict_onlymean(self, x, u, scale=True, only_prior=False):
        return self.predict(reshape_pt1(x), reshape_pt1(u), scale=scale,
                            only_prior=only_prior)[0]

    def predict_deriv_onlymean(self, x, u, scale=True, only_x=False,
                               only_prior=False):
        return self.predict_deriv(
            reshape_pt1(x), reshape_pt1(u), scale=scale, only_x=only_x,
            only_prior=only_prior)[0]
