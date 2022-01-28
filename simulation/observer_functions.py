import logging

import numpy as np
import torch

from functools import partial

from utils.utils import reshape_pt1, reshape_pt1_tonormal

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


# Possible observers (dynamics functions f(x_estim, u)) and functions to
# produce measured data from true data


# Simple linear Luenberger observer
class Linear_Luenberger:
    def __init__(self, device, kwargs):
        self.device = device
        self.n = kwargs.init_state_estim.shape[1]
        self.L = kwargs.L
        self.observe_data = kwargs.observe_data

    def __call__(self, t, xhat, u, y, t0, init_control, GP,
                 kwargs, impose_init_control=False):
        xhat = reshape_pt1(xhat)
        y = reshape_pt1(y(t, kwargs))
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        if GP:
            if 'GP' in GP.__class__.__name__:
                mean, var, lowconf, uppconf = GP.predict(reshape_pt1(xhat),
                                                         reshape_pt1(u))
                if not kwargs.get('continuous_model'):
                    # Continuous observer dynamics, but GP is discrete
                    # TODO better than Euler?
                    mean = (mean - reshape_pt1(xhat[:, -1])) / \
                           GP.prior_kwargs['dt']
            else:
                mean = GP(reshape_pt1(xhat), reshape_pt1(u), kwargs.get(
                    'prior_kwargs'))
        else:
            mean = torch.zeros_like(reshape_pt1(xhat[:, -1]))
        if np.any(kwargs.get('saturation')):
            # Saturate the estimate of the nonlinearity to guarantee contraction
            a_min = np.min([kwargs.get('saturation')])
            a_max = np.max([kwargs.get('saturation')])
            mean = torch.clamp(mean, a_min, a_max)
        LC = reshape_pt1(self.L * (y - self.observe_data(xhat)))
        xhatdot = reshape_pt1(mean + LC)
        return xhatdot


# Linear system used in the nonlinear KKL observer
class KKL:
    def __init__(self, device, kwargs):
        self.device = device
        self.n = kwargs.get('init_state_estim').shape[1]
        self._D = kwargs.get('D')
        self._alpha = 1.  # scalar for scaling D
        self.F = kwargs.get('F')
        self.eigvals = torch.linalg.eig(self.D).eigenvalues
        for x in self.eigvals:
            if np.abs(np.real(x.item())) < 1e-5:
                logging.error('The eigenvalues of the matrix D are '
                              'dangerously  small, low robustness of the '
                              'observer! Increase the gains.')
            elif np.real(x.item()) > 0:
                logging.error('Some of the eigenvalues of the matrix D '
                              'are positive. Change the gains to get a '
                              'Hurwitz matrix.')
        if 'Db' in kwargs.keys():
            self._Db = kwargs.get('Db')
            self._alphab = 1.  # scalar for scaling Db
            self.Fb = kwargs.get('Fb')
            self.eigvals_backward = torch.linalg.eig(self.Db).eigenvalues
            for x in self.eigvals_backward:
                if np.abs(np.real(x.item())) < 1e-5:
                    logging.error('The eigenvalues of the matrix Db are '
                                  'dangerously  small, low robustness of the '
                                  'observer! Increase the gains.')
                elif np.real(x.item()) > 0:
                    logging.error('Some of the eigenvalues of the matrix Db '
                                  'are positive. Change the gains to get a '
                                  'Hurwitz matrix.')
        self.z_config = kwargs

    @property
    def D(self):
        return self._D * self._alpha

    @D.setter
    def D(self, D):
        self._D = D
        self._alpha = 1

    @property
    def Db(self):
        return self._Db * self._alphab

    @Db.setter
    def Db(self, Db):
        self._Db = Db
        self._alphab = 1

    def set_alpha(self, alpha):
        self._alpha = alpha

    def set_alphab(self, alphab):
        self._alphab = alphab

    def __call__(self, t, xhat, u, y, t0, init_control, GP, kwargs,
                 impose_init_control=False):
        # Return xhatdot with controllable (D,F), D Hurwitz
        xhat = reshape_pt1(xhat)
        y = reshape_pt1(y(t, kwargs))
        return self.call_with_yu(xhat, y)

    def call_with_yu(self, xhat, yu):
        xhatdot = torch.matmul(xhat, self.D.t()) + torch.matmul(yu, self.F.t())
        return reshape_pt1(xhatdot)

    def backward(self, t, xhat, u, y, t0, init_control, GP, kwargs,
                 impose_init_control=False):
        # Return xhatdot with different controllable (Db,Fb), Db Hurwitz but
        # simulation/t_eval in backward time
        xhat = reshape_pt1(xhat)
        y = reshape_pt1(y(t, kwargs))
        return self.backward_with_yu(xhat, y)

    def backward_with_yu(self, xhat, yu):
        xhatdot = torch.matmul(xhat, self.Db.t()) + \
                  torch.matmul(yu, self.Fb.t())
        return xhatdot


# Linear system used in the nonlinear KKL observer, taking control input as
# an extra output generated by an extra dynamical system which we do not seek
# to estimate
class KKLu:
    def __init__(self, device, kwargs):
        self.device = device
        self.n = kwargs.get('init_state_estim').shape[1]
        self._D = kwargs.get('D')
        self._alpha = 1.  # scalar for scaling D
        self.F = kwargs.get('F')
        self.eigvals = torch.linalg.eig(self.D).eigenvalues
        for x in self.eigvals:
            if np.abs(np.real(x.item())) < 1e-5:
                logging.error('The eigenvalues of the matrix D are '
                              'dangerously  small, low robustness of the '
                              'observer! Increase the gains.')
            elif np.real(x.item()) > 0:
                logging.error('Some of the eigenvalues of the matrix D '
                              'are positive. Change the gains to get a '
                              'Hurwitz matrix.')
        if 'Db' in kwargs.keys():
            self._Db = kwargs.get('Db')
            self._alphab = 1.  # scalar for scaling Db
            self.Fb = kwargs.get('Fb')
            self.eigvals_backward = torch.linalg.eig(self.Db).eigenvalues
            for x in self.eigvals_backward:
                if np.abs(np.real(x.item())) < 1e-5:
                    logging.error('The eigenvalues of the matrix Db are '
                                  'dangerously  small, low robustness of the '
                                  'observer! Increase the gains.')
                elif np.real(x.item()) > 0:
                    logging.error('Some of the eigenvalues of the matrix Db '
                                  'are positive. Change the gains to get a '
                                  'Hurwitz matrix.')
        self.z_config = kwargs

    @property
    def D(self):
        return self._D * self._alpha

    @D.setter
    def D(self, D):
        self._D = D
        self._alpha = 1

    @property
    def Db(self):
        return self._Db * self._alphab

    @Db.setter
    def Db(self, Db):
        self._Db = Db
        self._alphab = 1

    def set_alpha(self, alpha):
        self._alpha = alpha

    def set_alphab(self, alphab):
        self._alphab = alphab

    def __call__(self, t, xhat, u, y, t0, init_control, GP, kwargs,
                 impose_init_control=False):
        # Return xhatdot with controllable (D,F), D Hurwitz
        xhat = reshape_pt1(xhat)
        y = reshape_pt1(y(t, kwargs))
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        yu = torch.cat((y, u), dim=1)
        return self.call_with_yu(xhat, yu)

    def call_with_yu(self, xhat, yu):
        xhatdot = torch.matmul(xhat, self.D.t()) + torch.matmul(yu, self.F.t())
        return reshape_pt1(xhatdot)

    def backward(self, t, xhat, u, y, t0, init_control, GP, kwargs,
                 impose_init_control=False):
        # Return xhatdot with different controllable (Db,Fb), Db Hurwitz but
        # simulation/t_eval in backward time
        xhat = reshape_pt1(xhat)
        y = reshape_pt1(y(t, kwargs))
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        yu = torch.cat((y, u), dim=1)
        return self.backward_with_yu(xhat, yu)

    def backward_with_yu(self, xhat, yu):
        xhatdot = torch.matmul(xhat, self.Db.t()) + \
                  torch.matmul(yu, self.Fb.t())
        return xhatdot


# High gain extended observer from Michelangelo for the mass-spring-mass system
# Using current GP estimation of dynamics for xi_dot, high gain observer
# from Michelangelo's paper, extended with extra state variable xi
class MSM_observer_Michelangelo_GP:
    def __init__(self, device, kwargs):
        self.device = device
        self.n = kwargs.init_state_estim[:, :-1].shape[1]
        # Gain (needs to be large enough)
        self.g = kwargs.get('prior_kwargs').get('observer_gains').get('g')
        self.K = torch.zeros(self.n, device=self.device)
        for i in range(self.n):
            self.K[i] = kwargs.get('prior_kwargs').get('observer_gains').get(
                'k' + str(i + 1))
        self.Gamma1 = self.K * torch.tensor([
            self.g ** (i + 1) for i in range(self.n)], device=self.device)
        self.Gamma2 = reshape_pt1([
            kwargs.get('prior_kwargs').get('observer_gains').get(
                'k' + str(self.n + 1)) * self.g ** (self.n + 1)])
        self.A = torch.diag_embed(torch.ones(self.n - 1, device=device),
                                  offset=1)
        self.B = torch.zeros((self.n, 1), device=self.device)
        self.B[-1] = 1
        self.C = torch.zeros((1, self.n), device=self.device)
        self.C[0] = 1
        self.M = self.A - torch.matmul(reshape_pt1(self.K).t(), self.C)
        # Also check eigenvalues of M for stability without high gain
        self.eigvals = torch.linalg.eig(self.M).eigenvalues
        for x in self.eigvals:
            if np.abs(np.real(x.item())) < 1e-5:
                logging.error('The eigenvalues of the matrix M are '
                              'dangerously  small, low robustness of the '
                              'observer! Increase the gains.')
            elif np.real(x.item()) > 0:
                logging.error('Some of the eigenvalues of the matrix M '
                              'are positive. Change the gains to get a '
                              'Hurwitz matrix.')

    def __call__(self, t, xhat, u, y, t0, init_control, GP, kwargs,
                 impose_init_control=False):
        device = xhat.device
        x = reshape_pt1(xhat)
        xhat = reshape_pt1(x[:, :-1])
        xi = reshape_pt1(x[:, -1])
        y = reshape_pt1(y(t, kwargs))
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        if GP:
            if 'GP' in GP.__class__.__name__:
                mean_deriv, var_deriv, lowconf_deriv, uppconf_deriv = \
                    GP.predict_deriv(reshape_pt1(xhat), reshape_pt1(u),
                                     only_x=True)
            else:
                mean_deriv = GP(reshape_pt1(xhat), reshape_pt1(u),
                                kwargs.get('prior_kwargs'))
        else:
            mean_deriv = torch.zeros_like(xhat)
        if np.any(kwargs.get('saturation')):
            # Saturate the derivative of the nonlinearity estimate to guarantee
            # contraction
            a_min = np.min([-kwargs.get('saturation')])
            a_max = np.max([kwargs.get('saturation')])
            mean_deriv = torch.clamp(mean_deriv, a_min, a_max)
        if kwargs.get('add_u_obs_justvelocity'):
            ABmult = (torch.matmul(self.A, xhat.t()) +
                      torch.matmul(self.B, xi + u)).t()
        else:
            ABmult = (torch.matmul(
                self.A, xhat.t()) + torch.matmul(self.B, xi)).t()
        DfA = torch.matmul(mean_deriv, ABmult.t())
        LC1 = reshape_pt1(self.Gamma1 * (y - xhat[:, 0]))
        LC2 = reshape_pt1(self.Gamma2 * (y - xhat[:, 0]))
        xhatdot = reshape_pt1(ABmult + LC1)
        xidot = reshape_pt1(DfA + LC2)

        return torch.cat((xhatdot, xidot), dim=1)


# High gain observer for the mass-spring-mass system
# Using current GP estimation of velocity for xdot, regular high gain
# observer but with GP only predicting velocity
class MSM_justvelocity_observer_highgain_GP:
    def __init__(self, device, kwargs):
        self.device = device
        self.n = kwargs.get('init_state_estim').shape[1]
        # Gain (needs to be large enough)
        self.g = kwargs.get('prior_kwargs').get('observer_gains').get('g')
        self.K = torch.zeros(self.n, device=self.device)
        for i in range(self.n):
            self.K[i] = kwargs.get('prior_kwargs').get('observer_gains').get(
                'k' + str(i + 1))
        self.Gamma1 = self.K * torch.tensor([
            self.g ** (i + 1) for i in range(self.n)], device=self.device)
        self.A = torch.diag_embed(torch.ones(self.n - 1, device=self.device),
                                  offset=1)
        self.B = torch.zeros((self.n, 1), device=self.device)
        self.B[-1] = 1
        self.C = torch.zeros((1, self.n), device=self.device)
        self.C[0] = 1
        self.M = self.A - torch.matmul(reshape_pt1(self.K).t(), self.C)
        # Also check eigenvalues of M for stability without high gain
        self.eigvals = torch.linalg.eig(self.M).eigenvalues
        for x in self.eigvals:
            if np.abs(np.real(x.item())) < 1e-5:
                logging.error('The eigenvalues of the matrix M are '
                              'dangerously  small, low robustness of the '
                              'observer! Increase the gains.')
            elif np.real(x.item()) > 0:
                logging.error('Some of the eigenvalues of the matrix M '
                              'are positive. Change the gains to get a '
                              'Hurwitz matrix.')
        if 'backward_observer_gains' in kwargs.get('prior_kwargs').keys():
            self.backward_g = kwargs.get('prior_kwargs').get(
                'backward_observer_gains').get('g')
            self.backward_K = torch.zeros(self.n, device=self.device)
            for i in range(self.n):
                self.backward_K[i] = kwargs.get('prior_kwargs').get(
                    'backward_observer_gains').get('k' + str(i + 1))
            self.backward_Gamma1 = \
                - self.backward_K * torch.tensor([self.backward_g ** (
                        i + 1) for i in range(self.n)], device=self.device)

    def __call__(self, t, xhat, u, y, t0, init_control, GP, kwargs,
                 impose_init_control=False, Gamma1=None):
        xhat = reshape_pt1(xhat)
        y = reshape_pt1(y(t, kwargs))
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        if not Gamma1:
            Gamma1 = self.Gamma1
        if GP:
            if 'GP' in GP.__class__.__name__:
                mean, var, lowconf, uppconf = GP.predict(reshape_pt1(xhat),
                                                         reshape_pt1(u))
                if not kwargs.get('continuous_model'):
                    # Continuous observer dynamics, but GP is discrete
                    # TODO better than Euler?
                    mean = (mean - reshape_pt1(xhat[:, -1])) / \
                           GP.prior_kwargs['dt']
            else:
                mean = GP(reshape_pt1(xhat), reshape_pt1(u), kwargs.get(
                    'prior_kwargs'))
        else:
            mean = torch.zeros_like(reshape_pt1(xhat[:, -1]))
        if np.any(kwargs.get('saturation')):
            # Saturate the estimate of the nonlinearity to guarantee contraction
            a_min = np.min([kwargs.get('saturation')])
            a_max = np.max([kwargs.get('saturation')])
            mean = torch.clamp(mean, a_min, a_max)
        if kwargs.get('add_u_obs_justvelocity'):
            ABmult = (torch.matmul(self.A, xhat.t()) + torch.matmul(
                self.B, mean + u)).t()
        else:
            ABmult = (torch.matmul(
                self.A, xhat.t()) + torch.matmul(self.B, mean)).t()
        LC1 = reshape_pt1(Gamma1 * (y - xhat[:, 0]))
        xhatdot = reshape_pt1(ABmult + LC1)
        return reshape_pt1(xhatdot)

    def backward(self, t, xhat, u, y, t0, init_control, GP, kwargs,
                 impose_init_control=False):
        # Return xhatdot but with -1 x correction term
        # forward_Gamma1 = torch.clone(self.Gamma1)
        # self.Gamma1 = self.backward_Gamma1
        xhatdot = self.__call__(t, xhat, u, y, t0, init_control, GP, kwargs,
                                impose_init_control, self.backward_Gamma1)
        # self.Gamma1 = forward_Gamma1
        return xhatdot


# High gain observer for the mass-spring-mass system
# Using current GP estimation of velocity for xdot, regular high gain
# observer but with GP only predicting velocity and with gain following a
# dynamical adaptation law
class MSM_justvelocity_observer_adaptive_highgain_GP:
    def __init__(self, device, kwargs):
        self.device = device
        self.n = kwargs.init_state_estim[:, :-1].shape[1]
        # Gain (needs to be large enough)
        self.g = kwargs.get('prior_kwargs').get('observer_gains').get('g0')
        self.K = torch.zeros(self.n, device=self.device)
        for i in range(self.n):
            self.K[i] = kwargs.get('prior_kwargs').get(
                'observer_gains').get('k' + str(i + 1))
        self.Gamma1 = self.K * torch.tensor([
            self.g ** (i + 1) for i in range(self.n)], device=self.device)
        self.A = torch.diag_embed(torch.ones(self.n - 1, device=self.device),
                                  offset=1)
        self.B = torch.zeros((self.n, 1), device=self.device)
        self.B[-1] = 1
        self.C = torch.zeros((1, self.n), device=self.device)
        self.C[0] = 1
        self.M = self.A - torch.matmul(reshape_pt1(self.K).t(), self.C)
        self.adaptation_law = kwargs.get('prior_kwargs').get(
            'observer_gains').get('adaptation_law')
        # Also check eigenvalues of M for stability without high gain
        self.eigvals = torch.linalg.eig(self.M).eigenvalues
        for x in self.eigvals:
            if np.abs(np.real(x.item())) < 1e-5:
                logging.error('The eigenvalues of the matrix M are '
                              'dangerously  small, low robustness of the '
                              'observer! Increase the gains.')
            elif np.real(x.item()) > 0:
                logging.error('Some of the eigenvalues of the matrix M '
                              'are positive. Change the gains to get a '
                              'Hurwitz matrix.')

    def __call__(self, t, xhat, u, y, t0, init_control, GP, kwargs,
                 impose_init_control=False):
        device = xhat.device
        x = reshape_pt1(xhat)
        xhat = reshape_pt1(x[:, :-1])
        g = float(x[:, -1])
        y = reshape_pt1(y(t, kwargs))
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))

        # Gain (needs to be large enough), depends on current state!
        K = torch.zeros(xhat.shape[1], device=device)
        for i in range(xhat.shape[1]):
            K[i] = kwargs.get('prior_kwargs').get('observer_gains').get(
                'k' + str(i + 1))
        Gamma1 = K * torch.tensor([g ** (i + 1) for i in range(xhat.shape[1])],
                                  device=device)
        if GP:
            if 'GP' in GP.__class__.__name__:
                mean, var, lowconf, uppconf = GP.predict(reshape_pt1(xhat),
                                                         reshape_pt1(u))
                if not kwargs.get('continuous_model'):
                    # Continuous observer dynamics, but GP is discrete
                    # TODO better than Euler?
                    mean = (mean - reshape_pt1(xhat[:, -1])) / \
                           GP.prior_kwargs['dt']
            else:
                mean = GP(reshape_pt1(xhat), reshape_pt1(u), kwargs.get(
                    'prior_kwargs'))
        else:
            mean = torch.zeros_like(reshape_pt1(xhat[:, -1]))
        if np.any(kwargs.get('saturation')):
            # Saturate the estimate of the nonlinearity to guarantee contraction
            a_min = np.min([kwargs.get('saturation')])
            a_max = np.max([kwargs.get('saturation')])
            mean = torch.clamp(mean, a_min, a_max)
        if kwargs.get('add_u_obs_justvelocity'):
            ABmult = (torch.matmul(self.A, xhat.t()) +
                      torch.matmul(self.B, mean + u)).t()
        else:
            ABmult = (torch.matmul(
                self.A, xhat.t()) + torch.matmul(self.B, mean)).t()
        LC1 = reshape_pt1(Gamma1 * (y - xhat[:, 0]))
        xhatdot = reshape_pt1(ABmult + LC1)
        gdot = reshape_pt1(self.adaptation_law(
            g=g, y=y, yhat=reshape_pt1(xhat[:, 0]), kwargs=kwargs.get(
                'prior_kwargs').get('observer_gains')))

        return reshape_pt1(torch.cat((xhatdot, gdot), dim=1))

    def call_noadapt(self, t, xhat, u, y, t0, init_control, GP, kwargs,
                     impose_init_control=False):
        return self.__call__(t, xhat, u, y, t0, init_control, GP, kwargs,
                             impose_init_control)


# EKF that takes distribution over current state, measurement and prior
# dynamics with their Jacobian, outputs distribution over whole next state,
# using linear measurement function. xhat contains mean and flattened covariance
# of distribution over state. Expects GP = a discrete-time dynamics model,
# and assumes state x = (mean, covar.flatten()). To use a continuous-time
# dynamics model, give it a function f(xt,ut) = predict xt+dt from xt, ut and a
# function predict_deriv = Jacobian of f(xt,ut).
class EKF_GP:
    def __init__(self, device, kwargs, ODE_dt=None):
        self.device = device
        self.n = kwargs.get('prior_kwargs').get('n')
        self.C = reshape_pt1(
            kwargs.get('prior_kwargs').get('observation_matrix'))
        self.Jc = self.C
        if ODE_dt is not None:
            self.ODE = True
            self.dt = ODE_dt

    def __call__(self, t, xhat, u, y, t0, init_control, GP, kwargs,
                 impose_init_control=False):
        y = reshape_pt1(y(t, kwargs))
        if ('ODE' not in GP.__class__.__name__) and not self.ODE:
            # evaluate continuous controller
            u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        return self.call_withyu(t, xhat, u, y, t0, init_control, GP, kwargs,
                                impose_init_control)

    def call_withyu(self, t, xhat, u, y, t0, init_control, GP, kwargs,
                 impose_init_control=False):
        device = xhat.device
        x = reshape_pt1(xhat)
        xhat = reshape_pt1(x[:, :self.n])
        covarhat = x[:, self.n:].view(self.n, self.n)
        # Prediction step: compute estimate of xhatnext before seeing
        # measurement and compute Jacobian of estimated dynamics at xhat, u
        if GP:
            if 'GP' in GP.__class__.__name__:
                mean, _, _, _ = GP.predict(reshape_pt1(xhat), reshape_pt1(u))
                mean_deriv, _, _, _ = GP.predict_deriv(xhat, u, only_x=True)
            elif ('ODE' in GP.__class__.__name__) or self.ODE:
                mean = GP.predict(xhat, u, t0=t0, dt=self.dt,
                                  init_control=init_control)
                mean_deriv = GP.predict_deriv(
                    xhat, partial(GP.predict, u=u, t0=t0, dt=self.dt,
                                  init_control=init_control))
            else:
                mean = GP.predict(reshape_pt1(xhat), reshape_pt1(u),
                                  kwargs.get('prior_kwargs'))
                mean_deriv = GP.predict_deriv(
                    xhat, u, kwargs.get('prior_kwargs'))
        else:
            mean = torch.zeros_like(xhat)
            mean_deriv = torch.zeros((self.n, self.n), device=device)
        if np.any(kwargs.get('saturation')):
            # Saturate the estimate of the nonlinearity to guarantee contraction
            a_min = np.min([-kwargs.get('saturation')])
            a_max = np.max([kwargs.get('saturation')])
            mean = torch.clamp(mean, a_min, a_max)
            mean_deriv = torch.clamp(mean_deriv, a_min, a_max)
        # Update step: correct previous estimation with measurement
        covar = torch.matmul(torch.matmul(
            mean_deriv, covarhat), mean_deriv.t()) + \
                kwargs.get('prior_kwargs').get('EKF_process_covar')
        S = torch.matmul(torch.matmul(self.Jc, covar), self.Jc.t()) + \
            kwargs.get('prior_kwargs').get('EKF_meas_covar')
        K = (torch.matmul(
            covar, torch.matmul(self.Jc.t(), torch.inverse(S))))
        xhatnext = mean + reshape_pt1(torch.matmul(K, torch.squeeze(
            y - torch.matmul(self.C, reshape_pt1_tonormal(mean).t()), dim=0)))
        covarhatnext = torch.matmul(torch.eye(xhat.shape[1]) - torch.matmul(
            K, reshape_pt1(self.Jc)), covar)
        return torch.cat((reshape_pt1(xhatnext),
                          reshape_pt1(torch.flatten(covarhatnext))), dim=1)

    def backward(self, t, xhat, xnext_forward, u, y, tf, final_control,
                       GP, kwargs, impose_init_control=False):
        if ('ODE' not in GP.__class__.__name__) and not self.ODE:
            # evaluate continuous controller
            u = reshape_pt1(u(t, kwargs, tf, final_control,
                              impose_init_control))
        return self.backward_withyu(t, xhat, xnext_forward, u, y, tf,
                                    final_control, GP, kwargs,
                                    impose_init_control)

    def backward_withyu(self, t, xhat, xnext_forward, u, y, tf, final_control,
                       GP, kwargs, impose_init_control=False):
        # RTS smoother after EKF
        device = xhat.device
        xnext_forward = reshape_pt1(xnext_forward)
        xhat_forward = reshape_pt1(xnext_forward[:, :self.n])
        covarhat_forward = xnext_forward[:, self.n:].view(self.n, self.n)
        x = reshape_pt1(xhat)
        xhat = reshape_pt1(x[:, :self.n])
        covarhat = x[:, self.n:].view(self.n, self.n)
        # Prediction step: compute estimate of xhatnext before seeing
        # measurement and compute Jacobian of estimated dynamics at xhat, u
        if GP:
            if 'GP' in GP.__class__.__name__:
                mean, _, _, _ = GP.predict(
                    reshape_pt1(xhat_forward), reshape_pt1(u))
                mean_deriv, _, _, _ = GP.predict_deriv(
                    xhat_forward, u, only_x=True)
            elif ('ODE' in GP.__class__.__name__) or self.ODE:
                mean = GP.predict(xhat, u, t0=tf, dt=self.dt,
                                  init_control=final_control)
                mean_deriv = GP.predict_deriv(
                    xhat, partial(GP.predict, u=u, t0=tf, dt=self.dt,
                                  init_control=final_control))
            else:
                mean = GP.predict(reshape_pt1(xhat_forward), reshape_pt1(u),
                                  kwargs.get('prior_kwargs'))
                mean_deriv = GP.predict_deriv(
                    xhat_forward, u, kwargs.get('prior_kwargs'))
        else:
            mean = torch.zeros_like(xhat_forward)
            mean_deriv = torch.zeros((self.n, self.n), device=device)
        if np.any(kwargs.get('saturation')):
            # Saturate the estimate of the nonlinearity to guarantee contraction
            a_min = np.min([-kwargs.get('saturation')])
            a_max = np.max([kwargs.get('saturation')])
            mean = torch.clamp(mean, a_min, a_max)
            mean_deriv = torch.clamp(mean_deriv, a_min, a_max)
        # Update step: correct forward estimation with backward estimation
        inv = torch.inverse(torch.matmul(mean_deriv, torch.matmul(
            covarhat_forward, mean_deriv.t())) +
                            kwargs.get('prior_kwargs').get('EKF_process_covar'))
        G = torch.matmul(torch.matmul(covarhat_forward, mean_deriv.t()), inv)
        xhatnext = xhat_forward + torch.matmul(G, torch.squeeze(xhat - mean))
        covarhatnext = covarhat_forward + torch.matmul(G, torch.matmul(
            covarhat, G.t())) - torch.matmul(
            G, torch.matmul(mean_deriv, covarhat_forward))
        return torch.cat((reshape_pt1(xhatnext),
                          reshape_pt1(torch.flatten(covarhatnext))), dim=1)


# Continuous-time EKF that takes distribution over current state, measurement
# and prior continuous-time dynamics with their Jacobian, outputs distribution
# over whole next state, using linear measurement function. xhat contains
# mean and flattened covariance of distribution over state. Expects ODE = a
# continuous-time dynamics model, and assumes state x = (mean, covar)
# https://en.wikipedia.org/wiki/Extended_Kalman_filter#Continuous-time_extended_Kalman_filter
class EKF_ODE:
    def __init__(self, device, kwargs):
        self.device = device
        self.n = kwargs.get('prior_kwargs').get('n')
        self.C = reshape_pt1(
            kwargs.get('prior_kwargs').get('observation_matrix'))

    def __call__(self, t, xhat, u, y, t0, init_control, ODE, kwargs,
                 impose_init_control=False):
        y = reshape_pt1(y(t, kwargs))
        return self.call_withyu(t, xhat, u, y, t0, init_control, ODE, kwargs,
                                impose_init_control)

    def call_withyu(self, t, xhat, u, y, t0, init_control, ODE, kwargs,
                 impose_init_control=False):
        device = xhat.device
        x = reshape_pt1(xhat)
        xhat = reshape_pt1(x[:, :self.n])
        covarhat = x[:, self.n:].view(self.n, self.n)
        # Prediction step: compute xhatdot without correction term and
        # Jacobian of estimated dynamics at (xhat, u)
        if ODE:
            if 'NODE' in ODE.__class__.__name__:
                mean = ODE.defunc.dyn_NODE(
                    t=t, x=xhat, u=u, t0=t0, init_control=init_control,
                    process_noise_var=0., kwargs=kwargs,
                    impose_init_control=impose_init_control)
                partial_ODE = lambda xt: ODE.defunc.dyn_NODE(
                    t=t, x=xt, u=u, t0=t0, init_control=init_control,
                    process_noise_var=0., kwargs=kwargs,
                    impose_init_control=impose_init_control)
                mean_deriv = ODE.predict_deriv(xhat, partial_ODE)
            else:
                mean = ODE(t=t, x=xhat, u=u, t0=t0, init_control=init_control,
                           process_noise_var=0., kwargs=kwargs,
                           impose_init_control=impose_init_control)
                mean_deriv = ODE.call_deriv(
                    t=t, x=xhat, u=u, t0=t0, init_control=init_control,
                    process_noise_var=0., kwargs=kwargs,
                    impose_init_control=impose_init_control)
        else:
            mean = torch.zeros_like(xhat)
            mean_deriv = torch.zeros((self.n, self.n), device=device)
        # Update step: compute correction term for xhatdot, K, and covarhatdot
        K = torch.matmul(torch.matmul(covarhat, self.C.t()), torch.inverse(
            kwargs.get('prior_kwargs').get('EKF_meas_covar')))
        S = torch.matmul(mean_deriv, covarhat)
        xhatdot = mean + \
                  torch.matmul(K, y.t() - torch.matmul(self.C, xhat.t())).t()
        covarhatdot = S + S.t() + \
                      kwargs.get('prior_kwargs').get('EKF_process_covar') - \
                      torch.matmul(torch.matmul(K, kwargs.get(
                          'prior_kwargs').get('EKF_meas_covar')), K.t())
        return torch.cat((reshape_pt1(xhatdot),
                          reshape_pt1(torch.flatten(covarhatdot))), dim=1)
