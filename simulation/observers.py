from torchdiffeq import odeint

from utils.utils import reshape_dim1
from .observer_functions import *

# Solver to simulate dynamics of observer. Takes an observer object as input
# which it calls, or a dynamics function of the observer. Also measurement
# functions

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


# Input x, u, version and parameters, output x over t_eval with torchdiffeq ODE
# solver (or manually if discrete)
def dynamics_traj_observer(x0, u, y, t0, dt, init_control, discrete=False,
                           version=None, method='dopri5', t_eval=[0.1],
                           GP=None, stay_GPU=False, lightning=False,
                           impose_init_control=False, **kwargs):
    # Go to GPU at the beginning of simulation
    if torch.cuda.is_available() and not lightning:
        x0 = x0.cuda()
    device = x0.device
    if kwargs['kwargs'].get('solver_options'):
        solver_options = kwargs['kwargs'].get('solver_options').copy()
        rtol = solver_options['rtol']
        atol = solver_options['atol']
        solver_options.pop('rtol')
        solver_options.pop('atol')
    else:
        solver_options = {}
        rtol = 1e-3
        atol = 1e-6
    x0 = reshape_pt1(x0)
    if not torch.is_tensor(t_eval):
        t_eval = torch.tensor(t_eval, device=device)
    if torch.cuda.is_available() and not lightning:
        t_eval = t_eval.cuda()
    if discrete:
        if torch.is_tensor(t0):
            t = torch.clone(t0)
        else:
            t = torch.tensor([t0], device=device)
        if len(t_eval) == 1:
            # Solve until reach final time in t_eval
            x = reshape_pt1(x0).clone()
            while t < t_eval[-1]:
                xnext = reshape_pt1(
                    version(t, x, u, y, t0, init_control, GP,
                            impose_init_control=impose_init_control, **kwargs))
                x = xnext
                t += dt
            xtraj = reshape_pt1(x)
        else:
            # Solve one time step at a time until end or length of t_eval
            # xtraj = torch.empty((len(t_eval), x0.shape[1]), device=device)
            xtraj = torch.empty(tuple([len(t_eval)] + list(x0.shape[1:])),
                                device=device)
            xtraj[0] = reshape_pt1(x0)
            i = 0
            while (i < len(t_eval) - 1) and (t < t_eval[-1]):
                i += 1
                xnext = reshape_pt1(
                    version(t, xtraj[i - 1], u, y, t0, init_control, GP,
                            impose_init_control=impose_init_control, **kwargs))
                xtraj[i] = xnext
                t += dt
    else:
        def f(tl, xl):
            return version(tl, xl, u, y, t0, init_control, GP,
                           impose_init_control=impose_init_control, **kwargs)

        if len(t_eval) == 1:
            # t0 always needed for odeint, then deleted
            t_eval = torch.cat((torch.tensor([t0], device=device), t_eval))
            xtraj = odeint(f, reshape_pt1_tonormal(x0), t_eval,
                           method=method, rtol=rtol, atol=atol,
                           options=solver_options)[1:, :]
        else:
            xtraj = odeint(f, reshape_pt1_tonormal(x0), t_eval,
                           method=method, rtol=rtol, atol=atol,
                           options=solver_options)
    # Go back to CPU at end of simulation
    if not stay_GPU:
        return reshape_pt1(xtraj.cpu())
    else:
        return reshape_pt1(xtraj)


# Same as before but in backward time: xf, tf, and t_eval=[tf,...,t0] are
# expected, [xf,...,x0] is returned. For continuous time systems,
# same dynamics function with flipped time vector but also change the sign of
# the gains matrix for a well-posed problem ("backward" function of "version").
# For discrete time systems, iterations are in backward time but a different
# transition map is also expected, which should be the inverse of the forward
# one. The "version" object given by the user should implement this
# "backward" transition map! The forward trajectory is also needed so that
# each forward point can be given to the backward transition map (e.g. EK
# Smoother needs KF estimates)
def dynamics_traj_observer_backward(xf, u, y, tf, xtraj_forward, dt,
                                    final_control, discrete=False, version=None,
                                    method='dopri5', t_eval=[-0.1], GP=None,
                                    stay_GPU=False, lightning=False,
                                    impose_final_control=False, **kwargs):
    # Go to GPU at the beginning of simulation
    if torch.cuda.is_available() and not lightning:
        xf = xf.cuda()
    device = xf.device
    if kwargs['kwargs'].get('solver_options'):
        solver_options = kwargs['kwargs'].get('solver_options').copy()
        rtol = solver_options['rtol']
        atol = solver_options['atol']
        solver_options.pop('rtol')
        solver_options.pop('atol')
    else:
        solver_options = {}
        rtol = 1e-3
        atol = 1e-6
    xf = reshape_pt1(xf)
    if not torch.is_tensor(t_eval):
        t_eval = torch.tensor(t_eval, device=device)
    if torch.cuda.is_available() and not lightning:
        t_eval = t_eval.cuda()
    if discrete:
        if torch.is_tensor(tf):
            t = torch.clone(tf)
        else:
            t = torch.tensor([tf], device=device)
        if len(t_eval) == 1:
            # Solve until reach initial time in t_eval
            x = reshape_pt1(xf).clone()
            i = 0
            while t > t_eval[-1]:
                i += 1
                xnext_forward = xtraj_forward[-i]
                xnext = reshape_pt1(version.backward(
                    t, x, xnext_forward, u, y, tf, final_control, GP,
                    impose_init_control=impose_final_control, **kwargs))
                x = xnext
                t -= dt
            xtraj = reshape_pt1(x)
        else:
            # Solve one time step at a time until end or length of t_eval
            # xtraj = torch.empty((len(t_eval), xf.shape[1]), device=device)
            xtraj = torch.empty(tuple([len(t_eval)] + list(xf.shape[1:])),
                                device=device)
            xtraj[0] = reshape_pt1(xf)
            i = 0
            while (i < len(t_eval) - 1) and (t > t_eval[-1]):
                i += 1
                xnext_forward = xtraj_forward[-i - 1]
                xnext = reshape_pt1(version.backward(
                    t, xtraj[i - 1], xnext_forward, u, y, tf, final_control,
                    GP, impose_init_control=impose_final_control, **kwargs))
                xtraj[i] = xnext
                t -= dt
    else:
        def f(tl, xl):
            return version.backward(tl, xl, u, y, tf, final_control, GP,
                                    impose_init_control=impose_final_control,
                                    **kwargs)

        if len(t_eval) == 1:
            # tf always needed for odeint, then deleted
            t_eval = torch.cat((torch.tensor([tf], device=device), t_eval))
            xtraj = odeint(f, reshape_pt1_tonormal(xf), t_eval,
                           method=method, rtol=rtol, atol=atol,
                           options=solver_options)[1:, :]
        else:
            xtraj = odeint(f, reshape_pt1_tonormal(xf), t_eval,
                           method=method, rtol=rtol, atol=atol,
                           options=solver_options)
    # Go back to CPU at end of simulation
    if not stay_GPU:
        return reshape_pt1(xtraj.cpu())
    else:
        return reshape_pt1(xtraj)


# Functions for observing experimental data from full data
def dim1_observe_data(xtraj):
    # return reshape_dim1(xtraj[..., 0])
    return torch.index_select(
        xtraj, dim=-1, index=torch.tensor([0], device=xtraj.device))

def dim14_observe_data(xtraj):
    return torch.index_select(
        xtraj, dim=-1, index=torch.tensor([0, 3], device=xtraj.device))


def dim16_observe_data(xtraj):
    return torch.index_select(
        xtraj, dim=-1, index=torch.tensor([0, 5], device=xtraj.device))

def dimlast_observe_data(xtraj):
    return reshape_dim1(xtraj[..., -1])
