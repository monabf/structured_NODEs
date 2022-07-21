import torch
from torchdiffeq import odeint
from scipy.integrate import solve_ivp

from .dynamics_functions import *
from utils.utils import rk4, euler

# Solver to simulate dynamics. Takes a dynamics object as input which it
# calls, or a dynamics function

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


# Input x, u, version and parameters, output x over t_eval with torchdiffeq ODE
# solver (or manually if discrete)
def dynamics_traj(x0, u, t0, dt, init_control, discrete=False, version=None,
                  meas_noise_var=0, process_noise_var=0, method='dopri5',
                  t_eval=[0.1], stay_GPU=False, lightning=False,
                  impose_init_control=False, **kwargs):
    if not torch.is_tensor(x0):
        return dynamics_traj_numpy(
            x0, u, t0, dt, init_control, discrete=discrete, version=version,
            meas_noise_var=meas_noise_var,
            process_noise_var=process_noise_var, method=method,
            t_span=[t_eval[0], t_eval[-1]], t_eval=t_eval,
            impose_init_control=impose_init_control, **kwargs)
    # Go to GPU at the beginning of simulation
    if torch.cuda.is_available() and not lightning:
        # lightning handles cuda itself
        x0 = x0.cuda()
    device = x0.device
    if kwargs['kwargs'].get('solver_options'):
        solver_options = kwargs['kwargs'].get('solver_options').copy()
        if solver_options.get('rtol'):
            rtol = solver_options['rtol']
            atol = solver_options['atol']
            solver_options.pop('rtol')
            solver_options.pop('atol')
        else:
            rtol = 1e-3
            atol = 1e-6
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
                    version(t, x, u, t0, init_control, process_noise_var,
                            impose_init_control=impose_init_control, **kwargs))
                x = xnext
                t += dt
            xtraj = reshape_pt1(x)
        else:
            # Solve one time step at a time until end or length of t_eval
            # xtraj = torch.zeros((len(t_eval), x0.shape[1]), device=device)
            xtraj = torch.empty(tuple([len(t_eval)] + list(x0.shape[1:])),
                                device=device)
            xtraj[0] = reshape_pt1(x0)
            i = 0
            while (i < len(t_eval) - 1) and (t < t_eval[-1]):
                i += 1
                xnext = reshape_pt1(
                    version(t, xtraj[i - 1], u, t0, init_control,
                            process_noise_var,
                            impose_init_control=impose_init_control, **kwargs))
                xtraj[i] = xnext
                t += dt
            # xtraj = xtraj[:i]
    else:

        def f(tl, xl):
            return version(tl, xl, u, t0, init_control, process_noise_var,
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
    if meas_noise_var != 0:
        xtraj += torch.normal(0, np.sqrt(meas_noise_var), size=xtraj.shape,
                              device=device)
    # Go back to CPU at end of simulation
    if not stay_GPU:
        return reshape_pt1(xtraj.cpu())
    else:
        return reshape_pt1(xtraj)


# Same as before but in backward time: xf, tf, and t_eval=[tf,...,t0] are
# expected, [xf,...,x0] is returned. For continuous time systems, no changes
# since same dynamics function with flipped time vector.
# For discrete time systems, iterations are in backward time but a different
# transition map is also expected, which should be the inverse of the forward
# one. The "version" object given by the user should implement this
# "backward" transition map! The forward trajectory is also needed so that
# each forward point can be given to the backward transition map (e.g. EK
# Smoother needs KF estimates)
def dynamics_traj_backward(xf, u, tf, xtraj_forward, dt, final_control,
                           discrete=False, version=None, meas_noise_var=0,
                           process_noise_var=0, method='dopri5',
                           t_eval=[-0.1], stay_GPU=False, lightning=False,
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
                    t, x, xnext_forward, u, tf, final_control,
                    process_noise_var,
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
                    t, xtraj[i - 1], xnext_forward, u, tf, final_control,
                    process_noise_var,
                    impose_init_control=impose_final_control, **kwargs))
                xtraj[i] = xnext
                t -= dt
            # xtraj = xtraj[:i]
    else:
        def f(tl, xl):
            return version(tl, xl, u, tf, final_control, process_noise_var,
                           impose_init_control=impose_final_control, **kwargs)

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


def dynamics_traj_numpy(x0, u, t0, dt, init_control, discrete=False,
                        version=None, meas_noise_var=0, process_noise_var=0,
                        method='RK45', t_span=[0, 1], t_eval=[0.1], **kwargs):
    if kwargs['kwargs'].get('solver_options'):
        solver_options = kwargs['kwargs'].get('solver_options')
    else:
        solver_options = {}
    x0 = reshape_pt1(x0)
    if discrete:
        t = t0
        if len(t_eval) == 1:
            # Solve until reach final time in t_eval
            x = reshape_pt1(x0).copy()
            while t < t_eval[-1]:
                xnext = reshape_pt1(
                    version(t, x, u, t0, init_control, process_noise_var,
                            **kwargs))
                x = xnext
                t += dt
            xtraj = reshape_pt1(x)
        else:
            # Solve one time step at a time until end or length of t_eval
            xtraj = np.empty((len(t_eval), x0.shape[1]))
            xtraj[0] = reshape_pt1(x0)
            i = 0
            while (i < len(t_eval) - 1) and (t < t_eval[-1]):
                i += 1
                xnext = reshape_pt1(
                    version(t, xtraj[i - 1], u, t0, init_control,
                            process_noise_var, **kwargs))
                xtraj[i] = xnext
                t += dt
            xtraj = xtraj[:i]
    elif method == 'my_RK4':
        t = t0
        if len(t_eval) == 1:
            # Solve until reach final time in t_eval
            x = reshape_pt1(x0).copy()
            while t < t_eval[-1]:
                f = lambda xl: version(t, xl, u, t0, init_control,
                                       process_noise_var, **kwargs)
                if dt > 1:  # 0.01:
                    # If dt too high, make intermediate steps
                    xnext = rk4(x, f, dt, accelerate=True,
                                accelerate_deltat=0.01)
                else:
                    xnext = rk4(x, f, dt)
                x = xnext
                t += dt
            xtraj = reshape_pt1(x)
        else:
            # Solve one time step at a time until end or length of t_eval
            xtraj = np.empty((len(t_eval), x0.shape[1]))
            xtraj[0] = reshape_pt1(x0)
            i = 0
            while (i < len(t_eval) - 1) and (t < t_eval[-1]):
                i += 1
                f = lambda xl: version(t, xl, u, t0, init_control,
                                       process_noise_var, **kwargs)
                if dt > 1:  # 0.01:
                    # If dt too high, make intermediate steps
                    xnext = rk4(xtraj[i - 1], f, dt, accelerate=True,
                                accelerate_deltat=0.01)
                else:
                    xnext = rk4(xtraj[i - 1], f, dt)
                xtraj[i] = xnext
                t += dt
            xtraj = xtraj[:i]
    elif method == 'my_Euler':
        t = t0
        if len(t_eval) == 1:
            # Solve until reach final time in t_eval
            x = reshape_pt1(x0).copy()
            while t < t_eval[-1]:
                f = lambda xl: version(t, xl, u, t0, init_control,
                                       process_noise_var, **kwargs)
                if dt > 1:  # 0.01:
                    # If dt too high, make intermediate steps
                    xnext = euler(x, f, dt, accelerate=True,
                                  accelerate_deltat=0.01)
                else:
                    xnext = euler(x, f, dt)
                x = xnext
                t += dt
            xtraj = reshape_pt1(x)
        else:
            # Solve one time step at a time until end or length of t_eval
            xtraj = np.empty((len(t_eval), x0.shape[1]))
            xtraj[0] = reshape_pt1(x0)
            i = 0
            while (i < len(t_eval) - 1) and (t < t_eval[-1]):
                i += 1
                f = lambda xl: version(t, xl, u, t0, init_control,
                                       process_noise_var, **kwargs)
                if dt > 1:  # 0.01:
                    # If dt too high, make intermediate steps
                    xnext = euler(xtraj[i - 1], f, dt, accelerate=True,
                                  accelerate_deltat=0.01)
                else:
                    xnext = euler(xtraj[i - 1], f, dt)
                xtraj[i] = xnext
                t += dt
            xtraj = xtraj[:i]
    else:
        sol = solve_ivp(
            lambda t, x: version(t, x, u, t0, init_control, process_noise_var,
                                 **kwargs), t_span=t_span,
            y0=reshape_pt1_tonormal(x0), method=method, t_eval=t_eval,
            **solver_options)
        xtraj = reshape_pt1(sol.y.T)
    if meas_noise_var != 0:
        xtraj += np.random.normal(0, np.sqrt(meas_noise_var), xtraj.shape)
    return reshape_pt1(xtraj)
