import logging

import numpy as np
import torch
from matplotlib import pyplot as plt

from utils.utils import reshape_pt1, reshape_dim1, interpolate_func
from .dynamics import dynamics_traj, dynamics_traj_backward
from .observers import dynamics_traj_observer, dynamics_traj_observer_backward


# Functions used in script to simulate dynamics, observed data, estimate
# data with observer, and feeding this data again to GP

# Simulate a dynamical system
def simulate_dynamics(t_eval, t0, dt, init_control, init_state, dynamics,
                      controller, method, dyn_config, meas_noise_var=0,
                      process_noise_var=0, discrete=False, verbose=False):
    xtraj = dynamics_traj(x0=reshape_pt1(init_state), u=controller, t0=t0,
                          dt=dt, init_control=init_control, discrete=discrete,
                          version=dynamics, meas_noise_var=meas_noise_var,
                          process_noise_var=process_noise_var,
                          method=method, t_eval=t_eval, kwargs=dyn_config)
    utraj = controller(t_eval, kwargs=dyn_config, t0=t0,
                       init_control=init_control)
    t_utraj = torch.cat((reshape_dim1(t_eval), utraj), dim=1)

    if verbose:
        # Trajectory
        plt.plot(t_eval, xtraj[:, 0], label='Position')
        plt.plot(t_eval, xtraj[:, 1], label='Velocity')
        plt.title('States over time')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.legend()
        plt.show()
        if xtraj.shape[1] > 2:
            for i in range(2, xtraj.shape[1]):
                plt.plot(t_eval, xtraj[:, i], label='True dim ' + str(i))
                plt.title('Dim ' + str(i + 1) + ' over time')
                plt.xlabel('t')
                plt.ylabel('x')
                plt.legend()
                plt.show()
        # Phase portrait
        plt.plot(xtraj[:, 0], xtraj[:, 1], label='Trajectory')
        plt.title('Phase portrait')
        plt.xlabel('x')
        plt.ylabel(r'$\dot{x}$')
        plt.legend()
        plt.show()
        plt.close('all')
        plt.clf()

    return xtraj, utraj, t_utraj


# Simulate the output of a dynamical observer given a true trajectory and a
# way to generate noisy, partial measurements from it
def simulate_estimations(system, observe_data, t_eval, t0, tf, dt,
                         meas_noise_var, init_control, init_state_estim,
                         controller, observer, method, dyn_config, xtraj,
                         GP=None, discrete=False, verbose=False):
    assert len(t_eval) == xtraj.shape[0], 'State trajectory and simulation ' \
                                          'time over which to estimate the ' \
                                          'states are not the same. Resample ' \
                                          'your true trajectory to plot it ' \
                                          'over the estimation time.'
    y_observed = observe_data(xtraj)
    if ('noisy_inputs' in system) and (meas_noise_var != 0):
        y_observed = reshape_pt1(
            y_observed + torch.normal(0, np.sqrt(meas_noise_var),
                                      size=y_observed.shape))
    return simulate_estimations_fromy(
        system, y_observed, t_eval, t0, tf, dt, init_control,
        init_state_estim, controller, observer, method, dyn_config,
        xtraj, GP, discrete, verbose)


# Simulate the output of a dynamical observer given a true trajectory and a
# way to generate noisy, partial measurements from it
def simulate_estimations_fromy(system, y_observed, t_eval, t0, tf, dt,
                               init_control, init_state_estim, controller,
                               observer, method, dyn_config,
                               xtraj=None, GP=None, discrete=False,
                               verbose=False):
    t_y = torch.cat((reshape_dim1(t_eval), y_observed), dim=1)
    # Put observed data to cuda for interpolation
    if torch.cuda.is_available():
        t_y = t_y.cuda()
    noisy_init_state_estim = torch.cat(
        (reshape_pt1(y_observed[0]), reshape_pt1(init_state_estim[:, 1:])),
        dim=1)
    if 'No_observer' in system:
        logging.info('No observer has been specified, using true data for '
                     'learning.')
        xtraj_estim = xtraj
    else:
        measurement = interpolate_func(x=t_y, t0=t0, init_value=y_observed[0])

        xtraj_estim = dynamics_traj_observer(
            x0=reshape_pt1(noisy_init_state_estim), u=controller,
            y=measurement, t0=t0, dt=dt, init_control=init_control,
            discrete=discrete, version=observer, method=method,
            t_eval=t_eval, GP=GP, kwargs=dyn_config)
        if dyn_config.backwards_after_estim:
            xtraj_estim = torch.flip(dynamics_traj_observer_backward(
                xf=reshape_pt1(xtraj_estim[-1]), u=controller, y=measurement,
                tf=t_eval[-1], xtraj_forward=xtraj_estim, dt=dt,
                final_control=reshape_pt1(controller(
                    t_eval[-1], dyn_config, t0, init_control)),
                discrete=discrete, version=observer, method=method,
                t_eval=torch.flip(t_eval, dims=[0, ]), GP=GP,
                kwargs=dyn_config), dims=[0, ])
            xtraj_estim = dynamics_traj_observer(
                x0=reshape_pt1(xtraj_estim[0]), u=controller, y=measurement,
                t0=t0, dt=dt, init_control=init_control, discrete=discrete,
                version=observer, method=method, t_eval=t_eval, GP=GP,
                kwargs=dyn_config)
        # Trajectory
        if verbose:
            dimmin = np.min([xtraj_estim.shape[1], xtraj.shape[1]])
            plt.plot(t_eval, xtraj[:, 0], 'g', label='True position')
            plt.plot(t_eval, y_observed, 'r', label='Observed position')
            plt.plot(t_eval, xtraj_estim[:, 0], 'orange',
                     label='Estimated position')
            plt.title('True and estimated position over time')
            plt.xlabel('t')
            plt.ylabel('x')
            plt.legend()
            plt.show()
            plt.plot(t_eval, xtraj[:, 1], 'g', label='True velocity')
            plt.plot(t_eval, xtraj_estim[:, 1], 'orange',
                     label='Estimated velocity')
            plt.title('True and estimated velocity over time')
            plt.xlabel('t')
            plt.ylabel('x')
            plt.legend()
            plt.show()
            if 'Michelangelo' in system:
                plt.plot(t_eval, xtraj_estim[:, -1], 'orange',
                         label='Estimated xi')
                plt.title('Estimated xi over time')
                plt.xlabel('t')
                plt.ylabel('xi')
                plt.legend()
                plt.show()
            elif 'adaptive' in system:
                plt.plot(t_eval, xtraj_estim[:, -1], 'orange', label='Gain')
                plt.title('Adaptive gain over time')
                plt.xlabel('t')
                plt.ylabel('g')
                plt.legend()
                plt.show()
            elif dimmin > 2:
                for i in range(2, dimmin):
                    plt.plot(t_eval, xtraj[:, i], 'g',
                             label='True dim ' + str(i))
                    plt.plot(t_eval, xtraj_estim[:, i], 'orange',
                             label='Estimated dim ' + str(i))
                    plt.title('True and estimated dim ' + str(i) + ' over time')
                    plt.xlabel('t')
                    plt.ylabel('x')
                    plt.legend()
                    plt.show()
            # Phase portrait
            plt.plot(xtraj[:, 0], xtraj[:, 1], 'g', label='True trajectory')
            plt.plot(xtraj_estim[:, 0], xtraj_estim[:, 1], 'orange',
                     label='Estimated trajectory')
            plt.title('True and estimated phase portrait')
            plt.xlabel('x')
            plt.ylabel(r'$\dot{x}$')
            plt.legend()
            plt.show()
            plt.close('all')
            plt.clf()
            # Error plot
            plt.plot(torch.sum(torch.square(
                xtraj[:, :dimmin] - xtraj_estim[:, :dimmin]), dim=1), 'orange',
                label='True trajectory')
            plt.title('Error plot')
            plt.xlabel('t')
            plt.ylabel(r'$|x - \hat{x}|$')
            plt.legend()
            plt.show()
            plt.close('all')
            plt.clf()
    return y_observed, t_y.cpu(), xtraj_estim


# Simulate the output of a dynamical observer given measurement data directly
# (no ground truth necessarily present/not relied upon)
def traj_from_data(system, measurement, controller, observer, xtraj, t_eval, t0,
                   tf, time, dt, meas_noise_var, init_control, init_state_estim,
                   method, dyn_config, GP=None, discrete=False,
                   verbose=False):
    assert len(t_eval) == xtraj.shape[0], 'State trajectory and simulation ' \
                                          'time over which to estimate the ' \
                                          'states are not the same. Resample ' \
                                          'your true trajectory to plot it ' \
                                          'over the estimation time.'
    if 'No_observer' in system:
        logging.info('No observer has been specified, using true data for '
                     'learning.')
        xtraj_estim = xtraj
        return xtraj_estim
    else:
        xtraj_estim = dynamics_traj_observer(x0=reshape_pt1(init_state_estim),
                                             u=controller, y=measurement,
                                             t0=t0, dt=dt,
                                             init_control=init_control,
                                             discrete=discrete,
                                             version=observer,
                                             method=method,
                                             t_eval=t_eval,
                                             GP=GP, kwargs=dyn_config)

        if dyn_config.backwards_after_estim:
            xtraj_estim = torch.flip(dynamics_traj_observer_backward(
                xf=reshape_pt1(xtraj_estim[-1]), u=controller, y=measurement,
                tf=t_eval[-1], xtraj_forward=xtraj_estim, dt=dt,
                final_control=reshape_pt1(controller(t_eval[-1])),
                discrete=discrete, version=observer, method=method,
                t_eval=torch.flip(t_eval, dims=[0, ]), GP=GP,
                kwargs=dyn_config), dims=[0, ])
            xtraj_estim = dynamics_traj_observer(
                x0=reshape_pt1(xtraj_estim[0]), u=controller, y=measurement,
                t0=t0, dt=dt, init_control=init_control, discrete=discrete,
                version=observer, method=method, t_eval=t_eval, GP=GP,
                kwargs=dyn_config)

        if verbose:
            # Trajectory
            dimmin = np.min([xtraj_estim.shape[1], xtraj.shape[1]])
            plt.plot(t_eval, xtraj[:, 0], 'g', label='True position')
            plt.plot(t_eval, xtraj_estim[:, 0], 'orange',
                     label='Estimated position')
            plt.title('True and estimated position over time')
            plt.xlabel('t')
            plt.ylabel('x')
            plt.legend()
            plt.show()
            plt.plot(t_eval, xtraj[:, 1], 'g', label='True velocity')
            plt.plot(t_eval, xtraj_estim[:, 1], 'orange',
                     label='Estimated velocity')
            plt.title('True and estimated velocity over time')
            plt.xlabel('t')
            plt.ylabel('x')
            plt.legend()
            plt.show()
            if 'Michelangelo' in system:
                plt.plot(t_eval, xtraj_estim[:, -1], 'orange',
                         label='Estimated xi')
                plt.title('Estimated xi over time')
                plt.xlabel('t')
                plt.ylabel('xi')
                plt.legend()
                plt.show()
            elif 'adaptive' in system:
                plt.plot(t_eval, xtraj_estim[:, -1], 'orange', label='Gain')
                plt.title('Adaptive gain over time')
                plt.xlabel('t')
                plt.ylabel('g')
                plt.legend()
                plt.show()
            # Phase portrait
            plt.plot(xtraj[:, 0], xtraj[:, 1], 'g', label='True trajectory')
            plt.plot(xtraj_estim[:, 0], xtraj_estim[:, 1], 'orange',
                     label='Estimated trajectory')
            plt.title('True and estimated phase portrait')
            plt.xlabel('x')
            plt.ylabel(r'$\dot{x}$')
            plt.legend()
            plt.show()
            plt.close('all')
            plt.clf()
            # Error plot
            plt.plot(torch.sum(
                torch.square(xtraj[:, :dimmin] - xtraj_estim[:, :dimmin]),
                dim=1), 'orange', label='Squared error')
            plt.title('Error plot')
            plt.xlabel('t')
            plt.ylabel(r'$|x - \hat{x}|$')
            plt.legend()
            plt.show()
            plt.close('all')
            plt.clf()
        return xtraj_estim


# Form X,U,Y data for a GP from trajectory data previously simulated,
# depending on the observer considered and on wether discrete GP (learns x_t,
# u_t -> x_t+1) or continuous (learns x_t, u_t -> xdot_t) using a derivative
# function corresponding to the dynamics
def form_GP_data(system, xtraj, xtraj_estim, utraj, meas_noise_var,
                 y_observed=None, derivative_function=None, model=None):
    if 'Discrete_model' in system:
        X, U, Y = form_discrete_GP_data(system, xtraj, xtraj_estim, utraj,
                                        meas_noise_var)
    elif 'Continuous_model' in system:
        X, U, Y = form_continuous_GP_data(system, xtraj, xtraj_estim, utraj,
                                          meas_noise_var, y_observed,
                                          derivative_function, model)
    else:
        logging.warning('Forming dataset of type (x_t, u_t) -> (x_t+1) '
                        'according to the discrete GP formalism, as no option '
                        'has been specified')
        X, U, Y = form_discrete_GP_data(system, xtraj, xtraj_estim, utraj,
                                        meas_noise_var)
    return X, U, Y


def form_discrete_GP_data(system, xtraj, xtraj_estim, utraj, meas_noise_var):
    if ('Michelangelo' in system) and ('noisy_inputs' in system):
        # GP learns (xhat_t) -> phi hat(xhat_t) = xi for Michelangelo high gain
        if 'Cross_val_test' in system:
            # For testing ignore dimension in phi, only keep test data (xt, ut)
            X = reshape_dim1(xtraj_estim[:-1, :-1])
            Y = reshape_dim1(xtraj_estim[1:, :-1])
            U = reshape_dim1(utraj[:-1, :])
        else:
            X = reshape_dim1(xtraj_estim[:, :-1])
            Y = reshape_dim1(xtraj_estim[:, -1])
            U = reshape_dim1(utraj)
    elif ('LS_justvelocity_highgain' in system) and ('noisy_inputs' in
                                                     system):
        # LS model learns (xhat_t) -> (xhat_n(t+1)) only last dim, but ignore
        # xi if using extended observer
        # X = reshape_dim1(xtraj_estim[:-1, :])
        # # Y = reshape_dim1(savgol_filter(
        # #     ( xtraj_estim[1:, -1] - xtraj_estim[:-1, -1]) / kwargs['dt'],
        # #     window_length=9, polyorder=5))
        # Y = reshape_dim1(xtraj_estim[1:, -1])
        X = reshape_dim1(xtraj_estim[:-1, :])  # or :-1 if extended obs
        Y = reshape_dim1(xtraj_estim[1:, -1])  # or -2 if using extended obs
        U = reshape_dim1(utraj[:-1, :])
    elif ('justvelocity_highgain' in system) and ('noisy_inputs' in
                                                  system):
        # GP learns (xhat_t) -> (xhat_n(t+1)) only last dim
        if 'Cross_val_test' in system:
            # For testing keep whole test data (xt, ut)
            X = reshape_dim1(xtraj_estim[:-1, :])
            Y = reshape_dim1(xtraj_estim[1:, :])
            U = reshape_dim1(utraj[:-1, :])
        else:
            X = reshape_dim1(xtraj_estim[:-1, :])
            Y = reshape_dim1(xtraj_estim[1:, -1])
            U = reshape_dim1(utraj[:-1, :])
    elif ('justvelocity_adaptive_highgain' in system) and ('noisy_inputs' in
                                                           system):
        # GP learns (xhat_t) -> (xhat_n(t+1)) only last dim + adaptive gain
        if 'Cross_val_test' in system:
            # For testing keep whole test data (xt, ut) but ignore last dim g
            X = reshape_dim1(xtraj_estim[:-1, :-1])
            Y = reshape_dim1(xtraj_estim[1:, :-1])
            U = reshape_dim1(utraj[:-1, :])
        else:
            # Always ignore last dim = gain for GP
            X = reshape_dim1(xtraj_estim[:-1, :-1])
            Y = reshape_dim1(xtraj_estim[1:, -2])
            U = reshape_dim1(utraj[:-1, :])
    elif ('EKF' in system) and ('noisy_inputs' in system):
        # GP learns (xhat_t) -> (xhat(t+1)) first n dims, ignore last nxn dims
        # that contain covar matrix from EKF
        n = xtraj.shape[1]
        X = reshape_dim1(xtraj_estim[:-1, :n])
        Y = reshape_dim1(xtraj_estim[1:, :n])
        U = reshape_dim1(utraj[:-1, :])
    elif 'noisy_inputs' in system:
        X = reshape_dim1(xtraj_estim[:-1, :])
        Y = reshape_dim1(xtraj_estim[1:, :])
        U = reshape_dim1(utraj[:-1, :])
    elif 'noise_after' in system:
        logging.warning('By using noise_after instead of noisy_inputs in the '
                        'title of your system, you add noise only to the '
                        'output Y of your dataset and not to the input X. Be '
                        'sure this is the desired behavior!')
        X = reshape_dim1(xtraj_estim[:-1, :])
        Y = reshape_dim1(xtraj_estim[1:, :])
        if meas_noise_var != 0:
            Y = Y + torch.normal(0, np.sqrt(meas_noise_var), size=Y.shape)
        U = reshape_pt1(utraj[:-1, :])
    else:
        raise Exception('System name must contain more information about '
                        'dataset to form for model learning')
    return X, U, Y


def form_continuous_GP_data(system, xtraj, xtraj_estim, utraj, meas_noise_var,
                            y_observed, derivative_function, model):
    if not derivative_function:
        raise Exception('Need to provide a function function (x,u) -> xdot for '
                        'learning a continuous model.')
    elif ('LS_justvelocity_highgain' in system) and ('noisy_inputs' in
                                                     system):
        # LS model learns (xhat_t) -> (xhat_n(t+1)) only last dim, but ignore
        # xi if using extended observer
        # X = reshape_dim1(xtraj_estim[:-1, :])
        # # Y = reshape_dim1(savgol_filter(
        # #     ( xtraj_estim[1:, -1] - xtraj_estim[:-1, -1]) / kwargs['dt'],
        # #     window_length=9, polyorder=5))
        # Y = reshape_dim1(xtraj_estim[1:, -1])
        X = reshape_dim1(xtraj_estim[:-1, :])  # or :-1 if extended obs
        U = reshape_dim1(utraj[:-1, :])
        Y = reshape_dim1(derivative_function(
            X, U, y_observed, model)[:, -1])  # or -2 if using extended obs
    elif ('justvelocity_highgain' in system) and ('noisy_inputs' in
                                                  system):
        # GP learns (xhat_t) -> (xhat_n(t+1)) only last dim
        if 'Cross_val_test' in system:
            # For testing keep whole test data (xt, ut)
            X = reshape_dim1(xtraj_estim[:-1, :])
            U = reshape_dim1(utraj[:-1, :])
            Y = reshape_dim1(derivative_function(X, U, y_observed, model))
        else:
            X = reshape_dim1(xtraj_estim[:-1, :])
            U = reshape_dim1(utraj[:-1, :])
            Y = reshape_dim1(
                derivative_function(X, U, y_observed, model)[:, -1])
    elif ('justvelocity_adaptive_highgain' in system) and ('noisy_inputs' in
                                                           system):
        # GP learns (xhat_t) -> (xhat_n(t+1)) only last dim + adaptive gain
        if 'Cross_val_test' in system:
            # For testing keep whole test data (xt, ut) but ignore last dim g
            X = reshape_dim1(xtraj_estim[:-1, :-1])
            U = reshape_dim1(utraj[:-1, :])
            Y = reshape_dim1(
                derivative_function(X, U, y_observed, model)[:, :-1])
        else:
            # Always ignore last dim = gain for GP
            X = reshape_dim1(xtraj_estim[:-1, :-1])
            U = reshape_dim1(utraj[:-1, :])
            Y = reshape_dim1(
                derivative_function(X, U, y_observed, model)[:, -2])
    elif ('EKF' in system) and ('noisy_inputs' in system):
        # GP learns (xhat_t) -> (xhat(t+1)) first n dims, ignore last nxn dims
        # that contain covar matrix from EKF
        n = xtraj.shape[1]
        X = reshape_dim1(xtraj_estim[:-1, :n])
        U = reshape_dim1(utraj[:-1, :])
        Y = reshape_dim1(
            derivative_function(X, U, y_observed, model))
    elif 'noisy_inputs' in system:
        X = reshape_dim1(xtraj_estim[:-1, :])
        U = reshape_dim1(utraj[:-1, :])
        Y = reshape_dim1(derivative_function(X, U, y_observed, model))
    elif 'noise_after' in system:
        logging.warning('By using noise_after instead of noisy_inputs in the '
                        'title of your system, you add noise only to the '
                        'output Y of your dataset and not to the input X. Be '
                        'sure this is the desired behavior!')
        X = reshape_dim1(xtraj_estim[:-1, :])
        U = reshape_pt1(utraj[:-1, :])
        Y = reshape_dim1(derivative_function(X, U, y_observed, model))
        if meas_noise_var != 0:
            Y = Y + torch.normal(0, np.sqrt(meas_noise_var), size=Y.shape)
    else:
        raise Exception('System name must contain more information about '
                        'dataset to form for model learning')
    return X, U, Y
