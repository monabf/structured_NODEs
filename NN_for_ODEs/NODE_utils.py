import numpy as np
import pytorch_lightning as pl
import scipy.signal
import torch
import torchdiffeq

from NN_for_ODEs.defunc_time import DEFunc_time
from simulation.controllers import Control_from_list
from simulation.observers import dynamics_traj_observer
from utils.config import Config
from utils.pytorch_utils import StandardScaler, get_parameters
from utils.utils import reshape_pt1, reshape_dim1, interpolate_func, \
    reshape_dim1_difftraj, Interpolate_func, reshape_pt1_difftraj


# Useful functions for learning dynamical systems with NODEs
# TODO for loops are not optimal, but only done once...

# Several functions for recognition models: first define inputs to
# recognition model (called init_state_obs, in single case and in case of
# difftraj), then update the data if needed (for forward KKL data needs to
# start at T after recognition model), then update config with everything
# needed for this recognition model

def set_DF(W0, dz, dy, method):
    # Set the KKL matrix D with different methods
    # First set the filter from which to take the eigenvalues, then how to
    # set D such that it has those eigenvalues
    if method.startswith('butter'):
        filter = scipy.signal.butter
        method = method.split('_', 1)[1]
    elif method.startswith('bessel'):
        filter = scipy.signal.bessel
        method = method.split('_', 1)[1]
    else:
        filter = scipy.signal.bessel  # default
    if method.startswith('indirect'):
        # Indirect method to place poles of D with given filter
        zO, pO, kO = filter(dz, W0, analog=True, output='zpk')
        pO = np.sort(pO)
        A = -np.array([[i] for i in range(1, dz + 1)]) * np.eye(dz)
        B = np.ones((dz, 1))
        whole_D = scipy.signal.place_poles(A, B, pO)
        if whole_D.rtol == 0 and B.shape[1] != 1:
            raise Exception('Pole placing failed')
        K = whole_D.gain_matrix
        D = torch.as_tensor(A - np.dot(B, K))
        F = torch.ones(dz, dy)
    elif method.startswith('direct'):
        # Direct method to place poles of D with given filter
        zO, pO, kO = filter(dz, W0, analog=True, output='zpk')
        pO = np.sort(pO)
        A = np.zeros((dz, dz))
        B = - np.eye(dz)
        whole_D = scipy.signal.place_poles(A, B, pO)
        if whole_D.rtol == 0 and B.shape[1] != 1:
            raise Exception('Pole placing failed')
        D = torch.as_tensor(whole_D.gain_matrix)
        F = torch.ones(dz, dy)
    elif method.startswith('companion'):
        # D in companion form of given filter denominator
        b, a = filter(dz, W0, analog=True, output='ba')
        D = torch.as_tensor(np.polynomial.polynomial.polycompanion(np.flip(
            a))).t()
        F = torch.zeros(dz, dy)
        F[-1] = torch.ones(dy)
    elif method.startswith('block_diag'):
        # D as block diagonal of real (block of dim 1) and complex conjugate
        # (block of dim 2) eigenvalues of given filter
        D = np.zeros((dz, dz))
        zO, pO, kO = filter(dz, W0, analog=True, output='zpk')
        pO = np.sort(pO)
        real_idx = -1
        complex_idx = 0
        ignore_next = False
        for i in range(len(pO)):
            if ignore_next:
                ignore_next = False
                continue
            v = pO[i]
            if v.imag == 0:
                D[real_idx, real_idx] = v.real
                real_idx -= 1
                ignore_next = False
            else:
                D[complex_idx, complex_idx] = v.real
                D[complex_idx, complex_idx + 1] = v.imag
                D[complex_idx + 1, complex_idx] = -v.imag
                D[complex_idx + 1, complex_idx + 1] = v.real
                complex_idx += 2
                ignore_next = True
        D = torch.as_tensor(D)
        F = torch.ones(dz, dy)
    elif method.startswith('block_companion'):
        # D as block diagonal of real (block of dim 1) and complex conjugate
        # (companion matrix of dim 2) eigenvalues of given filter
        D = np.zeros((dz, dz))
        F = np.zeros((dz, dy))
        zO, pO, kO = filter(dz, W0, analog=True, output='zpk')
        pO = np.sort(pO)
        real_idx = 0
        complex_idx = -1
        ignore_next = False
        for i in range(len(pO)):
            if ignore_next:
                ignore_next = False
                continue
            v = pO[i]
            if v.imag == 0:
                D[real_idx, real_idx] = v.real
                F[real_idx] = torch.ones(dy)
                real_idx += 1
                ignore_next = False
            else:
                D[complex_idx - 1, complex_idx - 1] = 0.
                D[complex_idx - 1, complex_idx] = 1.
                D[complex_idx, complex_idx - 1] = -(v.real ** 2 + v.imag ** 2)
                D[complex_idx, complex_idx] = 2 * v.real
                F[complex_idx] = torch.ones(dy)
                complex_idx -= 2
                ignore_next = True
        D = torch.as_tensor(D)
        F = torch.as_tensor(F)
    elif method.startswith('diag'):
        # Diagonal method
        D = -torch.tensor([[i * W0] for i in range(1, dz + 1)]) * \
            torch.eye(dz)
        F = torch.ones(dz, dy)
    elif method.startswith('id'):
        # Diagonal method
        D = - W0 * torch.eye(dz)
        F = torch.ones(dz, dy)
    elif method.startswith('randn'):
        D = torch.randn(dz, dz) / dz
        F = torch.ones(dz, dy)
    else:
        raise KeyError(f'Undefined method to set D: {method}')
    if method.endswith('noise'):
        D = D + torch.rand(D.shape) * 2 * 1e-2 - 1e-2
    return D, F


def make_init_state_obs(y_observed_true, utraj, init_state_x, time,
                        config: Config):
    # Define the inputs of the recognition model
    # Cut all inputs, then flip if necessary
    z_t_eval = time
    if config.KKL_l2_reg is None:  # KKL reg: KKL runs over whole traj!
        z_t_eval = z_t_eval[:config.init_state_obs_T]
        y_observed_true = y_observed_true[:config.init_state_obs_T]
        utraj = utraj[:config.init_state_obs_Tu]  # TODO cut in init_state_obs!
    if config.init_state_obs_method == 'x0':
        config['z_config'] = {}
        init_state_obs = init_state_x
    elif config.init_state_obs_method == 'y0':
        config['z_config'] = {}
        init_state_obs = reshape_pt1(y_observed_true[0])
    elif config.init_state_obs_method == 'y0_u0':
        config['z_config'] = {}
        init_state_obs = reshape_pt1(torch.cat((
            torch.flatten(y_observed_true[0]), torch.flatten(utraj[0]))))
    elif config.init_state_obs_method == 'y0T_u0T' or \
            config.init_state_obs_method == 'fixed_recognition_model':
        config['z_config'] = {}
        if config.no_control:
            init_state_obs = reshape_pt1(torch.flatten(y_observed_true))
        else:
            init_state_obs = reshape_pt1(torch.cat((
                torch.flatten(y_observed_true), torch.flatten(utraj))))
    elif 'optimD' in config.init_state_obs_method:
        if config.no_control or ('KKLu' in config.init_state_obs_method):
            init_state_obs = config['z_config']['init_state_estim']
        else:
            init_state_obs = reshape_pt1(torch.cat((
                torch.flatten(config['z_config']['init_state_estim']),
                torch.flatten(utraj))))
    elif config.init_state_obs_method.startswith('KKL_u0T_back'):
        z0 = config['z_config']['init_state_estim']
        t_y = torch.cat((reshape_dim1(z_t_eval), reshape_dim1(
            torch.flip(y_observed_true, dims=[0, ]))), dim=1)
        flipped_measurement = interpolate_func(
            x=t_y, t0=z_t_eval[0], init_value=y_observed_true[-1])
        ztraj_estim = dynamics_traj_observer(
            x0=z0, u=None, y=flipped_measurement, t0=z_t_eval[0], dt=config.dt,
            init_control=0., discrete=False, version=config.init_state_KKL,
            method=config.simu_solver, t_eval=z_t_eval, GP=None,
            kwargs=config.z_config)
        if config.no_control:
            init_state_obs = reshape_pt1(ztraj_estim[config.init_state_obs_T-1])
        else:
            init_state_obs = reshape_pt1(torch.cat((
                torch.flatten(reshape_pt1(ztraj_estim[
                                              config.init_state_obs_T-1])),
                torch.flatten(utraj[:config.init_state_obs_Tu]))))
        if config.KKL_l2_reg is not None:
            t_y = torch.cat((reshape_dim1(z_t_eval),
                             reshape_dim1(y_observed_true)), dim=1)
            measurement = interpolate_func(
                x=t_y, t0=z_t_eval[0], init_value=y_observed_true[0])
            KKL_traj = dynamics_traj_observer(
                x0=z0, u=None, y=measurement, t0=z_t_eval[0],
                dt=config.dt, init_control=0., discrete=False,
                version=config.init_state_KKL, method=config.simu_solver,
                t_eval=z_t_eval, GP=None, kwargs=config.z_config)
        # print(z_t_eval, init_state_obs, utraj[:config.init_state_obs_T])
        # import matplotlib.pyplot as plt
        # plt.plot(flipped_measurement(z_t_eval))
        # plt.plot(torch.flip(y_observed_true[:config.init_state_obs_T],
        #                     dims=[0, ]))
        # plt.plot(y_observed_true)
        # plt.show()
        # plt.plot(ztraj_estim)
        # plt.show()
    elif config.init_state_obs_method.startswith('KKLu_back'):
        z0 = config['z_config']['init_state_estim']
        t_y = torch.cat((reshape_dim1(z_t_eval), reshape_dim1(
            torch.flip(y_observed_true, dims=[0, ]))), dim=1)
        flipped_measurement = interpolate_func(
            x=t_y, t0=z_t_eval[0], init_value=y_observed_true[-1])
        t_u = torch.cat((reshape_dim1(z_t_eval), reshape_dim1(
            torch.flip(utraj, dims=[0, ]))), dim=1)
        flipped_controller = interpolate_func(
            x=t_u, t0=z_t_eval[0], init_value=utraj[-1])
        config.z_config.update(dict(controller_args=config.controller_args))
        ztraj_estim = dynamics_traj_observer(
            x0=z0, u=flipped_controller, y=flipped_measurement, t0=z_t_eval[0],
            dt=config.dt, init_control=config.init_control, discrete=False,
            version=config.init_state_KKL, method=config.simu_solver,
            t_eval=z_t_eval, GP=None, kwargs=config.z_config)
        init_state_obs = reshape_pt1(ztraj_estim[config.init_state_obs_T-1])
        if config.KKL_l2_reg is not None:
            t_y = torch.cat((reshape_dim1(z_t_eval),
                             reshape_dim1(y_observed_true)), dim=1)
            measurement = interpolate_func(
                x=t_y, t0=z_t_eval[0], init_value=y_observed_true[0])
            t_u = torch.cat((reshape_dim1(z_t_eval),
                             reshape_dim1(utraj)), dim=1)
            controller = interpolate_func(
                x=t_u, t0=z_t_eval[0], init_value=utraj[0])
            KKL_traj = dynamics_traj_observer(
                x0=z0, u=controller, y=measurement, t0=z_t_eval[0],
                dt=config.dt, init_control=config.init_control,
                discrete=False, version=config.init_state_KKL,
                method=config.simu_solver, t_eval=z_t_eval, GP=None,
                kwargs=config.z_config)
        # import matplotlib.pyplot as plt
        # for i in range(y_observed_true.shape[-1]):
        #     plt.plot(z_t_eval, flipped_measurement(z_t_eval, config)[..., i])
        #     plt.plot(z_t_eval,torch.flip(y_observed_true,
        #                         dims=[0, ])[...,i])
        #     plt.plot(z_t_eval,y_observed_true[...,i])
        #     plt.show()
        # print(z_t_eval, (config.init_state_obs_T - 1) * config.dt - z_t_eval,
        #       len(z_t_eval), config.dt, torch.arange(len(z_t_eval)) * config.dt)
        # plt.plot(z_t_eval, flipped_controller(z_t_eval, config.z_config,
        #                                      z_t_eval[0],
        #                             config.init_control,
        #                             impose_init_control=False))
        # plt.plot(z_t_eval, torch.flip(utraj, dims=[0, ]))
        # plt.plot(z_t_eval, utraj)
        # plt.show()
        # plt.plot(ztraj_estim)
        # plt.show()
    elif config.init_state_obs_method.startswith('KKL_u0T'):
        z0 = config['z_config']['init_state_estim']
        t_y = torch.cat((reshape_dim1(z_t_eval),
                         reshape_dim1(y_observed_true)), dim=1)
        measurement = interpolate_func(
            x=t_y, t0=z_t_eval[0], init_value=y_observed_true[0])
        ztraj_estim = dynamics_traj_observer(
            x0=z0, u=None, y=measurement, t0=z_t_eval[0], dt=config.dt,
            init_control=0., discrete=False, version=config.init_state_KKL,
            method=config.simu_solver, t_eval=z_t_eval, GP=None,
            kwargs=config.z_config)
        if config.no_control:
            init_state_obs = reshape_pt1(ztraj_estim[config.init_state_obs_T-1])
        else:
            init_state_obs = reshape_pt1(torch.cat((
                torch.flatten(reshape_pt1(ztraj_estim[
                                              config.init_state_obs_T-1])),
                torch.flatten(utraj[:config.init_state_obs_Tu]))))
        if config.KKL_l2_reg is not None:
            KKL_traj = ztraj_estim
        # import matplotlib.pyplot as plt
        # print(z_t_eval, init_state_obs, utraj[:config.init_state_obs_T])
        # for i in range(y_observed_true.shape[1]):
        #     plt.plot(measurement(z_t_eval)[:, i])
        #     plt.plot(y_observed_true[:config.init_state_obs_T, i])
        #     plt.show()
        # plt.plot(ztraj_estim)
        # plt.show()
    elif config.init_state_obs_method.startswith('KKLu'):
        z0 = config['z_config']['init_state_estim']
        t_y = torch.cat((reshape_dim1(z_t_eval),
                         reshape_dim1(y_observed_true)), dim=1)
        measurement = interpolate_func(
            x=t_y, t0=z_t_eval[0], init_value=y_observed_true[0])
        t_u = torch.cat((reshape_dim1(z_t_eval), reshape_dim1(utraj)), dim=1)
        controller = interpolate_func(
            x=t_u, t0=z_t_eval[0], init_value=utraj[0])
        config.z_config.update(dict(controller_args=config.controller_args))
        ztraj_estim = dynamics_traj_observer(
            x0=z0, u=controller, y=measurement, t0=z_t_eval[0], dt=config.dt,
            init_control=config.init_control, discrete=False,
            version=config.init_state_KKL, method=config.simu_solver,
            t_eval=z_t_eval, GP=None, kwargs=config.z_config)
        init_state_obs = reshape_pt1(ztraj_estim[config.init_state_obs_T-1])
        if config.KKL_l2_reg is not None:
            KKL_traj = ztraj_estim
        # print(init_state_obs)
        # import matplotlib.pyplot as plt
        # plt.plot(measurement(z_t_eval))
        # plt.plot(y_observed_true[:config.init_state_obs_T])
        # plt.show()
        # plt.plot(controller(z_t_eval, config.z_config, z_t_eval[0],
        #                     config.init_control))
        # plt.plot(controller(time, config, config.t0,
        #                     config.init_control)[:config.init_state_obs_T])
        # plt.show()
        # plt.plot(ztraj_estim)
        # plt.show()
    else:
        raise KeyError(f'No recognition model under the name '
                       f'{config.init_state_obs_method}')
    if ('KKL' in config.init_state_obs_method) and (
            'y0T' in config.init_state_obs_method):
        # TODO KKL_l2_reg not ready, a mess everywhere! Clean it up!
        init_state_obs = torch.cat((
            init_state_obs, reshape_pt1(torch.flatten(y_observed_true))), dim=1)
    if (config.KKL_l2_reg is not None) and (
            'KKL' in config.init_state_obs_method) and not (
            'optimD' in config.init_state_obs_method):
        # KKL reg: keep ztraj in config
        # if optimD: ztraj needs to be computed again at each iteration
        if config.KKL_traj is None:
            config['KKL_traj'] = torch.unsqueeze(KKL_traj, dim=0)
        else:
            config['KKL_traj'] = torch.cat((
                config['KKL_traj'], torch.unsqueeze(KKL_traj, dim=0)), dim=0)
    return init_state_obs


def make_diff_init_state_obs(diff_y_observed, diff_utraj, init_state_x, time,
                             config: Config):
    # Define the inputs of the recognition model for difftraj
    device = time.device
    for i in range(len(init_state_x)):
        init_state_obs = make_init_state_obs(
            diff_y_observed[i], diff_utraj[i], init_state_x[i], time, config)
        if i == 0:
            diff_init_state_obs = torch.zeros(0, 1, init_state_obs.shape[1],
                                              device=device)
        diff_init_state_obs = torch.cat((
            diff_init_state_obs, torch.unsqueeze(init_state_obs, 0)), dim=0)
    return diff_init_state_obs


def update_trajs_init_state_obs(xtraj_true, y_observed_true, utraj,
                                time, config: Config):
    # Model estimation starts at T if KKL used forward to estimate x0
    if ('KKL' in config.init_state_obs_method) and not \
            ('back' in config.init_state_obs_method):
        if len(xtraj_true.shape) == 3:  # difftraj
            xtraj_true = torch.transpose(xtraj_true, 0, 1)
            y_observed_true = torch.transpose(y_observed_true, 0, 1)
            utraj = torch.transpose(utraj, 0, 1)
        xtraj_true = xtraj_true[config.init_state_obs_T:]
        y_observed_true = y_observed_true[config.init_state_obs_T:]
        utraj = utraj[config.init_state_obs_T:]
        time = time[config.init_state_obs_T:]
        if len(xtraj_true.shape) == 3:
            xtraj_true = torch.transpose(xtraj_true, 1, 0)
            y_observed_true = torch.transpose(y_observed_true, 1, 0)
            utraj = torch.transpose(utraj, 1, 0)
    return xtraj_true, y_observed_true, utraj, time


def update_config_init_state_obs(diff_init_state_obs, init_state_model,
                                 diff_xtraj_true, diff_y_observed_true,
                                 diff_utraj, time, config: Config):
    if config.init_state_obs_method == 'x0':
        diff_init_state_estim = diff_init_state_obs
        config.update(dict(
            init_state_estim=diff_init_state_estim,
            init_state_estim_before=diff_init_state_estim.clone(),
            init_state_model=None))
        return update_trajs_init_state_obs(
            diff_xtraj_true, diff_y_observed_true, diff_utraj, time, config)
    elif 'optimD' in config.init_state_obs_method:
        init_state_model = KKL_optimD_model(
            init_state_model, time, diff_y_observed_true, diff_utraj, config)
        if config.KKL_l2_reg is not None:
            config['KKL_traj'] = init_state_model.simulate_ztraj(
                diff_init_state_obs[..., :init_state_model.KKL_ODE_model.n],
                init_state_model.z_t_eval)
            z = torch.squeeze(init_state_model.simulate_zu(
                diff_init_state_obs, ztraj=config['KKL_traj']), dim=1)
        else:
            z = torch.squeeze(init_state_model.simulate_zu(
                diff_init_state_obs), dim=1)
        diff_init_state_estim = reshape_pt1(
            init_state_model.init_state_model(z))
        scaler_Z = StandardScaler(z)
    else:
        diff_init_state_estim = reshape_pt1(
            init_state_model(diff_init_state_obs))
        scaler_Z = StandardScaler(torch.squeeze(diff_init_state_obs, dim=1))
    # if config.nb_difftraj is not None:
    #     init_state_x = config.init_state_x.contiguous().view(
    #         config.nb_difftraj, config.init_state_x.shape[-1])
    # else:
    #     init_state_x = config.init_state_x
    config['z_config'].update(dict(scaler_Z=scaler_Z))
    # scaler_Y = StandardScaler(init_state_x)
    # init_state_model.set_scalers(scaler_X=scaler_Z, scaler_Y=scaler_Y)
    n_param, param = get_parameters(init_state_model, verbose=True)
    d_init_state_model = diff_init_state_obs.shape[-1]
    config.update(dict(
        init_state_obs=diff_init_state_obs,
        init_state_estim=diff_init_state_estim,
        init_state_estim_before=diff_init_state_estim.clone(),
        init_state_model=init_state_model,
        n_param_init_state_model=n_param,
        d_init_state_model=d_init_state_model))
    return update_trajs_init_state_obs(
        diff_xtraj_true, diff_y_observed_true, diff_utraj, time, config)


class KKL_optimD_model(pl.LightningModule):
    def __init__(self, init_state_model, time, y_observed_true, utraj,
                 config: Config):
        super(KKL_optimD_model, self).__init__()
        self.scaler_X = None
        self.scaler_Y = None
        self.init_state_model = init_state_model
        self.config = config
        # Cut all inputs, then flip if necessary
        if self.config.KKL_l2_reg is None:
            max_time = config.init_state_obs_T
            max_timeu = config.init_state_obs_Tu
        else:
            max_time = len(time)
            max_timeu = len(time)
        self.z_t_eval = time[:max_time]
        if len(y_observed_true.shape) == 3:
            self.difftraj = True
            self.nb_difftraj = y_observed_true.shape[0]
            y_observed_true = reshape_dim1_difftraj(
                y_observed_true[:, :max_time, :])
            utraj = reshape_dim1_difftraj(
                utraj[:, :max_timeu, :])
        else:
            self.difftraj = False
            self.nb_difftraj = 1
            y_observed_true = reshape_dim1(
                y_observed_true[:max_time])
            utraj = reshape_dim1(utraj[:max_timeu])

        # Construct the time-dependent function t -> (y(t)) or (y(t),
        # u(t)) for autonomous/functional KKL, built as list over difftraj
        if self.difftraj:
            self.measurement_controller, self.init_yu = \
                self.create_diff_measurement_controller(
                    y_observed_true, utraj)
        else:
            self.measurement_controller = self.create_measurement_controller(
                y_observed_true, utraj)

        # KKL model
        self.KKL_ODE_model = KKL_optimD_submodel(self.config.init_state_KKL)
        self.defunc = DEFunc_time(
            model=self.KKL_ODE_model, controller=self.measurement_controller,
            t0=time[0], init_control=self.init_yu, config=self.config,
            sensitivity='autograd', intloss=None, order=1, force_control=True)

    def create_diff_measurement_controller(self, diff_y_observed_true,
                                           diff_utraj):
        device = diff_y_observed_true.device
        measurement_controller_list = []
        for i in range(self.nb_difftraj):
            measurement_controller, init_yu = \
                self.create_measurement_controller(diff_y_observed_true[i],
                                                   diff_utraj[i])
            measurement_controller_list.append(measurement_controller)
            if i == 0:
                init_diff_yu = torch.zeros(0, 1, init_yu.shape[1],
                                           device=device)
            init_diff_yu = torch.cat((
                init_diff_yu, torch.unsqueeze(init_yu, 0)), dim=0)
        return Control_from_list(measurement_controller_list, init_diff_yu), \
               init_diff_yu

    def create_measurement_controller(self, y_observed_true, utraj):
        if self.config.init_state_obs_method == 'KKL_u0T_optimD':
            t_y = torch.cat((reshape_dim1(self.z_t_eval),
                             reshape_dim1(y_observed_true)), dim=-1)
            init_yu = reshape_pt1(t_y[0, 1:])
            measurement_controller = Interpolate_func(
                x=t_y, t0=self.z_t_eval[0], init_value=init_yu)
        elif self.config.init_state_obs_method == 'KKLu_optimD':
            t_y = torch.cat((reshape_dim1(self.z_t_eval),
                             reshape_dim1(y_observed_true)), dim=-1)
            utraj = reshape_dim1(utraj)
            t_y_u = torch.cat((t_y, utraj), dim=-1)
            init_yu = reshape_pt1(t_y_u[0, 1:])
            measurement_controller = Interpolate_func(
                x=t_y_u, t0=self.z_t_eval[0], init_value=init_yu)
        elif self.config.init_state_obs_method == 'KKL_u0T_back_optimD':
            t_y = torch.cat((reshape_dim1(self.z_t_eval),
                             reshape_dim1(torch.flip(y_observed_true,
                                                     dims=[0, ]))), dim=-1)
            init_yu = reshape_pt1(t_y[0, 1:])
            measurement_controller = Interpolate_func(
                x=t_y, t0=self.z_t_eval[0], init_value=init_yu)
        elif self.config.init_state_obs_method == 'KKLu_back_optimD':
            t_y_u = torch.cat((
                reshape_dim1(self.z_t_eval),
                reshape_dim1(torch.flip(y_observed_true, dims=[0, ])),
                reshape_dim1(torch.flip(utraj, dims=[0, ]))), dim=-1)
            init_yu = reshape_pt1(t_y_u[0, 1:])
            measurement_controller = Interpolate_func(
                x=t_y_u, t0=self.z_t_eval[0], init_value=init_yu)
        else:
            raise KeyError(f'No recognition model under the name '
                           f'{self.config.init_state_obs_method}')
        return measurement_controller, init_yu

    def set_scalers(self, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.init_state_model.set_scalers(self.scaler_X, self.scaler_Y)

    def simulate_ztraj(self, z0, z_t_eval):
        # From z(0) and simulation time, simulate z and return whole trajectory
        ztraj = torchdiffeq.odeint(
                self.defunc, z0.to(self.device), z_t_eval.to(self.device),
                **self.config.optim_solver_options)
        if self.difftraj:
            return torch.transpose(torch.squeeze(ztraj, dim=2), 0, 1)
        else:
            return torch.squeeze(ztraj, dim=2)

    def simulate_only_z(self, z0, ztraj=None):
        # From z(0): simulate z, return z(T)
        # Cut simulation time to init_state_obs_T to return z(init_state_obs_T)
        z_t_eval = self.z_t_eval[:self.config.init_state_obs_T]
        if ztraj is None:
            ztraj = self.simulate_ztraj(z0, z_t_eval)
        if self.difftraj:
            z = torch.unsqueeze(ztraj[:, self.config.init_state_obs_T-1], dim=1)
            return z
        else:
            z = reshape_pt1(ztraj[self.config.init_state_obs_T-1])
            return z

    def simulate_zu(self, z0_u, ztraj=None):
        # From z(0): simulate z, retain z(T), and return (z(T), u(0), ..., u(T))
        u = reshape_dim1(
            z0_u[..., self.KKL_ODE_model.n:
                      self.KKL_ODE_model.n+self.config.init_state_obs_Tu])
        z0 = reshape_dim1(z0_u[..., :self.KKL_ODE_model.n])
        z = self.simulate_only_z(z0, ztraj)
        zu = torch.cat((torch.flatten(z, start_dim=1),
                        torch.flatten(u, start_dim=1)), dim=1)
        if self.difftraj:
            return reshape_pt1_difftraj(zu)
        else:
            return reshape_pt1(zu)

    def forward(self, z0_u):
        # Actually both forward and backward pass: runs forward pass =
        # simulation of z, only keeps final z(T) and applies recognition model
        # to it; retains autograd backward pass for optimization
        # Built on torchdyn.models.NeuralDE
        zu = self.simulate_zu(z0_u)
        return self.init_state_model(zu)


class KKL_optimD_submodel(torch.nn.Module):
    def __init__(self, init_state_KKL):
        super(KKL_optimD_submodel, self).__init__()
        self.init_state_KKL = init_state_KKL
        self.n = self.init_state_KKL.n

    def forward(self, x):
        yu = reshape_pt1(x[..., self.n:])
        x = reshape_pt1(x[..., :self.n])
        return self.init_state_KKL.call_with_yu(x, yu)
