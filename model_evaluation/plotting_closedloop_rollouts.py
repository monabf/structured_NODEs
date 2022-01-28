import logging
import os

import numpy as np
import pandas as pd
import seaborn as sb
import torch
from matplotlib import pyplot as plt

from simulation.observers import dynamics_traj_observer, \
    dynamics_traj_observer_backward
from utils.utils import RMS, log_multivariate_normal_likelihood, reshape_pt1, \
    reshape_dim1, interpolate_func, list_torch_to_numpy

sb.set_style('whitegrid')


# Some useful plotting functions to run closed-loop rollouts (estimated
# trajectory given noisy measurements and a GP model)


# Run rollouts also in closed loop, i.e. just estimating xhat(t) depending on
# y(t) with the current GP model
def model_closedloop_rollout(dyn_GP, observer, observe_data, init_state,
                             control_traj, true_mean, rollout_length=100,
                             no_GP_in_observer=False):
    device = true_mean.device
    rollout_length = int(np.min([rollout_length, len(true_mean) - 1]))
    time = torch.arange(0., (rollout_length + 1) * dyn_GP.dt, step=dyn_GP.dt,
                        device=device)
    kwargs = dyn_GP.config
    if no_GP_in_observer:
        GP = dyn_GP.observer_prior_mean
    else:
        GP = dyn_GP

    t_u = torch.cat((reshape_dim1(time[:-1]),
                     reshape_dim1(control_traj)), dim=1)
    y_observed = reshape_dim1(observe_data(true_mean))
    if 'noisy_inputs' in dyn_GP.system:
        y_observed = reshape_pt1(y_observed + torch.normal(0, np.sqrt(
            dyn_GP.true_meas_noise_var), y_observed.shape, device=device))
    init_state_estim = torch.cat((reshape_pt1(y_observed[0]), torch.zeros((
        1, init_state.shape[1] - 1))), dim=1)  # xhat0 = (y_0,0,...,0)
    if 'Michelangelo' in dyn_GP.system:
        init_state_estim = torch.cat((init_state_estim, reshape_pt1([0])),
                                     dim=1)  # initial guess of xi = 0
    elif 'EKF' in dyn_GP.system:
        # initial guess of covar = same as prior
        init_state_estim = torch.cat((
            init_state_estim, reshape_pt1(
                dyn_GP.prior_kwargs['EKF_process_covar'].flatten())), dim=1)
    t_y = torch.cat((reshape_dim1(time), reshape_dim1(y_observed)), dim=1)

    controller = interpolate_func(x=t_u, t0=time[0], init_value=control_traj[0])
    measurement = interpolate_func(x=t_y, t0=time[0], init_value=y_observed[0])

    if ('No_observer' in dyn_GP.system) or ('observer' not in dyn_GP.system):
        predicted_mean = true_mean
        logging.warning('No observer has been specified: the closed-loop '
                        'rollouts are simply the true trajectories.')
    else:
        predicted_mean = dynamics_traj_observer(
            x0=reshape_pt1(init_state_estim), u=controller, y=measurement,
            t0=time[0], dt=dyn_GP.dt, init_control=control_traj[0],
            discrete=dyn_GP.discrete, version=observer,
            method=dyn_GP.simu_solver, t_eval=time, GP=GP, kwargs=kwargs)
        if dyn_GP.backwards_after_estim:
            predicted_mean = torch.flip(dynamics_traj_observer_backward(
                xf=reshape_pt1(predicted_mean[-1]), u=controller, y=measurement,
                tf=time[-1], xtraj_forward=predicted_mean, dt=dyn_GP.dt,
                final_control=reshape_pt1(controller(time[-1])),
                discrete=dyn_GP.discrete, version=observer,
                method=dyn_GP.simu_solver,
                t_eval=torch.flip(time, dims=[0, ]), GP=GP, kwargs=kwargs),
                dims=[0, ])
            predicted_mean = dynamics_traj_observer(
                x0=reshape_pt1(predicted_mean[0]), u=controller, y=measurement,
                t0=time[0], dt=dyn_GP.dt, init_control=control_traj[0],
                discrete=dyn_GP.discrete, version=observer,
                method=dyn_GP.simu_solver, t_eval=time, GP=GP, kwargs=kwargs)
    if any(k in dyn_GP.system for k in ('Michelangelo', 'adaptive', 'EKF')):
        # get rid of covar/xi/other last part of state in trajs
        predicted_mean = predicted_mean[:, :init_state.shape[1]]
    predicted_lowconf = predicted_mean
    predicted_uppconf = predicted_mean
    predicted_var = torch.zeros((rollout_length + 1, 1), device=device)

    RMSE = RMS(predicted_mean - true_mean)
    log_likelihood = log_multivariate_normal_likelihood(true_mean[1:, :],
                                                        predicted_mean[1:, :],
                                                        predicted_var[1:, :])

    return init_state, control_traj, true_mean, predicted_mean, predicted_var, \
           predicted_lowconf, predicted_uppconf, RMSE, log_likelihood


# Save the results of closed-loop rollouts (different nnames)
def save_closedloop_rollout_variables(dyn_GP, results_folder, nb_rollouts,
                                      rollout_list, step,
                                      ground_truth_approx=False, plots=True,
                                      title=None):
    if title:
        folder = os.path.join(results_folder, title + '_' + str(step))
    else:
        folder = os.path.join(results_folder, 'Rollouts_' + str(step))
    os.makedirs(folder, exist_ok=True)
    rollout_list = list_torch_to_numpy(rollout_list)  # transfer all back to CPU
    for i in range(nb_rollouts):
        rollout_folder = os.path.join(folder, 'Rollout_' + str(i))
        filename = 'Predicted_mean_traj_closedloop.csv'
        file = pd.DataFrame(reshape_pt1(rollout_list[i][3]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'Predicted_var_traj_closedloop.csv'
        file = pd.DataFrame(reshape_pt1(rollout_list[i][4]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'Predicted_lowconf_traj_closedloop.csv'
        file = pd.DataFrame(reshape_pt1(rollout_list[i][5]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'Predicted_uppconf_traj_closedloop.csv'
        file = pd.DataFrame(reshape_pt1(rollout_list[i][6]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'RMSE_closedloop.csv'
        file = pd.DataFrame(reshape_pt1(rollout_list[i][7]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'SRMSE_closedloop.csv'
        file = pd.DataFrame(reshape_pt1(rollout_list[i][8]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'Log_likelihood_closedloop.csv'
        file = pd.DataFrame(reshape_pt1(rollout_list[i][9]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'Standardized_log_likelihood_closedloop.csv'
        file = pd.DataFrame(reshape_pt1(rollout_list[i][10]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        true_mean = reshape_dim1(rollout_list[i][2])
        predicted_mean = reshape_dim1(rollout_list[i][3])
        predicted_lowconf = reshape_dim1(rollout_list[i][5])
        predicted_uppconf = reshape_dim1(rollout_list[i][6])
        time = np.arange(0, len(true_mean))
        if dyn_GP.plot_output:
            y_observed = reshape_dim1(dyn_GP.observe_data(
                torch.as_tensor(true_mean)))
            if 'noisy_inputs' in dyn_GP.system:
                y_observed = reshape_pt1(y_observed + np.random.normal(
                    0, np.sqrt(dyn_GP.true_meas_noise_var),
                    y_observed.shape))
            y_observed_estim = reshape_dim1(dyn_GP.observe_data(
                torch.as_tensor(predicted_mean)))
            filename = 'Closedloop_rollout_output_predictions.csv'
            file = pd.DataFrame(y_observed_estim)
            file.to_csv(os.path.join(rollout_folder, filename),
                        header=False)
        if plots:
            for k in range(predicted_mean.shape[1]):
                name = 'Closedloop_rollout_model_predictions' + str(k) + '.pdf'
                plt.plot(time, true_mean[:, k], 'g', label='True trajectory')
                plt.plot(time, predicted_mean[:, k],
                         label='Estimated trajectory', c='orange', alpha=0.9)
                plt.fill_between(time,
                                 predicted_lowconf[:, k],
                                 predicted_uppconf[:, k],
                                 facecolor='orange', alpha=0.2)
                if not ground_truth_approx:
                    plt.title('Closedloop roll out of predicted and true '
                              'trajectory '
                              'over time, random start, random control')
                else:
                    plt.title('Closedloop roll out of predicted and true '
                              'trajectory '
                              'over time, random start, data control')
                plt.legend()
                plt.xlabel('Time steps')
                plt.ylabel('State')
                plt.savefig(os.path.join(rollout_folder, name),
                            bbox_inches='tight')
                plt.close('all')

            for k in range(predicted_mean.shape[1] - 1):
                name = 'Closedloop_rollout_phase_portrait' + str(k) + '.pdf'
                plt.plot(true_mean[:, k], true_mean[:, k + 1], 'g',
                         label='True trajectory')
                plt.plot(predicted_mean[:, k], predicted_mean[:, k + 1],
                         label='Estimated trajectory', c='orange', alpha=0.9)
                plt.fill_between(predicted_mean[:, k],
                                 predicted_lowconf[:, k + 1],
                                 predicted_uppconf[:, k + 1],
                                 facecolor='orange', alpha=0.2)
                if not ground_truth_approx:
                    plt.title('Closedloop roll out of predicted and true '
                              'phase portrait over time, random start, '
                              'random control')
                else:
                    plt.title('Closedloop roll out of predicted and true '
                              'phase portrait over time, random start, '
                              'data control')
                plt.legend()
                plt.xlabel('x_' + str(k))
                plt.ylabel('x_' + str(k + 1))
                plt.savefig(os.path.join(rollout_folder, name),
                            bbox_inches='tight')
                plt.close('all')

            if dyn_GP.plot_output:
                for k in range(y_observed.shape[1]):
                    name = 'Closedloop_rollout_output_predictions' + str(k) + \
                           '.pdf'
                    plt.plot(time, y_observed[:, k], 'g',
                             label='Observed output')
                    plt.plot(time, y_observed_estim[:, k],
                             label='Estimated output', c='orange', alpha=0.9)
                    if not dyn_GP.ground_truth_approx:
                        plt.title(
                            'Closed loop rollout of predicted and true output '
                            'over time, random start, random control')
                    else:
                        if title and ('Test' in title):
                            plt.title('Closed loop rollout of predicted and '
                                      'true output over time over testing data')
                        elif title and ('Val' in title):
                            plt.title('Closed loop rollout of predicted and '
                                      'true output over time over validation '
                                      'data')
                        else:
                            plt.title('Closed loop rollout of predicted and '
                                      'true output over time, random start, '
                                      'data control')
                    plt.legend()
                    plt.xlabel('Time steps')
                    plt.ylabel('Output')
                    plt.savefig(os.path.join(rollout_folder, name),
                                bbox_inches='tight')
                    plt.close('all')


# Plot quantities about closed-loop rollouts over time
def plot_closedloop_rollout_data(dyn_GP, folder):
    name = 'Closedloop_rollout_RMSE'
    RMSE_df = pd.DataFrame(dyn_GP.closedloop_rollout_RMSE.numpy())
    RMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.closedloop_rollout_RMSE[:, 0],
             dyn_GP.closedloop_rollout_RMSE[:, 1],
             'c', label='RMSE')
    plt.title('Closedloop rollout RMSE over time, over '
              + str(dyn_GP.nb_rollouts) + ' rollouts')
    plt.xlabel('Number of samples')
    plt.ylabel('RMSE over rollouts')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Closedloop_rollout_SRMSE'
    SRMSE_df = pd.DataFrame(dyn_GP.closedloop_rollout_SRMSE.numpy())
    SRMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.closedloop_rollout_SRMSE[:, 0],
             dyn_GP.closedloop_rollout_SRMSE[:, 1],
             'c', label='SRMSE')
    plt.title('Closedloop rollout SRMSE over time, over ' + str(
        dyn_GP.nb_rollouts) + ' rollouts')
    plt.xlabel('Number of samples')
    plt.ylabel('SRMSE over rollouts')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Closedloop_rollout_log_AL'
    log_AL_df = pd.DataFrame(dyn_GP.closedloop_rollout_log_AL.numpy())
    log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.closedloop_rollout_log_AL[:, 0],
             dyn_GP.closedloop_rollout_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Closedloop rollout average log likelihood over time, over ' +
              str(dyn_GP.nb_rollouts) + ' rollouts')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood over rollouts')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Closedloop_rollout_stand_log_AL'
    stand_log_AL_df = pd.DataFrame(
        dyn_GP.closedloop_rollout_stand_log_AL.numpy())
    stand_log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.closedloop_rollout_stand_log_AL[:, 0],
             dyn_GP.closedloop_rollout_stand_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Closedloop rollout average log likelihood over time, over ' +
              str(dyn_GP.nb_rollouts) + ' rollouts')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood over rollouts')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')


# Plot quantities about test rollouts over time
def plot_test_closedloop_rollout_data(dyn_GP, folder):
    name = 'Test_closedloop_rollout_RMSE'
    RMSE_df = pd.DataFrame(dyn_GP.test_closedloop_rollout_RMSE.numpy())
    RMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.test_closedloop_rollout_RMSE[:, 0],
             dyn_GP.test_closedloop_rollout_RMSE[:, 1],
             'c', label='RMSE')
    plt.title('Closedloop rollout RMSE over time, over testing data')
    plt.xlabel('Number of samples')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Test_closedloop_rollout_SRMSE'
    SRMSE_df = pd.DataFrame(dyn_GP.test_closedloop_rollout_SRMSE.numpy())
    SRMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.test_closedloop_rollout_SRMSE[:, 0],
             dyn_GP.test_closedloop_rollout_SRMSE[:, 1],
             'c', label='SRMSE')
    plt.title('Closedloop rollout SRMSE over time, over testing data')
    plt.xlabel('Number of samples')
    plt.ylabel('SRMSE')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Test_closedloop_rollout_log_AL'
    log_AL_df = pd.DataFrame(dyn_GP.test_closedloop_rollout_log_AL.numpy())
    log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.test_closedloop_rollout_log_AL[:, 0],
             dyn_GP.test_closedloop_rollout_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Closedloop rollout average log likelihood over time, '
              'over testing data')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'),
                bbox_inches='tight')
    plt.close('all')

    name = 'Test_closedloop_rollout_stand_log_AL'
    stand_log_AL_df = pd.DataFrame(
        dyn_GP.test_closedloop_rollout_stand_log_AL.numpy())
    stand_log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.test_closedloop_rollout_stand_log_AL[:, 0],
             dyn_GP.test_closedloop_rollout_stand_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Closedloop rollout average log likelihood over time, '
              'over testing data')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'),
                bbox_inches='tight')
    plt.close('all')


# Plot quantities about validation rollouts over time
def plot_val_closedloop_rollout_data(dyn_GP, folder):
    name = 'Val_closedloop_rollout_RMSE'
    RMSE_df = pd.DataFrame(dyn_GP.val_closedloop_rollout_RMSE.numpy())
    RMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.val_closedloop_rollout_RMSE[:, 0],
             dyn_GP.val_closedloop_rollout_RMSE[:, 1],
             'c', label='RMSE')
    plt.title('Closedloop rollout RMSE over time, over validation data')
    plt.xlabel('Number of samples')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Val_closedloop_rollout_SRMSE'
    SRMSE_df = pd.DataFrame(dyn_GP.val_closedloop_rollout_SRMSE.numpy())
    SRMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.val_closedloop_rollout_SRMSE[:, 0],
             dyn_GP.val_closedloop_rollout_SRMSE[:, 1],
             'c', label='SRMSE')
    plt.title('Closedloop rollout SRMSE over time, over validation data')
    plt.xlabel('Number of samples')
    plt.ylabel('SRMSE')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Val_closedloop_rollout_log_AL'
    log_AL_df = pd.DataFrame(dyn_GP.val_closedloop_rollout_log_AL.numpy())
    log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.val_closedloop_rollout_log_AL[:, 0],
             dyn_GP.val_closedloop_rollout_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Closedloop rollout average log likelihood over time, '
              'over validation data')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'),
                bbox_inches='tight')
    plt.close('all')

    name = 'Val_closedloop_rollout_stand_log_AL'
    stand_log_AL_df = pd.DataFrame(
        dyn_GP.val_closedloop_rollout_stand_log_AL.numpy())
    stand_log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.val_closedloop_rollout_stand_log_AL[:, 0],
             dyn_GP.val_closedloop_rollout_stand_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Closedloop rollout average log likelihood over time, '
              'over validation data')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'),
                bbox_inches='tight')
    plt.close('all')
