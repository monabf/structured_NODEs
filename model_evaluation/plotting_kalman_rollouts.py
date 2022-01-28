import os

import numpy as np
import pandas as pd
import seaborn as sb
import torch
from matplotlib import pyplot as plt
from simulation.observers import dynamics_traj_observer
from utils.utils import RMS, log_multivariate_normal_likelihood, reshape_pt1, \
    reshape_dim1, interpolate_func, list_torch_to_numpy

sb.set_style('whitegrid')


# Some useful plotting functions to run kalman-style rollouts (trajectory of GP
# predictions restarting from noiseless state estimation at each step)


# Run rollouts also in Kallman-style closed loop, i.e. state is observed at
# each step then next stae is predicted, combination of observer and GP model
def model_kalman_rollout(dyn_GP, observer, observe_data, discrete_observer,
                         init_state, control_traj, true_mean,
                         rollout_length=100, no_GP_in_observer=False,
                         only_prior=False):
    device = true_mean.device
    rollout_length = int(np.min([rollout_length, len(true_mean) - 1]))
    predicted_mean = torch.empty((rollout_length + 1, init_state.shape[1]),
                                 device=device)
    predicted_lowconf = torch.empty((rollout_length + 1, init_state.shape[1]),
                                    device=device)
    predicted_uppconf = torch.empty((rollout_length + 1, init_state.shape[1]),
                                    device=device)
    predicted_var = torch.empty((rollout_length + 1, init_state.shape[1]),
                                device=device)
    dt = dyn_GP.dt
    time = torch.arange(0., rollout_length + 1, device=device) * dyn_GP.dt
    t_u = torch.cat((reshape_dim1(time[:-1]), reshape_dim1(control_traj)),
                    dim=1)
    y_observed = reshape_dim1(observe_data(true_mean))
    if ('noisy_inputs' in dyn_GP.system) and (dyn_GP.true_meas_noise_var != 0):
        y_observed = reshape_pt1(y_observed + torch.normal(0, np.sqrt(
            dyn_GP.true_meas_noise_var), y_observed.shape, device=device))
    # xhat0 = (x0_1 + noise,0,...,0)
    init_state_estim = torch.cat((reshape_pt1(y_observed[0]), torch.zeros((
        1, init_state.shape[1] - 1), device=device)), dim=1)
    # if 'Michelangelo' in dyn_GP.system:  # TODO
    #     init_state_estim = torch.cat((init_state_estim, torch.tensor(
    #         [[0.]], device=device)), dim=1)  # initial guess of xi = 0
    # elif 'EKF' in dyn_GP.system:
    #     # initial guess of covar = same as prior
    #     init_state_estim = torch.cat((init_state_estim, reshape_pt1(
    #         dyn_GP.prior_kwargs['EKF_process_covar'].flatten())), dim=1)
    t_y = torch.cat((reshape_dim1(time), reshape_dim1(y_observed)), dim=1)

    controller = interpolate_func(x=t_u, t0=time[0], init_value=control_traj[0])
    measurement = interpolate_func(x=t_y, t0=time[0], init_value=y_observed[0])

    predicted_mean[0] = init_state_estim
    predicted_lowconf[0] = init_state_estim
    predicted_uppconf[0] = init_state_estim
    predicted_var[0] = torch.zeros((1, init_state.shape[1]), device=device)
    kwargs = dyn_GP.config
    kwargs.update(dict(kalman=True))
    if no_GP_in_observer:
        GP = dyn_GP.observer_prior_mean
    else:
        GP = dyn_GP

    for t in range(rollout_length):
        control = control_traj[t]
        if t == 0:
            if 'Michelangelo' in dyn_GP.system:
                # True and predicted trajectory over time (random start, random
                # control) with Euler to get xt+1 from GP xt, ut->phit
                mean_next, varnext, next_lowconf, next_uppconf = \
                    dyn_GP.predict_euler_Michelangelo(
                        predicted_mean[t], control, only_prior=only_prior)
            elif ('justvelocity' in dyn_GP.system) and \
                    not dyn_GP.continuous_model:
                # True and predicted trajectory over time (random start, random
                # control) with Euler to get xt+1 from GP xt, ut->x_nt+1
                mean_next, varnext, next_lowconf, next_uppconf = \
                    dyn_GP.predict_euler_discrete_justvelocity(
                        predicted_mean[t], control, only_prior=only_prior)
            elif ('justvelocity' in dyn_GP.system) and dyn_GP.continuous_model:
                # True and predicted trajectory over time (random start, random
                # control) with Euler to get xt+1 from GP xt, ut->xdot_nt
                mean_next, varnext, next_lowconf, next_uppconf = \
                    dyn_GP.predict_euler_continuous_justvelocity(
                        predicted_mean[t], control, only_prior=only_prior)
            else:
                # True and predicted trajectory over time (random start, random
                # control)
                mean_next, varnext, next_lowconf, next_uppconf = dyn_GP.predict(
                    predicted_mean[t], control, only_prior=only_prior)
            last_observation = reshape_pt1(init_state_estim.clone())
            if 'Michelangelo' in dyn_GP.system:
                last_observation = torch.cat((last_observation, torch.tensor(
                    [[0.]], device=device)), dim=1)
            elif 'EKF' in dyn_GP.system:
                previous_covar = dyn_GP.prior_kwargs['EKF_process_covar']
                last_observation = torch.cat((
                    last_observation, reshape_pt1(previous_covar.flatten())),
                    dim=1)
        else:
            # Solve the observer from t-1 to current t to restart GP
            # prediction from a point corrected by current measure, and at
            # next step restart observer solving from past corrected GP
            # prediction (KF style)
            # Best would be to restart observer from init_observation,
            # but takes too much time, so either solutions arguable (restart
            # from last observation or from last GP prediction)
            if 'Michelangelo' in dyn_GP.system:
                previous_xi = reshape_pt1((predicted_mean[t, -1]
                                           - last_observation[:, -2]) / dt)
                restart = torch.cat((reshape_pt1(predicted_mean[t - 1]),
                                     previous_xi), dim=1)
                last_observation = dynamics_traj_observer(
                    x0=reshape_pt1(restart),  # last_observation?
                    u=controller,
                    y=measurement,
                    t0=(t - 1) * dt, dt=dt,
                    init_control=reshape_pt1(control_traj[t - 1]),
                    discrete=discrete_observer,
                    version=observer,
                    method=dyn_GP.fixed_all_parameters,
                    t_eval=[t * dt],
                    GP=GP, stay_GPU=True, kwargs=dyn_GP.config)
            elif 'EKF' in dyn_GP.system:
                restart = torch.cat((
                    reshape_pt1(predicted_mean[t - 1]),
                    reshape_pt1(previous_covar.flatten())), dim=1)
                last_observation = dynamics_traj_observer(
                    x0=reshape_pt1(restart),  # last_observation?
                    u=controller,
                    y=measurement,
                    t0=(t - 1) * dt, dt=dt,
                    init_control=reshape_pt1(control_traj[t - 1]),
                    discrete=discrete_observer,
                    version=observer,
                    method=dyn_GP.fixed_all_parameters,
                    t_span=[(t - 1) * dt, t * dt],
                    t_eval=[t * dt],
                    GP=GP, kwargs=dyn_GP.config)
                previous_covar = last_observation[:,
                                 init_state.shape[1]:].clone()
                last_observation = last_observation[:, :init_state.shape[1]]
            elif ('No_observer' in dyn_GP.system) or ('observer' not in
                                                      dyn_GP.system):
                last_observation = true_mean[t]
            else:
                last_observation = dynamics_traj_observer(
                    x0=reshape_pt1(predicted_mean[t - 1]),  # last_observation?
                    u=controller,
                    y=measurement,
                    t0=(t - 1) * dt, dt=dt,
                    init_control=reshape_pt1(control_traj[t - 1]),
                    discrete=discrete_observer,
                    version=observer,
                    method=dyn_GP.fixed_all_parameters,
                    t_eval=[t * dt],
                    GP=GP, stay_GPU=True, kwargs=dyn_GP.config)
            if 'Michelangelo' in dyn_GP.system:
                # True and predicted trajectory over time (random start, random
                # control) with Euler to get xt+1 from GP xt, ut->phit
                mean_next, varnext, next_lowconf, next_uppconf = \
                    dyn_GP.predict_euler_Michelangelo(
                        last_observation[:, :-1], control,
                        only_prior=only_prior)
            elif ('justvelocity' in dyn_GP.system) and \
                    not dyn_GP.continuous_model:
                # True and predicted trajectory over time (random start, random
                # control) with Euler to get xt+1 from GP xt, ut->xt+1
                mean_next, varnext, next_lowconf, next_uppconf = \
                    dyn_GP.predict_euler_discrete_justvelocity(
                        last_observation, control, only_prior=only_prior)
            elif ('justvelocity' in dyn_GP.system) and dyn_GP.continuous_model:
                # True and predicted trajectory over time (random start, random
                # control) with Euler to get xt+1 from GP xt, ut->xdott
                mean_next, varnext, next_lowconf, next_uppconf = \
                    dyn_GP.predict_euler_continuous_justvelocity(
                        last_observation, control, only_prior=only_prior)
            else:
                # True and predicted trajectory over time (random start, random
                # control)
                mean_next, varnext, next_lowconf, next_uppconf = dyn_GP.predict(
                    last_observation, control, only_prior=only_prior)
        predicted_mean[t + 1] = mean_next
        predicted_lowconf[t + 1] = next_lowconf
        predicted_uppconf[t + 1] = next_uppconf
        predicted_var[t + 1] = varnext

    RMSE = RMS(predicted_mean - true_mean)
    log_likelihood = log_multivariate_normal_likelihood(true_mean[1:, :],
                                                        predicted_mean[1:, :],
                                                        predicted_var[1:, :])

    return init_state, control_traj, true_mean, predicted_mean, predicted_var, \
           predicted_lowconf, predicted_uppconf, RMSE, log_likelihood


# Save the results of closed-loop rollouts (different nnames)
def save_kalman_rollout_variables(dyn_GP, results_folder, nb_rollouts,
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
        filename = 'Predicted_mean_traj_kalman.csv'
        file = pd.DataFrame(reshape_pt1(rollout_list[i][3]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'Predicted_var_traj_kalman.csv'
        file = pd.DataFrame(reshape_pt1(rollout_list[i][4]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'Predicted_lowconf_traj_kalman.csv'
        file = pd.DataFrame(reshape_pt1(rollout_list[i][5]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'Predicted_uppconf_traj_kalman.csv'
        file = pd.DataFrame(reshape_pt1(rollout_list[i][6]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'RMSE_kalman.csv'
        file = pd.DataFrame(reshape_pt1(rollout_list[i][7]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'SRMSE_kalman.csv'
        file = pd.DataFrame(reshape_pt1(rollout_list[i][8]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'Log_likelihood_kalman.csv'
        file = pd.DataFrame(reshape_pt1(rollout_list[i][9]))
        file.to_csv(os.path.join(rollout_folder, filename),
                    header=False)
        filename = 'Standardized_log_likelihood_kalman.csv'
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
            filename = 'Kalman_rollout_output_predictions.csv'
            file = pd.DataFrame(dyn_GP.observe_data(predicted_mean))
            file.to_csv(os.path.join(rollout_folder, filename),
                        header=False)
        if plots:
            for k in range(predicted_mean.shape[1]):
                name = 'Kalman_rollout_model_predictions' + str(k) + '.pdf'
                plt.plot(time, true_mean[:, k], 'g',
                         label='True trajectory')
                plt.plot(time, predicted_mean[:, k], 'b',
                         label='Predicted trajectory', alpha=0.7)
                plt.fill_between(time,
                                 predicted_lowconf[:, k],
                                 predicted_uppconf[:, k],
                                 facecolor='blue', alpha=0.2)
                if not ground_truth_approx:
                    plt.title('Kalman roll out of predicted and true '
                              'trajectory '
                              'over time, random start, random control')
                else:
                    plt.title('Kalman roll out of predicted and true '
                              'trajectory '
                              'over time, random start, data control')
                plt.legend()
                plt.xlabel('Time steps')
                plt.ylabel('State')
                plt.savefig(os.path.join(rollout_folder, name),
                            bbox_inches='tight')
                plt.close('all')

            for k in range(predicted_mean.shape[1] - 1):
                name = 'Kalman_rollout_phase_portrait' + str(k) + '.pdf'
                plt.plot(true_mean[:, k], true_mean[:, k + 1], 'g',
                         label='True trajectory')
                plt.plot(predicted_mean[:, k], predicted_mean[:, k + 1],
                         'b', label='Predicted trajectory', alpha=0.7)
                plt.fill_between(predicted_mean[:, k],
                                 predicted_lowconf[:, k + 1],
                                 predicted_uppconf[:, k + 1],
                                 facecolor='blue', alpha=0.2)
                if not ground_truth_approx:
                    plt.title('Kalman roll out of predicted and true '
                              'phase portrait over time, random start, '
                              'random control')
                else:
                    plt.title('Kalman roll out of predicted and true '
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
                    name = 'Kalman_rollout_output_predictions' + str(k) + '.pdf'
                    plt.plot(time, y_observed[:, k], 'g',
                             label='Observed output')
                    plt.plot(time, y_observed_estim[:, k],
                             label='Estimated output', c='orange', alpha=0.9)
                    if not dyn_GP.ground_truth_approx:
                        plt.title(
                            'Kalman rollout of predicted and true output '
                            'over time, random start, random control')
                    else:
                        if title and ('Test' in title):
                            plt.title('Kalman rollout of predicted and true '
                                      'output over time over testing data')
                        elif title and ('Val' in title):
                            plt.title('Kalman rollout of predicted and true '
                                      'output over time over validation data')
                        else:
                            plt.title('Kalman rollout of predicted and true '
                                      'output over time, random start, '
                                      'data control')
                    plt.legend()
                    plt.xlabel('Time steps')
                    plt.ylabel('Output')
                    plt.savefig(os.path.join(rollout_folder, name),
                                bbox_inches='tight')
                    plt.close('all')


# Plot quantities about closed-loop rollouts over time
def plot_kalman_rollout_data(dyn_GP, folder):
    name = 'Kalman_rollout_RMSE'
    RMSE_df = pd.DataFrame(dyn_GP.kalman_rollout_RMSE.numpy())
    RMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.kalman_rollout_RMSE[:, 0],
             dyn_GP.kalman_rollout_RMSE[:, 1],
             'c', label='RMSE')
    plt.title('Kalman rollout RMSE over time, over '
              + str(dyn_GP.nb_rollouts) + ' rollouts')
    plt.xlabel('Number of samples')
    plt.ylabel('RMSE over rollouts')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Kalman_rollout_SRMSE'
    SRMSE_df = pd.DataFrame(dyn_GP.kalman_rollout_SRMSE.numpy())
    SRMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.kalman_rollout_SRMSE[:, 0],
             dyn_GP.kalman_rollout_SRMSE[:, 1],
             'c', label='SRMSE')
    plt.title('Kalman rollout SRMSE over time, over ' + str(
        dyn_GP.nb_rollouts) + ' rollouts')
    plt.xlabel('Number of samples')
    plt.ylabel('SRMSE over rollouts')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Kalman_rollout_log_AL'
    log_AL_df = pd.DataFrame(dyn_GP.kalman_rollout_log_AL.numpy())
    log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.kalman_rollout_log_AL[:, 0],
             dyn_GP.kalman_rollout_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Kalman rollout average log likelihood over time, over ' +
              str(dyn_GP.nb_rollouts) + ' rollouts')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood over rollouts')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Kalman_rollout_stand_log_AL'
    stand_log_AL_df = pd.DataFrame(dyn_GP.kalman_rollout_stand_log_AL.numpy())
    stand_log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.kalman_rollout_stand_log_AL[:, 0],
             dyn_GP.kalman_rollout_stand_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Kalman rollout average log likelihood over time, over ' +
              str(dyn_GP.nb_rollouts) + ' rollouts')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood over rollouts')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')


# Plot quantities about test rollouts over time
def plot_test_kalman_rollout_data(dyn_GP, folder):
    name = 'Test_kalman_rollout_RMSE'
    RMSE_df = pd.DataFrame(dyn_GP.test_kalman_rollout_RMSE.numpy())
    RMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.test_kalman_rollout_RMSE[:, 0],
             dyn_GP.test_kalman_rollout_RMSE[:, 1],
             'c', label='RMSE')
    plt.title('Kalman rollout RMSE over time, over testing data')
    plt.xlabel('Number of samples')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Test_kalman_rollout_SRMSE'
    SRMSE_df = pd.DataFrame(dyn_GP.test_kalman_rollout_SRMSE.numpy())
    SRMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.test_kalman_rollout_SRMSE[:, 0],
             dyn_GP.test_kalman_rollout_SRMSE[:, 1],
             'c', label='SRMSE')
    plt.title('Kalman rollout SRMSE over time, over testing data')
    plt.xlabel('Number of samples')
    plt.ylabel('SRMSE')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Test_kalman_rollout_log_AL'
    log_AL_df = pd.DataFrame(dyn_GP.test_kalman_rollout_log_AL.numpy())
    log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.test_kalman_rollout_log_AL[:, 0],
             dyn_GP.test_kalman_rollout_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Kalman rollout average log likelihood over time, '
              'over testing data')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'),
                bbox_inches='tight')
    plt.close('all')

    name = 'Test_kalman_rollout_stand_log_AL'
    stand_log_AL_df = pd.DataFrame(
        dyn_GP.test_kalman_rollout_stand_log_AL.numpy())
    stand_log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.test_kalman_rollout_stand_log_AL[:, 0],
             dyn_GP.test_kalman_rollout_stand_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Kalman rollout average log likelihood over time, '
              'over testing data')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'),
                bbox_inches='tight')
    plt.close('all')


# Plot quantities about validation rollouts over time
def plot_val_kalman_rollout_data(dyn_GP, folder):
    name = 'Val_kalman_rollout_RMSE'
    RMSE_df = pd.DataFrame(dyn_GP.val_kalman_rollout_RMSE.numpy())
    RMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.val_kalman_rollout_RMSE[:, 0],
             dyn_GP.val_kalman_rollout_RMSE[:, 1],
             'c', label='RMSE')
    plt.title('Kalman rollout RMSE over time, over validation data')
    plt.xlabel('Number of samples')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Val_kalman_rollout_SRMSE'
    SRMSE_df = pd.DataFrame(dyn_GP.val_kalman_rollout_SRMSE.numpy())
    SRMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.val_kalman_rollout_SRMSE[:, 0],
             dyn_GP.val_kalman_rollout_SRMSE[:, 1],
             'c', label='SRMSE')
    plt.title('Kalman rollout SRMSE over time, over validation data')
    plt.xlabel('Number of samples')
    plt.ylabel('SRMSE')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Val_kalman_rollout_log_AL'
    log_AL_df = pd.DataFrame(dyn_GP.val_kalman_rollout_log_AL.numpy())
    log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.val_kalman_rollout_log_AL[:, 0],
             dyn_GP.val_kalman_rollout_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Kalman rollout average log likelihood over time, '
              'over validation data')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'),
                bbox_inches='tight')
    plt.close('all')

    name = 'Val_kalman_rollout_stand_log_AL'
    stand_log_AL_df = pd.DataFrame(
        dyn_GP.val_kalman_rollout_stand_log_AL.numpy())
    stand_log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.val_kalman_rollout_stand_log_AL[:, 0],
             dyn_GP.val_kalman_rollout_stand_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Kalman rollout average log likelihood over time, '
              'over validation data')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'),
                bbox_inches='tight')
    plt.close('all')
