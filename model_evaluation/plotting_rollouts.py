import os
import logging

import numpy as np
import pandas as pd
import seaborn as sb
import torch
from matplotlib import pyplot as plt
from functools import partial

from simulation.observers import dynamics_traj_observer
from simulation.observer_functions import EKF_ODE
from NN_for_ODEs.NODE_utils import make_init_state_obs, \
    update_trajs_init_state_obs
from utils.utils import RMS, log_multivariate_normal_likelihood, reshape_pt1, \
    reshape_dim1, list_torch_to_numpy, interpolate_func

sb.set_style('whitegrid')


# Some useful plotting functions to run open-loop rollouts (trajectory of GP
# predictions given a true and a control trajectory)


# Run open-loop rollouts of GP model
def model_rollout(dyn_GP, init_state, control_traj, true_mean,
                  rollout_length=100, only_prior=False):
    device = true_mean.device
    rollout_length = int(np.min([rollout_length, len(true_mean) - 1]))
    predicted_mean = torch.zeros((rollout_length + 1, init_state.shape[1]),
                                 device=device)
    predicted_lowconf = torch.zeros((rollout_length + 1, init_state.shape[1]),
                                    device=device)
    predicted_uppconf = torch.zeros((rollout_length + 1, init_state.shape[1]),
                                    device=device)
    predicted_var = torch.zeros((rollout_length + 1, init_state.shape[1]),
                                device=device)
    predicted_mean[0] = init_state
    predicted_lowconf[0] = init_state
    predicted_uppconf[0] = init_state
    predicted_var[0] = torch.zeros((1, init_state.shape[1]), device=device)
    for t in range(rollout_length):
        control = control_traj[t]
        if 'Michelangelo' in dyn_GP.system:
            # True and predicted trajectory over time (random start, random
            # control) with Euler to get xt+1 from GP xt, ut->phit
            mean_next, varnext, next_lowconf, next_uppconf = \
                dyn_GP.predict_euler_Michelangelo(predicted_mean[t], control,
                                                  only_prior=only_prior)
        elif ('justvelocity' in dyn_GP.system) and not dyn_GP.continuous_model:
            # True and predicted trajectory over time (random start, random
            # control) with Euler to get xt+1 from GP xt, ut->xn_t+1
            mean_next, varnext, next_lowconf, next_uppconf = \
                dyn_GP.predict_euler_discrete_justvelocity(
                    predicted_mean[t], control, only_prior=only_prior)
        elif ('justvelocity' in dyn_GP.system) and dyn_GP.continuous_model:
            # True and predicted trajectory over time (random start, random
            # control) with Euler to get xt+1 from GP xt, ut->xdot_t
            mean_next, varnext, next_lowconf, next_uppconf = \
                dyn_GP.predict_euler_continuous_justvelocity(
                    predicted_mean[t], control, only_prior=only_prior)
        else:
            # True and predicted trajectory over time (random start, random
            # control)
            mean_next, varnext, next_lowconf, next_uppconf = dyn_GP.predict(
                predicted_mean[t], control, only_prior=only_prior)
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


# Run open-loop rollouts of NODE model
def NODE_rollout(NODE, init_state, control_traj, xtraj_true,
                 rollout_length=100, only_prior=False, scale=True):
    device = xtraj_true.device
    with torch.no_grad():
        rollout_length = int(np.min([rollout_length, len(xtraj_true)]))
        time = torch.arange(0., rollout_length * NODE.dt, step=NODE.dt,
                            device=device)
        if NODE.ground_truth_approx:
            y_observed_true = xtraj_true
        else:
            y_observed_true = NODE.observe_data(xtraj_true)
        if NODE.true_meas_noise_var != 0:
            y_observed_true += torch.normal(0, np.sqrt(
                NODE.true_meas_noise_var), size=y_observed_true.shape)
        t_u = torch.cat((reshape_dim1(time), reshape_dim1(control_traj)), dim=1)
        controller = interpolate_func(
            x=t_u, t0=time[0], init_value=reshape_pt1(control_traj[0]))
        if NODE.init_state_model:
            if NODE.ground_truth_approx:
                # If ground_truth_approx, init_state contains the test inputs
                # for the given recognition model (taken from Xtest)
                x0_estim = NODE.init_state_model(init_state)
            else:
                # Otherwise, need to create the inputs for the recognition
                # model given the rollout trajectories
                obs0 = make_init_state_obs(y_observed_true, control_traj,
                                           init_state, time, NODE.config)
                xtraj_true, y_observed_true, control_traj, time = \
                    update_trajs_init_state_obs(
                        xtraj_true, y_observed_true, control_traj, time,
                        NODE.config)
                x0_estim = NODE.init_state_model(obs0)
        else:
            x0_estim = init_state
        xtraj_estim = NODE.NODE_model.forward_traj(
            x0_estim, controller, time[0], time, reshape_pt1(control_traj[0]))
        y_pred = NODE.observe_data_x(xtraj_estim)
        if scale:
            y_pred = NODE.scaler_Y.transform(y_pred)
            y_observed_true = NODE.scaler_Y.transform(y_observed_true)
        RMSE_output = RMS(y_pred - y_observed_true)
        if NODE.ground_truth_approx:
            RMSE = RMSE_output
            RMSE_init = RMS(
                reshape_pt1(y_pred[0]) - reshape_pt1(y_observed_true[0]))
        else:
            RMSE = RMS(xtraj_estim - xtraj_true)
            RMSE_init = RMS(x0_estim - reshape_pt1(xtraj_true[0]))
        return init_state, control_traj, xtraj_true, xtraj_estim, RMSE, \
               RMSE_init, RMSE_output


# Run open-loop rollouts of NODE model
def NODE_EKF_rollout(NODE, init_state, control_traj, xtraj_true,
                     rollout_length=100, only_prior=False, scale=True):
    device = xtraj_true.device
    if NODE.config.get('prior_kwargs').get('EKF_added_meas_noise_var') is None:
        meas_noise_var = NODE.true_meas_noise_var
    else:
        meas_noise_var = \
            NODE.config.get('prior_kwargs').get('EKF_added_meas_noise_var')
    with torch.no_grad():
        rollout_length = int(np.min([rollout_length, len(xtraj_true)]))
        time = torch.arange(0., rollout_length * NODE.dt, step=NODE.dt,
                            device=device)
        if NODE.ground_truth_approx:
            y_observed_true = xtraj_true
        else:
            y_observed_true = NODE.observe_data(xtraj_true)
        if NODE.config.get('prior_kwargs').get(
                'EKF_observe_data') is not None:
            logging.info('New measurement function for testing EKF')
            y_EKF = NODE.config.get('prior_kwargs').get(
                'EKF_observe_data')(y_observed_true)
        else:
            y_EKF = y_observed_true
        if meas_noise_var != 0:
            y_EKF += torch.normal(0, np.sqrt(meas_noise_var), size=y_EKF.shape)
        t_y = torch.cat((reshape_dim1(time), reshape_dim1(y_EKF)), dim=1)
        measurement = interpolate_func(
            x=t_y, t0=time[0], init_value=reshape_pt1(y_EKF[0]))
        t_u = torch.cat((reshape_dim1(time), reshape_dim1(control_traj)), dim=1)
        controller = interpolate_func(
            x=t_u, t0=time[0], init_value=reshape_pt1(control_traj[0]))
        if NODE.init_state_model:
            if NODE.ground_truth_approx:
                # If ground_truth_approx, init_state contains the test inputs
                # for the given recognition model (taken from Xtest)
                x0_estim = NODE.init_state_model(init_state)
            else:
                # Otherwise, need to create the inputs for the recognition
                # model given the rollout trajectories
                obs0 = make_init_state_obs(y_observed_true, control_traj,
                                           init_state, time, NODE.config)
                xtraj_true, y_observed_true, control_traj, time = \
                    update_trajs_init_state_obs(
                        xtraj_true, y_observed_true, control_traj, time,
                        NODE.config)
                x0_estim = NODE.init_state_model(obs0)
        else:
            x0_estim = init_state
        covar0 = NODE.config.get('prior_kwargs').get('EKF_init_covar')
        x0_estim = torch.cat((x0_estim, reshape_pt1(torch.flatten(covar0))),
                             dim=1)  # EKF state = (x, covar)
        observer = EKF_ODE(device, NODE.config)
        xtraj_estim = dynamics_traj_observer(
            x0=x0_estim, u=controller, y=measurement, t0=time[0], dt=NODE.dt,
            init_control=reshape_pt1(control_traj[0]), discrete=False,
            version=observer, t_eval=time, GP=NODE.NODE_model,
            kwargs=NODE.config)
        xtraj_estim = reshape_pt1(xtraj_estim[:, :NODE.n])  # get rid of covar
        y_pred = NODE.observe_data_x(xtraj_estim)
        if scale:
            y_pred = NODE.scaler_Y.transform(y_pred)
            y_observed_true = NODE.scaler_Y.transform(y_observed_true)
        RMSE_output = RMS(y_pred - y_observed_true)
        if NODE.ground_truth_approx:
            RMSE = RMSE_output
            RMSE_init = RMS(
                reshape_pt1(y_pred[0]) - reshape_pt1(y_observed_true[0]))
        else:
            RMSE = RMS(xtraj_estim - xtraj_true)
            RMSE_init = RMS(x0_estim - reshape_pt1(xtraj_true[0]))
        return init_state, control_traj, xtraj_true, xtraj_estim, RMSE, \
               RMSE_init, RMSE_output


# Save the results of rollouts
def save_rollout_variables(model_object, results_folder, nb_rollouts,
                           rollout_list, step, results=False,
                           ground_truth_approx=False, plots=True, title=None,
                           NODE=False):
    """
    Save all rollout variables (true, predicted, upper and lower confidence
    bounds...), eventually plot them. If ground_truth_approx for NODE,
    plot only predicted values for full state because only observations are
    available on test trajectories.
    """
    if title:
        folder = os.path.join(results_folder, title + '_' + str(step))
    else:
        folder = os.path.join(results_folder, 'Rollouts_' + str(step))
    os.makedirs(folder, exist_ok=True)
    rollout_list = list_torch_to_numpy(rollout_list)  # transfer all back to CPU
    for i in range(len(rollout_list)):
        rollout_folder = os.path.join(folder, 'Rollout_' + str(i))
        os.makedirs(rollout_folder, exist_ok=True)
        if results:
            if not NODE:
                filename = 'Predicted_mean_traj.csv'
                file = pd.DataFrame(reshape_pt1(rollout_list[i][3]))
                file.to_csv(os.path.join(rollout_folder, filename),
                            header=False)
                filename = 'Predicted_var_traj.csv'
                file = pd.DataFrame(reshape_pt1(rollout_list[i][4]))
                file.to_csv(os.path.join(rollout_folder, filename),
                            header=False)
                filename = 'Predicted_lowconf_traj.csv'
                file = pd.DataFrame(reshape_pt1(rollout_list[i][5]))
                file.to_csv(os.path.join(rollout_folder, filename),
                            header=False)
                filename = 'Predicted_uppconf_traj.csv'
                file = pd.DataFrame(reshape_pt1(rollout_list[i][6]))
                file.to_csv(os.path.join(rollout_folder, filename),
                            header=False)
                if not os.path.isfile(
                        os.path.join(rollout_folder, 'True_traj.csv')):
                    filename = 'Init_state.csv'
                    file = pd.DataFrame(rollout_list[i][0])
                    file.to_csv(os.path.join(rollout_folder, filename),
                                header=False)
                    filename = 'Control_traj.csv'
                    file = pd.DataFrame(rollout_list[i][1])
                    file.to_csv(os.path.join(rollout_folder, filename),
                                header=False)
                    filename = 'True_traj.csv'
                    file = pd.DataFrame(rollout_list[i][2])
                    file.to_csv(os.path.join(rollout_folder, filename),
                                header=False)
                filename = 'RMSE.csv'
                file = pd.DataFrame(reshape_pt1(rollout_list[i][7]))
                file.to_csv(os.path.join(rollout_folder, filename),
                            header=False)
                filename = 'SRMSE.csv'
                file = pd.DataFrame(reshape_pt1(rollout_list[i][8]))
                file.to_csv(os.path.join(rollout_folder, filename),
                            header=False)
                filename = 'Log_likelihood.csv'
                file = pd.DataFrame(reshape_pt1(rollout_list[i][9]))
                file.to_csv(os.path.join(rollout_folder, filename),
                            header=False)
                filename = 'Standardized_log_likelihood.csv'
                file = pd.DataFrame(reshape_pt1(rollout_list[i][10]))
                file.to_csv(os.path.join(rollout_folder, filename),
                            header=False)
                true_mean = reshape_dim1(rollout_list[i][2])
                predicted_mean = reshape_dim1(rollout_list[i][3])
                predicted_lowconf = reshape_dim1(rollout_list[i][5])
                predicted_uppconf = reshape_dim1(rollout_list[i][6])
                time = torch.arange(0., len(true_mean))
            else:
                filename = 'Predicted_mean_traj.csv'
                file = pd.DataFrame(reshape_pt1(rollout_list[i][3]))
                file.to_csv(os.path.join(rollout_folder, filename),
                            header=False)
                if not os.path.isfile(
                        os.path.join(rollout_folder, 'True_traj.csv')):
                    filename = 'Init_state.csv'
                    file = pd.DataFrame(rollout_list[i][0])
                    file.to_csv(os.path.join(rollout_folder, filename),
                                header=False)
                    filename = 'Control_traj.csv'
                    file = pd.DataFrame(rollout_list[i][1])
                    file.to_csv(os.path.join(rollout_folder, filename),
                                header=False)
                    filename = 'True_traj.csv'
                    file = pd.DataFrame(rollout_list[i][2])
                    file.to_csv(os.path.join(rollout_folder, filename),
                                header=False)
                filename = 'RMSE.csv'
                file = pd.DataFrame(reshape_pt1(rollout_list[i][4]))
                file.to_csv(os.path.join(rollout_folder, filename),
                            header=False)
                filename = 'SRMSE.csv'
                file = pd.DataFrame(reshape_pt1(rollout_list[i][5]))
                file.to_csv(os.path.join(rollout_folder, filename),
                            header=False)
                true_mean = reshape_dim1(rollout_list[i][2])
                predicted_mean = reshape_dim1(rollout_list[i][3])
                predicted_lowconf = predicted_mean
                predicted_uppconf = predicted_mean
                time = torch.arange(0., len(true_mean))
            if model_object.plot_output:
                if ground_truth_approx and NODE:
                    y_observed = true_mean
                else:
                    if (title is not None) and ('EKF' in title) and \
                            (model_object.config.get('prior_kwargs').get(
                                'EKF_added_meas_noise_var') is not None):
                        meas_noise_var = \
                            model_object.config.get('prior_kwargs').get(
                                'EKF_added_meas_noise_var')
                    else:
                        meas_noise_var = model_object.true_meas_noise_var
                    y_observed = reshape_dim1(model_object.observe_data(
                        torch.as_tensor(true_mean)))
                    if meas_noise_var != 0:
                        y_observed = reshape_pt1(y_observed + np.random.normal(
                            0, np.sqrt(meas_noise_var), y_observed.shape))
                y_observed_estim = reshape_dim1(model_object.observe_data(
                    torch.as_tensor(predicted_mean)))
                filename = 'Rollout_output_predictions.csv'
                file = pd.DataFrame(y_observed_estim)
                file.to_csv(os.path.join(rollout_folder, filename),
                            header=False)
            if plots:
                length = int(np.min([len(true_mean), len(predicted_mean)]))
                time = time[-length:]
                true_mean = true_mean[-length:]
                predicted_mean = predicted_mean[-length:]
                predicted_lowconf = predicted_lowconf[-length:]
                predicted_uppconf = predicted_uppconf[-length:]
                if model_object.plot_output:
                    y_observed = y_observed[-length:]
                    y_observed_estim = y_observed_estim[-length:]
                for k in range(predicted_mean.shape[1]):
                    name = 'Rollout_model_predictions' + str(k) + '.pdf'
                    if ground_truth_approx and NODE:
                        plt.plot(time, predicted_mean[:, k], 'b',
                                 label='Predicted trajectory', alpha=0.7)
                        plt.fill_between(time,
                                         predicted_lowconf[:, k],
                                         predicted_uppconf[:, k],
                                         facecolor='blue', alpha=0.2)
                        plt.title('Predicted test trajectory')
                    else:
                        plt.plot(time, true_mean[:, k], 'g',
                                     label='True trajectory')
                        plt.plot(time, predicted_mean[:, k], 'b',
                                 label='Predicted trajectory', alpha=0.7)
                        plt.fill_between(time,
                                         predicted_lowconf[:, k],
                                         predicted_uppconf[:, k],
                                         facecolor='blue', alpha=0.2)
                        plt.title('Test trajectory')
                    plt.legend()
                    plt.xlabel('Time steps')
                    plt.ylabel('State')
                    plt.savefig(os.path.join(rollout_folder, name),
                                bbox_inches='tight')
                    plt.close('all')

                for k in range(predicted_mean.shape[1] - 1):
                    name = 'Rollout_phase_portrait' + str(k) + '.pdf'
                    if ground_truth_approx and NODE:
                        plt.plot(predicted_mean[:, k], predicted_mean[:, k + 1],
                                 'b', label='Predicted trajectory', alpha=0.7)
                        plt.fill_between(predicted_mean[:, k],
                                         predicted_lowconf[:, k + 1],
                                         predicted_uppconf[:, k + 1],
                                         facecolor='blue', alpha=0.2)
                        plt.title('Phase portrait of predicted test trajectory')
                    else:
                        plt.plot(true_mean[:, k], true_mean[:, k + 1], 'g',
                                 label='True trajectory')
                        plt.plot(predicted_mean[:, k], predicted_mean[:, k + 1],
                                 'b', label='Predicted trajectory', alpha=0.7)
                        plt.fill_between(predicted_mean[:, k],
                                         predicted_lowconf[:, k + 1],
                                         predicted_uppconf[:, k + 1],
                                         facecolor='blue', alpha=0.2)
                        plt.title('Phase portrait of test trajectory')
                    plt.legend()
                    plt.xlabel('x_' + str(k))
                    plt.ylabel('x_' + str(k + 1))
                    plt.savefig(os.path.join(rollout_folder, name),
                                bbox_inches='tight')
                    plt.close('all')

                if model_object.plot_output:
                    for k in range(y_observed_estim.shape[1]):
                        name = 'Rollout_output_predictions' + str(k) + '.pdf'
                        plt.plot(time, y_observed[:, k], 'g',
                                 label='Observed output')
                        plt.plot(time, y_observed_estim[:, k],
                                 label='Predicted output', c='orange',
                                 alpha=0.9)
                        plt.title('Output of test trajectory')
                        plt.legend()
                        plt.xlabel('Time steps')
                        plt.ylabel('Output')
                        plt.savefig(os.path.join(rollout_folder, name),
                                    bbox_inches='tight')
                        plt.close('all')
        else:
            filename = 'Init_state.csv'
            file = pd.DataFrame(rollout_list[i][0])
            file.to_csv(os.path.join(rollout_folder, filename),
                        header=False)
            filename = 'Control_traj.csv'
            file = pd.DataFrame(rollout_list[i][1])
            file.to_csv(os.path.join(rollout_folder, filename),
                        header=False)
            filename = 'True_traj.csv'
            file = pd.DataFrame(rollout_list[i][2])
            file.to_csv(os.path.join(rollout_folder, filename),
                        header=False)


# Plot quantities about rollouts over time
def plot_rollout_data(dyn_GP, folder):
    name = 'Rollout_RMSE'
    RMSE_df = pd.DataFrame(dyn_GP.rollout_RMSE.numpy())
    RMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.rollout_RMSE[:, 0], dyn_GP.rollout_RMSE[:, 1],
             'c', label='RMSE')
    plt.title(
        'Rollout RMSE over time, over ' + str(dyn_GP.nb_rollouts) + ' rollouts')
    plt.xlabel('Number of samples')
    plt.ylabel('RMSE over rollouts')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Rollout_SRMSE'
    SRMSE_df = pd.DataFrame(dyn_GP.rollout_SRMSE.numpy())
    SRMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.rollout_SRMSE[:, 0], dyn_GP.rollout_SRMSE[:, 1],
             'c', label='SRMSE')
    plt.title('Rollout SRMSE over time, over ' + str(dyn_GP.nb_rollouts) +
              ' rollouts')
    plt.xlabel('Number of samples')
    plt.ylabel('SRMSE over rollouts')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Rollout_log_AL'
    log_AL_df = pd.DataFrame(dyn_GP.rollout_log_AL.numpy())
    log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.rollout_log_AL[:, 0], dyn_GP.rollout_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Rollout average log likelihood over time, over ' + str(
        dyn_GP.nb_rollouts) + ' rollouts')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood over rollouts')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Rollout_stand_log_AL'
    stand_log_AL_df = pd.DataFrame(dyn_GP.rollout_stand_log_AL.numpy())
    stand_log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.rollout_stand_log_AL[:, 0],
             dyn_GP.rollout_stand_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Rollout average log likelihood over time, over ' + str(
        dyn_GP.nb_rollouts) + ' rollouts')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood over rollouts')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')


# Plot quantities about test rollouts over time
def plot_test_rollout_data(dyn_GP, folder):
    name = 'Test_rollout_RMSE'
    RMSE_df = pd.DataFrame(dyn_GP.test_rollout_RMSE.numpy())
    RMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.test_rollout_RMSE[:, 0], dyn_GP.test_rollout_RMSE[:, 1],
             'c', label='RMSE')
    plt.title('Rollout RMSE over time, over testing data')
    plt.xlabel('Number of samples')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Test_rollout_SRMSE'
    SRMSE_df = pd.DataFrame(dyn_GP.test_rollout_SRMSE.numpy())
    SRMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.test_rollout_SRMSE[:, 0], dyn_GP.test_rollout_SRMSE[:, 1],
             'c', label='SRMSE')
    plt.title('Rollout SRMSE over time, over testing data')
    plt.xlabel('Number of samples')
    plt.ylabel('SRMSE')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Test_rollout_log_AL'
    log_AL_df = pd.DataFrame(dyn_GP.test_rollout_log_AL.numpy())
    log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.test_rollout_log_AL[:, 0], dyn_GP.test_rollout_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Rollout average log likelihood over time, over testing data')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'),
                bbox_inches='tight')
    plt.close('all')

    name = 'Test_rollout_stand_log_AL'
    stand_log_AL_df = pd.DataFrame(dyn_GP.test_rollout_stand_log_AL.numpy())
    stand_log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.test_rollout_stand_log_AL[:, 0],
             dyn_GP.test_rollout_stand_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Rollout average log likelihood over time, over testing data')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'),
                bbox_inches='tight')
    plt.close('all')


# Plot quantities about validation rollouts over time
def plot_val_rollout_data(dyn_GP, folder):
    name = 'Val_rollout_RMSE'
    RMSE_df = pd.DataFrame(dyn_GP.val_rollout_RMSE.numpy())
    RMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.val_rollout_RMSE[:, 0], dyn_GP.val_rollout_RMSE[:, 1],
             'c', label='RMSE')
    plt.title('Rollout RMSE over time, over validation data')
    plt.xlabel('Number of samples')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Val_rollout_SRMSE'
    SRMSE_df = pd.DataFrame(dyn_GP.val_rollout_SRMSE.numpy())
    SRMSE_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.val_rollout_SRMSE[:, 0], dyn_GP.val_rollout_SRMSE[:, 1],
             'c', label='SRMSE')
    plt.title('Rollout SRMSE over time, over validation data')
    plt.xlabel('Number of samples')
    plt.ylabel('SRMSE')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')
    plt.close('all')

    name = 'Val_rollout_log_AL'
    log_AL_df = pd.DataFrame(dyn_GP.val_rollout_log_AL.numpy())
    log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.val_rollout_log_AL[:, 0], dyn_GP.val_rollout_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Rollout average log likelihood over time, over validation data')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'),
                bbox_inches='tight')
    plt.close('all')

    name = 'Val_rollout_stand_log_AL'
    stand_log_AL_df = pd.DataFrame(dyn_GP.val_rollout_stand_log_AL.numpy())
    stand_log_AL_df.to_csv(os.path.join(folder, name + '.csv'), header=False)
    plt.plot(dyn_GP.val_rollout_stand_log_AL[:, 0],
             dyn_GP.val_rollout_stand_log_AL[:, 1],
             'c', label='Average log likelihood')
    plt.title('Rollout average log likelihood over time, over validation data')
    plt.xlabel('Number of samples')
    plt.ylabel('Average log likelihood')
    plt.legend()
    plt.savefig(os.path.join(folder, name + '.pdf'),
                bbox_inches='tight')
    plt.close('all')
