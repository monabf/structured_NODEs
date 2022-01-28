import os

import numpy as np
import pandas as pd
import seaborn as sb
import torch
from matplotlib import pyplot as plt

from utils.utils import RMS, log_multivariate_normal_likelihood, reshape_pt1, \
    reshape_dim1
from .plotting_closedloop_rollouts import model_closedloop_rollout
from .plotting_kalman_rollouts import model_kalman_rollout
from .plotting_rollouts import model_rollout, NODE_rollout, NODE_EKF_rollout

sb.set_style('whitegrid')


# Some useful plotting functions to make nice graphs of dynamics GPs,
# and other general things such as rollouts, model evaluation


# Evaluate model over a grid
# https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size
# https://stackoverflow.com/questions/36607742/drawing-phase-space-trajectories-with-arrows-in-matplotlib
def plot_model_evaluation(Evaluation_grid, Grid_controls, Predicted_grid,
                          True_predicted_grid, folder,
                          ground_truth_approx=False, title=None,
                          quiver=False, l2_error_array=torch.tensor([]),
                          verbose=False):
    nb = 50000
    Evaluation_grid = Evaluation_grid[:nb]
    Grid_controls = Grid_controls[:nb]
    Predicted_grid = Predicted_grid[:nb]
    True_predicted_grid = True_predicted_grid[:nb]
    if torch.is_tensor(Evaluation_grid):
        Evaluation_grid, Grid_controls, Predicted_grid, True_predicted_grid, \
        l2_error_array = Evaluation_grid.cpu(), Grid_controls.cpu(), \
                         Predicted_grid.cpu(), True_predicted_grid.cpu(), \
                         l2_error_array.cpu()
    if ground_truth_approx:
        l2_error_array = torch.tensor([])
    else:
        quiver = False

    for i in range(Evaluation_grid.shape[1]):
        for j in range(True_predicted_grid.shape[1]):
            if title:
                name = title + str(i) + str(j) + '.pdf'
            else:
                name = 'Model_evaluation' + str(i) + str(j) + '.pdf'
            if quiver:
                plt.quiver(Evaluation_grid[:-1, i], True_predicted_grid[:-1, j],
                           Evaluation_grid[1:, i] - Evaluation_grid[:-1, i],
                           True_predicted_grid[1:, j] -
                           True_predicted_grid[:-1, j],
                           label='True evolution', color='green', alpha=0.9,
                           angles='xy', scale_units='xy', scale=1)
                plt.quiver(Evaluation_grid[:-1, i], Predicted_grid[:-1, j],
                           Evaluation_grid[1:, i] - Evaluation_grid[:-1, i],
                           Predicted_grid[1:, j] - Predicted_grid[:-1, j],
                           label='GP mean prediction', color='blue', alpha=0.6,
                           angles='xy', scale_units='xy', scale=1)
            elif len(torch.nonzero(l2_error_array, as_tuple=False)) != 0:
                plt.scatter(Evaluation_grid[:, i], Predicted_grid[:, j],
                            s=5, cmap='jet', c=l2_error_array,
                            label='Prediction', alpha=0.6)
                cbar = plt.colorbar()
                cbar.set_label('Squared L2 prediction error')
            else:
                plt.scatter(Evaluation_grid[:, i], True_predicted_grid[:, j],
                            s=5, c='g', label='True model', alpha=0.9)
                plt.scatter(Evaluation_grid[:, i], Predicted_grid[:, j],
                            s=5, c='b', label='Prediction', alpha=0.6)
            plt.title('Predicted and true model evaluation')
            plt.legend()
            plt.xlabel('Evaluation points ' + str(i))
            plt.ylabel('Predicted points ' + str(j))
            plt.savefig(os.path.join(folder, name), bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close('all')
    for i in range(Grid_controls.shape[1]):
        for j in range(True_predicted_grid.shape[1]):
            if title:
                name = title + str(i + Evaluation_grid.shape[1]) + str(j) + \
                       '.pdf'
            else:
                name = 'Model_evaluation' + str(i + Evaluation_grid.shape[1]) \
                       + str(j) + '.pdf'
            if quiver:
                plt.quiver(Grid_controls[:-1, i], True_predicted_grid[:-1, j],
                           Grid_controls[1:, i] - Grid_controls[:-1, i],
                           True_predicted_grid[1:, j] -
                           True_predicted_grid[:-1, j],
                           label='True evolution', color='green', alpha=0.9,
                           angles='xy', scale_units='xy', scale=1)
                plt.quiver(Grid_controls[:-1, i], Predicted_grid[:-1, j],
                           Grid_controls[1:, i] - Grid_controls[:-1, i],
                           Predicted_grid[1:, j] - Predicted_grid[:-1, j],
                           label='GP mean prediction', color='blue', alpha=0.6,
                           angles='xy', scale_units='xy', scale=1)
            elif len(torch.nonzero(l2_error_array, as_tuple=False)) != 0:
                plt.scatter(Grid_controls[:, i], Predicted_grid[:, j],
                            s=5, cmap='jet', c=l2_error_array,
                            label='Prediction', alpha=0.6)
                cbar = plt.colorbar()
                cbar.set_label('Squared L2 prediction error')
            else:
                plt.scatter(Grid_controls[:, i], True_predicted_grid[:, j],
                            s=5, c='g', label='True model', alpha=0.9)
                plt.scatter(Grid_controls[:, i], Predicted_grid[:, j],
                            s=5, c='b', label='Prediction', alpha=0.6)
            plt.title('Predicted and true model evaluation')
            plt.legend()
            plt.xlabel('Control points ' + str(i))
            plt.ylabel('Predicted points ' + str(j))
            plt.savefig(os.path.join(folder, name), bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close('all')


# Compute list of errors/model quality in rollouts
def run_rollouts(dyn_GP, input_rollout_list, folder, observer=None,
                 observe_data=None, discrete_observer=False,
                 closedloop=False, kalman=False, no_GP_in_observer=False,
                 only_prior=False):
    rollout_RMSE_list = torch.zeros((len(input_rollout_list), 1))
    rollout_SRMSE_list = torch.zeros((len(input_rollout_list), 1))
    rollout_log_AL_list = torch.zeros((len(input_rollout_list), 1))
    rollout_stand_log_AL_list = torch.zeros((len(input_rollout_list), 1))
    rollout_list = []
    device = input_rollout_list[0][0].device
    complete_length = int(np.sum(
        [len(reshape_pt1(input_rollout_list[i][2])[1:, :]) for i in
         range(len(input_rollout_list))]))
    complete_true_mean = torch.zeros((
        complete_length, input_rollout_list[0][2].shape[1]), device=device)
    for i in range(len(input_rollout_list)):
        current_mean = reshape_pt1(input_rollout_list[i][2])[1:, :]
        current_length = len(current_mean)
        complete_true_mean[i * current_length:(i + 1) * current_length] = \
            current_mean
    mean_test_var = torch.var(complete_true_mean)  # would be better det(covar)
    if reshape_pt1(dyn_GP.scaler_Y._mean).shape[1] == \
            complete_true_mean.shape[1]:
        mean_vector = reshape_pt1(torch.repeat_interleave(
            reshape_pt1(dyn_GP.scaler_Y._mean), len(complete_true_mean), dim=0))
        var_vector = reshape_pt1(torch.repeat_interleave(
            reshape_pt1(dyn_GP.scaler_Y._var), len(complete_true_mean), dim=0))
    else:
        mean_vector = reshape_pt1(torch.repeat_interleave(
            reshape_pt1(dyn_GP.scaler_X._mean), len(complete_true_mean), dim=0))
        var_vector = reshape_pt1(torch.repeat_interleave(
            reshape_pt1(dyn_GP.scaler_X._var), len(complete_true_mean), dim=0))

    # TODO parallelize running different rollouts?
    for i in range(len(input_rollout_list)):
        if kalman:
            init_state, control_traj, true_mean, predicted_mean, \
            predicted_var, predicted_lowconf, predicted_uppconf, RMSE, \
            log_likelihood = model_kalman_rollout(
                dyn_GP=dyn_GP, observer=observer, observe_data=observe_data,
                discrete_observer=discrete_observer,
                init_state=reshape_pt1(input_rollout_list[i][0]),
                control_traj=reshape_pt1(input_rollout_list[i][1]),
                true_mean=reshape_pt1(input_rollout_list[i][2]),
                rollout_length=len(input_rollout_list[i][1]),
                no_GP_in_observer=no_GP_in_observer, only_prior=only_prior)
        elif closedloop:
            init_state, control_traj, true_mean, predicted_mean, \
            predicted_var, predicted_lowconf, predicted_uppconf, RMSE, \
            log_likelihood = model_closedloop_rollout(
                dyn_GP=dyn_GP, observer=observer, observe_data=observe_data,
                init_state=reshape_pt1(input_rollout_list[i][0]),
                control_traj=reshape_pt1(input_rollout_list[i][1]),
                true_mean=reshape_pt1(input_rollout_list[i][2]),
                rollout_length=len(input_rollout_list[i][1]),
                no_GP_in_observer=no_GP_in_observer)
        else:
            init_state, control_traj, true_mean, predicted_mean, \
            predicted_var, predicted_lowconf, predicted_uppconf, RMSE, \
            log_likelihood = model_rollout(
                dyn_GP=dyn_GP, init_state=reshape_pt1(input_rollout_list[i][0]),
                control_traj=reshape_pt1(input_rollout_list[i][1]),
                true_mean=reshape_pt1(input_rollout_list[i][2]),
                rollout_length=len(input_rollout_list[i][1]),
                only_prior=only_prior)
        SRMSE = RMSE / mean_test_var
        stand_log_likelihood = \
            log_likelihood - log_multivariate_normal_likelihood(
                complete_true_mean, mean_vector, var_vector)
        rollout_RMSE_list[i] = RMSE
        rollout_SRMSE_list[i] = SRMSE
        rollout_log_AL_list[i] = log_likelihood
        rollout_stand_log_AL_list[i] = stand_log_likelihood
        rollout_list.append(
            [predicted_mean, predicted_var, predicted_lowconf,
             predicted_uppconf, RMSE, SRMSE, log_likelihood,
             stand_log_likelihood])
    rollout_RMSE = torch.mean(reshape_pt1(rollout_RMSE_list)).cpu()
    rollout_SRMSE = torch.mean(reshape_pt1(rollout_SRMSE_list)).cpu()
    rollout_log_AL = torch.mean(reshape_pt1(rollout_log_AL_list)).cpu()
    rollout_stand_log_AL = torch.mean(reshape_pt1(
        rollout_stand_log_AL_list)).cpu()
    return rollout_list, rollout_RMSE, rollout_SRMSE, rollout_log_AL, \
           rollout_stand_log_AL


# Same as run_rollouts but no probabilistic metrics
def run_rollouts_NODE(NODE, input_rollout_list, only_prior=False,
                      type='default'):
    rollout_RMSE_list = torch.zeros((len(input_rollout_list), 1))
    rollout_SRMSE_list = torch.zeros((len(input_rollout_list), 1))
    rollout_RMSE_init_list = torch.zeros((len(input_rollout_list), 1))
    rollout_SRMSE_init_list = torch.zeros((len(input_rollout_list), 1))
    rollout_RMSE_output_list = torch.zeros((len(input_rollout_list), 1))
    rollout_SRMSE_output_list = torch.zeros((len(input_rollout_list), 1))
    rollout_list = []
    device = input_rollout_list[0][0].device
    complete_length = int(np.sum(
        [len(reshape_pt1(input_rollout_list[i][2])[1:, :]) for i in
         range(len(input_rollout_list))]))
    complete_true_mean = torch.zeros((
        complete_length, input_rollout_list[0][2].shape[1]), device=device)
    inits = torch.zeros(len(input_rollout_list),
                        input_rollout_list[0][0].shape[1], device=device)
    for i in range(len(input_rollout_list)):
        current_mean = reshape_pt1(input_rollout_list[i][2])[1:, :]
        current_length = len(current_mean)
        complete_true_mean[i * current_length:(i + 1) * current_length] = \
            current_mean
        inits[i] = reshape_pt1(input_rollout_list[0][0])
    mean_test_var = torch.var(complete_true_mean)  # would be better det(covar)
    inits_var = torch.var(inits)
    if NODE.ground_truth_approx:
        mean_output_var = mean_test_var
    else:
        mean_output_var = torch.var(NODE.observe_data(complete_true_mean))

    # TODO parallelize running different rollouts?
    for i in range(len(input_rollout_list)):
        if type == 'default':
            init_state, control_traj, true_mean, predicted_mean, RMSE, \
            RMSE_init, RMSE_output = \
                NODE_rollout(NODE=NODE, init_state=reshape_pt1(
                    input_rollout_list[i][0]), control_traj=reshape_pt1(
                    input_rollout_list[i][1]), xtraj_true=reshape_pt1(
                    input_rollout_list[i][2]), rollout_length=len(
                    input_rollout_list[i][1]), only_prior=only_prior)
        elif type == 'EKF':
            init_state, control_traj, true_mean, predicted_mean, RMSE, \
            RMSE_init, RMSE_output = \
                NODE_EKF_rollout(NODE=NODE, init_state=reshape_pt1(
                    input_rollout_list[i][0]), control_traj=reshape_pt1(
                    input_rollout_list[i][1]), xtraj_true=reshape_pt1(
                    input_rollout_list[i][2]), rollout_length=len(
                    input_rollout_list[i][1]), only_prior=only_prior)
        SRMSE = RMSE / mean_test_var
        rollout_RMSE_list[i] = RMSE
        rollout_SRMSE_list[i] = SRMSE
        rollout_RMSE_init_list[i] = RMSE_init
        rollout_SRMSE_init_list[i] = RMSE_init / inits_var
        rollout_RMSE_output_list[i] = RMSE_output
        rollout_SRMSE_output_list[i] = RMSE_output / mean_output_var
        rollout_list.append([predicted_mean, RMSE, SRMSE])
    rollout_RMSE = torch.mean(reshape_pt1(rollout_RMSE_list)).cpu()
    rollout_SRMSE = torch.mean(reshape_pt1(rollout_SRMSE_list)).cpu()
    rollout_RMSE_init = torch.mean(reshape_pt1(rollout_RMSE_init_list)).cpu()
    rollout_SRMSE_init = torch.mean(reshape_pt1(rollout_SRMSE_init_list)).cpu()
    rollout_RMSE_output = torch.mean(reshape_pt1(
        rollout_RMSE_output_list)).cpu()
    rollout_SRMSE_output = torch.mean(reshape_pt1(
        rollout_SRMSE_output_list)).cpu()
    return rollout_list, rollout_RMSE, rollout_SRMSE, rollout_RMSE_init, \
           rollout_SRMSE_init, rollout_RMSE_output, rollout_SRMSE_output


# Plot raw data received by GP
def save_GP_data(dyn_GP, direct=True, verbose=False):
    for i in range(dyn_GP.X.shape[1]):
        name = 'GP_input' + str(i) + '.pdf'
        plt.plot(dyn_GP.GP_X[:, i].cpu(), label='GP_X' + str(i))
        plt.title('Visualization of state input data given to GP')
        plt.legend()
        plt.xlabel('Time steps')
        plt.ylabel('State')
        plt.savefig(os.path.join(dyn_GP.results_folder, name),
                    bbox_inches='tight')
        if verbose:
            plt.show()
        plt.close('all')

    for i in range(dyn_GP.X.shape[1], dyn_GP.GP_X.shape[1]):
        name = 'GP_input' + str(i) + '.pdf'
        plt.plot(dyn_GP.GP_X[:, i].cpu(), label='GP_X' + str(i), c='m')
        plt.title('Visualization of control input data given to GP')
        plt.legend()
        plt.xlabel('Time steps')
        plt.ylabel('State')
        plt.savefig(os.path.join(dyn_GP.results_folder, name),
                    bbox_inches='tight')
        if verbose:
            plt.show()
        plt.close('all')

    for i in range(dyn_GP.GP_Y.shape[1]):
        name = 'GP_output' + str(i) + '.pdf'
        plt.plot(dyn_GP.GP_Y[:, i].cpu(), label='GP_Y' + str(i), c='orange')
        plt.title('Visualization of output data given to GP')
        plt.legend()
        plt.xlabel('Time steps')
        plt.ylabel('State')
        plt.savefig(os.path.join(dyn_GP.results_folder, name),
                    bbox_inches='tight')
        if verbose:
            plt.show()
        plt.close('all')

    if direct:
        for i in range(dyn_GP.X.shape[1]):
            for j in range(dyn_GP.Y.shape[1]):
                name = 'GP_data' + str(i) + str(j) + '.pdf'
                plt.scatter(dyn_GP.X[:, i].cpu(), dyn_GP.Y[:, j].cpu(), c='c')
                plt.title('Direct visualization of GP data')
                plt.xlabel('GP_X_' + str(i))
                plt.ylabel('GP_Y_' + str(j))
                plt.savefig(os.path.join(dyn_GP.results_folder, name),
                            bbox_inches='tight')
                if verbose:
                    plt.show()
                plt.close('all')


# Plot GP predictions with control = 0, over a linspace of each input dim
# while keeping all other input dims at 0
def plot_GP(dyn_GP, grid, verbose=False):
    xdim = dyn_GP.X.shape[1]
    udim = dyn_GP.U.shape[1]
    for i in range(xdim + udim):
        dataplot = torch.linspace(torch.min(grid[:, i]), torch.max(grid[:, i]),
                                  dyn_GP.nb_plotting_pts, device=dyn_GP.device)
        data = torch.zeros((dyn_GP.nb_plotting_pts, xdim + udim),
                           device=dyn_GP.device)
        data[:, i] = dataplot
        x = data[:, :xdim]
        u = data[:, xdim:]
        predicted_mean, predicted_var, predicted_lowconf, \
        predicted_uppconf = dyn_GP.predict(x, u)
        if not dyn_GP.ground_truth_approx:
            true_predicted_mean = predicted_mean.clone()
            for idx, _ in enumerate(true_predicted_mean):
                true_predicted_mean[idx] = reshape_pt1(
                    dyn_GP.true_dynamics(reshape_pt1(x[idx]),
                                         reshape_pt1(u[idx])))
            df = pd.DataFrame(true_predicted_mean.cpu().numpy())
            df.to_csv(os.path.join(
                dyn_GP.results_folder, 'GP_plot_true' + str(i) + '.csv'),
                header=False)
        df = pd.DataFrame(predicted_mean.cpu().numpy())
        df.to_csv(os.path.join(dyn_GP.results_folder, 'GP_plot_estim' + str(i)
                               + '.csv'), header=False)
        for j in range(dyn_GP.Y.shape[1]):
            # Plot function learned by GP
            name = 'GP_plot' + str(i) + str(j) + '.pdf'
            if not dyn_GP.ground_truth_approx:
                plt.plot(data[:, i].cpu(), true_predicted_mean[:, j].cpu(),
                         label='True function', c='darkgreen')
            plt.plot(data[:, i].cpu(), predicted_mean[:, j].cpu(),
                     label='GP mean',
                     alpha=0.9)
            plt.fill_between(data[:, i].cpu(), predicted_lowconf[:, j].cpu(),
                             predicted_uppconf[:, j].cpu(),
                             facecolor='blue', alpha=0.2)
            if not dyn_GP.ground_truth_approx:
                plt.title('Visualization of GP posterior')
            else:
                plt.title('Visualization of GP posterior over training data')
            plt.legend()
            plt.xlabel('Input state ' + str(i))
            plt.ylabel('GP prediction ' + str(j) + '(x)')
            plt.savefig(os.path.join(dyn_GP.results_folder, name),
                        bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close('all')

    # if any(k in dyn_GP.system for k in ['Michelangelo', 'EKF']):
    #     for i in range(dyn_GP.X.shape[1]):
    #         dataplot = torch.linspace(
    #             torch.min(grid[:, i]), torch.max(grid[:, i]),
    #             dyn_GP.nb_plotting_pts)
    #         data = torch.zeros((dyn_GP.nb_plotting_pts, xdim + udim))
    #         data[:, i] = dataplot
    #         x = data[:, :xdim]
    #         u = data[:, xdim:]
    #         predicted_mean_deriv, predicted_var_deriv, \
    #         predicted_lowconf_deriv, predicted_uppconf_deriv = \
    #             dyn_GP.predict_deriv(x, u)
    #         df = pd.DataFrame(predicted_mean_deriv.numpy())
    #         df.to_csv(os.path.join(dyn_GP.results_folder, 'GP_plot_deriv' +
    #                                str(i) + '.csv'), header=False)
    #         for j in range(predicted_mean_deriv.shape[1]):
    #             # Plot derivative of function learned by GP
    #             name = 'GP_plot_deriv' + str(i) + str(j) + '.pdf'
    #             plt.plot(x[:, i], predicted_mean_deriv[:, j],
    #                      label='dGP_' + str(j) + '/dx')
    #             plt.fill_between(x[:, i], predicted_lowconf_deriv[:, j],
    #                              predicted_uppconf_deriv[:, j],
    #                              facecolor='blue', alpha=0.2)
    #             if not dyn_GP.ground_truth_approx:
    #                 plt.title('Visualization of GP posterior derivative')
    #             else:
    #                 plt.title(
    #                     'Visualization of GP posterior derivative over'
    #                     'training data')
    #             plt.legend()
    #             plt.xlabel('Input state ' + str(i))
    #             plt.ylabel('GP derivative prediction ' + str(j) + '(x)')
    #             plt.savefig(os.path.join(dyn_GP.results_folder, name),
    #                         bbox_inches='tight')
    #             if verbose:
    #                 plt.show()
    #             plt.close('all')


# Plot NN predictions over a linspace of each input dim while keeping all
# other input dims at 0
def plot_NODE(NODE, grid, verbose=False):
    xdim = NODE.n
    udim = NODE.nu
    device = NODE.X_train.device
    if NODE.no_control:
        udim = 0
    for i in range(xdim + udim):
        if grid.shape[1] < xdim + udim:
            grid_inf = NODE.grid_inf
            grid_sup = NODE.grid_sup
        else:
            grid_inf = torch.min(grid[:, i])
            grid_sup = torch.max(grid[:, i])
        dataplot = torch.linspace(grid_inf, grid_sup, NODE.nb_plotting_pts,
                                  device=device)
        data = torch.zeros((NODE.nb_plotting_pts, xdim + NODE.nu),
                           device=device)
        data[:, i] = dataplot
        x = data[:, :xdim]
        u = data[:, xdim:]
        control = lambda t, kwargs, t0, init_control, impose_init_control: \
            reshape_dim1(u)
        learned_f = NODE.NODE_model.defunc.dyn_NODE(
            t=NODE.t0, x=x, u=control, t0=NODE.t0,
            init_control=NODE.init_control, process_noise_var=0.,
            kwargs=NODE.config, verbose=False).detach()
        df = pd.DataFrame(learned_f.cpu().numpy())
        df.to_csv(os.path.join(NODE.results_folder, 'Model_plot_estim' + str(i)
                               + '.csv'), header=False)
        if not NODE.ground_truth_approx:
            true_f = NODE.true_dynamics(
                t=NODE.t0, x=x, u=control, t0=NODE.t0,
                init_control=NODE.init_control, process_noise_var=0.,
                kwargs=NODE.config).detach()
            df = pd.DataFrame(true_f.cpu().numpy())
            df.to_csv(os.path.join(
                NODE.results_folder, 'Model_plot_true' + str(i) + '.csv'),
                header=False)
        for j in range(NODE.n):
            # Plot function learned by GP
            name = 'Model_plot' + str(i) + str(j) + '.pdf'
            if not NODE.ground_truth_approx:
                plt.plot(data[:, i].cpu(), true_f[:, j].cpu(),
                         label='True function', c='darkgreen')
            plt.plot(data[:, i].cpu(), learned_f[:, j].cpu(),
                     label='Learned model', alpha=0.9)
            plt.title('Dynamics model')
            plt.legend()
            plt.xlabel(r'$x_{}$'.format(i))
            plt.ylabel(r'$f_{}(x)$'.format(j))
            plt.savefig(os.path.join(NODE.results_folder, name),
                        bbox_inches='tight')
            if verbose:
                plt.show()
            plt.close('all')


# Save data from outside the GP model into its results folder
def save_outside_data(dyn_GP, data_dic, outside_folder=None):
    if not outside_folder:
        outside_folder = os.path.join(dyn_GP.results_folder, 'Data_outside_GP')
    os.makedirs(outside_folder, exist_ok=True)
    for key, val in data_dic.items():
        df = pd.DataFrame(val.numpy())
        df.to_csv(os.path.join(outside_folder, key + '.csv'), header=False)


# Plot data from outside the GP model into its results folder, start function
def plot_outside_data_start(dyn_GP, data_dic, outside_folder=None):
    if not outside_folder:
        outside_folder = os.path.join(dyn_GP.results_folder, 'Data_outside_GP')
    os.makedirs(outside_folder, exist_ok=True)
    if all(k in data_dic for k in ('xtraj', 'xtraj_estim')):
        plot_xtraj_xtrajestim = True
    else:
        plot_xtraj_xtrajestim = False

    for key, val in data_dic.items():
        if key in ('xtraj', 'xtraj_estim') and plot_xtraj_xtrajestim:
            continue
        val = reshape_dim1(val)
        for i in range(val.shape[1]):
            name = key + str(i) + '.pdf'
            plt.plot(val[:, i], label=key + str(i))
            plt.title(key)
            plt.legend()
            plt.xlabel('Time steps')
            plt.ylabel(key)
            plt.savefig(os.path.join(outside_folder, name),
                        bbox_inches='tight')
            plt.close('all')

    if plot_xtraj_xtrajestim:
        xtraj = reshape_dim1(data_dic.get('xtraj'))
        xtraj_estim = reshape_dim1(data_dic.get('xtraj_estim'))
        if len(torch.nonzero(xtraj, as_tuple=False)) == 0:
            return 0, 0, 0, 0, 0
        dimmin = np.min([xtraj_estim.shape[1], xtraj.shape[1]])
        for i in range(dimmin):
            name = 'xtraj_xtrajestim_' + str(i) + '.pdf'
            plt.plot(xtraj[:, i], label='True state', c='g')
            plt.plot(xtraj_estim[:, i], label='Estimated state',
                     c='orange', alpha=0.9)
            plt.title('True and estimated position over time')
            plt.legend()
            plt.xlabel('Time steps')
            plt.ylabel('x_' + str(i))
            plt.savefig(os.path.join(outside_folder, name), bbox_inches='tight')
            plt.close('all')

        name = 'Estimation_RMSE_time'
        RMSE = RMS(xtraj[:, :dimmin] - xtraj_estim[:, :dimmin])
        SRMSE = RMSE / torch.var(xtraj)
        output_RMSE = RMS(xtraj[:, 0] - xtraj_estim[:, 0])
        output_SRMSE = RMSE / torch.var(xtraj[:, 0])
        error = torch.sqrt(
            torch.mean(
                torch.square(xtraj[:, :dimmin] - xtraj_estim[:, :dimmin]),
                dim=1))
        error_df = pd.DataFrame(error.numpy())
        error_df.to_csv(
            os.path.join(outside_folder, name + '_time.csv'),
            header=False)
        plt.plot(error, 'orange', label='Error')
        plt.title('State estimation RMSE over last cycle')
        plt.xlabel('Time steps')
        plt.ylabel(r'$|x - \hat{x}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        name = 'Energy_estimated_signal'
        energy = torch.sum(torch.square(xtraj_estim), dim=1)
        energy_df = pd.DataFrame(energy)
        energy_df.to_csv(
            os.path.join(outside_folder, name + '.csv'), header=False)
        plt.plot(energy, 'orange', label='Energy')
        plt.title('Energy of estimated signal over last cycle')
        plt.xlabel('Time steps')
        plt.ylabel(r'$|\hat{x}|^2$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        return error, RMSE, SRMSE, output_RMSE, output_SRMSE


# Plot data from outside the GP model into its results folder, complete
# function that also plots quantities about that outside data that have
# varied over time (such as estimation RMSE of that data over time)
def plot_outside_data(dyn_GP, data_dic, outside_folder=None):
    if not outside_folder:
        outside_folder = os.path.join(dyn_GP.results_folder, 'Data_outside_GP')
    os.makedirs(outside_folder, exist_ok=True)
    error, RMSE, SRMSE, output_RMSE, output_SRMSE = \
        plot_outside_data_start(dyn_GP=dyn_GP, data_dic=data_dic,
                                outside_folder=outside_folder)
    if all(k in data_dic for k in ('xtraj', 'xtraj_estim')):
        dyn_GP.observer_RMSE = torch.cat((
            dyn_GP.observer_RMSE, reshape_pt1(torch.tensor(
                [torch.tensor(dyn_GP.sample_idx), RMSE]))), dim=0)
        dyn_GP.observer_SRMSE = torch.cat((
            dyn_GP.observer_SRMSE, reshape_pt1(torch.tensor(
                [torch.tensor(dyn_GP.sample_idx), SRMSE]))), dim=0)

        name = 'Estimation_RMSE'
        RMSE_df = pd.DataFrame(dyn_GP.observer_RMSE.numpy())
        RMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                       header=False)
        plt.plot(dyn_GP.observer_RMSE[:, 0], dyn_GP.observer_RMSE[:, 1],
                 'c', label='RMSE')
        plt.title('State estimation RMSE over time')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|x - \hat{x}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        name = 'Estimation_SRMSE'
        SRMSE_df = pd.DataFrame(dyn_GP.observer_SRMSE.numpy())
        SRMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                        header=False)
        plt.plot(dyn_GP.observer_SRMSE[:, 0], dyn_GP.observer_SRMSE[:, 1],
                 'c', label='SRMSE')
        plt.title('State estimation SRMSE over time')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|x - \hat{x}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        dyn_GP.output_RMSE = torch.cat((
            dyn_GP.output_RMSE, reshape_pt1(torch.tensor(
                [torch.tensor(dyn_GP.sample_idx), output_RMSE]))), dim=0)
        dyn_GP.output_SRMSE = torch.cat((
            dyn_GP.output_SRMSE, reshape_pt1(torch.tensor(
                [torch.tensor(dyn_GP.sample_idx), output_SRMSE]))), dim=0)

        name = 'Output_RMSE'
        RMSE_df = pd.DataFrame(dyn_GP.output_RMSE.numpy())
        RMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                       header=False)
        plt.plot(dyn_GP.output_RMSE[:, 0], dyn_GP.output_RMSE[:, 1],
                 'c', label='RMSE')
        plt.title('Output RMSE over time')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|y - \hat{y}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        name = 'Output_SRMSE'
        SRMSE_df = pd.DataFrame(dyn_GP.output_SRMSE.numpy())
        SRMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                        header=False)
        plt.plot(dyn_GP.output_SRMSE[:, 0], dyn_GP.output_SRMSE[:, 1],
                 'c', label='SRMSE')
        plt.title('Output SRMSE over time')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|y - \hat{y}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')


# Plot data from outside the GP model into its results folder, complete
# function that also plots quantities about that outside data that have
# varied over time (such as estimation RMSE of that data over time),
# but targeted for validation data
def plot_outside_validation_data(dyn_GP, data_dic, outside_folder=None):
    if not outside_folder:
        outside_folder = os.path.join(dyn_GP.validation_folder,
                                      'Data_outside_GP')
    error, RMSE, SRMSE, output_RMSE, output_SRMSE = \
        plot_outside_data_start(dyn_GP=dyn_GP, data_dic=data_dic,
                                outside_folder=outside_folder)
    if all(k in data_dic for k in ('xtraj', 'xtraj_estim')):
        dyn_GP.observer_val_RMSE = torch.cat((
            dyn_GP.observer_val_RMSE, reshape_pt1(torch.tensor(
                [torch.tensor(dyn_GP.sample_idx), RMSE]))), dim=0)
        dyn_GP.observer_val_SRMSE = torch.cat((
            dyn_GP.observer_val_SRMSE, reshape_pt1(torch.tensor(
                [torch.tensor(dyn_GP.sample_idx), SRMSE]))), dim=0)

        name = 'Estimation_RMSE'
        val_RMSE_df = pd.DataFrame(dyn_GP.observer_val_RMSE.numpy())
        val_RMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                           header=False)
        plt.plot(dyn_GP.observer_val_RMSE[:, 0], dyn_GP.observer_val_RMSE[:, 1],
                 'c', label='RMSE')
        plt.title('State estimation RMSE over time over validation data')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|x - \hat{x}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        name = 'Estimation_SRMSE'
        val_SRMSE_df = pd.DataFrame(dyn_GP.observer_val_SRMSE.numpy())
        val_SRMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                            header=False)
        plt.plot(dyn_GP.observer_val_SRMSE[:, 0],
                 dyn_GP.observer_val_SRMSE[:, 1],
                 'c', label='SRMSE')
        plt.title('State estimation SRMSE over time over validation data')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|x - \hat{x}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        dyn_GP.output_val_RMSE = torch.cat((
            dyn_GP.output_val_RMSE, reshape_pt1(torch.tensor(
                [torch.tensor(dyn_GP.sample_idx), output_RMSE]))), dim=0)
        dyn_GP.output_val_SRMSE = torch.cat((
            dyn_GP.output_val_SRMSE, reshape_pt1(torch.tensor(
                [torch.tensor(dyn_GP.sample_idx), output_SRMSE]))), dim=0)

        name = 'Output_RMSE'
        RMSE_df = pd.DataFrame(dyn_GP.output_val_RMSE.numpy())
        RMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                       header=False)
        plt.plot(dyn_GP.output_val_RMSE[:, 0], dyn_GP.output_val_RMSE[:, 1],
                 'c', label='RMSE')
        plt.title('Output RMSE over time')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|y - \hat{y}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        name = 'Output_SRMSE'
        SRMSE_df = pd.DataFrame(dyn_GP.output_val_SRMSE.numpy())
        SRMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                        header=False)
        plt.plot(dyn_GP.output_val_SRMSE[:, 0], dyn_GP.output_val_SRMSE[:, 1],
                 'c', label='SRMSE')
        plt.title('Output SRMSE over time')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|y - \hat{y}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')


# Save outside data into validation results folder
def save_outside_validation_data(dyn_GP, data_dic, outside_folder=None):
    if not outside_folder:
        outside_folder = os.path.join(dyn_GP.validation_folder,
                                      'Data_outside_GP')
    save_outside_data(dyn_GP=dyn_GP, data_dic=data_dic,
                      outside_folder=outside_folder)


# Plot data from outside the GP model into its results folder, complete
# function that also plots quantities about that outside data that have
# varied over time (such as estimation RMSE of that data over time),
# but targeted for test data
def plot_outside_test_data(dyn_GP, data_dic, outside_folder=None):
    if not outside_folder:
        outside_folder = os.path.join(dyn_GP.test_folder, 'Data_outside_GP')
    error, RMSE, SRMSE, output_RMSE, output_SRMSE = \
        plot_outside_data_start(dyn_GP=dyn_GP, data_dic=data_dic,
                                outside_folder=outside_folder)
    if all(k in data_dic for k in ('xtraj', 'xtraj_estim')):
        dyn_GP.observer_test_RMSE = torch.cat((
            dyn_GP.observer_test_RMSE, reshape_pt1(torch.tensor(
                [torch.tensor(dyn_GP.sample_idx), RMSE]))), dim=0)
        dyn_GP.observer_test_SRMSE = torch.cat((
            dyn_GP.observer_test_SRMSE, reshape_pt1(torch.tensor(
                [torch.tensor(dyn_GP.sample_idx), SRMSE]))), dim=0)

        name = 'Estimation_RMSE'
        test_RMSE_df = pd.DataFrame(dyn_GP.observer_test_RMSE.numpy())
        test_RMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                            header=False)
        plt.plot(dyn_GP.observer_test_RMSE[:, 0],
                 dyn_GP.observer_test_RMSE[:, 1],
                 'c', label='RMSE')
        plt.title('State estimation RMSE over time over test data')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|x - \hat{x}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        name = 'Estimation_SRMSE'
        test_SRMSE_df = pd.DataFrame(dyn_GP.observer_test_SRMSE.numpy())
        test_SRMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                             header=False)
        plt.plot(dyn_GP.observer_test_SRMSE[:, 0],
                 dyn_GP.observer_test_SRMSE[:, 1],
                 'c', label='SRMSE')
        plt.title('State estimation SRMSE over time over test data')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|x - \hat{x}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        dyn_GP.output_test_RMSE = torch.cat((
            dyn_GP.output_test_RMSE, reshape_pt1(torch.tensor(
                [torch.tensor(dyn_GP.sample_idx), output_RMSE]))), dim=0)
        dyn_GP.output_test_SRMSE = torch.cat((
            dyn_GP.output_test_SRMSE, reshape_pt1(torch.tensor(
                [torch.tensor(dyn_GP.sample_idx), output_SRMSE]))), dim=0)

        name = 'Output_RMSE'
        RMSE_df = pd.DataFrame(dyn_GP.output_test_RMSE.numpy())
        RMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                       header=False)
        plt.plot(dyn_GP.output_test_RMSE[:, 0], dyn_GP.output_test_RMSE[:, 1],
                 'c', label='RMSE')
        plt.title('Output RMSE over time')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|y - \hat{y}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')

        name = 'Output_SRMSE'
        SRMSE_df = pd.DataFrame(dyn_GP.output_test_SRMSE.numpy())
        SRMSE_df.to_csv(os.path.join(outside_folder, name + '.csv'),
                        header=False)
        plt.plot(dyn_GP.output_test_SRMSE[:, 0], dyn_GP.output_test_SRMSE[:, 1],
                 'c', label='SRMSE')
        plt.title('Output SRMSE over time')
        plt.xlabel('Number of samples')
        plt.ylabel(r'$|y - \hat{y}|$')
        plt.legend()
        plt.savefig(os.path.join(outside_folder, name + '.pdf'),
                    bbox_inches='tight')
        plt.close('all')


# Save outside data into test results folder
def save_outside_test_data(dyn_GP, data_dic, outside_folder=None):
    if not outside_folder:
        outside_folder = os.path.join(dyn_GP.test_folder, 'Data_outside_GP')
    save_outside_data(dyn_GP=dyn_GP, data_dic=data_dic,
                      outside_folder=outside_folder)
