import logging
# Pickle cannot be used with GPyTorch:
# https://github.com/cloudpipe/cloudpickle/issues/381
# https://stackoverflow.com/questions/23582489/python-pickle-protocol-choice
import os
import sys
import time as timep

import dill as pkl
import numpy as np
import pandas as pd
import pickle5 as pickle
import pytorch_lightning as pl
import seaborn as sb
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append('.')

from simulation.controllers import Control_from_list
from simulation.observers import dim14_observe_data, dim1_observe_data
from simulation.observer_functions import KKL, KKLu
from wandercraft_id_NN_models import MLPn, WDC_simple_recog4, \
    WDC_single_deformation_model
from eval_WDC_data import plot_NODE_lin_rollouts
from NN_for_ODEs.learn_neural_ODE_several_exp_datasets import \
    Learn_NODE_difftraj_exp_datasets
from NN_for_ODEs.NODE_utils import make_diff_init_state_obs, \
    update_config_init_state_obs, set_DF
from utils.utils import start_log, stop_log, reshape_pt1, reshape_dim1, \
    Interpolate_func
from utils.pytorch_utils import get_parameters
from utils.config import Config

sb.set_style('whitegrid')

# Script to learn structured NODEs on the robotic exoskeleton data from
# Wandercraft (further WDC)
# No structure

# Logging
# https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(
            '../Figures/Logs', 'log' + str(sys.argv[1]))),
        logging.StreamHandler(sys.stdout)
    ])

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # no GPU
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'  # more explicit cuda errors

if __name__ == '__main__':
    start_log()
    start = timep.time()
    train = True
    test = True
    colab = False
    dataset_on_GPU = False
    if torch.cuda.is_available():
        gpus = -1
    else:
        gpus = 0
    print(gpus, torch.cuda.is_available())

    # Read data
    vars = dict.fromkeys(['u', 'theta', 'w'])
    dt = 1e-3
    dt_before_subsampling = 1e-3
    data_folder = 'Data/preprocessing/' \
                  '7-29-2difftraj_0.001subsampling_200samples_filter_gaussian50_butteru/'
    for key in vars.keys():
        data_file = data_folder + 'data_' + key + '.pkl'
        if colab:
            with open(data_file, "rb") as fh:
                data = pickle.load(fh)
                df = pd.DataFrame(data)
        else:
            df = pd.read_pickle(data_file).T
        vars[key] = torch.unsqueeze(torch.as_tensor(df.values), dim=-1)
    nb_samples = df.shape[1]
    t0 = 0.
    tf = t0 + (nb_samples - 1) * dt
    U_train = vars['u']
    X_train = torch.cat((vars['theta'], vars['w']), dim=-1)
    # Shorten nb of difftraj: get rid of some data to shorten computation
    nb_difftraj_max = int(np.min([600, len(X_train)]))
    random_idx = torch.randint(0, len(X_train), size=(nb_difftraj_max,))
    X_train = X_train[random_idx]
    U_train = U_train[random_idx]
    nb_difftraj = len(X_train)
    print(X_train.shape, U_train.shape, dt, tf)

    # General params
    config = Config(system='Continuous/Wandercraft_id/MLPn_noisy_inputs',
                    sensitivity='autograd',
                    intloss=None,
                    order=1,
                    nb_samples=nb_samples,
                    nb_difftraj=nb_difftraj,
                    t0_span=t0,
                    tf_span=tf,
                    t0=t0,
                    tf=tf,
                    dt=dt,
                    init_state_obs_method=str(sys.argv[2]),
                    setD_method='diag',
                    init_state_obs_T=100,
                    data_folder=data_folder,
                    NODE_file='../Figures/Continuous/Wandercraft_id/'
                              'MLPn_noisy_inputs/7-29-2difftraj_0.001subsampling_200samples_filter_gaussian50/'
                              'KKL_u0T_back/Best1_diag10_EKF/Learn_NODE.pkl',
                    true_meas_noise_var=0.,
                    process_noise_var=0.,
                    extra_noise=5e-5,
                    simu_solver='dopri5',
                    optim_solver_options={'method': 'dopri5'},
                    scalerX_method='meanY_obs',
                    trainer_options={'max_epochs': 2000, 'gpus': gpus,
                                     'progress_bar_refresh_rate': 1},
                    optim_method=torch.optim.Adam,
                    optim_lr=8e-3,
                    optim_minibatch_size=10,
                    optim_shuffle=True,
                    optim_options={'weight_decay': 1e-12},
                    optim_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    optim_scheduler_options={
                        'mode': 'min', 'factor': 0.7, 'patience': 3,
                        'threshold': 0.01, 'verbose': True},
                    optim_stopper=pl.callbacks.early_stopping.EarlyStopping(
                        monitor='val_loss', min_delta=0.001, patience=7,
                        verbose=False, mode='min'),
                    verbose=False,
                    monitor_experiment=False,
                    prior_mean=None,
                    prior_mean_deriv=None,
                    continuous_model=False,
                    plot_output=True)

    # Train whole model
    if train:
        # Add noise scaled to var of signal to make more robust
        if config.extra_noise:
            # https://stackoverflow.com/questions/64074698/how-to-add-5-gaussian-noise-to-the-signal-data
            true_meas_noise_var_Xtrain = \
                config.extra_noise * torch.mean(
                    torch.var(X_train, dim=1), dim=0)
            true_meas_noise_var_Utrain = \
                config.extra_noise * torch.mean(
                    torch.var(U_train, dim=1), dim=0)
            config.update(dict(
                true_meas_noise_var_Xtrain=true_meas_noise_var_Xtrain,
                true_meas_noise_var_Utrain=true_meas_noise_var_Utrain))
            distrib_Xtrain = \
                torch.distributions.multivariate_normal.MultivariateNormal(
                    torch.zeros(X_train.shape[-1]), torch.diag(
                        config.true_meas_noise_var_Xtrain))
            X_train += distrib_Xtrain.sample(X_train.shape[:-1])
            distrib_Utrain = \
                torch.distributions.multivariate_normal.MultivariateNormal(
                    torch.zeros(U_train.shape[-1]), torch.diag(
                        config.true_meas_noise_var_Utrain))
            U_train += distrib_Utrain.sample(U_train.shape[:-1])
        init_state = torch.unsqueeze(torch.cat((
            reshape_dim1(X_train[:, 0, 0]),
            torch.zeros_like(reshape_dim1(X_train[:, 0, 0])),
            torch.zeros_like(reshape_dim1(X_train[:, 0, 1])),
            reshape_dim1(X_train[:, 0, 1])), dim=-1), dim=1)
        init_control = torch.unsqueeze(U_train[:, 0], dim=1)

        # System params
        config.update(dict(
            discrete=False,
            dt_before_subsampling=dt_before_subsampling,
            no_control=False,
            init_control=init_control,
            init_state=init_state,
            init_state_x=init_state.clone(),
            init_state_estim=init_state.clone(),
            n=init_state.shape[2],
            observe_data=dim14_observe_data,
            observe_data_x=dim14_observe_data,
            prior_kwargs={'dt': config.dt,
                          'dt_before_subsampling': config.dt}))
        if dataset_on_GPU:
            X_train, U_train, init_state, init_control = \
                X_train.to(config.cuda_device), U_train.to(config.cuda_device), \
                init_state.to(config.cuda_device), \
                init_control.to(config.cuda_device)
        controller_list = []
        time = torch.arange(0., config.nb_samples * config.dt, config.dt,
                            device=U_train.device)
        for i in range(config.nb_difftraj):
            t_u = torch.cat((reshape_dim1(time), reshape_dim1(U_train[i])),
                            dim=1)
            control = Interpolate_func(
                x=t_u, t0=time[0], init_value=reshape_pt1(U_train[i, 0]))
            controller_list.append(control)
        controller = Control_from_list(controller_list, config.init_control)
        controller_args = [{} for i in range(len(controller_list))]
        config.update(dict(controller=controller,
                           controller_args=controller_args))

        # Create NN submodel of dynamics, then pack into Learn_NODE
        if config.no_control:
            nu_submodel = 0
            if config.init_state_obs_method == 'KKLu':
                logging.warning('KKLu without control: switching to '
                                'KKL_u0T to ignore the terms in u')
                config.init_state_obs_method = 'KKL_u0T'
        else:
            nu_submodel = config.init_control.shape[2]
        submodel = MLPn(num_hl=5, n_in=config.n + nu_submodel,
                        n_hl=100, n_out=config.n, activation=nn.SiLU())
        n_param, param = get_parameters(submodel, verbose=True)
        config.update(dict(
            n_param=n_param, nu=config.init_control.shape[2],
            constrain_u=[-300, 300], constrain_x=[-0.5, 0.5],
            grid_inf=-0.5, grid_sup=0.5))

        # Recognition model to estimate x0 jointly with the dynamics
        # First define some configs for the inputs of the recognition model
        if 'KKL_u0T' in config.init_state_obs_method:
            dz = X_train.shape[2] * (config.n + 1)
            W0 = 10
            # D, F = set_DF(W0, dz, X_train.shape[-1], config.setD_method)
            D = torch.tensor([[-1.8474e+01, -4.2819e+00, -4.2610e+00, -3.6734e+00, -8.0365e+00,
         -8.7800e+00, -1.0980e+01, -2.2556e+00,  1.0720e+01, -3.8795e-01],
        [-2.1512e+00, -2.6660e+01, -3.0679e+00, -3.9345e+00, -5.7781e+00,
         -4.1870e+00, -8.3926e+00, -6.0282e+00,  5.7000e+00, -6.4706e-01],
        [-1.3528e+00, -3.4638e+00, -3.4646e+01, -2.7246e+00, -5.8220e+00,
         -5.0300e+00, -1.2631e+01, -3.3517e+00,  6.7464e+00, -7.7474e-01],
        [-2.2409e+00, -9.4713e-01, -8.8194e-01, -4.5214e+01, -6.0542e+00,
         -5.2720e+00, -9.4771e+00, -1.6663e-01,  5.6003e+00, -4.9630e-01],
        [ 1.1417e+01,  4.8758e+00,  8.3621e+00,  3.9322e+00, -5.4376e+01,
         -4.9032e-01, -4.7565e+00,  5.2328e+00,  2.9748e+01,  1.1097e+01],
        [ 7.7382e+00,  1.1984e+01,  1.0785e+01,  9.1461e+00,  3.5487e+00,
         -6.0973e+01, -2.2682e+00,  1.4288e+01,  3.3318e+01,  1.2111e+01],
        [ 1.7837e+01,  1.2790e+01,  1.9590e+01,  1.8734e+01,  1.0185e+01,
          1.3191e+01, -8.1477e+01,  1.4725e+01,  2.8341e+01,  2.3872e+01],
        [ 1.2653e+00, -1.5949e+00, -1.5168e+00,  5.5937e-02, -4.6892e+00,
         -1.1371e+00,  1.1692e-01, -9.1007e+01,  5.5278e+00,  1.0514e-01],
        [-6.9920e+00, -6.3658e+00, -7.2090e+00, -9.0941e+00, -1.0793e+01,
         -7.8995e+00, -1.5955e+01, -6.1372e+00, -1.0304e+02, -8.6409e+00],
        [ 1.7628e+00,  1.4198e+00, -1.1623e-01,  8.8766e-02, -1.4570e+00,
         -6.5718e-01, -5.5938e+00,  1.5948e+00,  1.7220e+00, -1.0060e+02]])
            F = torch.ones(dz, X_train.shape[-1])
            z0 = torch.zeros(1, dz, device=X_train.device)
            z_config = {'D': D, 'F': F, 'init_state_estim': z0,
                        'Bessel_W0': W0, 'dz': dz}
            if 'optimD' in config.init_state_obs_method:
                z_config['init_D'] = D.clone()
            KKL = KKL(config.device, z_config)
            config.update(dict(z_config=z_config, init_state_KKL=KKL))
            print(f'D: {D}, eigvals: {torch.linalg.eigvals(D)}')
        elif 'KKLu' in config.init_state_obs_method:
            dw = 3  # to generate sinusoidal control with varying amplitude
            dz = (X_train.shape[2] + config.init_control.shape[1]) * (
                    config.n + dw + 1)
            W0 = 10
            D, F = set_DF(W0, dz, X_train.shape[-1] +
                          config.init_control.shape[-1], config.setD_method)
            z0 = torch.zeros(1, dz, device=X_train.device)
            z_config = {'D': D, 'F': F, 'init_state_estim': z0,
                        'controller_args': config.controller_args,
                        'Bessel_W0': W0, 'dz': dz}
            if 'optimD' in config.init_state_obs_method:
                z_config['init_D'] = D.clone()
            KKL = KKLu(config.device, z_config)
            config.update(dict(z_config=z_config, init_state_KKL=KKL))
            print(f'D: {D}, eigvals: {torch.linalg.eigvals(D)}')

        # Define the inputs of the recognition model
        diff_init_state_obs = make_diff_init_state_obs(
            X_train, U_train, config.init_state_x, config.t_eval, config)

        # Define the actual recognition model (same for all init)
        if config.init_state_obs_method == 'fixed_recognition_model':
            init_state_model = WDC_simple_recog4(config.n, config.dt)
        else:
            init_state_model = MLPn(num_hl=5,
                                    n_in=torch.numel(diff_init_state_obs[0]),
                                    n_hl=50, n_out=config.n,
                                    activation=nn.SiLU())
        _, X_train, U_train, config.t_eval = update_config_init_state_obs(
            diff_init_state_obs, init_state_model, X_train, X_train, U_train,
            config.t_eval, config)
        # Create Learn_NODE object
        NODE = Learn_NODE_difftraj_exp_datasets(
            X_train, U_train, submodel, config,
            sensitivity=config.sensitivity, dataset_on_GPU=dataset_on_GPU)
        # Train
        # To see logger in tensorboard, type in terminal then click on weblink:
        # tensorboard --logdir=../Figures/Continuous/Wandercraft_id/MLPn_noisy_inputs/ --port=8080
        logger = TensorBoardLogger(save_dir=NODE.results_folder + '/tb_logs')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              save_weights_only=True)
        if config.sensitivity == 'forward':
            traj_estim = NODE.train_forward_sensitivity()
        else:
            if config.optim_stopper:
                trainer = pl.Trainer(
                    callbacks=[config.optim_stopper, checkpoint_callback],
                    **config.trainer_options, logger=logger,
                    log_every_n_steps=1, check_val_every_n_epoch=2)
            else:
                trainer = pl.Trainer(
                    callbacks=[checkpoint_callback], **config.trainer_options,
                    logger=logger, log_every_n_steps=1,
                    check_val_every_n_epoch=2)
            trainer.fit(NODE)
        # Save and update estim x0
        NODE.save_model(checkpoint_path=checkpoint_callback.best_model_path)
    else:
        # Recover everything from pickle
        NODE = pkl.load(open(config.NODE_file, 'rb'))
        NODE.results_folder = config.NODE_file.rsplit('/', 1)[0]
        config = NODE.config

    if test:

        # Compare with models identified at WDC
        lin_model = WDC_single_deformation_model()
        verbose = False
        save = True
        vars = ['u_gaussian50', 'theta_gaussian50', 'w_gaussian50']
        dt = NODE.dt_before_subsampling
        n = len(NODE.rollout_list)
        # Compare rollouts
        RMSE_lin_train = plot_NODE_lin_rollouts(n, NODE, NODE.rollout_list,
                                                lin_model=lin_model,
                                                verbose=verbose, save=save)
        # Plot few training trajs
        os.makedirs(os.path.join(NODE.results_folder, 'Training_trajs'),
                    exist_ok=True)
        control_idx = NODE.train_val_idx[:n]
        xtraj_estim = NODE.NODE_model.forward_traj(
            NODE.init_state_estim[:n], NODE.controller[control_idx], NODE.t0,
            NODE.t_eval, NODE.init_control)
        y_observed = NODE.X_train[:n]
        y_pred = NODE.observe_data_x(xtraj_estim)
        print(xtraj_estim.shape, y_pred.shape, y_observed.shape,
              len(NODE.t_eval))
        import matplotlib.pyplot as plt

        for j in range(n):
            for i in range(y_pred.shape[2]):
                name = 'Training_trajs/y_pred' + str(j) + \
                       str(i) + '.pdf'
                plt.plot(y_observed[j, :, i], label='True')
                plt.plot(y_pred.detach()[j, :, i], label='Estimated')
                plt.title('Output')
                plt.legend()
                plt.savefig(os.path.join(NODE.results_folder, name),
                            bbox_inches='tight')
                plt.clf()
                plt.close('all')

        # Read test data
        vars = dict.fromkeys(['u', 'theta', 'w'])
        dt = 1e-3
        dt_before_subsampling = 1e-3
        data_folder = 'Data/preprocessing/' \
                      '6-30-2difftraj_0.001subsampling_2000samples_filter_gaussian50_butteru/'
        for key in vars.keys():
            data_file = data_folder + 'data_' + key + '.pkl'
            if colab:
                with open(data_file, "rb") as fh:
                    data = pickle.load(fh)
                    df = pd.DataFrame(data)
            else:
                df = pd.read_pickle(data_file).T
            vars[key] = torch.unsqueeze(torch.as_tensor(df.values), dim=-1)
        nb_samples = df.shape[1]
        t0 = 0.
        tf = t0 + (nb_samples - 1) * dt
        U_test = vars['u']
        X_test = torch.cat((vars['theta'], vars['w']), dim=-1)
        nb_rollouts = len(X_test)
        print(X_test.shape, U_test.shape, dt, tf)
        # Make init state obs for X_test, U_test
        init_state_x = torch.unsqueeze(torch.cat((
            reshape_dim1(X_test[:, 0, 0]),
            torch.zeros_like(reshape_dim1(X_test[:, 0, 0])),
            torch.zeros_like(reshape_dim1(X_test[:, 0, 1])),
            reshape_dim1(X_test[:, 0, 1])), dim=-1), dim=1)
        diff_init_state_obs = make_diff_init_state_obs(
            X_test, U_test, init_state_x, config.t_eval, config)
        _, X_test, U_test, config.t_eval = update_config_init_state_obs(
            diff_init_state_obs, NODE.init_state_model, X_test, X_test,
            U_test, config.t_eval, config)
        # Make test rollout data
        rollout_list = []
        i = 0
        while i < nb_rollouts:
            if NODE.init_state_model:
                # For ground_truth_approx, init_state in rollout_list
                # contains the inputs to the recognition model for Xtest
                # since it is known anyway, so that it can be used in NODE
                # rollouts directly
                init_state = reshape_pt1(diff_init_state_obs[i, 0])
            else:
                init_state = reshape_pt1(X_test[i, 0])
            control_traj = reshape_dim1(U_test[i])
            true_mean = reshape_dim1(X_test[i])
            rollout_list.append([init_state, control_traj, true_mean])
            i += 1
        NODE.step += 1
        NODE.evaluate_rollouts(rollout_list, plots=True)
        # # Compare with WDC lin model
        lin_model = WDC_single_deformation_model()
        verbose = False
        save = True
        RMSE_lin_test = plot_NODE_lin_rollouts(nb_rollouts, NODE, rollout_list,
                                               lin_model=lin_model,
                                               verbose=verbose, save=save)
        specs_file = os.path.join(NODE.results_folder, 'Specifications.txt')
        with open(specs_file, 'a') as f:
            print('\n', file=f)
            print('\n', file=f)
            print('\n', file=f)
            print(f'Test rollouts: {data_folder}', file=f)
            print(f'Number of test rollouts: {nb_rollouts}', file=f)
            print(f'Prior kwargs for rollouts (including EKF): '
                  f'{NODE.config.prior_kwargs}', file=f)
            for key, val in NODE.specs.items():
                if 'rollout' in key:
                    print(key, ': ', val, file=f)
            print(f'RMSE of linear model on test rollouts used during training:'
                  f' {RMSE_lin_train}', file=f)
            print(f'RMSE of linear model on open-loop test rollouts:'
                  f' {RMSE_lin_test}', file=f)

    end = timep.time()
    logging.info('Script ran in ' + str(end - start))
    stop_log()
