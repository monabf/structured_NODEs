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

from eval_WDC_data import plot_NODE_lin_rollouts
from simulation.controllers import Control_from_list
from simulation.observers import dim14_observe_data
from simulation.observer_functions import KKL, KKLu
from wandercraft_id_NN_models import MLPn, WDC_simple_recog4, \
    WDC_single_deformation_model, WDC_x1dotx2_submodel, RNNn, MLPn_xin
from NN_for_ODEs.learn_neural_ODE_several_exp_datasets import \
    Learn_NODE_difftraj_exp_datasets
from NN_for_ODEs.NODE_utils import make_diff_init_state_obs, \
    update_config_init_state_obs, set_DF
from utils.utils import start_log, stop_log, reshape_pt1, reshape_dim1, \
    Interpolate_func
from utils.pytorch_utils import get_parameters
from utils.config import Config

sb.set_style("whitegrid")

# Script to learn physical dynamics (backPhi) from canonical observations
# using a NN model, on a harmonic oscillator with unknown frequency

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

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # TODO no GPU
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'  # more explicit cuda errors

if __name__ == '__main__':
    start_log()
    start = timep.time()
    train = True  # TODO
    test = True
    colab = False
    dataset_on_GPU = False
    if torch.cuda.is_available():
        gpus = -1
        # strategy = pl.plugins.DDPPlugin(find_unused_parameters=False)
    else:
        gpus = 0
        # strategy = None
    print(gpus, torch.cuda.is_available())

    # Read data
    vars = dict.fromkeys(['u', 'theta', 'w'])
    dt = 1e-3  # TODO
    dt_before_subsampling = 1e-3
    data_folder = 'Data//Wandercraft_id/preprocessing/' \
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
    # nb_difftraj = df.shape[0]
    nb_samples = df.shape[1]
    t0 = 0.
    tf = t0 + (nb_samples - 1) * dt
    U_train = vars['u']
    X_train = torch.cat((vars['theta'], vars['w']), dim=-1)
    # Shorten nb of difftraj: get rid of some data to shorten computation
    nb_difftraj_max = int(np.min([600, len(X_train)]))  # TODO
    random_idx = torch.randperm(len(X_train))[:nb_difftraj_max]
    X_train = X_train[random_idx]
    U_train = U_train[random_idx]
    nb_difftraj = len(X_train)
    print(X_train.shape, U_train.shape, dt, tf)

    # General params
    config = Config(system='Continuous/Wandercraft_id/MLPn_noisy_inputs/'
                           'x1dotx2',
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
                    init_state_obs_T=100,  # TODO
                    data_folder=data_folder,
                    NODE_file='../Figures/Continuous/Wandercraft_id/'
                              'MLPn_noisy_inputs/x1dotx2/7-29-2difftraj_0.001subsampling_200samples_filter_gaussian50_butteru/'
                              'KKLu_back/Ok1_optimD1_autograd_200samples_noise0'
                              '.0_NODE_difftraj265_Adam0.008_KKLu_back100/Learn_NODE.pkl',
                    true_meas_noise_var=0.,
                    process_noise_var=0.,
                    extra_noise=5e-5,
                    simu_solver='dopri5',
                    optim_solver_options={'method': 'dopri5'},  # for adjoint
                    # 'rtol': 1e-8, 'atol': 1e-8},
                    scalerX_method='meanY_obs',
                    trainer_options={'max_epochs': 2000, 'gpus': gpus,
                                     'progress_bar_refresh_rate': 1},
                    optim_method=torch.optim.Adam,
                    optim_lr=8e-3,
                    optim_minibatch_size=10,  # TODO
                    optim_shuffle=True,
                    optim_options={'weight_decay': 1e-12},
                    # l1_reg=1.,
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
    if dataset_on_GPU:  # TODO
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
    f = MLPn(num_hl=5, n_in=config.n + nu_submodel, n_hl=100, n_out=2,
             activation=nn.SiLU())
    submodel = WDC_x1dotx2_submodel(f, config)  # TODO
    n_param, param = get_parameters(submodel, verbose=True)
    config.update(dict(
        n_param=n_param, nu=config.init_control.shape[2],
        constrain_u=[-300, 300], constrain_x=[-0.5, 0.5],
        grid_inf=-0.5, grid_sup=0.5))

    # Recognition model to estimate x0 jointly with the dynamics
    # First define some configs for the inputs of the recognition model
    if 'KKL_u0T' in config.init_state_obs_method:
        dz = X_train.shape[2] * (config.n + 1)
        # dz += 40
        W0 = 10  # 2 * np.pi * 10
        # D, F = set_DF(W0, dz, X_train.shape[-1], config.setD_method)
        D = torch.tensor(
            [[-3.7186e+01, 1.6680e+01, -7.3743e+00, -5.7511e+00, 1.3829e+00,
              -6.2140e+00, -1.2042e+01, -4.9374e+00, -3.6362e+00,
              1.7192e+00],
             [-3.0775e+00, -4.2098e+01, -6.2486e+00, -1.0527e+01,
              -8.1801e+00,
              -1.0426e+01, -1.0999e+01, -6.1369e+00, -2.8191e+00,
              -1.1575e+01],
             [-7.3258e+00, -4.8695e+00, -3.8232e+01, -6.1654e+00,
              -6.3253e+00,
              -8.7638e+00, -1.0839e+01, -4.9910e+00, -1.4824e+00,
              -6.2062e+00],
             [5.5861e+00, 7.6686e+00, 8.0404e+00, -4.0586e+01, 1.0563e+01,
              4.5876e+00, 5.6495e+00, 8.6370e+00, 9.1084e+00, 1.1907e+01],
             [-5.8743e+00, -7.1924e+00, -5.1019e+00, -3.4136e+00,
              -5.8130e+01,
              -3.9638e+00, -3.5521e+00, 1.0272e-01, -4.2360e+00,
              -6.3878e+00],
             [1.0270e+01, 1.5673e+01, 1.4131e+01, 3.9136e+00, 1.1429e+01,
              -5.7596e+01, 5.7571e+00, 1.0754e+01, 7.2754e+00, 1.1584e+01],
             [1.5175e+00, 2.2040e+01, 8.4386e+00, -9.4462e-01, 1.6971e+01,
              -1.2891e+00, -7.8455e+01, 9.0068e+00, 1.8835e+01, 1.1461e+01],
             [4.8975e+00, 2.7655e+00, 5.6610e+00, -5.5993e+00, 4.8439e+00,
              -3.5150e+00, -1.9509e+00, -8.2745e+01, 4.6534e-01,
              4.1363e+00],
             [-3.5310e-01, 2.4500e-01, 6.9539e-01, -6.2407e+00, 7.3483e-01,
              -4.9217e+00, -6.3861e+00, -6.8566e+00, -1.1423e+02,
              -1.0697e+01],
             [-2.7481e+00, -3.3673e+00, -1.6452e+00, -6.1161e+00,
              -6.2164e+00,
              -5.5431e+00, -5.6969e+00, -2.2723e+00, -7.1125e+00,
              -1.1765e+02]])
        F = torch.ones(dz, X_train.shape[-1])
        z0 = torch.zeros(1, dz, device=X_train.device)
        z_config = {'D': D, 'F': F, 'init_state_estim': z0,
                    'Bessel_W0': W0, 'dz': dz}
        if 'optimD' in config.init_state_obs_method:
            z_config['init_D'] = D.clone()
        KKL = KKL(config.device, z_config)
        config.update(dict(z_config=z_config, init_state_KKL=KKL))
        print(D, torch.linalg.eigvals(D))
    elif 'KKLu' in config.init_state_obs_method:
        dw = 3  # to generate sinusoidal control with varying amplitude
        # dz = (X_train.shape[2] + config.init_control.shape[1]) * (
        #         config.n + dw + 1)
        dz = 50
        W0 = 2
        # D, F = set_DF(W0, dz, X_train.shape[-1] +
        #               config.init_control.shape[-1], config.setD_method)
        data = pd.read_csv('Data//Wandercraft_id/KKLu_back_D1_x1dotx2.csv',
                           sep=',', header=None)
        D = torch.from_numpy(data.drop(data.columns[0], axis=1).values)
        F = torch.ones(dz, X_train.shape[-1] +
                       config.init_control.shape[-1])
        z0 = torch.zeros(1, dz, device=X_train.device)
        z_config = {'D': D, 'F': F, 'init_state_estim': z0,
                    'controller_args': config.controller_args,
                    'Bessel_W0': W0, 'dz': dz}
        if 'optimD' in config.init_state_obs_method:
            z_config['init_D'] = D.clone()
        KKL = KKLu(config.device, z_config)
        config.update(dict(z_config=z_config, init_state_KKL=KKL))
        print(D, torch.linalg.eigvals(D))

    # Define the inputs of the recognition model
    diff_init_state_obs = make_diff_init_state_obs(
        X_train, U_train, config.init_state_x, config.t_eval, config)

    # Define the actual recognition model (same for all init)
    if config.init_state_obs_method == 'fixed_recognition_model':
        init_state_model = WDC_simple_recog4(config.n, config.dt)
    elif config.init_state_obs_method.startswith('y0T_u0T_RNN_outNN'):
        dz = 110  # same as KKL_u0T
        init_state_model1 = RNNn(
            n_in=X_train.shape[-1] + config.nu,  # TODO
            n_out=dz, n_hl=1, RNN=torch.nn.GRU)
        init_state_model = MLPn_xin(
            n_in=dz, num_hl=5, n_hl=50, n_out=config.n,
            model_in=init_state_model1, activation=nn.SiLU())
    else:
        init_state_model = MLPn(num_hl=5,
                                n_in=torch.numel(diff_init_state_obs[0]),
                                n_hl=50, n_out=config.n,
                                activation=nn.SiLU())  # TODO

    if train:
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
        # # Recover from checkpoint: first need to recreate X_train, submodel...
        # checkpoint_path = '../Figures/Continuous/Wandercraft_id/MLPn_noisy_inputs/x1dotx2/' \
        #                   '10-11difftraj_0.005subsampling_40samples_filter_gaussian50/KKL_u0T/0.02923947397964699_autograd_40samples_noise0.0_NODE_difftraj24_Adam0.001_KKL_u0T20/tb_logs/default/version_0/checkpoints/epoch=289-step=289.ckpt'
        # checkpoint_model = torch.load(checkpoint_path)
        # print(checkpoint_model)
        # NODE = Learn_NODE_difftraj_exp_datasets(
        #     X_train, U_train, submodel, config,
        #     sensitivity=config.sensitivity, dataset_on_GPU=dataset_on_GPU)
        # NODE.load_state_dict(checkpoint_model['state_dict'])
        # NODE.results_folder = checkpoint_path.rsplit('/tb_logs', 1)[0]
        # config = NODE.config
        # NODE.save_model()

        # Or recover everything from pickle
        NODE = pkl.load(open(config.NODE_file, 'rb'))
        NODE.results_folder = config.NODE_file.rsplit('/', 1)[0]
        config = NODE.config
        # NODE.evaluate_rollouts(NODE.rollout_list)

    if test:
        # Compare with models identified at WDC
        lin_model = WDC_single_deformation_model()
        verbose = False
        save = True  # TODO
        vars = ['u_gaussian50', 'theta_gaussian50', 'w_gaussian50']
        dt = NODE.dt_before_subsampling
        n = len(NODE.rollout_list)
        # # Save predictions over full data trajectories
        # # plot_NODE_full_rollouts(NODE, dt, vars, verbose=verbose, save=save)
        # # Compare rollouts: problem with initial condition, makes little sense!
        RMSE_lin_train = plot_NODE_lin_rollouts(n, NODE, NODE.rollout_list,
                                                lin_model=None,
                                                verbose=verbose, save=save)
        # # # Compare Bode diagrams: same procedure as for experimental data
        # # # plot_NODE_lin_Bode(NODE, dt, lin_model, vars=vars, verbose=verbose,
        # # #                    save=save, two_deformation=False)
        # Plot few training trajs  # TODO
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
        data_folder = 'Data//Wandercraft_id/preprocessing/' \
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
        # nb_difftraj = df.shape[0]
        nb_samples = df.shape[1]
        t0 = 0.
        tf = t0 + (nb_samples - 1) * dt
        U_test = vars['u']
        X_test = torch.cat((vars['theta'], vars['w']), dim=-1)
        # # Shorten nb of difftraj: get rid of some data to shorten computation
        # nb_difftraj_max = int(np.min([3, len(X_test)]))  # TODO
        # # random_idx = torch.randperm(len(X_test))[:nb_difftraj_max]
        # random_idx = torch.tensor([0])
        # X_test = X_test[random_idx, 5000:15000]
        # U_test = U_test[random_idx, 5000:15000]
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
        NODE.evaluate_rollouts(rollout_list, plots=True)  # TODO
        # Compare with WDC lin model
        lin_model = WDC_single_deformation_model()
        verbose = False
        save = True
        RMSE_lin_test = plot_NODE_lin_rollouts(nb_rollouts, NODE, rollout_list,
                                               lin_model=lin_model,
                                               verbose=verbose, save=save)
        # Also run these test rollouts with EKF
        C = torch.tensor([[1, 0, 0., 0],
                          [0, 0, 0, 1.]])
        meas = torch.tensor([1e-3, 1e-1])
        process = torch.tensor([1e3, 1e3, 1e-1, 1e-1])
        init = torch.tensor([1e2, 1e2, 1e2, 1e2])
        # C = torch.tensor([[1, 0, 0., 0]])  # TODO estimate x4 from y=x1
        # meas = torch.tensor([1e-4])
        # process = torch.tensor([1e-1, 1e-1, 1e-2, 1e-2])
        # init = torch.tensor([1e-1, 1e-1, 1e-1, 1e-1])
        # C = torch.tensor([[0, 0, 0, 1.]])  # TODO estimate x4 from y=x4
        # meas = torch.tensor([5e-5])
        # process = torch.tensor([1e-3, 1e-3, 1e-1, 1e-1])
        # init = torch.tensor([1e-3, 1e-2, 1e-1, 1e-1])
        from simulation.observers import dim1_observe_data, dimlast_observe_data
        NODE.config.update(
            {'prior_kwargs': {'n': NODE.n, 'observation_matrix': C,
                              'EKF_meas_covar': meas * torch.eye(C.shape[0]),
                              'EKF_process_covar': process * torch.eye(NODE.n),
                              'EKF_init_covar': init * torch.eye(NODE.n),
                              'EKF_added_meas_noise_var': 1e-9, }})
                              # 'EKF_observe_data': dim1_observe_data}})
                              # 'EKF_observe_data': dimlast_observe_data}})
        NODE.evaluate_EKF_rollouts(rollout_list, plots=True)  # TODO
        # Compare with WDC lin model
        lin_model = WDC_single_deformation_model()
        verbose = False
        save = True
        vars = ['u_gaussian50', 'theta_gaussian50', 'w_gaussian50']
        dt = NODE.dt_before_subsampling
        # Compare EKF rollouts
        RMSE_lin_EKF = plot_NODE_lin_rollouts(
            nb_rollouts, NODE, rollout_list, rollouts_title='EKF_rollouts',
            lin_model=lin_model, type='EKF', verbose=verbose, save=save)
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
                  f' {RMSE_lin_train}', file=f)  # TODO
            print(f'RMSE of linear model on open-loop test rollouts:'
                  f' {RMSE_lin_test}', file=f)
            print(f'RMSE of linear model on EKF test rollouts:'
                  f' {RMSE_lin_EKF}', file=f)

    end = timep.time()
    logging.info('Script ran in ' + str(end - start))
    stop_log()
