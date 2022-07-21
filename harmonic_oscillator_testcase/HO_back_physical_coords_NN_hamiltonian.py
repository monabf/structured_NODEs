import logging
# Pickle cannot be used with GPyTorch:
# https://github.com/cloudpipe/cloudpickle/issues/381
# https://stackoverflow.com/questions/23582489/python-pickle-protocol-choice
import os
import sys
import time as timep

import dill as pkl
import numpy as np
import pytorch_lightning as pl
import scipy.signal
import seaborn as sb
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from smt.sampling_methods import LHS

sys.path.append('.')

from simulation.controllers import Control_from_dict
from simulation.dynamics_functions import Harmonic_oscillator
from simulation.simulation_functions import simulate_dynamics
from simulation.observers import dim1_observe_data
from simulation.observer_functions import KKL, KKLu
from NN_for_ODEs.learn_neural_ODE import Learn_NODE_difftraj
from NN_for_ODEs.NODE_utils import make_diff_init_state_obs, \
    update_config_init_state_obs, set_DF
from HO_NN_models import MLP2, MLP2_xin, RNNn
from utils.utils import start_log, stop_log, reshape_dim1
from utils.config import Config

# To avoid Type 3 fonts for submission https://tex.stackexchange.com/questions/18687/how-to-generate-pdf-without-any-type3-fonts
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}\usepackage{cmbright}')
plt.rc('font', family='serif')

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
    train = True
    # torch.manual_seed(0)
    if torch.cuda.is_available():
        gpus = -1
    else:
        gpus = 0

    # General params
    config = Config(system='Continuous/Harmonic_Oscillator/MLP2_noisy_inputs'
                           '/Hamiltonian',
                    sensitivity='autograd',
                    intloss=None,
                    order=1,
                    nb_samples=50,  # TODO
                    nb_difftraj=72,
                    t0_span=0,
                    tf_span=3,
                    t0=0,
                    tf=3,
                    init_state_obs_method=str(sys.argv[2]),
                    setD_method='id',
                    init_state_obs_T=20,
                    NODE_file='../Figures/Continuous/Harmonic_Oscillator/Discrete_model/'
                              'Back_physical/MLP1_noisy_inputs/'
                              'Test2_100samples_noise0.001_MLP1_Adam0.05/Learn_NODE.pkl',
                    true_meas_noise_var=1e-4,
                    process_noise_var=0,
                    simu_solver='dopri5',
                    optim_solver_options={'method': 'dopri5'},  # for adjoint
                    # 'rtol': 1e-8, 'atol': 1e-8},
                    trainer_options={'max_epochs': 2500, 'gpus': gpus,
                                     'progress_bar_refresh_rate': 1},
                    optim_method=torch.optim.Adam,
                    optim_lr=5e-3,
                    optim_minibatch_size=100,
                    optim_shuffle=False,
                    optim_options={'weight_decay': 1e-4},
                    optim_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    optim_scheduler_options={
                        'mode': 'min', 'factor': 0.2, 'patience': 20,
                        'threshold': 0.01, 'verbose': True},
                    optim_stopper=pl.callbacks.early_stopping.EarlyStopping(
                        monitor='val_loss', min_delta=0.001, patience=50,
                        verbose=False, mode='min'),
                    # l1_reg=0.001,
                    nb_rollouts=100,
                    rollout_length=500,
                    rollout_controller={'null_controller': 100},
                    rollout_controller_args={'controller_args': [{}] * 100},
                    max_rollout_value=5.,
                    verbose=False,
                    monitor_experiment=True,
                    prior_mean=None,
                    prior_mean_deriv=None,
                    plot_output=True)
    if 'Continuous_model' in config.system:
        config.update(dict(continuous_model=True))
    else:
        config.update(dict(continuous_model=False))

    # Train whole model
    if train:
        # System params
        if 'Continuous/Harmonic_Oscillator' in config.system:
            config.update(dict(
                discrete=False,
                pulse=1,#np.random.uniform(0.5, 2.),
                dt=config.dt,
                dt_before_subsampling=0.001,
                dynamics_function=Harmonic_oscillator,
                no_control=True,
                controller_dict={'null_controller': config.nb_difftraj},
                controller_args=[{}] * config.nb_difftraj,
                init_control=torch.zeros((config.nb_difftraj, 1, 1)),
                observe_data=dim1_observe_data,
                observe_data_x=dim1_observe_data,
                # observe_data=lambda x: x,  # TODO
                # observe_data_x=lambda x: x,
                prior_kwargs={'dt': config.dt,
                              'dt_before_subsampling': config.dt}))
            dynamics = config.dynamics_function(config.cuda_device, config)
            config.update(dict(
                dynamics=dynamics,
                dynamics_x=dynamics.harmonic_oscillator_dynamics,
                true_dynamics=dynamics.harmonic_oscillator_dynamics))
            print(config.pulse)

            # Set initial states of x, xi for all difftrajs with LHS
            # https://smt.readthedocs.io/en/latest/_src_docs/sampling_methods/lhs.html
            xlimits = np.array([[-1., 1.], [-1., 1.]])
            sampling = LHS(xlimits=xlimits)
            if config.nb_difftraj == 1:
                init_state_x = torch.unsqueeze(torch.rand((1, 2)) * 2 - 1, 1)
            else:
                init_state_x = torch.unsqueeze(torch.as_tensor(sampling(
                    config.nb_difftraj)), 1)
            init_state = init_state_x.clone()
            init_state_estim = init_state_x.clone()  # init state model after
            config.update(dict(xlimits=xlimits, init_state_x=init_state_x,
                               init_state=init_state,
                               init_state_estim=init_state_estim,
                               n=init_state_x.shape[2]))
            if config.no_control:
                nu_submodel = 0
                if config.init_state_obs_method == 'KKLu':
                    logging.warning('KKLu without control: switching to '
                                    'KKL_u0T to ignore the terms in u')
                    config.init_state_obs_method = 'KKL_u0T'
            else:
                nu_submodel = config.init_control.shape[2]


            # Create NN submodel of dynamics, then pack into Learn_NODE
            class Submodel(nn.Module):
                def __init__(self, Hamiltonian):
                    super(Submodel, self).__init__()
                    self.n = config.n + nu_submodel
                    M = torch.eye(self.n)
                    self.M = torch.cat([M[self.n // 2:], -M[:self.n // 2]])
                    self.H = Hamiltonian

                def set_scalers(self, scaler_X=None, scaler_Y=None):
                    self.scaler_X = scaler_X
                    self.scaler_Y = scaler_Y

                def forward(self, x):
                    with torch.set_grad_enabled(True):
                        # TODO compute graph only once? But changes?
                        x = reshape_dim1(x.requires_grad_(True))
                        gradH = torch.autograd.grad(
                            self.H(x).sum(), x, allow_unused=False,
                            create_graph=True)[0]  # sum grad over difftraj
                    xdot = torch.matmul(gradH, self.M.t())
                    return xdot


            H = MLP2(n_in=config.n + nu_submodel, n_h1=25, n_h2=25, n_out=1,
                     activation=nn.SiLU)  # TODO
            config.update(dict(Hamiltonian=H))
            submodel = Submodel(H)
            # n_param, param = get_parameters(submodel, verbose=True)
            n_param = 0
            config.update(dict(
                n_param=n_param, reg_coef=1., nu=config.init_control.shape[1],
                constrain_u=[-1., 1.],
                constrain_x=[],
                grid_inf=[-1, -1],  # TODO
                grid_sup=[1, 1]))
            controller = Control_from_dict(config.controller_dict,
                                           config.init_control,
                                           config.constrain_u)
            config.update(controller=controller)
        else:
            raise Exception('Unknown system')

        # Simulate system in x for all difftrajs
        diff_xtraj_true = torch.zeros(0, len(config.t_eval), config.n)
        diff_utraj = torch.zeros(0, len(config.t_eval), config.nu)
        diff_y_observed_true = torch.zeros(
            0, len(config.t_eval),
            config.observe_data(init_state_x[0]).shape[1])
        for i in range(len(config.init_state_x)):
            xtraj_true, utraj, t_utraj = \
                simulate_dynamics(t_eval=config.t_eval, t0=config.t0,
                                  dt=config.dt,
                                  init_control=config.init_control,
                                  init_state=config.init_state_x[i],
                                  dynamics=config.dynamics_x,
                                  controller=config.controller[i],
                                  process_noise_var=config.process_noise_var,
                                  method=config.simu_solver,
                                  dyn_config=config,
                                  discrete=config.discrete,
                                  verbose=config.verbose)
            y_observed_true = config.observe_data(xtraj_true)
            if config.true_meas_noise_var != 0:
                y_observed_true += torch.normal(0, np.sqrt(
                    config.true_meas_noise_var), size=y_observed_true.shape)
            diff_xtraj_true = torch.cat((diff_xtraj_true, torch.unsqueeze(
                xtraj_true, 0)), dim=0)
            diff_utraj = torch.cat((
                diff_utraj, torch.unsqueeze(utraj, 0)), dim=0)
            diff_y_observed_true = torch.cat((
                diff_y_observed_true,
                torch.unsqueeze(y_observed_true, 0)), dim=0)

        # Recognition model to estimate x0 jointly with the dynamics
        # First define some configs for the inputs of the recognition model
        if 'KKL_u0T' in config.init_state_obs_method:
            dz = y_observed_true.shape[1] * (config.n + 1)
            # dz += 30
            W0 = 1#2 * np.pi * 1  # TODO
            # D, F = set_DF(W0, dz, y_observed_true.shape[-1], config.setD_method)
            if 'back' in config.init_state_obs_method:
                D = torch.tensor([[-1.7172e+00, -4.9691e+00, 3.2359e-05],
                                  [4.2332e+00, -1.7829e+00, 1.5964e+00],
                                  [2.0029e+00, -6.6328e-02, -6.4910e+00]])
                F = torch.ones(dz, y_observed_true.shape[1])
            else:
                D = torch.tensor([[-2.4558, -4.2001, 0.7883],
                                  [5.6656, -4.0468, -0.8448],
                                  [2.5478, -0.2983, -6.8766]])
                F = torch.ones(dz, y_observed_true.shape[1])
            z0 = torch.zeros(1, dz)
            z_config = {'D': D, 'F': F, 'init_state_estim': z0,
                        'Db': D, 'Fb': F, 'Bessel_W0': W0, 'dz': dz}
            if 'optimD' in config.init_state_obs_method:
                z_config['init_D'] = D.clone()
            KKL = KKL(config.device, z_config)
            config.update(dict(z_config=z_config,
                               init_state_KKL=KKL))
            print(D, torch.linalg.eigvals(D))
        elif 'KKLu' in config.init_state_obs_method:
            dw = 1  # to generate constant control
            dz = (y_observed_true.shape[1] +
                  config.init_control.shape[1]) * (config.n + dw + 1)
            W0 = 2 * np.pi * 1
            D, F = set_DF(W0, dz, y_observed_true.shape[-1] +
                          config.init_control.shape[-1], config.setD_method)
            z0 = torch.zeros(1, dz)
            z_config = {'D': D, 'F': F, 'init_state_estim': z0,
                        'controller_args': config.controller_args,
                        'Db': D, 'Fb': F, 'Bessel_W0': W0, 'dz': dz}
            if 'optimD' in config.init_state_obs_method:
                z_config['init_D'] = D.clone()
            KKL = KKLu(config.device, z_config)
            config.update(dict(z_config=z_config,
                               init_state_KKL=KKL))
            print(D, torch.linalg.eigvals(D))

        # Define the inputs of the recognition model
        diff_init_state_obs = make_diff_init_state_obs(
            diff_y_observed_true, diff_utraj, init_state_x, config.t_eval,
            config)

        # Define the actual recognition model (same for all init)
        if config.init_state_obs_method.startswith('y0T_u0T_RNN_outNN'):
            dz = y_observed_true.shape[1] * (config.n + 1)  # same as KKLu
            init_state_model1 = RNNn(
                n_in=diff_y_observed_true.shape[-1],  # TODO
                n_out=dz, n_hl=1, RNN=torch.nn.GRU)
            init_state_model = MLP2_xin(
                n_in=dz, n_h1=10, n_h2=10, n_out=config.n,
                model_in=init_state_model1, activation=nn.SiLU)
        elif config.init_state_obs_method.startswith('y0T_u0T_RNN'):
            init_state_model = RNNn(
                n_in=diff_y_observed_true.shape[-1],  # TODO
                n_out=config.n, n_hl=1, RNN=torch.nn.GRU)
        else:
            init_state_model = MLP2(n_in=torch.numel(diff_init_state_obs[0]),
                                    n_h1=10, n_h2=10, n_out=config.n,
                                    activation=nn.SiLU)  # TODO
        diff_xtraj_true, diff_y_observed_true, diff_utraj, config.t_eval = \
            update_config_init_state_obs(
                diff_init_state_obs, init_state_model, diff_xtraj_true,
                diff_y_observed_true, diff_utraj, config.t_eval, config)

        # Create Learn_NODE object
        NODE = Learn_NODE_difftraj(
            diff_y_observed_true, diff_utraj, submodel, config,
            sensitivity=config.sensitivity)
        # Train
        # To see logger in tensorboard, type in terminal then click on weblink:
        # tensorboard --logdir=../Figures/Continuous/Harmonic_Oscillator/Discrete_model/Back_physical/MLP2_noisy_inputs/ --port=8080
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
                    log_every_n_steps=1, check_val_every_n_epoch=10)
            else:
                trainer = pl.Trainer(
                    callbacks=[checkpoint_callback], **config.trainer_options,
                    logger=logger, log_every_n_steps=1,
                    check_val_every_n_epoch=10)
            trainer.fit(NODE)
        # Save and update estim x0
        NODE.save_model(checkpoint_path=checkpoint_callback.best_model_path)
        # Plot additional results
        xtraj_estim = NODE.NODE_model.forward_traj(
            NODE.init_state_estim, config.controller, config.t0,
            config.t_eval, config.init_control)
        if NODE.partial_obs:
            for j in range(config.nb_difftraj):
                for i in range(xtraj_estim.shape[2]):
                    name = 'Training_trajs/xtraj_estim' + str(j) + \
                           str(i) + '.pdf'
                    plt.plot(diff_xtraj_true[j, :, i], label='True')
                    plt.plot(xtraj_estim.detach()[j, :, i],
                             label='Estimated')
                    plt.title('State trajectory')
                    plt.legend()
                    plt.savefig(os.path.join(NODE.results_folder, name),
                                bbox_inches='tight')
                    plt.clf()
                    plt.close('all')
    else:
        NODE = pkl.load(open(config.NODE_file, 'rb'))
        NODE.results_folder = config.NODE_file.rsplit('/', 1)[0]
        config = NODE.config

    # # Evaluate model
    # utraj = config.controller(config.t_eval, config, config.t0,
    #                           config.init_control)
    # x0 = torch.tensor([[0.5, 1.5]])
    # xtraj_true = dynamics_traj(x0=x0,
    #                            u=config.controller,
    #                            t0=config.t0, dt=config.dt,
    #                            init_control=config.init_control,
    #                            discrete=config.discrete,
    #                            version=config.dynamics_x,
    #                            meas_noise_var=0,
    #                            process_noise_var=0,
    #                            method=config.simu_solver,
    #                            t_eval=config.t_eval,
    #                            kwargs=config)
    # _, _, _, xtraj_estim, RMSE = NODE_rollout(
    #     NODE, x0, utraj, xtraj_true, rollout_length=config.rollout_length)
    # for i in range(xtraj_estim.shape[1]):
    #     name = 'xtraj_estim' + str(i) + '.pdf'
    #     plt.plot(xtraj_true[:, i], label='True')
    #     plt.plot(xtraj_estim.detach()[:, i], label='Estimated')
    #     plt.title('State trajectory')
    #     plt.legend()
    #     plt.show()
    #     plt.clf()
    #     plt.close('all')
    # x0 = torch.tensor([[1., 3.5]])
    # xtraj_true = dynamics_traj(x0=x0,
    #                            u=config.controller,
    #                            t0=config.t0, dt=config.dt,
    #                            init_control=config.init_control,
    #                            discrete=config.discrete,
    #                            version=config.dynamics_x,
    #                            meas_noise_var=0,
    #                            process_noise_var=0,
    #                            method=config.simu_solver,
    #                            t_eval=config.t_eval,
    #                            kwargs=config)
    # _, _, _, xtraj_estim, RMSE = NODE_rollout(
    #     NODE, x0, utraj, xtraj_true, rollout_length=config.rollout_length)
    # for i in range(xtraj_estim.shape[1]):
    #     name = 'xtraj_estim' + str(i) + '.pdf'
    #     plt.plot(xtraj_true[:, i], label='True')
    #     plt.plot(xtraj_estim.detach()[:, i], label='Estimated')
    #     plt.title('State trajectory')
    #     plt.legend()
    #     plt.show()
    #     plt.clf()
    #     plt.close('all')
    # x0 = torch.tensor([[-0.2, -0.8]])
    # xtraj_true = dynamics_traj(x0=x0,
    #                            u=config.controller,
    #                            t0=config.t0, dt=config.dt,
    #                            init_control=config.init_control,
    #                            discrete=config.discrete,
    #                            version=config.dynamics_x,
    #                            meas_noise_var=0,
    #                            process_noise_var=0,
    #                            method=config.simu_solver,
    #                            t_eval=config.t_eval,
    #                            kwargs=config)
    # _, _, _, xtraj_estim, RMSE = NODE_rollout(
    #     NODE, x0, utraj, xtraj_true, rollout_length=config.rollout_length)
    # for i in range(xtraj_estim.shape[1]):
    #     name = 'xtraj_estim' + str(i) + '.pdf'
    #     plt.plot(xtraj_true[:, i], label='True')
    #     plt.plot(xtraj_estim.detach()[:, i], label='Estimated')
    #     plt.title('State trajectory')
    #     plt.legend()
    #     plt.show()
    #     plt.clf()
    #     plt.close('all')

    end = timep.time()
    logging.info('Script ran in ' + str(end - start))
    stop_log()
