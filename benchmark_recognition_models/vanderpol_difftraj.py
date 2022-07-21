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
import seaborn as sb
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from smt.sampling_methods import LHS

sys.path.append('.')

from simulation.controllers import Control_from_dict
from simulation.dynamics_functions import VanDerPol
from simulation.simulation_functions import simulate_dynamics
from simulation.observers import dim1_observe_data
from simulation.observer_functions import KKL, KKLu
from benchmark_NN_models import MLP2, RNNn, MLP2_xin
from NN_for_ODEs.learn_neural_ODE import Learn_NODE_difftraj
from NN_for_ODEs.NODE_utils import make_diff_init_state_obs, \
    update_config_init_state_obs
from utils.utils import start_log, stop_log, reshape_pt1
from utils.config import Config

# To avoid Type 3 fonts for submission https://tex.stackexchange.com/questions/18687/how-to-generate-pdf-without-any-type3-fonts
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}\usepackage{cmbright}')
plt.rc('font', family='serif')

sb.set_style("whitegrid")

# Script to learn a recognition model (estimates the  initial condition using
# NODE settings and observation data) for the controlled Van der Pol oscillator

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
    config = Config(system='Continuous/Benchmark_recognition/'
                           'VanderPol/MLP2_noisy_inputs',
                    sensitivity='autograd',
                    intloss=None,
                    order=1,
                    nb_samples=100,  # TODO
                    nb_difftraj=72,
                    t0_span=0,
                    tf_span=3,
                    t0=0,
                    tf=3,
                    init_state_obs_method=str(sys.argv[2]),
                    setD_method='butter_block_diag',
                    init_state_obs_T=40,
                    NODE_file='../Figures/Continuous/Benchmark_recognition/'
                              'VanderPol/MLP2_noisy_inputs/'
                              'Test2_100samples_noise0.001_MLP1_Adam0.05/Learn_NODE.pkl',
                    true_meas_noise_var=1e-3,
                    process_noise_var=0,
                    simu_solver='dopri5',
                    # simu_solver='rk4',
                    # solver_options={'step_size': 1e-3},
                    optim_solver_options={'method': 'dopri5'},  # for adjoint
                    # 'rtol': 1e-8, 'atol': 1e-8},
                    trainer_options={'max_epochs': 2500, 'gpus': gpus,
                                     'progress_bar_refresh_rate': 1},
                    optim_method=torch.optim.Adam,
                    optim_lr=5e-3,
                    optim_minibatch_size=100,
                    optim_shuffle=False,
                    optim_options={'weight_decay': 1e-5},
                    # l1_reg=1.,
                    optim_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    optim_scheduler_options={
                        'mode': 'min', 'factor': 0.2, 'patience': 50,
                        'threshold': 0.1, 'verbose': True},
                    optim_stopper=pl.callbacks.early_stopping.EarlyStopping(
                        monitor='val_loss', min_delta=0.001, patience=20,
                        verbose=False, mode='min'),
                    nb_rollouts=100,  # TODO
                    rollout_length=100,
                    rollout_controller={'sin_controller_1D': 100},
                    rollout_controller_args={'controller_args': [
                        {'gamma': 1.2,
                         'omega': (2 * np.pi) / (torch.rand(1) * (5 - 3) + 3.)}
                        for i in range(100)]},
                    max_rollout_value=10.,
                    verbose=False,
                    monitor_experiment=True,
                    prior_mean=None,
                    prior_mean_deriv=None,
                    continuous_model=False,
                    plot_output=True)

    # Train whole model
    if train:
        # System params
        config.update(dict(
            discrete=False,
            mu=1.,
            dt_before_subsampling=0.1,
            dynamics_function=VanDerPol,
            no_control=False,
            controller_dict={'sin_controller_1D': config.nb_difftraj},
            controller_args=[
                {'gamma': 1.2,
                 'omega': (2 * np.pi) / (torch.rand(1) * (5 - 3) + 3.)}
                for i in range(config.nb_difftraj)],
            observe_data=dim1_observe_data,
            observe_data_x=dim1_observe_data,
            # observe_data=lambda x: x,  # TODO
            # observe_data_x=lambda x: x,
            prior_kwargs={'dt': config.dt,
                          'dt_before_subsampling': config.dt}))
        dynamics = config.dynamics_function(config.cuda_device, config)
        config.update(dict(
            dynamics=dynamics, dynamics_x=dynamics, true_dynamics=dynamics,
            init_control=torch.zeros(config.nb_difftraj, 1, 1)))

        # Set initial states of x, xi for all difftrajs with LHS
        # https://smt.readthedocs.io/en/latest/_src_docs/sampling_methods/lhs.html
        xlimits = np.array([[-1., 1.], [-1., 1.]])
        sampling = LHS(xlimits=xlimits)
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
            def __init__(self, config):
                super(Submodel, self).__init__()
                self.config = config

            def set_scalers(self, scaler_X=None, scaler_Y=None):
                self.scaler_X = scaler_X
                self.scaler_Y = scaler_Y

            def forward(self, x):
                u = reshape_pt1(x[..., config.n:])
                x = reshape_pt1(x[..., :config.n])
                return config.dynamics.VanderPol_dynamics_xu(x, u)


        submodel = Submodel(config)
        n_param = 0
        config.update(dict(
            n_param=n_param, reg_coef=1., nu=config.init_control.shape[2],
            constrain_u=[-10., 10.],
            constrain_x=[],
            grid_inf=-1.,
            grid_sup=1.))
        controller = Control_from_dict(config.controller_dict,
                                       config.init_control,
                                       config.constrain_u)
        config.update(controller=controller)

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
            W0 = 2 * np.pi * 1  # TODO
            # D, F = set_DF(W0, dz, y_observed_true.shape[-1], config.setD_method)
            if 'back' in config.init_state_obs_method:
                D = torch.tensor([[-2.2163, -4.5076, -0.1335],
                                  [3.6112, -2.8001, 1.2039],
                                  [-0.7560, 0.3002, -6.0045]])
                F = torch.ones(dz, y_observed_true.shape[1])
            else:
                D = torch.tensor([[-1.9616, -4.4288, 0.7020],
                                  [3.6782, -3.1710, 0.3702],
                                  [1.5975, 0.6336, -5.9863]])
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
            dw = 3  # to generate sinusoidal control with varying amplitude
            dz = (y_observed_true.shape[1] +
                  config.init_control.shape[1]) * (config.n + dw + 1)
            W0 = 2 * np.pi * 1  # TODO
            # D, F = set_DF(W0, dz, y_observed_true.shape[-1] +
            #               config.init_control.shape[-1], config.setD_method)
            if 'back' in config.init_state_obs_method:
                D = torch.tensor([[-8.5902e+00, -1.6184e+00, -2.0686e+00,
                                   -8.3462e-01, -1.7119e+00,
                                   -9.2733e-01, -4.9820e-01, -1.6199e+00,
                                   1.5250e-02, -1.2626e+00,
                                   1.7575e+00, -1.1424e+00],
                                  [4.3440e-01, -7.3393e+00, -6.0891e-01,
                                   -1.0540e+00, -5.3724e-01,
                                   -1.0755e+00, -3.2499e-02, -1.0693e+00,
                                   -6.9446e-01, -1.1289e+00,
                                   1.4231e+00, -7.7276e-01],
                                  [-1.1489e+00, -9.9407e-01, -7.3433e+00,
                                   -3.5770e+00, -1.5456e+00,
                                   -1.2976e+00, -2.4511e-01, -1.3765e+00,
                                   5.8008e-01, -7.1727e-01,
                                   1.4079e+00, -1.0678e+00],
                                  [1.4918e+00, -1.5618e-01, 3.8070e+00,
                                   -6.9404e+00, 1.9946e+00,
                                   -9.8787e-01, 2.1276e-01, -6.7116e-01,
                                   -2.4010e+00, -8.6128e-01,
                                   3.1510e-01, -8.5061e-01],
                                  [-9.0442e-01, -7.9915e-01, -1.0777e+00,
                                   -6.6492e-01, -6.3143e+00,
                                   -4.3378e+00, 6.3433e-02, -6.4709e-01,
                                   1.7579e-01, -5.1120e-01,
                                   3.5650e-01, -3.4556e-01],
                                  [1.2475e+00, 5.2368e-01, 1.3250e+00,
                                   -2.9272e-01, 5.7339e+00,
                                   -5.4401e+00, 8.8259e-02, 7.9868e-02,
                                   -2.8609e+00, -7.7589e-02,
                                   1.1356e-01, -2.6805e-01],
                                  [8.4694e-02, -6.4996e-01, -8.2105e-01,
                                   -3.2513e-01, -3.8423e-01,
                                   -4.8946e-01, -4.3505e+00, -5.6455e+00,
                                   -4.5486e-01, -6.1077e-01,
                                   -9.0144e-01, -1.9758e-01],
                                  [-3.8626e-01, 1.8258e-01, 7.1724e-02,
                                   -1.7570e-01, 4.6693e-01,
                                   -5.1111e-01, 5.8317e+00, -4.3352e+00,
                                   -1.1097e+00, -2.8564e-01,
                                   2.9625e-01, -9.1554e-01],
                                  [1.1866e+00, -6.5851e-01, -2.6161e-01,
                                   -9.7662e-01, -4.7887e-01,
                                   -1.0380e+00, -3.1414e-01, -1.0217e+00,
                                   -4.4309e+00, -7.5467e+00,
                                   -3.5852e-01, -3.2818e-01],
                                  [-2.8059e+00, -9.0343e-01, -1.7911e+00,
                                   -6.7638e-01, -1.0873e+00,
                                   -8.2031e-01, -1.1049e-01, -9.0316e-01,
                                   7.6557e+00, -3.5422e+00,
                                   6.5372e-01, -4.9194e-01],
                                  [4.2662e-02, -2.4656e-01, -2.4264e-01,
                                   -3.7344e-01, -1.9030e-03,
                                   -2.3371e-01, 4.2457e-02, -2.0726e-01,
                                   4.3737e-01, 1.0111e-01,
                                   -1.7995e-01, -6.2651e+00],
                                  [4.7598e-01, -2.6444e-01, -3.3932e-02,
                                   -5.4276e-01, -1.1726e-01,
                                   -1.2217e-01, -2.9784e-01, -1.7136e-01,
                                   -6.2651e-01, -2.3436e-01,
                                   6.2785e+00, -7.3127e-01]])
                F = torch.ones(dz, y_observed_true.shape[1] +
                               config.init_control.shape[1])
            else:
                D = torch.tensor([[-6.6329e+00, -9.0111e-01, 1.2657e-01,
                                   4.7199e-02, -5.1071e-01,
                                   -2.8627e-02, -1.2032e+00, 1.2795e-01,
                                   9.3822e-02, -4.6530e-01,
                                   2.0834e-01, -1.4045e-01],
                                  [7.2277e-02, -7.2410e+00, -1.0284e+00,
                                   -8.5539e-01, -3.6938e-02,
                                   -8.4501e-01, -2.9947e-01, -7.8933e-01,
                                   6.9925e-01, -6.8982e-01,
                                   5.3793e-01, 6.4277e-04],
                                  [1.1162e-01, -3.1608e-01, -6.4987e+00,
                                   -2.2911e+00, -5.8605e-01,
                                   -5.1605e-02, 4.0669e-01, 8.6615e-02,
                                   -3.0642e-02, -1.7896e-01,
                                   1.0968e-01, -5.9914e-01],
                                  [-7.1442e-01, -3.1982e-01, 2.5175e+00,
                                   -6.7074e+00, 6.1836e-02,
                                   -6.5401e-01, -1.6476e+00, -5.8812e-01,
                                   2.3210e-01, -1.1333e+00,
                                   1.2511e+00, -9.0449e-01],
                                  [-2.4725e-01, -4.1079e-01, -3.6573e-01,
                                   -9.5079e-02, -6.6755e+00,
                                   -3.8646e+00, 1.5068e-01, 5.2230e-02,
                                   -2.2975e-02, -1.0199e-01,
                                   -3.1579e-01, -1.2144e+00],
                                  [-2.7155e-01, -2.9456e-01, -4.2040e-01,
                                   -5.0798e-01, 5.3872e+00,
                                   -5.5279e+00, -1.5994e-01, -6.3129e-01,
                                   1.7693e-01, -5.8441e-01,
                                   4.7226e-01, -9.9723e-02],
                                  [-4.1572e-01, -6.1853e-01, -7.7175e-01,
                                   -4.9271e-01, 2.3964e+00,
                                   -7.1191e-01, -6.0158e+00, -5.7227e+00,
                                   2.4995e-01, -5.9222e-01,
                                   1.8987e-01, 6.0628e-01],
                                  [-5.7597e-01, -5.3931e-01, -3.9829e-01,
                                   -5.6007e-01, -1.8732e+00,
                                   -3.1483e-01, 7.5084e+00, -4.4825e+00,
                                   5.3018e-01, -6.9153e-01,
                                   5.8394e-01, -1.1226e+00],
                                  [-3.4546e-01, -5.3041e-01, -8.5630e-01,
                                   -6.3427e-01, 4.7420e-02,
                                   -6.2982e-01, 6.6720e-01, -9.4707e-01,
                                   -2.1630e+00, -5.9952e+00,
                                   2.2339e-01, 1.2186e+00],
                                  [-4.3301e-01, -3.0302e-01, -7.8377e-01,
                                   -2.0719e-01, -7.6046e-01,
                                   -2.8440e-01, -1.2740e+00, -2.8683e-01,
                                   5.7115e+00, -2.5697e+00,
                                   -4.0828e-01, -2.1568e-02],
                                  [-3.9901e-01, -2.5967e-01, 4.6458e-01,
                                   5.6668e-01, -5.6932e-01,
                                   -9.8378e-02, 8.4337e-01, 5.6019e-01,
                                   1.1007e+00, -3.2985e-01,
                                   2.6762e-01, -5.8345e+00],
                                  [-3.3007e-01, -1.4712e-01, -7.1274e-01,
                                   -2.2338e-01, -9.2018e-01,
                                   -2.5641e-01, 2.1001e-01, -3.7569e-01,
                                   4.3387e-01, 7.1579e-01,
                                   5.8095e+00, 2.5262e-01]])
                F = torch.ones(dz, y_observed_true.shape[1] +
                               config.init_control.shape[1])
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
            dw = 3  # same as KKLu
            dz = (y_observed_true.shape[1] +
                  config.init_control.shape[1]) * (config.n + dw + 1)
            init_state_model1 = RNNn(
                n_in=diff_y_observed_true.shape[-1] + config.nu,  # TODO
                n_out=dz, n_hl=1, RNN=torch.nn.GRU)
            init_state_model = MLP2_xin(
                n_in=dz, n_h1=50, n_h2=50, n_out=config.n,
                model_in=init_state_model1, activation=nn.SiLU)
        elif config.init_state_obs_method.startswith('y0T_u0T_RNN'):
            init_state_model = RNNn(
                n_in=diff_y_observed_true.shape[-1] + config.nu,  # TODO
                n_out=config.n, n_hl=1, RNN=torch.nn.GRU)
        else:
            init_state_model = MLP2(n_in=torch.numel(diff_init_state_obs[0]),
                                    n_h1=50, n_h2=50, n_out=config.n,
                                    activation=nn.SiLU)
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
        # tensorboard --logdir=../Figures/Continuous/Benchmark_recognition/VanderPol/MLP2_noisy_inputs/ --port=8080
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
                    plt.plot(config.t_eval, diff_xtraj_true[j, :, i],
                             label='True')
                    plt.plot(config.t_eval, xtraj_estim.detach()[j, :, i],
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

    end = timep.time()
    logging.info('Script ran in ' + str(end - start))
    stop_log()
