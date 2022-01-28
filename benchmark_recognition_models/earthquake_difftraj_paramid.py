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
from simulation.dynamics_functions import Earthquake_building
from simulation.simulation_functions import simulate_dynamics
from simulation.observers import dim1_observe_data
from simulation.observer_functions import KKL, KKLu
from benchmark_NN_models import MLP2, Recog_ignore
from NN_for_ODEs.learn_neural_ODE import Learn_NODE_difftraj
from NN_for_ODEs.NODE_utils import make_diff_init_state_obs, \
    update_config_init_state_obs, set_DF
from utils.utils import start_log, stop_log, reshape_pt1
from utils.pytorch_utils import get_parameters, StandardScaler
from utils.config import Config

sb.set_style('whitegrid')

# Script to learn a recognition model (estimates the  initial condition using
# NODE settings and observation data) for the earthquake model with a known
# external disturbance (linear but non autonomous) (unrealistic to assume
# disturbance caused by earthquake is known individually, only output should be)

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
    # torch.manual_seed(0)
    if torch.cuda.is_available():
        gpus = -1
    else:
        gpus = 0

    # General params
    config = Config(system='Continuous/Benchmark_recognition/'
                           'Earthquake_building_paramid/MLP2_noisy_inputs',
                    sensitivity='autograd',
                    intloss=None,
                    order=1,
                    nb_samples=100,
                    nb_difftraj=72,
                    t0_span=0,
                    tf_span=3,
                    t0=0,
                    tf=3,
                    init_state_obs_method=str(sys.argv[2]),
                    setD_method='butter_block_diag',
                    init_state_obs_T=40,
                    init_state_obs_Tu=1,
                    NODE_file='../Figures/Continuous/Benchmark_recognition/'
                              'Earthquake_building/MLP2_noisy_inputs/'
                              'Test2_100samples_noise0.001_MLP1_Adam0.05/Learn_NODE.pkl',
                    true_meas_noise_var=1e-4,
                    process_noise_var=0,
                    simu_solver='dopri5',
                    optim_solver_options={'method': 'dopri5'},
                    trainer_options={'max_epochs': 2500, 'gpus': gpus,
                                     'progress_bar_refresh_rate': 1},
                    optim_method=torch.optim.Adam,
                    optim_lr=5e-2,
                    optim_minibatch_size=100,
                    optim_shuffle=False,
                    optim_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    optim_scheduler_options={
                        'mode': 'min', 'factor': 0.8, 'patience': 20,
                        'threshold': 0.1, 'verbose': True},
                    optim_stopper=pl.callbacks.early_stopping.EarlyStopping(
                        monitor='val_loss', min_delta=0.001, patience=20,
                        verbose=False, mode='min'),
                    nb_rollouts=100,
                    rollout_length=100,
                    rollout_controller={'cos_controller_1D': 100},
                    rollout_controller_args={'controller_args': [
                        {'gamma': torch.rand(1) * (1.5 - 0.5) + 0.5,
                         'omega': torch.rand(1) * (1.5 - 0.5) + 0.5}
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
            k=1e4,#np.random.uniform(5e3, 5e4),
            m=1e3,#np.random.uniform(5e2, 5e3),
            estimated_km=torch.as_tensor(np.random.uniform(8, 12)),
            umin=0.5,
            umax=1.5,
            dt_before_subsampling=0.1,
            dynamics_function=Earthquake_building,
            no_control=False,
            controller_dict={'cos_controller_1D': config.nb_difftraj},
            controller_args=[{'gamma': torch.rand(1) * (1.5 - 0.5) + 0.5,
                              'omega': torch.rand(1) * (1.5 - 0.5) + 0.5}
                             for i in range(config.nb_difftraj)],
            observe_data=dim1_observe_data,
            observe_data_x=dim1_observe_data,
            prior_kwargs={'dt': config.dt,
                          'dt_before_subsampling': config.dt}))
        dynamics = config.dynamics_function(config.cuda_device, config)
        config.update(dict(
            dynamics=dynamics, dynamics_x=dynamics, true_dynamics=dynamics,
            init_control=torch.tensor([
                config.controller_args[i]['gamma'] for i in range(
                    config.nb_difftraj)]).unsqueeze(1).unsqueeze(1)))
        print(config.k / config.m, config.estimated_km)

        # Set initial states of x, xi for all difftrajs with LHS
        # https://smt.readthedocs.io/en/latest/_src_docs/sampling_methods/lhs.html
        xlimits = np.array([[-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.]])
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
            def __init__(self, cf):
                super(Submodel, self).__init__()
                self.config = cf
                self.km0norm = torch.linalg.norm(cf.estimated_km.clone())
                kminit = cf.estimated_km / self.km0norm
                self.km = nn.parameter.Parameter(kminit, requires_grad=True)

            def set_scalers(self, scaler_X=None, scaler_Y=None):
                self.scaler_X = scaler_X
                self.scaler_Y = scaler_Y

            def forward(self, x):
                u = reshape_pt1(x[..., config.n:])
                x = reshape_pt1(x[..., :config.n])
                xdot = torch.zeros_like(x)
                xdot[..., 0] = x[..., 1]
                xdot[..., 1] += self.km * self.km0norm * (
                        x[..., 2] - 2 * x[..., 0]) - u[..., 0]
                xdot[..., 2] = x[..., 3]
                xdot[..., 3] += self.km * self.km0norm * (
                        x[..., 0] - x[..., 2]) - u[..., 0]
                return xdot


        submodel = Submodel(config)
        n_param = 0
        config.update(dict(
            n_param=n_param, reg_coef=1., nu=config.init_control.shape[2],
            constrain_u=[config.umin, config.umax],
            constrain_x=[],
            grid_inf=-1,
            grid_sup=1))
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
            W0 = 2 * np.pi * 1
            # D, F = set_DF(W0, dz, y_observed_true.shape[-1], config.setD_method)
            if 'back' in config.init_state_obs_method:
                D = torch.tensor([[-5.1170, -5.1501, -0.6393, -1.1812, -1.8479],
                                  [2.7349, -3.7487, -0.8182, -0.8294, 2.1080],
                                  [0.3598, 0.1438, -0.7842, -5.2801, -0.0140],
                                  [-0.0272, -0.0064, 5.1830, -0.6535, -0.3714],
                                  [-0.8910, 0.4529, -0.9746, -0.9187, -5.4204]])
                F = torch.ones(dz, y_observed_true.shape[1])
            else:
                D = torch.tensor([[-4.9150, -5.6215, -0.8632, -1.3044, -1.8419],
                                  [4.2384, -3.8208, -1.0166, -1.1344, 1.1855],
                                  [0.2262, -0.2933, -0.5137, -5.2573, -0.3749],
                                  [1.7877, -0.1911, 5.3581, -0.2396, -0.0558],
                                  [-0.4391, 0.4878, -0.4415, -0.1295, -6.9281]])
                F = torch.ones(dz, y_observed_true.shape[1])
            z0 = torch.zeros(1, dz)
            z_config = {'D': D, 'F': F, 'init_state_estim': z0,
                        'Db': D, 'Fb': F, 'Bessel_W0': W0, 'dz': dz}
            if 'optimD' in config.init_state_obs_method:
                z_config['init_D'] = D.clone()
            KKL = KKL(config.device, z_config)
            config.update(dict(z_config=z_config,
                               init_state_KKL=KKL))
            print(f'D: {D}, eigvals: {torch.linalg.eigvals(D)}')
        elif 'KKLu' in config.init_state_obs_method:
            dw = 3  # to generate sinus with diff freqs, amplitudes
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
            print(f'D: {D}, eigvals: {torch.linalg.eigvals(D)}')

        # Define the inputs of the recognition model
        diff_init_state_obs = make_diff_init_state_obs(
            diff_y_observed_true, diff_utraj, init_state_x, config.t_eval,
            config)

        # Define the actual recognition model (same for all init)
        init_state_model1 = MLP2(
            n_in=torch.numel(diff_init_state_obs[0]) - config.nu, n_h1=50,
            n_h2=50, n_out=config.n, activation=nn.SiLU)  # TODO
        init_state_model = Recog_ignore(
            model=init_state_model1, idx1=0,
            idx2=torch.numel(diff_init_state_obs[0]) - config.nu)
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
        # tensorboard --logdir=../Figures/Continuous/Benchmark_recognition/Earthquake_building/MLP2_noisy_inputs/ --port=8080
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
        specs_file = os.path.join(NODE.results_folder, 'Specifications.txt')
        with open(specs_file, 'a') as f:
            print(f'Initial estimated k/m = {config.estimated_km}', file=f)
            print(f'Final estimated k/m = '
                  f'{NODE.submodel.km.item() * NODE.submodel.km0norm.item()}',
                  file=f)
            print(f'True k/m = {config.k / config.m}', file=f)
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
