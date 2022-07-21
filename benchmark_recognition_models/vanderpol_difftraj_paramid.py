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
# Dynamics known up to parameters, jointly optimized with recognition model

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
                           'VanderPol_paramid/MLP2_noisy_inputs',
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
                              'VanderPol_paramid/MLP2_noisy_inputs/'
                              'Test2_100samples_noise0.001_MLP1_Adam0.05/Learn_NODE.pkl',
                    true_meas_noise_var=1e-3,
                    process_noise_var=0,
                    simu_solver='dopri5',
                    optim_solver_options={'method': 'dopri5'},  # for adjoint
                    # 'rtol': 1e-8, 'atol': 1e-8},
                    trainer_options={'max_epochs': 2500, 'gpus': gpus,
                                     'progress_bar_refresh_rate': 1},
                    optim_method=torch.optim.Adam,
                    optim_lr=1e-2,
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
            estimated_mu=torch.as_tensor(np.random.uniform(0.5, 1.5)),
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
        print(config.mu, config.estimated_mu)

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
            def __init__(self, cf):
                super(Submodel, self).__init__()
                self.config = cf
                self.munorm = torch.linalg.norm(cf.estimated_mu.clone())
                muinit = cf.estimated_mu / self.munorm
                self.mu = nn.parameter.Parameter(muinit, requires_grad=True)

            def set_scalers(self, scaler_X=None, scaler_Y=None):
                self.scaler_X = scaler_X
                self.scaler_Y = scaler_Y

            def forward(self, x):
                u = reshape_pt1(x[..., config.n:])
                x = reshape_pt1(x[..., :config.n])
                xdot = torch.zeros_like(x)
                xdot[..., 0] += x[..., 1]
                xdot[..., 1] += (self.mu * self.munorm) * (
                        1 - x[..., 0] ** 2) * x[..., 1] - x[..., 0] + u[..., 0]
                return xdot


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
                if config.true_meas_noise_var == 1e-1:
                    D = torch.tensor([[-0.7421, -3.4769, 1.0164],
                                      [2.9113, -2.7234, 1.6180],
                                      [-0.4309, 1.2862, -5.2873]])
                elif config.true_meas_noise_var == 1e-2:
                    D = torch.tensor([[-2.3905e+00, -4.6025e+00, -4.5189e-03],
                                      [3.4194e+00, -2.9026e+00, 1.2529e+00],
                                      [-1.8715e+00, 6.4854e-02, -5.6816e+00]])
                elif config.true_meas_noise_var == 1e-4:
                    D = torch.tensor([[-2.7966, -3.9505, 1.2050],
                                      [4.3081, -2.6728, 0.6819],
                                      [1.8017, -0.3623, -7.4790]])
                elif config.true_meas_noise_var == 1e-5:
                    D = torch.tensor([[-3.0639, -4.3926, 0.5134],
                                      [3.5742, -3.0799, 0.9933],
                                      [1.9671, 0.3727, -6.8439]])
                else:
                    D = torch.tensor([[-2.2163, -4.5076, -0.1335],
                                      [3.6112, -2.8001, 1.2039],
                                      [-0.7560, 0.3002, -6.0045]])
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
                if config.true_meas_noise_var == 1e-1:
                    D = torch.tensor([[-9.1243e+00, -2.0797e+00, -2.5827e+00,
                                       -1.0651e+00, -1.9754e+00,
                                       -8.1730e-01, -8.3320e-01, -1.4471e+00,
                                       -6.7043e-01, -8.9951e-01,
                                       1.3533e+00, -1.5554e+00],
                                      [1.0527e-01, -7.8802e+00, -1.2263e+00,
                                       -1.1828e+00, -1.1074e+00,
                                       -9.5968e-01, -7.5056e-01, -7.5770e-01,
                                       -9.0952e-01, -1.3921e+00,
                                       1.2941e+00, -1.7052e+00],
                                      [-1.4905e+00, -1.3047e+00, -8.4770e+00,
                                       -3.8241e+00, -1.6691e+00,
                                       -1.4812e+00, 2.3520e-01, -1.5944e+00,
                                       5.6768e-01, -1.0425e+00,
                                       7.3759e-01, -7.0772e-01],
                                      [1.2986e+00, -7.1499e-01, 4.3169e+00,
                                       -7.2748e+00, 2.1840e+00,
                                       -1.1810e+00, -3.5889e-01, -6.9902e-01,
                                       -1.9834e+00, -8.1147e-01,
                                       3.6595e-01, -1.6558e+00],
                                      [-1.0037e+00, -1.1543e+00, -9.0501e-01,
                                       -9.7439e-01, -6.3872e+00,
                                       -4.6109e+00, -1.9777e-01, -8.5602e-01,
                                       -3.8210e-01, -6.8298e-01,
                                       9.0275e-01, -8.9042e-01],
                                      [1.4431e+00, 1.0409e+00, 1.1617e+00,
                                       1.3627e-01, 5.8053e+00,
                                       -5.1204e+00, -4.0680e-01, 4.2933e-01,
                                       -3.0389e+00, -1.1412e-01,
                                       2.5590e-01, -4.1430e-03],
                                      [1.3972e-01, -7.4281e-01, -4.5357e-01,
                                       9.5530e-02, -3.7936e-02,
                                       -2.2177e-01, -4.7126e+00, -5.6178e+00,
                                       -8.5187e-01, 5.5356e-01,
                                       -6.9783e-01, -6.2061e-01],
                                      [-7.0420e-02, 4.5269e-01, 2.2581e-01,
                                       -3.6124e-01, 5.1711e-01,
                                       -8.0634e-01, 5.4621e+00, -4.6616e+00,
                                       -1.0058e+00, -9.3004e-01,
                                       6.6862e-01, -7.9110e-01],
                                      [1.4029e+00, -3.3807e-01, 1.4658e-01,
                                       -7.7698e-01, 5.1280e-01,
                                       -1.4408e+00, -2.8327e-01, -1.6236e+00,
                                       -4.1656e+00, -8.0489e+00,
                                       -3.6855e-01, 1.6453e-01],
                                      [-2.9577e+00, -1.2687e+00, -8.9962e-01,
                                       -9.3516e-01, -5.8799e-01,
                                       -1.2853e+00, 4.5676e-01, -1.0494e+00,
                                       9.6819e+00, -3.2572e+00,
                                       3.5065e-01, -5.8097e-01],
                                      [-4.1947e-02, -2.5127e-01, -4.3753e-01,
                                       -5.9419e-01, -1.5520e-01,
                                       1.4301e-01, -1.7535e-01, -2.5652e-01,
                                       2.4496e-01, -7.5197e-01,
                                       -2.4262e-01, -6.1323e+00],
                                      [5.1511e-01, -7.6510e-01, 5.3288e-01,
                                       -1.8907e+00, -3.5734e-01,
                                       -5.1269e-01, -3.9188e-01, -3.5767e-01,
                                       7.9862e-01, -4.6006e-01,
                                       6.2059e+00, -8.4529e-01]])
                elif config.true_meas_noise_var == 1e-2:
                    D = torch.tensor([[-8.8628, -1.7760, -2.3787, -0.8740,
                                       -2.3457, -0.7128, -0.7639, -1.4477,
                                       -1.0778, -1.7627, 1.3570, -1.8192],
                                      [0.6840, -8.1434, -0.2884, -1.2922,
                                       -0.8763, -1.1195, -0.2822, -1.2522,
                                       0.4804, -1.6851, 1.6680, -0.3375],
                                      [-1.3736, -1.3065, -8.3353, -3.7192,
                                       -1.2723, -1.5044, -0.0114, -1.5223,
                                       -0.0119, 0.2126, 0.7547, -1.1360],
                                      [1.3118, -2.2049, 4.4970, -7.4238, 1.7101,
                                       -1.0140, -0.0872, -0.7067,
                                       -2.5160, -1.1753, -0.1530, -1.0707],
                                      [-1.0855, -0.8838, -0.6598, -0.8206,
                                       -6.3946, -4.2458, 0.0324, -0.4777,
                                       -1.1369, -0.1668, 0.2358, -0.6120],
                                      [0.9708, 0.4894, 0.1523, -0.2918, 5.7912,
                                       -5.5886, 0.0518, -0.0523,
                                       -2.5766, -1.1107, 0.6769, -0.4318],
                                      [0.0980, -0.4551, -0.4627, -0.8421,
                                       -0.5575, -0.9458, -4.8684, -5.8433,
                                       -2.3738, -0.4259, -0.3827, -0.8659],
                                      [0.2414, 0.5574, -0.6116, 0.0864, 0.4842,
                                       -0.2902, 5.6828, -4.2826,
                                       -0.4182, -0.3875, 0.4041, -0.4128],
                                      [1.6092, -0.4696, 0.9886, -1.1414,
                                       -0.0692, -1.5736, 0.0582, -1.4732,
                                       -4.2026, -8.2498, 0.2837, -0.5166],
                                      [-2.4016, -1.2705, -1.4086, -1.0646,
                                       -0.4557, -1.7327, 0.0506, -1.7327,
                                       9.6068, -4.2971, 0.7903, -0.3198],
                                      [-0.2152, -0.2046, -0.1977, -0.3625,
                                       0.1133, -0.0449, -0.0172, -0.3741,
                                       -0.3869, -0.1438, -0.1801, -6.5229],
                                      [0.6938, -0.0943, 0.2141, -1.3877,
                                       -0.3541, -0.2625, -0.1399, -0.4202,
                                       -0.6447, -1.1979, 6.5613, -1.0085]])
                elif config.true_meas_noise_var == 1e-4:
                    D = torch.tensor([[-1.0982e+01, -3.5934e+00, -2.8385e+00,
                                       -2.8226e+00, -2.8014e+00,
                                       1.2642e-01, -3.1098e-01, -4.0371e-01,
                                       -1.4751e+00, 1.8812e-01,
                                       1.6553e-01, -2.0394e+00],
                                      [-1.2537e+00, -1.0652e+01, -1.1243e+00,
                                       -4.5475e+00, -2.6314e+00,
                                       -7.7839e-01, -5.5740e-01, -3.5778e-01,
                                       6.0975e-01, -2.6365e+00,
                                       -1.1206e-01, -2.5736e+00],
                                      [-2.4679e+00, -2.7616e+00, -8.6506e+00,
                                       -5.2426e+00, -2.9052e-01,
                                       -1.5710e+00, 1.2777e+00, -1.0600e+00,
                                       -1.3053e+00, -5.1778e-01,
                                       7.3291e-01, -3.9298e-01],
                                      [2.8666e+00, 8.0050e-01, 5.9061e+00,
                                       -6.3422e+00, 9.3655e-01,
                                       -1.1548e+00, -1.3786e+00, -1.2469e+00,
                                       -6.0044e-01, -2.4416e+00,
                                       1.2809e+00, -1.7099e+00],
                                      [8.2452e-01, 1.1491e+00, 7.7986e-01,
                                       3.8047e-01, -8.5510e+00,
                                       -4.9930e+00, -4.4955e-01, -1.8566e+00,
                                       3.1476e-01, -1.7009e+00,
                                       1.3085e+00, 4.3331e-02],
                                      [4.8479e-01, -5.2948e-01, 9.1837e-01,
                                       -2.1244e+00, 8.4313e+00,
                                       -6.9334e+00, 1.0557e+00, -3.8838e-01,
                                       6.8064e-01, -3.0971e-01,
                                       -1.8194e-01, -1.1319e+00],
                                      [9.3640e-01, 1.2193e+00, 4.2017e-01,
                                       1.6157e+00, 8.7837e-01,
                                       -1.1142e+00, -5.3283e+00, -6.4316e+00,
                                       1.7881e+00, -4.7855e-01,
                                       -2.2403e-01, -1.6593e+00],
                                      [1.9186e+00, 1.6784e+00, -1.2993e+00,
                                       1.9461e+00, 1.9603e+00,
                                       3.7793e-01, 7.2465e+00, -5.6738e+00,
                                       -2.1301e+00, 1.3930e+00,
                                       2.0212e+00, -3.6406e-01],
                                      [2.6846e+00, 1.2663e+00, 8.0597e-01,
                                       -2.5068e-01, -4.5468e-01,
                                       -2.3579e+00, -1.8642e+00, -2.2820e+00,
                                       -5.1008e+00, -9.4283e+00,
                                       1.7615e+00, -1.3822e+00],
                                      [-1.0354e+00, 3.5864e-01, -1.1894e-01,
                                       2.7614e-01, 2.9580e-01,
                                       -1.9844e+00, -1.4828e+00, -2.7625e+00,
                                       1.0167e+01, -5.5239e+00,
                                       2.1718e+00, -6.2150e-01],
                                      [5.3329e-01, 6.9701e-02, 5.4252e-01,
                                       6.7127e-03, -1.9937e-01,
                                       -7.0450e-01, -8.7331e-01, -1.0504e+00,
                                       1.1925e+00, 4.3777e-01,
                                       -1.0315e-01, -6.8847e+00],
                                      [1.6182e+00, 7.3374e-01, 6.8964e-01,
                                       3.3574e-01, -3.3333e-01,
                                       -5.4183e-01, -1.1378e+00, -1.4936e+00,
                                       -1.4883e+00, 6.2508e-01,
                                       6.9375e+00, -7.2906e-01]])
                elif config.true_meas_noise_var == 1e-5:
                    D = torch.tensor([[-10.5401, -1.9965, -4.2123, -1.1583,
                                       -2.8456, -0.1333, 0.2832,
                                       -1.8946, 1.0349, -3.0052, 0.7769,
                                       0.2584],
                                      [1.2890, -8.3878, 2.0065, -3.0810,
                                       -2.0404, -1.5169, -1.0739,
                                       0.1171, -1.8899, 0.2475, 2.7163,
                                       -2.0523],
                                      [-1.9258, -0.2399, -8.8871, -2.7809,
                                       -0.6369, -0.5178, 1.0587,
                                       -2.6203, 2.3265, -1.9723, 0.0193,
                                       0.3031],
                                      [1.4791, -0.9241, 3.2848, -8.2403, 0.5728,
                                       -0.5541, -0.6742,
                                       0.6512, -3.6431, 0.5552, 1.4504,
                                       -2.2086],
                                      [-1.8025, -1.3334, -1.6187, -2.3164,
                                       -8.8966, -4.4068, -0.6419,
                                       0.2131, -1.3204, 0.6153, -1.4590,
                                       -2.1557],
                                      [1.9353, -0.2262, 2.2211, -0.6897, 8.1307,
                                       -6.6330, -1.3429,
                                       -1.1908, -4.3823, -1.1346, 1.3760,
                                       -1.8255],
                                      [1.1703, -3.5369, 0.5808, -2.4499, 1.5505,
                                       -2.1838, -5.3337,
                                       -4.9609, -1.7078, 1.1652, 0.1527,
                                       -1.4937],
                                      [0.6702, 0.9417, 1.4444, 0.6052, 1.7308,
                                       0.6979, 4.9168,
                                       -5.6185, -2.9068, -1.9346, 1.6497,
                                       -0.0993],
                                      [2.5039, -0.1263, 0.8975, -0.3970,
                                       -1.7091, 0.1160, -1.1759,
                                       -2.4171, -5.7919, -8.5163, 0.6112,
                                       1.0235],
                                      [-4.4238, -2.1543, -3.1171, -1.6191,
                                       -2.3468, -1.6221, 1.0779,
                                       -2.0156, 8.8046, -5.4849, -0.6760,
                                       0.3130],
                                      [-1.1505, -1.3184, -1.4733, -1.5317,
                                       -1.2774, -1.3015, 1.2453,
                                       2.0953, 3.3291, -1.0708, -1.3956,
                                       -7.6066],
                                      [-1.1802, -1.9058, 1.3156, -2.2082,
                                       1.1485, -1.9126, -1.6920,
                                       -1.6558, 0.7412, -1.3251, 7.6139,
                                       -1.9481]])
                else:
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
        specs_file = os.path.join(NODE.results_folder, 'Specifications.txt')
        with open(specs_file, 'a') as f:
            print(f'Initial estimated mu = {config.estimated_mu}',
                  file=f)
            print(f'Final estimated mu = '
                  f'{NODE.submodel.mu.item() * NODE.submodel.munorm.item()}',
                  file=f)
            print(f'True mu = {config.mu}', file=f)
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
