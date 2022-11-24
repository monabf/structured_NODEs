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
    update_config_init_state_obs, set_DF
from utils.pytorch_utils import get_parameters
from utils.utils import start_log, stop_log
from utils.config import Config

# To avoid Type 3 fonts for submission https://tex.stackexchange.com/questions/18687/how-to-generate-pdf-without-any-type3-fonts
# https://jwalton.info/Matplotlib-latex-PGF/
# https://stackoverflow.com/questions/12322738/how-do-i-change-the-axis-tick-font-in-a-matplotlib-plot-when-rendering-using-lat
sb.set_style('whitegrid')
plot_params = {
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.serif': 'Palatino',
    'font.size': 16,
    "pgf.preamble": "\n".join([
        r'\usepackage{bm}',
    ]),
    'text.latex.preamble': [r'\usepackage{amsmath}',
                            r'\usepackage{amssymb}',
                            r'\usepackage{cmbright}'],
}
plt.rcParams.update(plot_params)

# Script to learn a recognition model (estimates the  initial condition using
# NODE settings and observation data) for the controlled Van der Pol
# oscillator, jointly with the unknown dynamics (full NN dynamics model)

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
                           'VanderPol_fullNODE/MLP2_noisy_inputs',
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
                    trainer_options={'max_epochs': 2500, 'gpus': gpus},
                    optim_method=torch.optim.Adam,
                    optim_lr=1e-2,  # TODO
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
        submodel = MLP2(n_in=config.n + nu_submodel,  # TODO
                        n_h1=50, n_h2=50, n_out=config.n,
                        activation=nn.SiLU)
        n_param, param = get_parameters(submodel, verbose=True)
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
            if 'back' in config.init_state_obs_method:
                if config.true_meas_noise_var == 1e-1:
                    D = torch.tensor([[0.4951, -3.2781, -0.4123],
                                      [1.3609, -3.1310, 1.3386],
                                      [0.0916, 0.7717, -6.2826]])
                elif config.true_meas_noise_var == 1e-2:
                    D = torch.tensor([[-1.6647, -3.7385, 1.7853],
                                      [2.2123, -2.5724, 0.1564],
                                      [4.5115, -1.0837, -6.6678]])
                elif config.true_meas_noise_var == 1e-3:
                    D = torch.tensor([[-2.2089, -4.4443, 0.0911],
                                      [3.6874, -2.5779, 1.0648],
                                      [0.3112, -0.5733, -6.4832]])
                elif config.true_meas_noise_var == 1e-4:
                    D = torch.tensor([[-2.8428, -4.3078, 0.4150],
                                      [3.7846, -2.7191, 1.0334],
                                      [1.4078, 0.1500, -6.5988]])
                elif config.true_meas_noise_var == 1e-5:
                    D = torch.tensor([[-1.0686, -4.8044, -2.9830],
                                      [4.7799, -1.0973, 0.0916],
                                      [-1.3979, -0.0865, -5.5783]])
                else:
                    D, F = set_DF(W0, dz, y_observed_true.shape[-1],
                                  config.setD_method)
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
                if config.true_meas_noise_var == 1e-1:
                    D = torch.tensor([[-8.2224e+00, -1.8697e+00, -1.1659e+00,
                                       -6.8622e-01, 3.5549e-01,
                                       -8.7810e-01, -1.4893e+00, -8.7515e-01,
                                       1.4181e+00, -2.8171e+00,
                                       1.1263e+00, -7.5933e-02],
                                      [-1.8898e-01, -8.5073e+00, -7.8542e-01,
                                       -1.1871e+00, 7.3397e-01,
                                       -1.0125e+00, 1.0811e-01, -9.2110e-01,
                                       5.4240e-01, -1.1754e+00,
                                       -3.1307e-02, 2.0997e-01],
                                      [-1.4820e+00, -7.7488e-01, -6.8286e+00,
                                       -3.0678e+00, -7.5459e-02,
                                       -8.2970e-01, -2.1959e+00, -8.2080e-01,
                                       8.5513e-01, -1.5538e+00,
                                       5.7390e-01, -9.4546e-01],
                                      [-4.1004e-01, -1.8709e+00, 1.6857e+00,
                                       -7.1647e+00, 5.2612e-01,
                                       -1.1802e+00, 1.1240e+00, -1.1261e+00,
                                       1.0123e+00, -1.5041e-01,
                                       6.2872e-01, -6.2694e-02],
                                      [-1.2479e+00, -3.9127e-01, -3.7749e-01,
                                       -3.7953e-02, -5.1688e+00,
                                       -4.3669e+00, 7.3262e-01, 9.9636e-02,
                                       -4.9120e-01, -1.0970e+00,
                                       6.7002e-01, -3.0353e-01],
                                      [-8.8047e-01, -4.2365e-02, -3.8225e-01,
                                       -1.8183e-01, 4.3474e+00,
                                       -4.9677e+00, -2.6302e+00, -1.3297e-01,
                                       3.8348e-01, -4.5735e-01,
                                       1.9258e-01, 2.0881e-01],
                                      [-4.0327e-01, 5.7245e-02, -7.4707e-01,
                                       -6.0737e-01, 7.3236e-01,
                                       -7.3923e-01, -5.3897e+00, -5.6575e+00,
                                       8.4145e-01, 1.2343e+00,
                                       1.1819e+00, 9.6166e-01],
                                      [3.0301e-02, -3.9838e-01, 2.9088e-01,
                                       9.2450e-02, -8.2270e-01,
                                       2.4264e-01, 6.0474e+00, -3.6921e+00,
                                       -4.1853e-01, -4.3907e-01,
                                       -6.4423e-01, -9.1670e-01],
                                      [-1.5185e+00, -9.4063e-01, -1.3429e+00,
                                       3.6170e-02, -7.5634e-01,
                                       -9.2564e-01, 4.5812e-01, -3.7835e-01,
                                       -2.2927e+00, -6.3333e+00,
                                       1.1149e+00, -1.5841e-01],
                                      [-1.2348e+00, -8.2440e-01, -1.0387e+00,
                                       -1.0872e+00, 8.9337e-01,
                                       -1.0689e+00, -1.4384e+00, -7.2142e-01,
                                       6.0431e+00, -3.3072e+00,
                                       1.1797e-01, 7.0917e-01],
                                      [-3.7059e-01, 4.1795e-01, -8.3884e-02,
                                       1.0364e+00, -7.9306e-02,
                                       1.4448e-01, 1.0084e+00, 3.8025e-01,
                                       8.6729e-02, 4.0263e-01,
                                       -4.9467e-01, -5.8594e+00],
                                      [-1.6927e+00, -6.7191e-01, -1.2595e+00,
                                       -1.4592e+00, 6.9642e-03,
                                       -1.4074e+00, -8.4378e-01, -3.1736e-01,
                                       -3.4955e-02, -3.4908e-02,
                                       6.0472e+00, -6.1282e-01]])
                elif config.true_meas_noise_var == 1e-2:
                    D = torch.tensor([[-7.3660, -1.7948, 1.4970, -0.8798,
                                       -1.3728, -0.6032, -0.6132, -0.5061,
                                       1.1109, -0.7196, 0.1892, -0.0639],
                                      [0.0674, -6.7823, 1.2893, -0.4349,
                                       -1.5324, -0.2226, -0.4912, -0.1174,
                                       0.8736, -0.3746, -0.7268, 0.7517],
                                      [-1.2430, -0.6999, -7.6853, -3.1173,
                                       -1.3081, -0.6090, 0.3516, -0.9260,
                                       0.0854, -1.9409, 1.4049, -1.0629],
                                      [-0.1059, 0.0507, 3.4050, -5.8080,
                                       -1.0678, 0.2360, -0.6488, -0.6999,
                                       0.1786, -1.3118, 0.9206, 0.7720],
                                      [-0.4351, -0.5397, 0.9064, -0.3241,
                                       -6.3858, -4.2181, 0.3432, 0.3081,
                                       0.7770, 0.1879, -0.6469, -0.6284],
                                      [0.0809, 0.2065, -0.8868, 0.0571, 4.6551,
                                       -4.8561, -0.4141, -0.1657,
                                       -0.1560, 0.3261, 0.1964, -0.5418],
                                      [-0.8471, -0.9482, -0.3519, -1.2534,
                                       1.7444, -1.5553, -5.7107, -6.5416,
                                       0.0522, 0.1015, -1.3488, -0.0137],
                                      [-0.4259, -0.5828, -1.2557, -0.3480,
                                       0.7054, -0.3224, 7.3689, -4.2865,
                                       0.1370, -0.7624, 0.6439, -1.6699],
                                      [-0.8222, -0.8069, -0.9077, -0.8995,
                                       -0.1587, -0.6938, 1.8154, -0.1773,
                                       -3.2852, -6.7669, 0.6721, -0.9829],
                                      [-1.1664, -0.9474, 1.1744, -0.8832,
                                       -1.8470, -0.5303, -0.2969, -0.4471,
                                       6.8939, -3.6448, 1.2715, 0.4217],
                                      [-0.3709, -0.6855, -0.1087, -0.4809,
                                       0.0094, -1.4961, -0.2910, -1.0987,
                                       0.4966, 0.3430, -0.9339, -6.2564],
                                      [-0.5463, -0.8918, -0.2962, -0.8664,
                                       -0.8974, -1.1498, 0.0832, -0.6948,
                                       -0.4241, -0.2346, 6.2074, -0.9434]])
                elif config.true_meas_noise_var == 1e-4:
                    D = torch.tensor([[-7.6158e+00, -1.5192e+00, -1.1143e+00,
                                       -1.0538e+00, -4.2904e-02,
                                       -1.3729e+00, -8.7660e-01, -1.5216e+00,
                                       -2.8174e-01, -1.2291e+00,
                                       8.0297e-01, -8.8638e-01],
                                      [-3.7378e-03, -8.1729e+00, -8.1395e-01,
                                       -1.4861e+00, 6.6458e-01,
                                       -9.1671e-01, 1.3635e+00, -1.1319e+00,
                                       1.2360e+00, -1.3681e+00,
                                       1.2726e+00, -5.8077e-01],
                                      [-1.6564e+00, 1.2161e-01, -8.7591e+00,
                                       -2.2390e+00, 3.1285e+00,
                                       -1.8433e+00, 4.0059e-01, -1.3062e+00,
                                       -1.2026e+00, -1.5805e+00,
                                       6.2198e-01, 4.6711e-01],
                                      [-1.5446e+00, -1.8343e+00, 8.3906e-01,
                                       -7.8282e+00, 1.1549e+00,
                                       -9.3718e-01, 1.2857e+00, -1.1443e+00,
                                       1.5380e+00, -1.4823e+00,
                                       1.0542e+00, -5.2666e-01],
                                      [-1.0120e+00, -3.7693e-01, -7.4716e-01,
                                       -4.7620e-01, -5.8168e+00,
                                       -4.1150e+00, 2.3294e-01, -8.2627e-01,
                                       9.1015e-01, -3.7178e-01,
                                       5.7768e-01, -8.5557e-01],
                                      [4.7181e-01, 5.3384e-01, 9.3448e-02,
                                       7.7106e-01, 5.1696e+00,
                                       -5.0299e+00, -3.5752e-01, 1.5283e-01,
                                       -1.7946e+00, -9.2307e-03,
                                       -2.4264e-01, 3.3111e-01],
                                      [-4.2836e-01, -4.3948e-01, -3.8058e-01,
                                       -3.5241e-01, -1.0485e+00,
                                       -7.3384e-01, -5.1864e+00, -6.9103e+00,
                                       1.3960e+00, -7.2202e-01,
                                       2.5043e-01, -1.3837e+00],
                                      [-2.0263e-01, -1.6747e-01, -5.6762e-01,
                                       -2.4248e-01, 1.6479e+00,
                                       -3.9558e-01, 6.9154e+00, -4.0818e+00,
                                       -2.8194e-01, -1.0947e-01,
                                       1.0328e+00, 3.9233e-01],
                                      [-7.6742e-01, 6.1024e-01, -3.9326e-01,
                                       4.7758e-01, 1.4349e-01,
                                       -5.6522e-01, -1.4430e+00, -7.3772e-02,
                                       -4.0622e+00, -7.7693e+00,
                                       3.9252e-03, -1.0878e+00],
                                      [-1.0263e+00, -1.3856e+00, -1.1401e+00,
                                       -1.4737e+00, -4.8511e-02,
                                       -1.2388e+00, 1.1632e+00, -1.5196e+00,
                                       7.7864e+00, -3.5365e+00,
                                       1.4250e+00, -7.4160e-01],
                                      [-8.9272e-01, -1.0751e+00, -2.0333e-01,
                                       -8.9582e-01, -5.0878e-03,
                                       -1.9739e+00, 6.1490e-01, -1.1200e+00,
                                       5.2804e-01, -9.6694e-01,
                                       -5.9934e-02, -6.6948e+00],
                                      [-1.1400e+00, 6.1675e-03, -7.7920e-01,
                                       -8.5725e-02, 9.5661e-01,
                                       -5.5604e-01, 2.6661e-01, -6.0825e-01,
                                       2.6994e-01, -9.9841e-01,
                                       6.5861e+00, -2.6721e-01]])
                elif config.true_meas_noise_var == 1e-5:
                    D = torch.tensor([[-9.2045e+00, -2.8790e+00, -1.8161e+00,
                                       -1.5856e+00, 8.5454e-01,
                                       -1.8534e+00, -2.0144e-01, -1.7380e+00,
                                       1.4386e+00, -1.9264e+00,
                                       5.8142e-01, -1.8557e-01],
                                      [-1.2262e+00, -8.8629e+00, -1.5630e+00,
                                       -1.3115e+00, 1.1953e+00,
                                       -1.7832e+00, -6.2466e-01, -1.6784e+00,
                                       1.8457e+00, -1.3971e+00,
                                       -4.7631e-01, -6.6038e-01],
                                      [7.4425e-01, 5.6816e-01, -6.7954e+00,
                                       -2.7862e+00, 1.0721e+00,
                                       -3.9522e-01, -1.7117e+00, -5.3914e-01,
                                       -1.1194e+00, -3.3898e-01,
                                       -8.0396e-02, 1.0137e-01],
                                      [7.0748e-02, -3.9346e-01, 2.5915e+00,
                                       -6.5700e+00, -1.4468e+00,
                                       -2.4216e-01, 1.3848e+00, -3.5326e-01,
                                       1.3826e+00, -6.0500e-01,
                                       1.0265e+00, -7.9442e-01],
                                      [-1.2157e+00, -1.0160e+00, -9.3564e-01,
                                       -1.0105e+00, -6.4040e+00,
                                       -5.0562e+00, -7.1544e-01, -1.0707e+00,
                                       1.2432e+00, -1.6253e+00,
                                       4.7383e-01, -3.8943e-01],
                                      [1.9767e-01, 2.9107e-02, -9.5550e-01,
                                       -5.4871e-02, 5.7742e+00,
                                       -5.4788e+00, -1.0901e+00, -4.6079e-01,
                                       -1.2339e-01, -1.8923e-01,
                                       -2.7840e-01, -6.4161e-03],
                                      [9.1287e-01, 5.3866e-01, -6.7510e-01,
                                       -6.2780e-01, 3.0998e-01,
                                       -5.6077e-01, -5.6469e+00, -5.6642e+00,
                                       -4.0028e-01, -8.0072e-01,
                                       -3.7370e-01, 5.7000e-01],
                                      [-1.3808e+00, -1.2382e+00, 8.3980e-03,
                                       -3.1702e-01, -1.3942e-01,
                                       -4.9827e-01, 6.8025e+00, -4.2666e+00,
                                       1.7591e+00, -5.9273e-01,
                                       1.1788e-01, -7.5163e-01],
                                      [-4.0137e-01, -1.4266e+00, -1.0885e+00,
                                       -8.8519e-01, 5.8278e-01,
                                       -1.0290e+00, -8.5300e-01, -1.6857e+00,
                                       -2.9195e+00, -7.8471e+00,
                                       8.3664e-01, -4.0567e-01],
                                      [-1.1459e+00, -1.6574e+00, -1.5184e+00,
                                       -1.7025e+00, 5.6009e-01,
                                       -1.5693e+00, 7.4730e-01, -1.7099e+00,
                                       6.4000e+00, -4.7358e+00,
                                       8.5888e-01, 3.2117e-01],
                                      [-7.2728e-01, -6.5679e-01, -6.2457e-01,
                                       -5.8309e-01, 1.1577e+00,
                                       -5.6245e-01, 1.1947e+00, -6.3276e-01,
                                       1.5760e-01, -5.2467e-01,
                                       -3.6941e-01, -5.6783e+00],
                                      [-4.9228e-01, -1.3897e+00, -8.6504e-01,
                                       -1.4495e+00, 1.5493e-01,
                                       -1.2346e+00, 1.1320e-01, -7.1822e-01,
                                       -2.2923e-01, -2.8954e-01,
                                       5.8749e+00, -4.1603e-01]])
                else:
                    D = torch.tensor([[-8.7100e+00, -1.8646e+00, -1.1653e+00,
                                       -1.1661e+00, -4.3520e-01,
                                       -1.0684e+00, 2.0304e-01, -1.7324e+00,
                                       1.2107e+00, -1.6507e+00,
                                       8.2774e-01, -5.2847e-01],
                                      [1.5394e-01, -8.8038e+00, -1.4662e+00,
                                       -1.3007e+00, -1.0768e+00,
                                       -7.3577e-01, 1.9213e-01, -6.8502e-01,
                                       1.5417e+00, -2.5735e+00,
                                       1.0708e+00, -1.1387e+00],
                                      [-1.5518e+00, -2.0036e+00, -7.7063e+00,
                                       -4.2104e+00, -1.6919e+00,
                                       -1.3364e+00, -2.0365e-01, -1.5178e+00,
                                       1.0952e+00, -1.7436e+00,
                                       9.3766e-01, -3.6644e-01],
                                      [-1.0855e-01, 1.4399e-01, 4.3644e+00,
                                       -6.8929e+00, -2.2497e+00,
                                       -9.9601e-01, -6.0753e-01, -7.4794e-01,
                                       6.5551e-01, -2.7581e-01,
                                       -7.4851e-01, -8.2931e-01],
                                      [1.4687e-01, -1.4001e-01, 4.8713e-01,
                                       -7.1538e-01, -6.2892e+00,
                                       -4.3948e+00, -2.3903e-01, -3.9078e-01,
                                       3.3914e-01, -4.7219e-01,
                                       -5.9172e-01, -6.0809e-01],
                                      [-7.8706e-01, -4.1636e-01, -7.2255e-01,
                                       -1.1614e-01, 4.3886e+00,
                                       -5.1957e+00, 2.9621e-01, -3.3442e-01,
                                       1.2491e-01, -2.4065e-01,
                                       7.3729e-01, -3.2074e-01],
                                      [-6.2613e-01, 4.2294e-01, 1.7137e-01,
                                       -2.9256e-01, 6.3219e-02,
                                       -1.0115e+00, -5.0820e+00, -5.5734e+00,
                                       -4.1726e-01, 7.7486e-02,
                                       1.9459e-01, -1.1217e+00],
                                      [-9.0956e-01, -1.4294e+00, -1.3464e+00,
                                       -1.7249e-01, 1.5092e-01,
                                       -1.4076e-02, 6.1693e+00, -4.8460e+00,
                                       -2.3902e-01, -7.3049e-01,
                                       1.2833e+00, -2.9253e-01],
                                      [-7.2575e-01, -1.2221e+00, -7.8205e-01,
                                       -8.1391e-01, -4.2138e-02,
                                       -8.7561e-01, -1.2719e-02, -1.3472e+00,
                                       -2.4380e+00, -6.9024e+00,
                                       -1.2279e-01, -3.4541e-02],
                                      [-1.3263e+00, -5.6605e-01, -5.8207e-01,
                                       -8.4142e-01, -1.3161e+00,
                                       -5.6928e-01, 4.7881e-01, -8.2307e-01,
                                       6.4783e+00, -3.4236e+00,
                                       -2.5034e-03, -7.5368e-01],
                                      [-7.8732e-01, -4.8401e-01, 2.4539e-01,
                                       -1.2213e+00, -1.1800e+00,
                                       -1.4164e+00, 2.8916e-01, -1.1316e+00,
                                       4.6737e-01, -9.5394e-01,
                                       -4.4074e-01, -6.7053e+00],
                                      [-9.5297e-01, -1.2306e+00, -4.1916e-01,
                                       -1.1035e+00, -9.9518e-01,
                                       -1.2778e+00, 1.1115e+00, -8.3802e-01,
                                       1.1254e+00, -2.6228e-01,
                                       6.7707e+00, -1.0745e+00]])
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
