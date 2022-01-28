import logging
import os
import sys
import time

import dill as pkl
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sb
import torch
import torch.nn as nn
import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from model_evaluation.plotting_functions import run_rollouts_NODE, \
    plot_model_evaluation, plot_NODE
from model_evaluation.plotting_rollouts import save_rollout_variables
from simulation.controllers import Control_from_dict
from simulation.dynamics import dynamics_traj
from utils.config import Config
from utils.pytorch_utils import StandardScaler
from utils.utils import reshape_pt1, reshape_pt1_tonormal, save_log, \
    kronecker, remove_outlier, reshape_dim1, concatenate_lists, RMS, \
    interpolate_func
from .neural_ODE import NODE, NODE_difftraj

sb.set_style('whitegrid')

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

# Class to learn ODEs with NNs. Optimization problem to train the NN on one
# or several trajectories of solutions: minimize a loss on these
# trajectories by enforcing the ODE as a constraint and computing the loss
# gradient analytically using the forward or adjoint sensitivity method.

# This is the learner class, which relies mainly on pytorch-lightning: it
# receives the data and optimizes the given NODE model
# https://torchdyn.readthedocs.io/en/latest/tutorials/01_neural_ode_cookbook.html
# https://pytorch-lightning.readthedocs.io/en/1.2.2/starter/converting.html
# https://pytorch-lightning.readthedocs.io/en/0.8.1/lightning-module.html
# On logging with pytorch-lightning: training_step must return at least
# 'loss' tag, self.log enables quick logging with progress bar, the logger (
# default tensorboard) is attached to the trainer, saves checkpoints which
# can be loaded, and can be seen in browser by typing in terminal
# tensorboard --logdir=log_directory/ --port=8080
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
# https://bleepcoder.com/pytorch-lightning/587991476/train-loss-vs-loss-on-progress-bar
# https://neptune.ai/blog/pytorch-lightning-neptune-integration
# https://www.pytorchlightning.ai/blog/tensorboard-with-pytorch-lightning
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.loggers.TensorBoardLogger.html#pytorch_lightning.loggers.TensorBoardLogger
# https://learnopencv.com/tensorboard-with-pytorch-lightning/

# Pytorch good practices:
# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

LOCAL_PATH = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
LOCAL_PATH_TO_SRC = LOCAL_PATH.split(os.sep + 'src', 1)[0]


class Learn_NODE(pl.LightningModule):
    """
    Default Learner class: trains on whole trajectory x0 -> xN
    All tensors concerning trajectories assumed of size N x n
    """

    # Inherit and overwrite training_step, loss, save_model for new models

    def __getattr__(self, item):
        # self.config[item] can be called directly as self.item
        # for val in self.__dict__.values():
        #     if isinstance(val, dict):
        #         if item in val.keys():
        #             return val[item]
        if item in self.__dict__.keys():
            return self.__dict__[item]
        elif item in self._modules:
            return self._modules[item]
        elif item in self.config.keys():
            return self.config[item]
        else:
            raise AttributeError(item)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __init__(self, X_train, U_train, submodel: nn.Module, config: Config,
                 sensitivity='autograd', ground_truth_approx=False,
                 validation=True, dataset_on_GPU=False):
        """
        Define learner class for NODE: trains on x0 -> xN by default,
        receives data and handles optimization

        :param X_train: trajectory of observations
        :type X_train: torch.tensor
        param U_train: trajectory of control inputs
        :type U_train: torch.tensor
        param submodel: vector field model to be optimized
        :type submodel: torch.nn.Module
        """
        assert len(X_train) == len(U_train), \
            'X_train and U_train must have the same length'
        super().__init__()

        # Set options of inherited classes to False by default
        if self.__class__.__name__ == 'Learn_NODE':
            self.difftraj = False
            self.subtraj = False

        # Create all important objects: config, data, submodel, model
        self.config = config
        self.specs = self.config
        self.X_train = reshape_pt1(X_train)
        self.U_train = reshape_pt1(U_train)
        self.ground_truth_approx = ground_truth_approx
        self.sensitivity = sensitivity
        self.submodel = submodel  # .to(self.device)
        if self.difftraj:
            self.p = self.X_train.shape[2]
        else:
            self.p = self.X_train.shape[1]
        if self.p < self.n or self.config.partial_obs:
            self.partial_obs = True
        else:
            self.partial_obs = False
        if self.config.KKL_traj is None:
            self.config['KKL_traj'] = torch.zeros((
                self.nb_difftraj, self.nb_samples, 1))

        logging.warning(
            'Using pytorch: all data used by the NN must be torch tensors and '
            'not numpy data. Conversions should be avoided at best. Only '
            'functions that convert to numpy are for saving data to csv and '
            'plots.')
        if self.dt > 0.1:
            logging.warning(
                'Time step is larger than 0.1s! This might be too much, '
                'most of all for all small operations that rely on Euler '
                'discretization to obtain continuous solutions from discrete '
                'GP models, such as rollouts, continuous observers...')
        if self.init_state_model:
            assert any(self.init_state_obs_method == k for k in (
                'x0', 'y0', 'y0_u0', 'y0T_u0T', 'KKL_u0T', 'KKLu',
                'KKL_u0T_back', 'KKLu_back', 'KKL_u0T_optimD', 'KKLu_optimD',
                'KKL_u0T_back_optimD', 'KKLu_back_optimD',
                'fixed_recognition_model', 'KKL_u0T_y0T', 'KKLu_y0T',
                'KKL_u0T_back_y0T', 'KKLu_back_y0T', 'KKL_u0T_optimD_y0T',
                'KKLu_optimD_y0T', 'KKL_u0T_back_optimD_y0T',
                'KKLu_back_optimD_y0T')), \
                'Only possible options for init_state_obs_method are: x0 for ' \
                'true x0; y0, y0_u0, y0T_u0T for these inputs; KKL_u0T, ' \
                'KKL_u0T_back for the inputs z0, u0:T with z0 obtained by ' \
                'autonomous KKL forward or backward; KKLu, KKLu_back for the ' \
                'input z0 obtained by functional KKL forward and backward; ' \
                'same with _optimD appended at the end for a KKL-based ' \
                'recognition model where D is optimized jointly with the ' \
                'nonlinear transformation; fixed_recognition_model for a ' \
                'fixed recognition model using directly y0:T and u0:T. A ' \
                'term y0T can also be appended to the KKL-based models in ' \
                'order to use both the KKL state and y0:T as input.'

        # Create model based on sensitivity
        if self.sensitivity not in ['forward', 'autograd', 'adjoint']:
            raise NotImplementedError('Only available sensitivity methods: '
                                      'froward, autograd, adjoint.')
        if self.sensitivity == 'forward':
            self.model = self.submodel
            # Prepare init extended state, scaler and optimizer
            self.init_ext = torch.cat((
                self.init_state_estim, torch.zeros((1, self.n * self.n_param))),
                dim=1)
        else:
            layers = []
            if self.init_state_model:
                layers.append(self.init_state_model)
            if self.difftraj:
                self.NODE_model = NODE_difftraj(self.submodel, self.config,
                                                self.order)
            else:
                self.NODE_model = NODE(self.submodel, self.config, self.order)
            layers.append(self.NODE_model)
            self.model = torch.nn.Sequential(*layers)
            if self.init_state_model and 'optimD' in self.init_state_obs_method:
                # Require grad for D so that gets optimized, set it for KKL
                # Also scale it: D = D / ||D0||, KKL ODE with D * ||D0||
                D_0 = torch.linalg.norm(
                    self.config['z_config']['D'].detach().clone())
                self.config['z_config']['D0'] = D_0
                D_init = self.config['z_config']['D'].detach().clone() / D_0
                self.model.init_state_KKL_Dscaled = \
                    torch.nn.parameter.Parameter(D_init, requires_grad=True)
                self.init_state_model.KKL_ODE_model.init_state_KKL.D = \
                    self.model.init_state_KKL_Dscaled
                self.init_state_model.KKL_ODE_model.init_state_KKL.set_alpha(
                    D_0)
                self.config['z_config']['D'] = \
                    self.init_state_model.KKL_ODE_model.init_state_KKL.D

        # If dataset fits in memory, you can put data + model on GPU
        if dataset_on_GPU:
            self.X_train, self.U_train, self.model = \
                self.X_train.to(self.cuda_device), self.U_train.to(
                    self.cuda_device), self.model.to(self.cuda_device)

        # Scale data
        with torch.no_grad():
            if self.difftraj:
                # print('train data', self.X_train.shape, self.U_train.shape)
                # plt.plot(self.X_train[:, :, 0].t())
                # plt.show()
                # plt.plot(self.X_train[:, :, 1].t())
                # plt.show()
                # plt.plot(self.U_train[:, :, 0].t())
                # plt.show()  # TODO
                self.X_train, self.U_train = self.view_difftraj(
                    self.X_train, self.U_train)
            self.scaler_Y = StandardScaler(self.X_train)
            self.scaler_U = StandardScaler(self.U_train)
            self.scaler_XU = StandardScaler(self.X_train)
            self.specs['Y_mean'] = self.scaler_Y._mean
            self.specs['Y_var'] = self.scaler_Y._var
            if self.partial_obs:
                if (self.config.scalerX_method is None) or (
                        self.scalerX_method == 'meanY_obs'):
                    self.config['scalerX_method'] = 'meanY_obs'
                    # Set scaler_X to scaler_Y in dimensions that are
                    # observed, mean of scaler_Y elsewhere
                    mean = torch.mean(self.scaler_Y._mean)
                    var = torch.mean(self.scaler_Y._var)
                    x_mean = mean.repeat(self.n)
                    x_var = var.repeat(self.n)
                    obs_dims = [int(i) for i in self.observe_data.__name__ if
                                i.isdigit()]
                    for i in range(len(obs_dims)):
                        dim = obs_dims[i] - 1
                        x_mean[dim] = self.scaler_Y._mean[i]
                        x_var[dim] = self.scaler_Y._var[i]
                    self.scaler_X = StandardScaler(mean=x_mean, var=x_var)
                elif self.scalerX_method == 'meanY':
                    # Set scaler_X to mean of scaler_Y
                    mean = torch.mean(self.scaler_Y._mean)
                    var = torch.mean(self.scaler_Y._var)
                    x_mean = mean.repeat(self.n)
                    x_var = var.repeat(self.n)
                    self.scaler_X = StandardScaler(mean=x_mean, var=x_var)
                elif self.scalerX_method == 'init_xtraj':
                    # Set scaler_X using the initial NODE
                    xtraj = self.NODE_model.forward_traj(
                        self.init_state_estim, self.controller, self.t0,
                        self.t_eval, self.init_control)
                    xtraj = xtraj.contiguous().view(-1, xtraj.shape[-1])
                    self.scaler_X = StandardScaler(xtraj)
                elif self.scalerX_method == 'concatY':
                    # Scaler is concatenation of scaler_Y
                    x_mean = self.scaler_Y._mean.repeat(
                        int(np.ceil(self.n / self.p)))
                    x_mean = x_mean[:self.n]
                    x_var = self.scaler_Y._var.repeat(
                        int(np.ceil(self.n / self.p)))
                    x_var = x_var[:self.n]
                    self.scaler_X = StandardScaler(mean=x_mean, var=x_var)
                self.specs['scalerX_method'] = self.scalerX_method
                logging.info(f'Partial observations: scaling of x is based on '
                             'a heuristic that depends on the scaling of y, '
                             'using method: %s. Make sure it makes sense for '
                             'your use case!', self.scalerX_method)
            else:
                self.scaler_X = StandardScaler(self.X_train)

        if self.no_control:
            self.submodel.set_scalers(scaler_X=self.scaler_X,
                                      scaler_Y=self.scaler_X)
        else:
            self.scaler_XU.set_scaler(
                torch.cat((self.scaler_X._mean, self.scaler_U._mean)),
                torch.cat((self.scaler_X._var, self.scaler_U._var)))
            self.submodel.set_scalers(scaler_X=self.scaler_XU,
                                      scaler_Y=self.scaler_X)
        if self.init_state_model:
            # self.init_state_model.set_scalers(
            #     scaler_X=self.init_state_model.scaler_X, scaler_Y=self.scaler_X)
            self.init_state_model.set_scalers(
                scaler_X=self.config['z_config']['scaler_Z'],
                scaler_Y=self.scaler_X)
        if self.difftraj:
            self.X_train, self.U_train = self.unview_difftraj(
                self.X_train, self.U_train)

        # Split training and validation data (test = rollouts/model evaluation)
        self.validation = validation
        if self.ground_truth_approx:
            assert self.validation, 'If ground_truth_approx, we are using ' \
                                    'experimental data, hence need ' \
                                    'train/val/test sets.'
        if self.validation:
            if self.ground_truth_approx:
                val_size = 0.21  # 0.3 of remaining 0.7 after test split
            else:
                val_size = 0.3
            self.train_val_split = train_test_split(
                np.arange(self.X_train.shape[0], dtype=int), test_size=val_size,
                shuffle=True)
            self.train_idx = self.train_val_split[0]
            self.val_idx = self.train_val_split[1]
            train_split = torch.as_tensor(self.train_idx,
                                          device=self.X_train.device)
            val_split = torch.as_tensor(self.val_idx,
                                        device=self.X_train.device)
            self.X_train, self.X_val = \
                torch.index_select(self.X_train, dim=0, index=train_split), \
                torch.index_select(self.X_train, dim=0, index=val_split)
            self.U_train, self.U_val = \
                torch.index_select(self.U_train, dim=0, index=train_split), \
                torch.index_select(self.U_train, dim=0, index=val_split)
            self.init_state_estim, self.init_state_estim_val = \
                torch.index_select(
                    self.init_state_estim, dim=0, index=train_split), \
                torch.index_select(
                    self.init_state_estim, dim=0, index=val_split)
            if self.init_state_model:
                self.init_state_obs, self.init_state_obs_val = \
                    torch.index_select(
                        self.init_state_obs, dim=0, index=train_split), \
                    torch.index_select(
                        self.init_state_obs, dim=0, index=val_split)
            if self.difftraj:
                self.nb_difftraj = len(train_split)
                self.nb_difftraj_val = len(val_split)
        else:
            self.train_idx = np.arange(
                self.X_train.shape[0], dtype=int).tolist()

        # Specs
        self.specs['X_mean'] = self.scaler_X._mean
        self.specs['X_var'] = self.scaler_X._var
        self.specs['U_mean'] = self.scaler_U._mean
        self.specs['U_var'] = self.scaler_U._var
        self.specs['submodel'] = self.submodel
        if self.init_state_model:
            self.specs['init_state_obs_mean'] = \
                self.init_state_model.scaler_X._mean
            self.specs['init_state_obs_var'] = \
                self.init_state_model.scaler_X._var
        if self.difftraj:
            if self.validation:
                self.specs['nb_difftraj_train'] = len(train_split)
                self.specs['nb_difftraj_val'] = len(val_split)
            else:
                self.specs['nb_difftraj_train'] = self.nb_difftraj
                self.specs['nb_difftraj_val'] = 0

        # Metrics to evaluate learned model
        if self.config.grid_inf is None:
            logging.warning('No grid was predefined by the user for one step '
                            'ahead model evaluation and rollouts, so using '
                            'min and max of state data.')
            self.grid_inf = torch.min(self.X_train, dim=0).values
            self.grid_sup = torch.max(self.X_train, dim=0).values
        self.grid_inf = torch.as_tensor(self.grid_inf)
        self.grid_sup = torch.as_tensor(self.grid_sup)
        self.step = 0
        self.sample_idx = 0
        if ground_truth_approx:
            # Data rollouts cannot be longer than data
            self.rollout_length = int(
                np.min([self.rollout_length, self.nb_samples - 1]))
        self.prior_kwargs = self.config.prior_kwargs
        if self.ground_truth_approx:
            logging.warning('True dynamics are approximated from data or '
                            'from a simplified model: there is actually no '
                            'ground truth, the true dynamics are only used as '
                            'a comparison to the GP model! Hence, model '
                            'evaluation tools such as GP_plot, rollouts or '
                            'model_evaluation are only indicative; true '
                            'evaluation of the model can only be obtained by '
                            'predicting on a test set and comparing to the '
                            'true data.')
        self.init_time = time.time()
        self.time = torch.zeros((1, 1))
        self.grid_RMSE = torch.zeros((0, 2))
        self.grid_SRMSE = torch.zeros((0, 2))
        self.rollout_RMSE = torch.zeros((0, 2))
        self.rollout_SRMSE = torch.zeros((0, 2))
        self.rollout_RMSE_init = torch.zeros((0, 2))
        self.rollout_SRMSE_init = torch.zeros((0, 2))
        self.rollout_RMSE_output = torch.zeros((0, 2))
        self.rollout_SRMSE_output = torch.zeros((0, 2))
        self.train_loss = torch.zeros((0,))
        self.val_loss = torch.zeros((0,))

        # Create rollouts for evaluation
        self.rollout_list = self.create_rollout_list()
        # Create grid of (x_t, u_t) to evaluate model quality (true dynamics
        # needed to compare true x_t+1 to predicted)
        self.grid, self.grid_controls = \
            self.create_grid(self.constrain_u, self.grid_inf, self.grid_sup)
        self.true_predicted_grid = \
            self.create_true_predicted_grid(self.grid, self.grid_controls)
        if not self.ground_truth_approx:
            # Reject outliers from grid
            true_predicted_grid_df = pd.DataFrame(
                self.true_predicted_grid.cpu().numpy())
            grid_df = pd.DataFrame(self.grid.cpu().numpy())
            grid_controls_df = pd.DataFrame(self.grid_controls.cpu().numpy())
            mask = remove_outlier(true_predicted_grid_df)
            true_predicted_grid_df = true_predicted_grid_df[mask]
            grid_df = grid_df[mask]
            grid_controls_df = grid_controls_df[mask]
            self.true_predicted_grid = \
                true_predicted_grid_df.values
            self.grid = grid_df.values
            self.grid_controls = \
                grid_controls_df.values
        # Update variables
        self.variables = {'X_train': self.X_train, 'U_train': self.U_train,
                          'Computation_time': self.time}
        self.variables['grid_RMSE'] = self.grid_RMSE
        self.variables['grid_SRMSE'] = self.grid_SRMSE
        self.variables['rollout_RMSE'] = self.rollout_RMSE
        self.variables['rollout_SRMSE'] = self.rollout_SRMSE
        self.variables['rollout_RMSE_init'] = self.rollout_RMSE_init
        self.variables['rollout_SRMSE_init'] = self.rollout_SRMSE_init
        self.variables['rollout_RMSE_output'] = self.rollout_RMSE_output
        self.variables['rollout_SRMSE_output'] = self.rollout_SRMSE_output
        if self.validation:
            self.variables['X_val'] = self.X_val
            self.variables['U_val'] = self.U_val
            self.variables['train_idx'] = self.train_idx
            self.variables['val_idx'] = self.val_idx

        # Create unique results folder and save test data
        params = str(np.random.uniform()) + '_' + sensitivity + '_' + str(
            self.nb_samples) + 'samples_noise' + str(
            self.true_meas_noise_var) + '_' + str(
            # self.model.__class__.__name__)
            self.NODE_model.__class__.__name__)
        if 'difftraj' in self.__class__.__name__:
            params = params + str(self.nb_difftraj)
        params = params + '_' + str(
            self.optim_method.__name__) + str(self.optim_lr)
        if self.init_state_model:
            params = params + '_' + str(self.init_state_obs_method) + str(
                self.init_state_obs_T)
            params = str(self.init_state_obs_method) + '/' + params
        else:
            params = 'x0/' + params
        if self.ground_truth_approx:
            params = self.data_folder.split('/')[-2] + '/' + params
        elif self.nb_rollouts > 0:
            params = str(self.nb_rollouts) + '_rollouts/' + params
        self.results_folder = os.path.join(
            str(LOCAL_PATH_TO_SRC), 'Figures', str(self.system), params)
        os.makedirs(self.results_folder, exist_ok=False)
        self.save_grid_variables(self.grid, self.grid_controls,
                                 self.true_predicted_grid,
                                 self.results_folder)
        self.true_predicted_grid = torch.as_tensor(
            self.true_predicted_grid, device=self.device)
        self.grid = torch.as_tensor(self.grid, device=self.device)
        self.grid_controls = torch.as_tensor(self.grid_controls,
                                             device=self.device)
        self.grid_variables = {
            'Evaluation_grid': self.grid,
            'Grid_controls': self.grid_controls,
            'True_predicted_grid': self.true_predicted_grid}
        save_rollout_variables(self, self.results_folder, self.nb_rollouts,
                               self.rollout_list, step=self.step,
                               ground_truth_approx=self.ground_truth_approx,
                               plots=self.monitor_experiment, NODE=True)
        # Save log in results folder
        os.rename(str(LOCAL_PATH_TO_SRC) + '/Figures/Logs/' + 'log' +
                  str(sys.argv[1]) + '.log',
                  os.path.join(self.results_folder,
                               'log' + str(sys.argv[1]) + '.log'))
        save_log(self.results_folder)
        if self.verbose:
            logging.info(self.results_folder)
        # self.save_hyperparameters()  # Good practice but deepcopy hard...

    def forward(self, x):
        # Make predictions after training
        return self.model(x)

    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/stable/common/optimizers.html
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2976
        if self.config.optim_options:
            optim_options = self.optim_options
        else:
            optim_options = {}
        parameters = self.model.parameters()
        optimizer = self.optim_method(parameters, self.optim_lr,
                                      **optim_options)
        if self.config.optim_scheduler:
            if self.config.optim_scheduler_options:
                optim_scheduler_options = self.optim_scheduler_options
            else:
                optim_scheduler_options = {}
            scheduler = {
                'scheduler': self.optim_scheduler(optimizer,
                                                  **optim_scheduler_options),
                'monitor': 'train_loss'}
            return [optimizer], [scheduler]
        else:
            return optimizer

    def train_dataloader(self):
        # Data = (idx, observations). If partial observations, use X_train =
        # observations and observation function in self.model.forward_traj_obs
        # Minibatches (over length of X_train) only appear after forward
        # simulation of x(t) so no impact
        train_dataset = TensorDataset(
            torch.arange(len(self.X_train)), self.X_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.optim_minibatch_size,
            shuffle=self.optim_shuffle)
        return train_loader

    def training_step(self, batch, batch_idx):
        # Forward pass NODE x0 -> xN, get samples matching batch, compute loss
        idx_batch, y_batch = batch
        if self.init_state_model:
            init_state_estim = self.init_state_obs
        else:
            init_state_estim = self.init_state_estim
        if (self.config.KKL_l2_reg is not None) and (
                'optimD' in self.init_state_obs_method):
            # Compute KKL traj to keep it for loss, then use same simulation
            # to compute z(T) -> x0 -> x(t)
            KKL_traj = self.init_state_model.simulate_ztraj(
                init_state_estim[...,
                :self.init_state_model.KKL_ODE_model.n],
                self.init_state_model.z_t_eval)
            z = torch.squeeze(self.init_state_model.simulate_zu(
                init_state_estim, ztraj=KKL_traj))
            init_state_estim = reshape_pt1(
                self.init_state_model.init_state_model(z))
            x_estim = self.NODE_model(init_state_estim)[idx_batch]
            KKL_traj = KKL_traj[idx_batch]
        else:
            x_estim = self.model(init_state_estim)[idx_batch]
            if self.config.KKL_l2_reg is not None:
                KKL_traj = self.config.KKL_traj[idx_batch]
            else:
                KKL_traj = None
        y_estim = self.observe_data_x(x_estim)
        if self.no_control:
            xu_estim = x_estim
        else:
            u_estim = self.config.controller(
                self.t_eval, self.config, self.t0, self.init_control)[idx_batch]
            xu_estim = torch.cat((x_estim, u_estim), dim=-1)
        losses = self.NODE_model.loss(
            y_estim=y_estim, y_true=y_batch, xu_estim=xu_estim,
            KKL_traj=KKL_traj, scaler_Y=self.scaler_Y, scaler_X=self.scaler_X)
        loss = sum(losses.values())
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        self.train_loss = torch.cat((self.train_loss, torch.tensor([loss])))
        self.time = torch.cat((
            self.time, torch.tensor([[time.time() - self.init_time]])), dim=0)
        logs = {'train_loss': loss.detach()}
        for key, val in losses.items():
            logs.update({key: val.detach()})
            self.log(key, val, prog_bar=False, logger=True)
        return {'loss': loss, 'log': logs}

    def val_dataloader(self):
        # For now validation same as training, just for early stopping
        if self.validation:
            val_dataset = TensorDataset(
                torch.arange(len(self.X_val)), self.X_val)
            val_loader = DataLoader(
                val_dataset, batch_size=self.optim_minibatch_size,
                shuffle=self.optim_shuffle)
        else:
            val_dataset = TensorDataset(
                torch.arange(len(self.X_train)), self.X_train)
            val_loader = DataLoader(
                val_dataset, batch_size=self.optim_minibatch_size,
                shuffle=self.optim_shuffle)
        return val_loader

    def validation_step(self, batch, batch_idx):
        # Validation is same as training, but on validation data if exists
        # (otherwise just training loss again, used for early stopping)
        with torch.no_grad():
            if self.validation:
                idx_batch, y_batch = batch
                if self.init_state_model:
                    init_state_estim = self.init_state_obs_val
                else:
                    init_state_estim = self.init_state_estim_val
                if (self.config.KKL_l2_reg is not None) and (
                        'optimD' in self.init_state_obs_method):
                    # Compute KKL traj to keep it for loss, then use same simulation
                    # to compute z(T) -> x0 -> x(t)
                    KKL_traj = self.init_state_model.simulate_ztraj(
                        init_state_estim[...,
                        :self.init_state_model.KKL_ODE_model.n],
                        self.init_state_model.z_t_eval)
                    z = torch.squeeze(self.init_state_model.simulate_zu(
                        init_state_estim, ztraj=KKL_traj))
                    init_state_estim = reshape_pt1(
                        self.init_state_model.init_state_model(z))
                    x_estim = self.NODE_model(init_state_estim)[idx_batch]
                    KKL_traj = KKL_traj[idx_batch]
                    if (self.config.KKL_l2_reg is not None) and (
                            'optimD' in self.init_state_obs_method):
                        # Compute KKL traj to keep it for loss, then use same simulation
                        # to compute z(T) -> x0 -> x(t)
                        KKL_traj = self.init_state_model.simulate_ztraj(
                            init_state_estim[...,
                            :self.init_state_model.KKL_ODE_model.n],
                            self.init_state_model.z_t_eval)
                        z = torch.squeeze(self.init_state_model.simulate_zu(
                            init_state_estim, ztraj=KKL_traj))
                        init_state_estim = reshape_pt1(
                            self.init_state_model.init_state_model(z))
                        x_estim = self.NODE_model(init_state_estim)[idx_batch]
                        KKL_traj = KKL_traj[idx_batch]
                    else:
                        x_estim = self.model(init_state_estim)[idx_batch]
                        if self.config.KKL_l2_reg is not None:
                            KKL_traj = self.config.KKL_traj[idx_batch]
                        else:
                            KKL_traj = None
                y_estim = self.observe_data_x(x_estim)
                if self.no_control:
                    xu_estim = x_estim
                else:
                    u_estim = self.config.controller(
                        self.t_eval, self.config, self.t0,
                        self.init_control)[idx_batch]
                    xu_estim = torch.cat((x_estim, u_estim), dim=-1)
                losses = self.NODE_model.loss(
                    y_estim=y_estim, y_true=y_batch, xu_estim=xu_estim,
                    KKL_traj=KKL_traj, scaler_Y=self.scaler_Y,
                    scaler_X=self.scaler_X)
                loss = sum(losses.values())
                self.log('val_loss', loss, on_step=True, prog_bar=True,
                         logger=True)
                self.val_loss = torch.cat((self.val_loss, torch.tensor([loss])))
                logs = {'val_loss': loss.detach()}
                for key, val in losses.items():
                    logs.update({key: val.detach()})
                    self.log(key, val, prog_bar=False, logger=True)
            else:
                if len(self.train_loss) == 0:
                    loss = torch.tensor(np.nan)
                else:
                    loss = self.train_loss[-1]
                self.log('val_loss', loss, on_step=True, prog_bar=True,
                         logger=True)
                logs = {'val_loss': loss.detach()}
            return {'loss': loss, 'log': logs}

    def train_forward_sensitivity(self):
        # Train model with forward sensitivity method
        # Only on full state data for autonomous systems!
        self.losses = []
        if self.config.optim_scheduler:
            optimizer, scheduler = \
                self.configure_optimizers()[0][0], \
                self.configure_optimizers()[1][0]['scheduler']
        else:
            optimizer = self.configure_optimizers()[0]
        # Prepare dataset, dataloader and iterators through data
        xtraj_true = self.X_train
        epochs_iter = tqdm.tqdm(range(self.trainer_options['max_epochs']),
                                desc="Epoch", leave=True)
        # Forward pass: in each epoch and each minibatch, solve training data
        # (x_estim, lambda_estim) for current weights, then optimize loss to get
        # new weights. Minibatches go over fixed data = (idx_yobs, yobs)
        for k in epochs_iter:
            # Simulate x, lambda
            traj_estim = dynamics_traj(x0=self.init_ext,
                                       u=self.controller,
                                       t0=self.t0, dt=self.dt,
                                       init_control=self.init_control,
                                       discrete=self.discrete,
                                       version=self.dynext_forward_sensitivity,
                                       meas_noise_var=0,
                                       process_noise_var=0,
                                       method=self.simu_solver,
                                       t_eval=self.t_eval,
                                       kwargs=self.config)
            # Organize simulation results in (x, lambda) and keep only minibatch
            xtraj_estim = traj_estim[:, :self.n]
            lambdatraj_estim = traj_estim[:, self.n:]
            lambdatraj_estim = lambdatraj_estim.reshape(
                -1, self.n_param, self.n).permute(0, 2, 1)

            # Compute loss, its gradient, step of optimizer and optimize param
            loss = 1. / 2 * torch.sum(
                torch.square(xtraj_estim - xtraj_true))
            self.losses.append(loss.item())
            dloss = torch.sum(
                torch.matmul((xtraj_estim - xtraj_true).unsqueeze(1),
                             lambdatraj_estim).squeeze(), dim=0)
            # Take a step of gradient descent, this time grad is needed
            # Manually set grad for each param
            param_beg = 0
            for name, parameter in self.model.named_parameters():
                param_end = param_beg + parameter.numel()
                parameter.grad = dloss[param_beg:param_end].reshape(
                    parameter.shape).clone()
                param_beg = param_end
            optimizer.step()
            # Manually zero the gradients after updating weights
            optimizer.zero_grad()

            # Update the learning rate and the early stopping at each epoch
            epochs_iter.set_postfix(loss=loss.item())
            if self.config.optim_scheduler:
                scheduler.step(loss)
        return traj_estim

    # Dynamics function to simulate a trajectory (x, lambd) where the ODE on
    # x is self.model and the ODE on lambd is the sensitivity equation for
    # the current self.model.
    # Uses autograd for sensitivity equation
    # https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
    def dynext_forward_sensitivity(self, t, x, u, t0, init_control,
                                   process_noise_var, kwargs,
                                   impose_init_control=False, verbose=False):
        start = time.time()
        x = reshape_pt1(x)
        n = kwargs.n
        n_param = kwargs.n_param
        lamb = reshape_pt1(x[:, n:])
        x = reshape_pt1(x[:, :n])
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        if not x.requires_grad:
            x.requires_grad = True  # at init, give x a grad field
        else:
            x = x.detach()  # detach new input from previous graph
            x.requires_grad = True
        xdot = self.model(x)  # output of NN
        # Compute grads of NN wrt inputs and parameters, one Jac per output dim!
        # https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
        # dfdx = torch.autograd.functional.jacobian(
        #     kwargs.model, x, create_graph=True)  # alternative computation
        dfdx = torch.zeros((n, n))
        dfdparam = torch.zeros((n, n_param))
        m = torch.eye(n)  # helper: args to compute vect-Jac prod of each dim
        for i in range(n):
            # Start by zeroing out the grad to avoid summing them over dims
            x.grad = None
            self.model.zero_grad()
            xdot.backward(reshape_pt1(m[i]), retain_graph=True)
            dfdx[i, :] = x.grad
            param_grad = torch.zeros((0,))
            for name, parameter in self.model.named_parameters():
                param_grad = torch.cat((param_grad, parameter.grad.flatten()))
            dfdparam[i, :] = param_grad
        lambdot = reshape_pt1(torch.matmul(kronecker(
            torch.eye(n_param), dfdx), reshape_pt1_tonormal(lamb)) +
                              dfdparam.t().flatten())
        end = time.time()
        if verbose:
            logging.info(
                f'Extended state simulation at {t}s in {end - start} s')
        return torch.cat((xdot, lambdot), dim=1)

    def create_grid(self, constrain_u, grid_inf, grid_sup):
        # Create random grid for evaluation
        # https://stackoverflow.com/questions/45583274/how-to-generate-an-n-dimensional-grid-in-python
        if constrain_u:
            umin = np.min(constrain_u)
            umax = np.max(constrain_u)
        else:
            logging.warning('No interval was predefined by the user for one '
                            'step ahead model evaluation and rollouts, '
                            'so using min and max of control data.')
            umin = torch.min(self.U_train, dim=0).values
            umax = torch.max(self.U_train, dim=0).values
        nb_points = int(np.ceil(np.max([10 ** 4 / self.n, 500])))
        grid = reshape_pt1(torch.rand((
            nb_points, self.n), device=self.device)
                           * (grid_sup - grid_inf) + grid_inf)
        grid_controls = reshape_pt1(torch.rand(
            (nb_points, self.nu), device=self.device)
                                    * (umax - umin) + umin)
        return grid, grid_controls

    def create_true_predicted_grid(self, grid, grid_controls):
        true_predicted_grid = torch.zeros((len(grid), self.n),
                                          device=self.device)
        for idx, x in enumerate(true_predicted_grid):
            control = reshape_pt1(grid_controls[idx])
            x = reshape_pt1(grid[idx])
            controller = \
                lambda t, kwargs, t0, init_control, impose_init_control: \
                    control
            xnext = dynamics_traj(
                x0=x, u=controller, t0=self.t0, dt=self.dt,
                init_control=control, discrete=self.discrete,
                version=self.true_dynamics, meas_noise_var=0,
                process_noise_var=0, method=self.simu_solver,
                t_eval=[self.dt], kwargs=self.config, lightning=True)
            true_predicted_grid[idx] = reshape_pt1(xnext)
        return true_predicted_grid

    def create_rollout_list(self):
        rollout_list = []
        if self.constrain_u:
            umin = np.min(self.constrain_u)
            umax = np.max(self.constrain_u)
        else:
            logging.warning('No interval was predefined by the user for one '
                            'step ahead model evaluation and rollouts, '
                            'so using min and max of control data.')
            umin = torch.min(self.U_train, dim=0).values
            umax = torch.max(self.U_train, dim=0).values
        # TODO handling of rollouts is slow (for loop instead of parallel
        #  simulation), should run them all in parallel like regular NODE!
        u0 = torch.zeros((self.nb_rollouts, 1, self.nu))
        controllers = Control_from_dict(self.rollout_controller, u0,
                                        [umin, umax])
        i = 0
        while i < len(controllers.controller_functions):
            time_vector = torch.arange(
                0., self.rollout_length * self.dt, step=self.dt)
            init_state = reshape_pt1(torch.rand((1, self.n)) * (
                    self.grid_sup - self.grid_inf) + self.grid_inf)
            if 'Reverse_Duffing' in self.system and torch.abs(
                    init_state[:, 0]).numpy() < 0.1:
                # If trajectory starts too close to 0, ignore this rollout
                logging.warning(
                    'Ignored a rollout with initial state ' + str(
                        init_state) + ' too close to 0')
                continue
            # Define control_traj depending on current controller
            control_traj = controllers[i](
                time_vector, self.config.rollout_controller_args,
                self.t0, u0)
            if 'Earthquake_building_extended' in self.system:
                # Initialize the extended states using constant control
                init_state[:, -2] = control_traj[0, 0]
                init_state[:, -1] = torch.tensor([0.])
            t_u = torch.cat((reshape_dim1(time_vector), reshape_dim1(
                control_traj)), dim=1)
            controller = interpolate_func(
                x=t_u, t0=time_vector[0],
                init_value=reshape_pt1(control_traj[0]))
            true_mean = dynamics_traj(
                x0=init_state, u=controller, t0=time_vector[0],
                dt=self.dt, init_control=reshape_pt1(control_traj[0]),
                discrete=self.discrete, version=self.true_dynamics,
                meas_noise_var=0, process_noise_var=0,
                method=self.simu_solver, t_eval=time_vector,
                kwargs=self.config, lightning=True)
            max = torch.max(torch.abs(true_mean))
            if max.numpy() > self.max_rollout_value:
                # If true trajectory diverges, ignore this rollout
                logging.warning(
                    'Ignored a rollout with diverging true '
                    'trajectory, with initial state ' + str(init_state) +
                    ' and maximum reached absolute value ' + str(max))
                continue
            rollout_list.append([init_state, control_traj, true_mean])
            i += 1
        return rollout_list

    def read_rollout_list(self, results_folder, nb_rollouts, step,
                          folder_title=None, save=None):
        if not folder_title:
            folder = os.path.join(results_folder, 'Rollouts_' + str(step))
        else:
            folder = os.path.join(results_folder, folder_title + '_' +
                                  str(step))
        rollout_list = []
        for i in range(nb_rollouts):
            rollout_folder = os.path.join(folder, 'Rollout_' + str(i))
            filename = 'Init_state.csv'
            data = pd.read_csv(os.path.join(rollout_folder, filename), sep=',',
                               header=None)
            init_state = data.drop(data.columns[0], axis=1).values
            filename = 'Control_traj.csv'
            data = pd.read_csv(os.path.join(rollout_folder, filename), sep=',',
                               header=None)
            control_traj = data.drop(data.columns[0], axis=1).values
            filename = 'True_traj.csv'
            data = pd.read_csv(os.path.join(rollout_folder, filename), sep=',',
                               header=None)
            true_mean = data.drop(data.columns[0], axis=1).values
            if save:
                # Save these variables to another folder
                if not folder_title:
                    save_folder = os.path.join(save, 'Rollouts_' + str(step))
                else:
                    save_folder = os.path.join(
                        save, folder_title + '_' + str(step))
                save_rollout_folder = os.path.join(
                    save_folder, 'Rollout_' + str(i))
                os.makedirs(save_rollout_folder, exist_ok=True)
                filename = 'Init_state.csv'
                file = pd.DataFrame(init_state)
                file.to_csv(os.path.join(save_rollout_folder, filename),
                            header=False)
                filename = 'Control_traj.csv'
                file = pd.DataFrame(control_traj)
                file.to_csv(os.path.join(save_rollout_folder, filename),
                            header=False)
                filename = 'True_traj.csv'
                file = pd.DataFrame(true_mean)
                file.to_csv(os.path.join(save_rollout_folder, filename),
                            header=False)
            rollout_list.append([
                torch.as_tensor(init_state, device=self.X_train.device),
                torch.as_tensor(control_traj, device=self.X_train.device),
                torch.as_tensor(true_mean, device=self.X_train.device)])
        return rollout_list

    def evaluate_model(self, scale=True):
        # Record RMSE over grid
        # Avoid if ground_truth_approx, only have measurement data and would
        # need to wait for recognition to converge to x0, no sense here
        if self.ground_truth_approx:
            return torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]), \
                   torch.zeros_like(self.true_predicted_grid)

        with torch.no_grad():
            controller = \
                lambda t, kwargs, t0, init_control, impose_init_control: \
                    self.grid_controls
            x = reshape_dim1(self.grid)
            xnext = self.NODE_model.forward_traj(
                x, controller, torch.tensor([0.]), torch.tensor([self.dt]),
                self.grid_controls)
            predicted_grid = torch.squeeze(xnext, dim=1)
            if scale:
                l2_error_array = torch.square(
                    self.scaler_Y.transform(self.true_predicted_grid) -
                    self.scaler_Y.transform(predicted_grid))
            else:
                l2_error_array = torch.square(
                    self.true_predicted_grid - predicted_grid)
            self.variables['l2_error_array'] = l2_error_array.detach()
            RMSE_array_dim = torch.sqrt(torch.mean(l2_error_array, dim=0))
            RMSE = RMS(self.true_predicted_grid - predicted_grid)
            var = torch.var(self.true_predicted_grid)  # det(covar) better
            SRMSE = RMSE / var
            self.grid_RMSE = torch.cat((self.grid_RMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), RMSE]))), dim=0)
            self.grid_SRMSE = torch.cat((self.grid_SRMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), SRMSE]))), dim=0)
            return RMSE_array_dim, RMSE, SRMSE, predicted_grid

    def evaluate_rollouts(self, input_rollout_list, only_prior=False,
                          plots=False):
        if len(input_rollout_list) == 0:
            return 0
        if not plots:
            plots = self.monitor_experiment
        # Rollout several trajectories from random start with random control,
        # get mean prediction error over whole trajectory
        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_RMSE_init, \
        rollout_SRMSE_init, rollout_RMSE_output, rollout_SRMSE_output = \
            run_rollouts_NODE(self, input_rollout_list, only_prior=only_prior)
        self.specs['nb_rollouts'] = len(input_rollout_list)
        self.specs['rollout_length'] = self.rollout_length
        self.specs['rollout_RMSE'] = rollout_RMSE
        self.specs['rollout_SRMSE'] = rollout_SRMSE
        self.specs['rollout_RMSE_init'] = rollout_RMSE_init
        self.specs['rollout_SRMSE_init'] = rollout_SRMSE_init
        self.specs['rollout_RMSE_output'] = rollout_RMSE_output
        self.specs['rollout_SRMSE_output'] = rollout_SRMSE_output
        self.rollout_RMSE = \
            torch.cat((self.rollout_RMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx), rollout_RMSE]))),
                      dim=0)
        self.rollout_SRMSE = \
            torch.cat((self.rollout_SRMSE, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_SRMSE]))), dim=0)
        self.rollout_RMSE_init = \
            torch.cat((self.rollout_RMSE_init, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_RMSE_init]))), dim=0)
        self.rollout_SRMSE_init = \
            torch.cat((self.rollout_SRMSE_init, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_SRMSE_init]))), dim=0)
        self.rollout_RMSE_output = \
            torch.cat((self.rollout_RMSE_output, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_RMSE_output]))), dim=0)
        self.rollout_SRMSE_output = \
            torch.cat((self.rollout_SRMSE_output, reshape_pt1(
                torch.tensor([torch.tensor(self.sample_idx),
                              rollout_SRMSE_output]))), dim=0)
        self.variables['rollout_RMSE'] = self.rollout_RMSE
        self.variables['rollout_SRMSE'] = self.rollout_SRMSE
        self.variables['rollout_RMSE_init'] = self.rollout_RMSE_init
        self.variables['rollout_SRMSE_init'] = self.rollout_SRMSE_init
        self.variables['rollout_RMSE_output'] = self.rollout_RMSE_output
        self.variables['rollout_SRMSE_output'] = self.rollout_SRMSE_output
        if self.nb_rollouts > 0:
            complete_rollout_list = concatenate_lists(input_rollout_list,
                                                      rollout_list)
            save_rollout_variables(
                self, self.results_folder, self.nb_rollouts,
                complete_rollout_list, step=self.step, results=True,
                ground_truth_approx=self.ground_truth_approx,
                plots=plots, NODE=True)

    def evaluate_EKF_rollouts(self, input_rollout_list, only_prior=False,
                              plots=False):
        if len(input_rollout_list) == 0:
            return 0
        if not plots:
            plots = self.monitor_experiment
        # Rollout several trajectories from random start with random control,
        # get mean prediction error over whole trajectory
        rollout_list, rollout_RMSE, rollout_SRMSE, rollout_RMSE_init, \
        rollout_SRMSE_init, rollout_RMSE_output, rollout_SRMSE_output = \
            run_rollouts_NODE(self, input_rollout_list, only_prior=only_prior,
                              type='EKF')
        self.specs['nb_EKF_rollouts'] = len(input_rollout_list)
        self.specs['EKF_rollout_length'] = self.rollout_length
        self.specs['EKF_rollout_RMSE'] = rollout_RMSE
        self.specs['EKF_rollout_SRMSE'] = rollout_SRMSE
        self.specs['EKF_rollout_RMSE_init'] = rollout_RMSE_init
        self.specs['EKF_rollout_SRMSE_init'] = rollout_SRMSE_init
        self.specs['EKF_rollout_RMSE_output'] = rollout_RMSE_output
        self.specs['EKF_rollout_SRMSE_output'] = rollout_SRMSE_output
        if self.nb_rollouts > 0:
            complete_rollout_list = concatenate_lists(input_rollout_list,
                                                      rollout_list)
            save_rollout_variables(
                self, self.results_folder, self.nb_rollouts,
                complete_rollout_list, step=self.step, results=True,
                ground_truth_approx=self.ground_truth_approx,
                plots=plots, title='EKF_rollouts', NODE=True)

    def view_difftraj(self, X, U=None, nb_difftraj=None):
        # View X, U as (nb_difftraj * N, n) easier for slicing and saving
        if nb_difftraj is None:
            nb_difftraj = self.nb_difftraj
        X = X.contiguous().view(nb_difftraj * len(self.t_eval), self.p)
        if U is None:
            return X
        else:
            U = U.contiguous().view(nb_difftraj * len(self.t_eval), self.nu)
            return X, U

    def unview_difftraj(self, X, U=None, nb_difftraj=None):
        # View X, U back as (nb_difftraj, N, n)
        if nb_difftraj is None:
            nb_difftraj = self.nb_difftraj
        X = X.contiguous().view(nb_difftraj, len(self.t_eval), self.p)
        if U is None:
            return X
        else:
            U = U.contiguous().view(nb_difftraj, len(self.t_eval), self.nu)
            return X, U

    def save_grid_variables(self, grid, grid_controls, true_predicted_grid,
                            results_folder):
        if torch.is_tensor(grid):
            grid = pd.DataFrame(grid.cpu().numpy())
            grid.to_csv(os.path.join(results_folder, 'Evaluation_grid.csv'),
                        header=False)
            grid_controls = pd.DataFrame(grid_controls.cpu().numpy())
            grid_controls.to_csv(os.path.join(
                results_folder, 'Grid_controls.csv'), header=False)
            true_predicted_grid = pd.DataFrame(
                true_predicted_grid.cpu().numpy())
            true_predicted_grid.to_csv(os.path.join(
                results_folder, 'True_predicted_grid.csv'), header=False)
        else:
            grid = pd.DataFrame(grid)
            grid.to_csv(os.path.join(results_folder, 'Evaluation_grid.csv'),
                        header=False)
            grid_controls = pd.DataFrame(grid_controls)
            grid_controls.to_csv(os.path.join(
                results_folder, 'Grid_controls.csv'), header=False)
            true_predicted_grid = pd.DataFrame(true_predicted_grid)
            true_predicted_grid.to_csv(os.path.join(
                results_folder, 'True_predicted_grid.csv'), header=False)

    def save_model_folder(self, checkpoint_path=None):
        # At end of training, retrieve model from best checkpoint, evaluate
        # it and save it with pickle (general)

        if checkpoint_path is not None:
            # Loading from checkpoint creates a new instance of class so either
            # need args of __init__, or self.save_hyperparameters() in __init__
            # https://pytorch-lightning.readthedocs.io/en/stable/common/weights_loading.html
            # https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint
            # Instead of new instance, can just reload state_dict of checkpoint
            # https://stackoverflow.com/questions/63243359/loading-model-from-checkpoint-is-not-working
            # https://github.com/PyTorchLightning/pytorch-lightning/issues/3426
            checkpoint_model = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint_model['state_dict'])

        # Save train/val/test data
        for key, val in self.variables.items():
            if any(key.startswith(k) for k in ('X_', 'U_')):
                filename = str(key) + '.csv'
                file = pd.DataFrame(val.cpu().numpy())
                file.to_csv(os.path.join(self.results_folder, filename),
                            header=False)

        # Put training and validation data back together for plots/evaluation
        if self.validation:
            split_idx = torch.cat((
                torch.as_tensor(self.train_idx, device=self.X_train.device),
                torch.as_tensor(self.val_idx, device=self.X_train.device)))
            self.train_val_idx, self.train_val_args = torch.sort(split_idx)

            self.X_train = torch.index_select(
                torch.cat((self.X_train, self.X_val)), dim=0,
                index=self.train_val_args)
            self.U_train = torch.index_select(
                torch.cat((self.U_train, self.U_val)), dim=0,
                index=self.train_val_args)
            self.init_state_estim = torch.index_select(torch.cat(
                (self.init_state_estim, self.init_state_estim_val)), dim=0,
                index=self.train_val_args)
            if self.init_state_model:
                self.init_state_obs = torch.index_select(torch.cat(
                    (self.init_state_obs, self.init_state_obs_val)), dim=0,
                    index=self.train_val_args)
            if self.difftraj:
                self.nb_difftraj = self.nb_difftraj + self.nb_difftraj_val
        else:
            self.train_val_idx = self.train_idx

        # Update initial state estimations
        if self.init_state_model:
            if 'optimD' in self.init_state_obs_method:
                # Make optimD recog a standard KKL recog again (no KKL ODE)
                self.model.init_state_KKL_Dscaled.requires_grad = False
                self.init_state_model.KKL_ODE_model.init_state_KKL.D = \
                    self.init_state_model.KKL_ODE_model.init_state_KKL.D.detach()
                self.config['z_config']['D'] = \
                    self.init_state_model.KKL_ODE_model.init_state_KKL.D
                self.init_state_model.defunc.set_idx(self.train_val_idx)
                self.init_state_obs = \
                    self.init_state_model.simulate_zu(self.init_state_obs)
                self.init_state_model.defunc.reset_idx()
                if self.ground_truth_approx:
                    self.init_state_model.defunc.set_idx(self.test_idx)
                    self.init_state_obs_test = \
                        self.init_state_model.simulate_zu(
                            self.init_state_obs_test)
                    self.init_state_model.defunc.reset_idx()
                    for i in range(len(self.rollout_list)):
                        self.rollout_list[i][0] = reshape_pt1(
                            self.init_state_obs_test[i, 0])
                self.init_state_model = self.init_state_model.init_state_model
                self.config['init_state_obs_method'] = \
                    self.init_state_obs_method.split('_optimD')[0]
                self.specs['z_config']['D'] = self.config['z_config']['D']
                if self.monitor_experiment:
                    name = 'OptimD_eigvals.pdf'
                    eig0 = torch.linalg.eigvals(
                        self.specs['z_config']['init_D'])
                    eig = torch.linalg.eigvals(self.specs['z_config']['D'])
                    plt.plot(eig0.real, eig0.imag, 'x', label='Initial')
                    plt.plot(eig.real, eig.imag, 'o', label='Final')
                    plt.title('Eigenvalues of optimized D')
                    plt.xlabel(r'$\mathbb{R}$')
                    plt.ylabel(r'$i\mathbb{R}$')
                    plt.legend()
                    plt.savefig(os.path.join(self.results_folder, name),
                                bbox_inches='tight')
                    plt.clf()
                    plt.close('all')
            self.init_state_estim = reshape_pt1(
                self.init_state_model(self.init_state_obs))
            self.specs['init_state_estim'] = self.init_state_estim
            if self.ground_truth_approx:
                self.init_state_estim_test = reshape_pt1(
                    self.init_state_model(self.init_state_obs_test))
                self.specs['init_state_estim_test'] = self.init_state_estim_test

        # Evaluate experiment
        if self.monitor_experiment:
            l2_error, RMSE, SRMSE, self.grid_variables['Predicted_grid'] = \
                self.evaluate_model()
            self.grid_variables['Predicted_grid'] = self.grid_variables[
                'Predicted_grid'].detach()
            self.specs['l2_error'] = l2_error
            self.specs['grid_RMSE'] = RMSE
            self.specs['grid_SRMSE'] = SRMSE
            filename = 'Predicted_grid.csv'
            file = pd.DataFrame(
                self.grid_variables['Predicted_grid'].cpu().numpy())
            file.to_csv(os.path.join(self.results_folder, filename),
                        header=False)
        self.evaluate_rollouts(self.rollout_list)

        # Update all evaluation variables
        self.variables['Computation_time'] = self.time
        self.variables['train_loss'] = self.train_loss
        self.variables['val_loss'] = self.val_loss
        self.variables['grid_RMSE'] = self.grid_RMSE
        self.variables['grid_SRMSE'] = self.grid_SRMSE
        self.variables['rollout_RMSE'] = self.rollout_RMSE
        self.variables['rollout_SRMSE'] = self.rollout_SRMSE
        self.variables['rollout_RMSE_init'] = self.rollout_RMSE_init
        self.variables['rollout_SRMSE_init'] = self.rollout_SRMSE_init
        self.variables['rollout_RMSE_output'] = self.rollout_RMSE_output
        self.variables['rollout_SRMSE_output'] = self.rollout_SRMSE_output

        # Save all variables
        if self.monitor_experiment:
            for key, val in self.variables.items():
                if any(key.startswith(k) for k in ('X_', 'U_')):
                    continue  # already saved
                if key == 'train_idx':
                    filename = 'train_idx.csv'
                    file = pd.DataFrame(np.array(self.train_idx))
                    file.to_csv(os.path.join(self.results_folder, filename),
                                header=False)
                    continue
                elif key == 'val_idx':
                    filename = 'val_idx.csv'
                    file = pd.DataFrame(np.array(self.val_idx))
                    file.to_csv(os.path.join(self.results_folder, filename),
                                header=False)
                    continue
                elif key == 'test_idx':
                    filename = 'test_idx.csv'
                    file = pd.DataFrame(np.array(self.test_idx))
                    file.to_csv(os.path.join(self.results_folder, filename),
                                header=False)
                    continue
                if key.startswith('test_') or ('split' in key) or (
                        key.startswith('val_') and not key == 'val_loss'):
                    # Avoid saving all test and validation variables
                    continue
                filename = str(key) + '.csv'
                file = pd.DataFrame(val.cpu().numpy())
                file.to_csv(os.path.join(self.results_folder, filename),
                            header=False)

        # Save specs, including final init_state_estim
        specs_file = os.path.join(self.results_folder, 'Specifications.txt')
        with open(specs_file, 'w') as f:
            print(sys.argv[0], file=f)
            for key, val in self.specs.items():
                print(key, ': ', val, file=f)
        with open(self.results_folder + '/model.pkl', 'wb') as f:
            pkl.dump(self.model, f, protocol=4)
        with open(self.results_folder + '/Learn_NODE.pkl', 'wb') as f:
            pkl.dump(self, f, protocol=4)
        logging.info(f'Saved model in {self.results_folder}')

    def save_model(self, checkpoint_path=None):
        # Save general model, put training and validation data back together
        with torch.no_grad():
            self.save_model_folder(checkpoint_path=checkpoint_path)
            if self.monitor_experiment:
                # Plot results on train trajectory and of evaluation
                if self.ground_truth_approx:
                    xtraj_estim = self.NODE_model.forward_traj(
                        self.init_state_estim, self.controller[
                            self.train_val_idx], self.t0, self.t_eval,
                        self.init_control)
                    y_observed = self.X_train
                else:
                    xtraj_estim = self.NODE_model.forward_traj(
                        self.init_state_estim, self.controller, self.t0,
                        self.t_eval, self.init_control)
                    y_observed = self.observe_data_x(self.X_train)
                y_pred = self.observe_data_x(xtraj_estim)
                # Plots
                name = 'Loss.pdf'
                plt.plot(self.train_loss, '+-', label='loss')
                plt.title('Training loss over time')
                plt.yscale('log')
                plt.legend()
                plt.savefig(os.path.join(self.results_folder, name),
                            bbox_inches='tight')
                plt.clf()
                plt.close('all')

                if self.validation:
                    name = 'Val_loss.pdf'
                    plt.plot(self.val_loss, '+-', label='loss')
                    plt.title('Validation loss over time')
                    plt.yscale('log')
                    plt.legend()
                    plt.savefig(os.path.join(self.results_folder, name),
                                bbox_inches='tight')
                    plt.clf()
                    plt.close('all')

                name = 'Computation_time.pdf'
                plt.plot(self.time, '+-', label='time')
                plt.title('Computation time')
                plt.yscale('log')
                plt.legend()
                plt.savefig(os.path.join(self.results_folder, name),
                            bbox_inches='tight')
                plt.clf()
                plt.close('all')
                os.makedirs(os.path.join(self.results_folder, 'Training_trajs'))

                plot_NODE(self, grid=torch.cat((
                    self.grid_variables['Evaluation_grid'],
                    self.grid_variables['Grid_controls']), dim=1),
                          verbose=self.verbose)
                if not self.ground_truth_approx:
                    plot_model_evaluation(
                        self.grid_variables['Evaluation_grid'],
                        self.grid_variables['Grid_controls'],
                        self.grid_variables['Predicted_grid'],
                        self.grid_variables['True_predicted_grid'],
                        self.results_folder,
                        ground_truth_approx=self.ground_truth_approx,
                        l2_error_array=torch.mean(
                            self.variables['l2_error_array'], dim=1),
                        verbose=False)
                # Back to CPU for final plots
                self.X_train, self.t_eval, y_observed, y_pred, self.U_train = \
                    self.X_train.cpu(), self.t_eval.cpu(), y_observed.cpu(), \
                    y_pred.cpu(), self.U_train.cpu()
                if not self.partial_obs:
                    for i in range(xtraj_estim.shape[1]):
                        name = 'Training_trajs/xtraj_estim' + str(i) + '.pdf'
                        plt.plot(self.t_eval, self.X_train[:, i], label='True')
                        plt.plot(self.t_eval, xtraj_estim.detach()[:, i],
                                 label='Estimated')
                        plt.title('State trajectory')
                        plt.legend()
                        plt.savefig(os.path.join(self.results_folder, name),
                                    bbox_inches='tight')
                        plt.clf()
                        plt.close('all')
                for i in range(y_pred.shape[1]):
                    name = 'Training_trajs/y_pred' + str(i) + '.pdf'
                    plt.plot(self.t_eval, y_observed[:, i], label='True')
                    plt.plot(self.t_eval, y_pred.detach()[:, i],
                             label='Estimated')
                    plt.title('Output')
                    plt.legend()
                    plt.savefig(os.path.join(self.results_folder, name),
                                bbox_inches='tight')
                    plt.clf()
                    plt.close('all')
                if not self.no_control:
                    for i in range(self.U_train.shape[2]):
                        name = 'Training_trajs/u_train' + str(i) + '.pdf'
                        plt.plot(self.t_eval, self.U_train[:, i], label='u')
                        plt.title('Control input')
                        plt.legend()
                        plt.savefig(os.path.join(self.results_folder, name),
                                    bbox_inches='tight')
                        plt.clf()
                        plt.close('all')


class Learn_NODE_difftraj(Learn_NODE):
    """
    Learner but train on different training trajectories of given size,
    i.e. on several trajectories x0 -> xN of same length

    All tensors concerning trajectories assumed of size nb_difftraj x N x n,
    i.e. all trajectories assumed of same length with same sampling times!
    Necessary for parallel simulations.
    """

    def __init__(self, X_train, U_train, submodel: nn.Module, config: Config,
                 sensitivity='autograd', ground_truth_approx=False,
                 validation=True, dataset_on_GPU=False):
        self.difftraj = True
        super().__init__(X_train=X_train, U_train=U_train, submodel=submodel,
                         config=config, sensitivity=sensitivity,
                         ground_truth_approx=ground_truth_approx,
                         validation=validation, dataset_on_GPU=dataset_on_GPU)

    def train_dataloader(self):
        # Data = (idx, observations). If partial observations, use X_train =
        # observations and observation function in self.model.forward_traj_obs
        # Minibatches are taken over difftraj
        train_dataset = TensorDataset(torch.arange(self.X_train.shape[0]),
                                      self.X_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.optim_minibatch_size,
            shuffle=self.optim_shuffle)
        return train_loader

    def training_step(self, batch, batch_idx):
        # Forward pass NODE x0 -> xN: get difftrajs matching batch, compute loss
        idx_batch, y_batch = batch
        if self.init_state_model:
            init_state_estim = self.init_state_obs[idx_batch]
        else:
            init_state_estim = self.init_state_estim[idx_batch]
        # Set idx of controller in NODE_model: given a list of controllers,
        # only simulate controllers of this batch
        if self.validation:  # TODO slow to compute u[idx_batch](t)!! What now?
            controller_idx = self.train_idx[idx_batch.tolist()]
        else:
            controller_idx = idx_batch
        self.NODE_model.defunc.set_idx(controller_idx)
        if self.init_state_model:
            if 'optimD' in self.init_state_obs_method:
                self.init_state_model.defunc.set_idx(controller_idx)
        if (self.config.KKL_l2_reg is not None) and (
                'optimD' in self.init_state_obs_method):
            # Compute KKL traj to keep it for loss, then use same simulation
            # to compute z(T) -> x0 -> x(t)
            KKL_traj = self.init_state_model.simulate_ztraj(
                init_state_estim[...,
                :self.init_state_model.KKL_ODE_model.n],
                self.init_state_model.z_t_eval)
            z = torch.squeeze(self.init_state_model.simulate_zu(
                init_state_estim, ztraj=KKL_traj))
            init_state_estim = reshape_pt1(
                self.init_state_model.init_state_model(z))
            x_estim = self.NODE_model(init_state_estim)
        else:
            x_estim = self.model(init_state_estim)
            if self.config.KKL_l2_reg is not None:
                KKL_traj = self.config.KKL_traj[idx_batch]
            else:
                KKL_traj = None
        y_estim = self.observe_data_x(x_estim)
        if self.no_control:
            xu_estim = x_estim
        else:
            u_estim = self.config.controller[controller_idx](
                self.t_eval, self.config, self.t0, self.init_control)
            xu_estim = torch.cat((x_estim, u_estim), dim=-1)
        losses = self.NODE_model.loss(
            y_estim=y_estim, y_true=y_batch, xu_estim=xu_estim,
            KKL_traj=KKL_traj, scaler_Y=self.scaler_Y, scaler_X=self.scaler_X)
        loss = sum(losses.values())
        self.train_loss = torch.cat((
            self.train_loss, torch.tensor([loss.detach()])))
        self.time = torch.cat((
            self.time, torch.tensor([[time.time() - self.init_time]])), dim=0)
        logs = {'train_loss': loss.detach()}
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        for key, val in losses.items():
            logs.update({key: val.detach()})
            self.log(key, val, prog_bar=False, logger=True)
        self.NODE_model.defunc.reset_idx()
        if self.init_state_model:
            if 'optimD' in self.init_state_obs_method:
                self.init_state_model.defunc.reset_idx()
        return {'loss': loss, 'log': logs}

    def val_dataloader(self):
        # Validation is same as training, but on validation data if exists
        # (otherwise just training loss again, used for early stopping)
        if self.validation:
            val_dataset = TensorDataset(torch.arange(self.X_val.shape[0]),
                                        self.X_val)
            val_loader = DataLoader(
                val_dataset, batch_size=self.optim_minibatch_size,
                shuffle=self.optim_shuffle)
        else:
            val_dataset = TensorDataset(torch.arange(self.X_train.shape[0]),
                                        self.X_train)
            val_loader = DataLoader(
                val_dataset, batch_size=self.optim_minibatch_size,
                shuffle=self.optim_shuffle)
        return val_loader

    def validation_step(self, batch, batch_idx):
        # For now validation same as training, just for early stopping
        with torch.no_grad():
            if self.validation:
                idx_batch, y_batch = batch
                if self.init_state_model:
                    init_state_estim = self.init_state_obs_val[idx_batch]
                else:
                    init_state_estim = self.init_state_estim_val[idx_batch]
                # Set idx of controller in NODE_model: given a list of
                # controllers, only simulate controllers of this batch
                controller_idx = self.val_idx[idx_batch.tolist()]
                self.NODE_model.defunc.set_idx(controller_idx)
                if self.init_state_model:
                    if 'optimD' in self.init_state_obs_method:
                        self.init_state_model.defunc.set_idx(controller_idx)
                if (self.config.KKL_l2_reg is not None) and (
                        'optimD' in self.init_state_obs_method):
                    # Compute KKL traj to keep it for loss, then use same
                    # simulation to compute z(T) -> x0 -> x(t)
                    KKL_traj = self.init_state_model.simulate_ztraj(
                        init_state_estim[...,
                        :self.init_state_model.KKL_ODE_model.n],
                        self.init_state_model.z_t_eval)
                    z = torch.squeeze(self.init_state_model.simulate_zu(
                        init_state_estim, ztraj=KKL_traj))
                    init_state_estim = reshape_pt1(
                        self.init_state_model.init_state_model(z))
                    x_estim = self.NODE_model(init_state_estim)
                else:
                    x_estim = self.model(init_state_estim)
                    if self.config.KKL_l2_reg is not None:
                        KKL_traj = self.config.KKL_traj[idx_batch]
                    else:
                        KKL_traj = None
                y_estim = self.observe_data_x(x_estim)
                if self.no_control:
                    xu_estim = x_estim
                else:
                    u_estim = self.config.controller[controller_idx](
                        self.t_eval, self.config, self.t0, self.init_control)
                    xu_estim = torch.cat((x_estim, u_estim), dim=-1)
                losses = self.NODE_model.loss(
                    y_estim=y_estim, y_true=y_batch, xu_estim=xu_estim,
                    KKL_traj=KKL_traj, scaler_Y=self.scaler_Y,
                    scaler_X=self.scaler_X)
                loss = sum(losses.values())
                self.log('val_loss', loss, on_step=True, prog_bar=True,
                         logger=True)
                self.val_loss = torch.cat((self.val_loss, torch.tensor([loss])))
                logs = {'val_loss': loss.detach()}
                for key, val in losses.items():
                    logs.update({key: val.detach()})
                    self.log(key, val, prog_bar=False, logger=True)
                self.NODE_model.defunc.reset_idx()
                if self.init_state_model:
                    if 'optimD' in self.init_state_obs_method:
                        self.init_state_model.defunc.reset_idx()
            else:
                if len(self.train_loss) == 0:
                    loss = torch.tensor(np.nan)
                else:
                    loss = self.train_loss[-1]
                self.log('val_loss', loss, on_step=True, prog_bar=True,
                         logger=True)
                logs = {'val_loss': loss.detach()}
            return {'loss': loss, 'log': logs}

    def save_model(self, checkpoint_path=None):
        # Save some more plots (can be easily overwritten)
        with torch.no_grad():
            self.variables['X_train'], self.variables['U_train'] = \
                self.view_difftraj(
                    self.variables['X_train'], self.variables['U_train'])
            if self.validation:
                self.variables['X_val'], self.variables['U_val'] = \
                    self.view_difftraj(
                        self.variables['X_val'], self.variables['U_val'],
                        self.nb_difftraj_val)
            if self.ground_truth_approx:
                self.variables['X_test'], self.variables['U_test'] = \
                    self.view_difftraj(
                        self.variables['X_test'], self.variables['U_test'],
                        self.nb_difftraj_test)
            # Save general model, put training and validation data back together
            self.save_model_folder(checkpoint_path=checkpoint_path)
            if self.validation:
                self.variables['X_train'] = self.X_train
                self.variables['U_train'] = self.U_train
            else:
                self.variables['X_train'], self.variables['U_train'] = \
                    self.unview_difftraj(self.X_train, self.U_train)

            if self.monitor_experiment:
                # Plot results on train trajectory and of evaluation
                if self.ground_truth_approx:
                    xtraj_estim = self.NODE_model.forward_traj(
                        self.init_state_estim, self.controller[
                            self.train_val_idx], self.t0, self.t_eval,
                        self.init_control)
                    y_observed = self.X_train
                else:
                    xtraj_estim = self.NODE_model.forward_traj(
                        self.init_state_estim, self.controller, self.t0,
                        self.t_eval, self.init_control)
                    y_observed = self.observe_data_x(self.X_train)
                y_pred = self.observe_data_x(xtraj_estim)
                # Plots
                name = 'Loss.pdf'
                plt.plot(self.train_loss, '+-', label='loss')
                plt.title('Training loss over time')
                plt.yscale('log')
                plt.legend()
                plt.savefig(os.path.join(self.results_folder, name),
                            bbox_inches='tight')
                plt.clf()
                plt.close('all')

                if self.validation:
                    name = 'Val_loss.pdf'
                    plt.plot(self.val_loss, '+-', label='loss')
                    plt.title('Validation loss over time')
                    plt.yscale('log')
                    plt.legend()
                    plt.savefig(os.path.join(self.results_folder, name),
                                bbox_inches='tight')
                    plt.clf()
                    plt.close('all')

                name = 'Computation_time.pdf'
                plt.plot(self.time, '+-', label='time')
                plt.title('Computation time')
                plt.yscale('log')
                plt.legend()
                plt.savefig(os.path.join(self.results_folder, name),
                            bbox_inches='tight')
                plt.clf()
                plt.close('all')
                os.makedirs(os.path.join(self.results_folder, 'Training_trajs'))

                plot_NODE(self, grid=torch.cat((
                    self.grid_variables['Evaluation_grid'],
                    self.grid_variables['Grid_controls']), dim=1),
                          verbose=self.verbose)
                if not self.ground_truth_approx:
                    plot_model_evaluation(
                        self.grid_variables['Evaluation_grid'],
                        self.grid_variables['Grid_controls'],
                        self.grid_variables['Predicted_grid'],
                        self.grid_variables['True_predicted_grid'],
                        self.results_folder,
                        ground_truth_approx=self.ground_truth_approx,
                        l2_error_array=torch.mean(
                            self.variables['l2_error_array'], dim=1),
                        verbose=False)
                # Back to CPU for final plots
                self.X_train, self.t_eval, y_observed, y_pred, self.U_train = \
                    self.X_train.cpu(), self.t_eval.cpu(), y_observed.cpu(), \
                    y_pred.cpu(), self.U_train.cpu()
                for j in range(self.nb_difftraj):
                    if not self.partial_obs:
                        for i in range(xtraj_estim.shape[2]):
                            name = 'Training_trajs/xtraj_estim' + str(j) + \
                                   str(i) + '.pdf'
                            plt.plot(self.X_train[j, :, i], label='True')
                            plt.plot(xtraj_estim.detach()[j, :, i],
                                     label='Estimated')
                            plt.title('State trajectory')
                            plt.legend()
                            plt.savefig(os.path.join(self.results_folder, name),
                                        bbox_inches='tight')
                            plt.clf()
                            plt.close('all')
                    for i in range(y_pred.shape[2]):
                        name = 'Training_trajs/y_pred' + str(j) + \
                               str(i) + '.pdf'
                        plt.plot(y_observed[j, :, i], label='True')
                        plt.plot(y_pred.detach()[j, :, i], label='Estimated')
                        plt.title('Output')
                        plt.legend()
                        plt.savefig(os.path.join(self.results_folder, name),
                                    bbox_inches='tight')
                        plt.clf()
                        plt.close('all')
                    if not self.no_control:
                        for i in range(self.U_train.shape[2]):
                            name = 'Training_trajs/u_train' + str(j) + str(i) \
                                   + '.pdf'
                            plt.plot(self.U_train[j, :, i], label='u')
                            plt.title('Control input')
                            plt.legend()
                            plt.savefig(os.path.join(self.results_folder, name),
                                        bbox_inches='tight')
                            plt.clf()
                            plt.close('all')
