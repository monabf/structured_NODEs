import os

import seaborn as sb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchdiffeq

from simulation.dynamics import dynamics_traj
from utils.config import Config
from utils.utils import MSE, MS
from .defunc_time import DEFunc_time
from .sensitivity_methods import My_Adjoint, My_Adjoint_difftraj

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

# Subclass to learn ODEs with NNs. Optimization problem to train the NN on one
# or several trajectories of solutions: minimize a loss on these
# trajectories by enforcing the ODE as a constraint and computing the loss
# gradient analytically using the forward or adjoint sensitivity method.

# This is the NODE model class, which mimics torchdyn but does not rely on it
# because it makes it difficult to have a control input u(t) and to have
# different kinds of forward passes with different simulations. It defines a
# forward pass that simulates the NODE, a loss depending on the simulation,
# and the sensitivity methods for optimization (from torchdyn.sensitivity)
# https://torchdyn.readthedocs.io/en/latest/tutorials/01_neural_ode_cookbook.html
# https://pytorch-lightning.readthedocs.io/en/1.2.2/starter/converting.html
# https://pytorch-lightning.readthedocs.io/en/0.8.1/lightning-module.html

LOCAL_PATH = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
LOCAL_PATH_TO_SRC = LOCAL_PATH.split(os.sep + 'src', 1)[0]


class NODE(pl.LightningModule):
    """ Default NODE class: train on whole trajectory x0 -> xN
    """
    # Inherit and overwrite forward/loss/adjoint for new models

    def __getattr__(self, item):
        # self.config[item] can be called directly as self.item
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

    def __init__(self, submodel: nn.Module, config: Config, order=1):
        super().__init__()
        self.config = config
        self.submodel = submodel
        # Define the NODE function that holds forward and adjoint backward pass
        self.defunc = DEFunc_time(
            model=self.submodel, controller=self.config.controller,
            t0=self.config.t0, init_control=self.config.init_control,
            config=self.config, sensitivity=self.sensitivity,
            intloss=self.intloss, order=order)
        if self.__class__.__name__ == 'NODE':
            # Specific adjoint for different inherited classes
            if self.sensitivity == 'adjoint':
                # Adjoint does both forward and backward pass!
                self.adjoint = My_Adjoint(self.intloss)

    def forward(self, x):
        # Actually both forward and backward pass: runs forward pass =
        # simulation of xtraj with config options, and retains either custom
        # adjoint or autograd backward pass for optimization
        # Built on torchdyn.models.NeuralDE
        if self.sensitivity == 'adjoint':
            return torch.squeeze(self.adjoint(
                self.defunc, x.to(self.device), self.t_eval.to(self.device),
                **self.optim_solver_options))
        elif self.sensitivity == 'autograd':
            return torch.squeeze(torchdiffeq.odeint(
                self.defunc, x.to(self.device), self.t_eval.to(self.device),
                **self.optim_solver_options))

    def loss(self, y_estim, y_true, xu_estim, KKL_traj=None,
             scaler_Y=None, scaler_X=None):
        # Custom loss function, can be overwritten easily
        # Pay attention to scaling!! All computations involved in
        # optimization should always be done on scaled variables!!
        if scaler_Y is not None:
            y_estim = scaler_Y.transform(y_estim)
            y_true = scaler_Y.transform(y_true)
        # loss = 1. / 2 * torch.sum(torch.square(y_estim - y_true))
        loss = 1. / 2 * MSE(y_estim, y_true)
        # Regularization: less efficient than weight_decay in optim_options
        # https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf 8.5.2
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        losses = {'loss': loss}
        if self.config.l1_reg:
            reg_l1 = self.config.l1_reg / 2. * sum(
                p.abs().sum() for p in self.model.parameters())
            losses.update({'reg_l1': reg_l1})
        if self.config.l2_reg:
            reg_l2 = self.config.l2_reg / 2. * sum(
                p.pow(2).sum() for p in self.model.parameters())
            losses.update({'reg_l2': reg_l2})
        if self.config.prior_l2_reg:
            reg_prior = self.config.prior_l2_reg / 2. * MS(
                self.submodel.resmodel.scaler_Y.transform(
                    self.submodel.resmodel(xu_estim)))
            losses.update({'reg_prior': reg_prior})
        if self.config.KKL_l2_reg is not None:
            n = len(scaler_X._mean)
            x_estim = scaler_X.transform(xu_estim[..., :n])
            if 'optimD' in self.init_state_obs_method:
                x_recog = self.init_state_model.init_state_model(KKL_traj)
            else:
                x_recog = self.init_state_model(KKL_traj)
            reg_KKL = self.config.KKL_l2_reg / 2. * MSE(
                x_estim, scaler_X.transform(x_recog))
            losses.update({'reg_KKL': reg_KKL})
        return losses

    def forward_traj(self, x0, controller, t0, t_eval, init_control,
                     simu_solver=None, ):
        # Simulate traj of observations on given t_eval
        device = x0.device
        t_eval = t_eval.to(device)
        init_control = init_control.to(device)
        self.submodel = self.submodel.to(device)
        if not simu_solver:
            simu_solver = self.simu_solver
        xtraj = dynamics_traj(x0=x0, u=controller, t0=t0, dt=self.dt,
                              init_control=init_control, discrete=self.discrete,
                              version=self.defunc.dyn_NODE, meas_noise_var=0,
                              process_noise_var=0, method=simu_solver,
                              t_eval=t_eval, stay_GPU=True, lightning=True,
                              kwargs=self.config)
        return xtraj

    def forward_traj_obs(self, x0, controller, t0, t_eval, init_control):
        xtraj = self.forward_traj(x0, controller, t0, t_eval, init_control)
        return self.observe_data_x(xtraj)

    # Predict xt+dt from xt and the function u. Useful for EKF!
    # Simple solver because several steps of that for each EKF step...
    def predict(self, x, u, t0, dt, init_control):
        xtraj = self.forward_traj(x0=x, controller=u, t0=t0,
                                  t_eval=torch.tensor([dt]),
                                  init_control=init_control,
                                  simu_solver='rk4')
        return xtraj

    # Jacobian of function that predicts xt+dt (or other) from xt and the
    # function u. Useful for EKF!
    def predict_deriv(self, x, f):
        # Compute Jacobian of f with respect to input x
        dfdh = torch.autograd.functional.jacobian(
            f, x, create_graph=False, strict=False, vectorize=False)
        # dfdx reshape (nb_difftraj, len(t_eval), n)
        dfdx = torch.transpose(torch.transpose(
            torch.diagonal(dfdh, dim1=0, dim2=2), 1, 2), 0, 1)
        return torch.squeeze(dfdx)



class NODE_difftraj(NODE):
    """
    NODE but learn from different training trajectories of given size,
    i.e. on several trajectories x0 -> xN of same length
    """

    def __init__(self, submodel: nn.Module, config: Config, order=1):
        super().__init__(submodel=submodel, config=config, order=order)
        if self.sensitivity == 'adjoint':
            # Adjoint does both forward and backward pass!
            self.adjoint = My_Adjoint_difftraj(self.intloss)

    def forward(self, x):
        # Actually both forward and backward pass: runs forward pass =
        # simulation of xtraj with config options, and retains either custom
        # adjoint or autograd backward pass for optimization
        # Built on torchdyn.models.NeuralDE
        if self.sensitivity == 'adjoint':
            return torch.transpose(torch.squeeze(self.adjoint(
                self.defunc, x.to(self.device), self.t_eval.to(self.device),
                **self.optim_solver_options), dim=2), 0, 1)
        elif self.sensitivity == 'autograd':
            return torch.transpose(torch.squeeze(torchdiffeq.odeint(
                self.defunc, x.to(self.device), self.t_eval.to(self.device),
                **self.optim_solver_options), dim=2), 0, 1)

    def forward_traj(self, x0, controller, t0, t_eval, init_control,
                     simu_solver=None):
        # Simulate traj of observations on given t_eval
        device = x0.device
        t_eval = t_eval.to(device)
        init_control = init_control.to(device)
        self.submodel = self.submodel.to(device)
        if not simu_solver:
            simu_solver = self.simu_solver
        xtraj = dynamics_traj(x0=x0, u=controller, t0=t0, dt=self.dt,
                              init_control=init_control, discrete=self.discrete,
                              version=self.defunc.dyn_NODE, meas_noise_var=0,
                              process_noise_var=0, method=simu_solver,
                              t_eval=t_eval, stay_GPU=True, lightning=True,
                              kwargs=self.config)
        if len(xtraj.shape) == 4:
            return torch.transpose(torch.squeeze(xtraj, dim=2), 0, 1)
        elif len(xtraj.shape) == 3:
            return torch.transpose(xtraj, 0, 1)
        else:
            return xtraj
