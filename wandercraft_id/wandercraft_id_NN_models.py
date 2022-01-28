import control
import numpy as np
import torch
import torch.nn as nn

from simulation.dynamics import dynamics_traj
from utils.utils import reshape_pt1, reshape_dim1_tonormal
from utils.pytorch_utils import StandardScaler


# Classes of NN models used to learn structured NODEs on WDC data
# Contains the prior linear models identified at WDC

# Normalize data in forward function: all that goes into the NN is normalized
# then denormalized. This step is taken into account in grad of output of NN
# so grads are still all good!

# Simple MLP model with n hidden layers. Can pass StandardScaler to
# normalize in and output in forward function.
class MLPn(nn.Module):

    def __init__(self, num_hl, n_in, n_hl, n_out, activation=nn.Tanh(),
                 init=None, init_args={}, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        super(MLPn, self).__init__()
        # Initialize weights using "He Init" if ReLU after, "Xavier" otherwise
        if not init:
            init = nn.init.xavier_uniform_
        # Create ModuleList and add first layer with input dimension
        # Layers: input * activation, hidden * activation, output
        if isinstance(n_hl, int):
            n_hl = [n_hl] * (num_hl + 1)
        layers = nn.ModuleList()
        layers.append(nn.Linear(n_in, n_hl[0]))
        init(layers[-1].weight, *init_args)
        if 'xavier' not in init.__name__:
            init(layers[-1].bias, *init_args)  # not for tensors dim < 2
        # Add num_hl layers of size n_hl with chosen activation
        for i in range(num_hl):
            layers.append(activation)
            layers.append(nn.Linear(n_hl[i], n_hl[i + 1]))
            init(layers[-1].weight, *init_args)
            if 'xavier' not in init.__name__:
                init(layers[-1].bias, *init_args)  # not for tensors dim < 2
        # Append last layer with output dimension (linear activation)
        layers.append(nn.Linear(n_hl[-1], n_out))
        init(layers[-1].weight, *init_args)
        if 'xavier' not in init.__name__:
            init(layers[-1].bias, *init_args)  # not for tensors dim < 2
        self.layers = layers

    def set_scalers(self, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

    def forward(self, x):
        # Compute output through all layers. Normalize in, denormalize out
        if self.scaler_X:
            x = self.scaler_X.transform(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        if self.scaler_Y:
            x = self.scaler_Y.inverse_transform(x)
        return x


# More general recognition model that ignores part of the input
# Returns model(in[idx1:idx2]) from input in
class Recog_ignore(nn.Module):
    def __init__(self, model, idx1, idx2, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        super(Recog_ignore, self).__init__()
        self.model = model
        self.idx1 = idx1
        self.idx2 = idx2

    def set_scalers(self, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

    def forward(self, x):
        if self.scaler_X:
            x = self.scaler_X.transform(x)
        x_res = self.model(x[..., self.idx1:self.idx2])  # outputs x(0)
        if self.scaler_Y:
            x_res = self.scaler_Y.inverse_transform(x_res)
        return x_res


# Simplistic recognition model for WDC data
class WDC_simple_recog4(nn.Module):
    def __init__(self, n, dt):
        super(WDC_simple_recog4, self).__init__()
        self.n = n
        self.dt = dt

    def set_scalers(self, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

    def forward(self, y):
        if len(y.shape) == 3:
            x0 = torch.zeros((y.shape[0], 1, self.n), device=y.device)
        elif len(y.shape) == 2:
            x0 = torch.zeros((y.shape[0], self.n), device=y.device)
        x0[..., 0] = y[..., 0]
        x0[..., 1] = (y[..., 2] - y[..., 0]) / self.dt
        x0[..., 2] = y[..., 0]
        x0[..., 3] = y[..., 1]
        return x0


# Double deformation model identified at WDC
class WDC_two_deformation_model(nn.Module):
    def __init__(self):
        super(WDC_two_deformation_model, self).__init__()
        # Parameters - with the dummy
        J = 4.6
        d = 0.38  # Knee position
        # Patient 52, empty.
        I = 6.06
        I2 = 1.66
        m = 22.3
        m2 = 13.5
        l2 = 0.569 - d
        l_tot = 0.419  # Position of CoM of the whole leg.
        # Recompute m1 parameters from (m1 + m2) and m2
        m1 = m - m2
        l1 = (l_tot * m - (d + l2) * m2) / m1
        # Stiffness, damping.
        k1 = 10000
        k2 = 10000
        nu1 = 100
        nu2 = 100
        nu_m = 40
        g = 9.81
        # Inertia matrix in theta, linearized.
        H = np.array([[I + J, I, I2],
                      [I, I, I2],
                      [I2, I2, I2]])
        H_inv = np.linalg.inv(H)
        # Right-hand side: linearized block.
        # State: theta thetadot alpha1 alpha1dot q qdot
        g1 = m1 * g * l1 + m2 * g * d
        g2 = m2 * g * l2
        rhs = np.array([[-g1, -g1, -g2, - nu_m, 0, 0],
                        [-g1, -g1 - k1, -g2, 0, - nu1, 0],
                        [k2, k2, -g2 - k2, nu2, nu2, - nu2]])
        b = np.array([1, 0, 0])
        Ap = H_inv @ rhs
        self.A = np.vstack((np.zeros((3, 6)), Ap))
        self.A[:3, 3:] = np.identity(3)
        self.A[-1] = np.dot(np.ones(3), Ap)
        self.A[[0, 1, 2, 3, 4, 5], :] = self.A[[0, 3, 1, 4, 2, 5], :]
        self.A[:, [0, 1, 2, 3, 4, 5]] = self.A[:, [0, 3, 1, 4, 2, 5]]
        Bp = H_inv @ b
        self.B = np.hstack((np.zeros((3)), Bp))
        self.B[-1] = np.dot(np.ones(3), Bp)
        self.B[[0, 1, 2, 3, 4, 5]] = self.B[[0, 3, 1, 4, 2, 5]]
        self.C = torch.tensor([[.1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1]])
        self.A = torch.as_tensor(self.A)
        self.b = torch.as_tensor(self.B)
        self.n = self.C.shape[1]
        self.p = self.C.shape[0]

    def dynamics(self, x, u):
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        # Matrix multiplication by A and b along dimension -1 of x and u
        xdot = torch.einsum('ij,k...j->k...i', self.A, x) + \
               torch.einsum('ij,k...j->k...i', self.b, u)
        return xdot

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        device = x.device
        u = reshape_dim1_tonormal(u(t, kwargs, t0, init_control,
                                    impose_init_control))
        xdot = self.dynamics(x, u)
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    def call_withu(self, x):
        u = reshape_pt1(x[..., self.n:])
        x = reshape_pt1(x[..., :self.n])
        return self.dynamics(x, u)


# Single deformation model identified at WDC
class WDC_single_deformation_model(nn.Module):
    def __init__(self):
        super(WDC_single_deformation_model, self).__init__()
        # State = (theta, thetadot, q, qdot)
        # Parameters - with the dummy
        J = 4.6
        d = 0.38  # Knee position
        # Patient 52, empty.
        I = 6.06
        I2 = 1.66
        m = 22.3
        m2 = 13.5
        l2 = 0.569 - d
        l_tot = 0.419  # Position of CoM of the whole leg.
        # Recompute m1 parameters from (m1 + m2) and m2
        m1 = m - m2
        l1 = (l_tot * m - (d + l2) * m2) / m1
        # Stiffness, damping
        k1 = 10000
        k2 = 10000
        nu1 = 100
        nu_m = 40
        g = 9.81
        # Right-hand side: linearized block
        # State: theta thetadot q qdot
        k = k1 * k2 / (k1 + k2)
        nu_f = nu1
        gravity = (m1 * l1 + m2 * l2) * g / I
        self.A = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0],
             [- k / J, - nu_f / J - nu_m / J, k / J, nu_f / J],
             [0.0, 0.0, 0.0, 1.0],
             [k / I, nu_f / I, - gravity - k / I, - nu_f / I]])
        self.b = torch.tensor([[0., 1. / J, 0., 0.]]).T
        self.C = torch.tensor([[1, 0, 0., 0],
                               [0, 0, 0, 1.]])
        self.n = self.C.shape[1]
        self.p = self.C.shape[0]

    def dynamics(self, x, u):
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        # Matrix multiplication by A and b along dimension -1 of x and u
        # xdot = torch.einsum('ij,k...j->k...i', self.A, x) + \
        #        torch.einsum('ij,k...j->k...i', self.b, u)
        xdot = torch.matmul(x, self.A.t()) + torch.matmul(u, self.b.t())
        return xdot

    def dynamics_deriv(self, x, u):
        xdot = self.A.expand(x.shape[0], -1, -1)
        return torch.squeeze(xdot)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        device = x.device
        u = reshape_dim1_tonormal(u(t, kwargs, t0, init_control,
                                    impose_init_control))
        xdot = self.dynamics(x, u)
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    def call_withu(self, x):
        u = reshape_pt1(x[..., self.n:])
        x = reshape_pt1(x[..., :self.n])
        return self.dynamics(x, u)

    # Returns d dynamics / dx (x,u). Useful for EKF!
    def call_deriv(self, t, x, u, t0, init_control, process_noise_var,
                   kwargs, impose_init_control=False):
        device = x.device
        u = reshape_dim1_tonormal(u(t, kwargs, t0, init_control,
                                    impose_init_control))
        xdot = self.dynamics_deriv(x, u)
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    # Predict xt+dt from xt and the function u. Useful for EKF!
    # Simple solver because several steps of that for each EKF step...
    def predict(self, x, u, t0, dt, init_control):
        xtraj = torch.as_tensor(dynamics_traj(
            x0=x, u=u, t0=t0, dt=dt, init_control=init_control,
            discrete=False, version=self, meas_noise_var=0.,
            process_noise_var=0., method='rk4', t_eval=torch.tensor([dt]),
            kwargs={}))
        return xtraj

    # Jacobian of function that predicts xt+dt (or other) from xt and the
    # function u. Useful for EKF!
    def predict_deriv(self, x, f):
        # Compute Jacobian of f with respect to input x
        dfdh = torch.autograd.functional.jacobian(
            f, x, create_graph=False, strict=False, vectorize=False)
        dfdx = torch.squeeze(dfdh)
        return dfdx


# Residuals model for WDC data: output = output of linear model identified at
# WDC(x) + unscale(output NODE(scale x))
class WDC_resmodel(nn.Module):

    def __init__(self, resmodel, config, single=True, scaler_X=None,
                 scaler_Y=None):
        super(WDC_resmodel, self).__init__()
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.resmodel = resmodel
        self.config = config
        if single:
            self.WDC_model = WDC_single_deformation_model()
        else:
            self.WDC_model = WDC_two_deformation_model()

    def set_scalers(self, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.resmodel.set_scalers(self.scaler_X, self.scaler_Y)

    def forward(self, x):
        xdot_WDC = self.WDC_model.call_withu(x)
        xdot_NODE = self.resmodel.forward(x)
        return xdot_WDC + xdot_NODE


# NODE model for WDC data: imposing x1dot = x2, x3dot = x4, learning the
# remaining dynamics
class WDC_x1dotx2_submodel(nn.Module):
    def __init__(self, f, config):
        super(WDC_x1dotx2_submodel, self).__init__()
        self.f = f
        self.config = config

    def set_scalers(self, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        scaler_fY = StandardScaler(
            mean=self.scaler_Y._mean[1:self.config.n:2],
            var=self.scaler_Y._var[1:self.config.n:2])
        self.f.set_scalers(self.scaler_X, scaler_fY)

    def forward(self, x):
        xdot = torch.zeros_like(x[..., :self.config.n])  # shape (N, 1, n)
        xdot[..., 0] = x[..., 1]
        xdot[..., 2] = x[..., 3]
        res = self.f(x).view(tuple(list(xdot.shape[:-1]) + [-1]))
        xdot[..., 1] = res[..., 0]
        xdot[..., 3] = res[..., 1]
        return xdot

# NODE model for WDC data: imposing x1dot = x2, x3dot = x4, u affine in x2dot,
# learning the remaining dynamics
class WDC_x1dotx2_uaffine_submodel(nn.Module):
    def __init__(self, f, k, config):
        super(WDC_x1dotx2_uaffine_submodel, self).__init__()
        self.f = f
        self.k0norm = torch.linalg.norm(k.clone())
        kinit = k.clone() / self.k0norm
        self.k = nn.parameter.Parameter(kinit, requires_grad=True)
        self.config = config

    def set_scalers(self, scaler_X=None, scaler_Y=None):
        self.scaler_Y = scaler_Y
        self.scaler_X = StandardScaler(mean=scaler_X._mean[:self.config.n],
                                       var=scaler_Y._var[:self.config.n])
        scaler_fY = StandardScaler(
            mean=self.scaler_Y._mean[1:self.config.n:2],
            var=self.scaler_Y._var[1:self.config.n:2])
        self.f.set_scalers(self.scaler_X, scaler_fY)

    def forward(self, x):
        u = reshape_pt1(x[..., self.config.n:])
        x = reshape_pt1(x[..., :self.config.n])
        xdot = torch.zeros_like(x[..., :self.config.n])  # shape (N, 1, n)
        xdot[..., 0] = x[..., 1]
        xdot[..., 2] = x[..., 3]
        res = self.f(x).view(tuple(list(xdot.shape[:-1]) + [-1]))
        xdot[..., 1] = res[..., 0] + self.k * self.k0norm * \
                       u.view(xdot[..., 1].shape)
        xdot[..., 3] = res[..., 1]
        return xdot

# NODE model for WDC data: imposing x1dot = x2, x3dot = x4, and learning the
# residuals of the linear prior identified at WDC for the remaining dynamics
# (constraint x1dot = x2 already included in prior model, so impose zero there)
class WDC_x1dotx2_resmodel(nn.Module):
    def __init__(self, resmodel, config, single=True, scaler_X=None,
                 scaler_Y=None):
        super(WDC_x1dotx2_resmodel, self).__init__()
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.resmodel = resmodel
        self.config = config
        if single:
            self.WDC_model = WDC_single_deformation_model()
        else:
            self.WDC_model = WDC_two_deformation_model()

    def set_scalers(self, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        scaler_fY = StandardScaler(
            mean=self.scaler_Y._mean[1:self.config.n:2],
            var=self.scaler_Y._var[1:self.config.n:2])
        self.resmodel.set_scalers(self.scaler_X, scaler_fY)

    def forward(self, x):
        xdot_WDC = self.WDC_model.call_withu(x)  # already contains x1dot = x2
        xdot_NODE = torch.zeros_like(x[..., :self.config.n])  # shape (N, 1, n)
        res = self.resmodel(x).view(tuple(list(xdot_NODE.shape[:-1]) + [-1]))
        xdot_NODE[..., 1] = res[..., 0]
        xdot_NODE[..., 3] = res[..., 1]
        return xdot_WDC + xdot_NODE
