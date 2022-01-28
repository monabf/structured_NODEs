import numpy as np
import torch

from utils.utils import reshape_pt1, reshape_pt1_tonormal, reshape_dim1_tonormal

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


# Possible dynamics functions as classes, calling the object of each class
# returns dx/dt


# Dynamics of the continuous time Duffing oscillator, with control law u(t)
class Duffing:

    def __init__(self, device, kwargs):
        self.device = device
        self.alpha = kwargs.get('alpha')
        self.beta = kwargs.get('beta')
        self.delta = kwargs.get('delta')
        self.A = torch.tensor([[0., 1.], [-self.alpha, -self.delta]],
                              device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.duffing_dynamics(t, x, u, t0, init_control,
                                     process_noise_var, kwargs,
                                     impose_init_control)

    def duffing_dynamics(self, t, x, u, t0, init_control, process_noise_var,
                         kwargs, impose_init_control=False):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        F = torch.tensor([
            torch.zeros_like(x[:, 0]),
            - self.beta * torch.pow(x[:, 0], 3) + reshape_pt1_tonormal(u)],
            device=device)
        xdot = torch.matmul(self.A, x.t()).t() + F
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot


# Dynamics of the continuous time Van der Pol oscillator, with control law u(t)
# See http://www.tangentex.com/VanDerPol.htm
class VanDerPol:

    def __init__(self, device, kwargs):
        self.device = device
        self.mu = kwargs.get('mu')
        self.A = torch.tensor([[0., 1.], [-1., 0.]], device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.VanderPol_dynamics(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def VanderPol_dynamics(self, t, x, u, t0, init_control,
                           process_noise_var, kwargs,
                           impose_init_control=False):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        F = torch.tensor(
            [[0, self.mu * (1 - x[:, 0] ** 2) * x[:, 1]]],
            device=device)
        xdot = torch.matmul(self.A, x.t()).t() + F + u
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot


# Dynamics of a simple inverted pendulum, with control law u(t), continuous time
# http://www.matthewpeterkelly.com/tutorials/simplePendulum/index.html
class Pendulum:

    def __init__(self, device, kwargs):
        self.device = device
        self.k = kwargs.get('k')
        self.m = kwargs.get('m')
        self.g = kwargs.get('g')
        self.l = kwargs.get('l')
        self.A = torch.tensor([[0., 1.], [0., 0.]], device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.pendulum_dynamics(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def pendulum_dynamics(self, t, x, u, t0, init_control, process_noise_var,
                          kwargs, impose_init_control=False):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        theta_before = x[:, 0]
        thetadot_before = x[:, 1]
        F = torch.tensor([[np.zeros_like(x[:, 0]),
                           - self.g / self.l * np.sin(theta_before) -
                           self.k / self.m * thetadot_before]], device=device)
        xdot = torch.matmul(self.A, x.t()).t() + F + u
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot


# Dynamics of a harmonic oscillator, with control law u(t), continuous time,
# pulsation/angular frequency w (2pi/period)
# https://en.wikipedia.org/wiki/Harmonic_oscillator
class Harmonic_oscillator:

    def __init__(self, device, kwargs):
        self.device = device
        self.w = kwargs.get('pulse')
        self.A = torch.tensor([[0., 1.], [- self.w ** 2, 0.]],
                              device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.harmonic_oscillator_dynamics(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def harmonic_oscillator_dynamics(self, t, x, u, t0, init_control,
                                     process_noise_var, kwargs,
                                     impose_init_control=False):
        device = x.device
        self.A = self.A.to(device)
        x = reshape_pt1(x)
        u = reshape_dim1_tonormal(u(t, kwargs, t0, init_control,
                                    impose_init_control))
        xdot = torch.matmul(self.A, x.t()).t()
        xdot[:, 1] += reshape_pt1_tonormal(u)
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    def harmonic_oscillator_extended_dynamics(self, t, x, u, t0,
                                              init_control,
                                              process_noise_var, kwargs,
                                              impose_init_control=False):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_dim1_tonormal(
            u(t, kwargs, t0, init_control, impose_init_control))
        xdot = torch.zeros_like(x)
        xdot[:, 0] = x[:, 1]
        xdot[:, 1] = - x[:, 2] * x[:, 0] + reshape_pt1_tonormal(u)
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot


# Classic form of the mass-spring-mass system
class MSM:

    def __init__(self, device, kwargs):
        self.device = device
        self.nx = 4
        self.n = 4
        self.m1 = kwargs.get('m1')
        self.m2 = kwargs.get('m2')
        self.k1 = kwargs.get('k1')
        self.k2 = kwargs.get('k2')
        self.Ax = torch.tensor(
            [[0., 1, 0, 0], [0., 0, 0, 0], [0, 0, 0, 1.], [0, 0, 0, 0.]],
            device=self.device)
        self.Bx = torch.tensor([[0.], [0], [0], [1]], device=self.device)
        self.A = torch.tensor(
            [[0., 1, 0, 0], [0., 0, 1, 0], [0, 0, 0, 1.], [0, 0, 0, 0.]],
            device=self.device)
        self.F1 = reshape_pt1(torch.zeros((self.n - 1,), device=self.device))

    def __call__(self, t, z, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.mass_spring_mass_dynamics_z(t, z, u, t0, init_control,
                                                process_noise_var, kwargs,
                                                impose_init_control)

    def mass_spring_mass_dynamics_x(self, t, x, u, t0, init_control,
                                    process_noise_var, kwargs,
                                    impose_init_control=False):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        F = torch.tensor([
            torch.zeros_like(x[:, 0]),
            self.k1 / self.m1 * (x[:, 2] - x[:, 0]) +
            self.k2 / self.m1 * (x[:, 2] - x[:, 0]) ** 3,
            torch.zeros_like(x[:, 2]),
            -self.k1 / self.m2 * (x[:, 2] - x[:, 0]) -
            self.k2 / self.m2 * (x[:, 2] - x[:, 0]) ** 3],
            device=device)
        xdot = (torch.matmul(self.Ax.double(), x.double().t()) + F.double() +
                torch.matmul(self.Bx.double(), u.double())).t()
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), xdot.shape,
                                 device=device)
        return xdot

    # Canonical form of the mass-spring-mass system using x1 as flat output
    def mass_spring_mass_dynamics_z(self, t, z, u, t0, init_control,
                                    process_noise_var, kwargs,
                                    impose_init_control=False):
        device = z.device
        z = reshape_pt1(z)
        z3 = reshape_pt1(z[:, 2])
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        v = reshape_pt1_tonormal(self.mass_spring_mass_v(z, kwargs))
        vdot = reshape_pt1_tonormal(self.mass_spring_mass_vdot(z, kwargs))
        F2 = torch.tensor([[
            self.k1 / (self.m1 * self.m2) * (u - (self.m1 + self.m2) * z3) +
            (3 * self.k2) / (self.m1 * self.m2) * (u - (self.m1 + self.m2) * z3)
            * v ** 2 + (6 * self.k2) / self.m1 * v * vdot ** 2]], device=device)
        F = reshape_pt1(torch.cat((self.F1, F2), dim=1))
        zdot = torch.matmul(self.A, z.double().t()).t() + F
        if process_noise_var != 0:
            zdot += torch.normal(0, np.sqrt(process_noise_var), size=zdot.shape,
                                 device=device)
        return zdot

    # Canonical form of the mass-spring-mass system using x1 as flat output,
    # only last dimension
    def mass_spring_mass_dynamics_z_justvelocity(self, t, z, u, t0,
                                                 init_control,
                                                 process_noise_var, kwargs,
                                                 impose_init_control=False):
        device = z.device
        z = reshape_pt1(z)
        z3 = reshape_pt1(z[:, 2])
        u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
        v = reshape_pt1_tonormal(self.mass_spring_mass_v(z, kwargs))
        vdot = reshape_pt1_tonormal(self.mass_spring_mass_vdot(z, kwargs))
        zdot = torch.tensor([[
            self.k1 / (self.m1 * self.m2) * (u - (self.m1 + self.m2) * z3) +
            (3 * self.k2) / (self.m1 * self.m2) * (u - (self.m1 + self.m2) * z3)
            * v ** 2 + (6 * self.k2) / self.m1 * v * vdot ** 2]], device=device)
        if process_noise_var != 0:
            zdot += torch.normal(0, np.sqrt(process_noise_var), size=zdot.shape,
                                 device=device)
        return zdot

    # Utility function for the mass-spring-mass system
    # Solution obtained with http://eqworld.ipmnet.ru/en/solutions/ae/ae0103.pdf
    def mass_spring_mass_v(self, z, kwargs):
        z = reshape_pt1(z)
        x1d2 = reshape_pt1(z[:, 2])
        p = self.k1 / self.k2
        q = - self.m1 / self.k2 * x1d2
        D = np.power(p / 3, 3) + np.power(q / 2, 2)
        A = np.cbrt(-q / 2 + np.sqrt(D))  # np.power not with negative floats!
        B = np.cbrt(-q / 2 - np.sqrt(D))
        v = reshape_pt1(A + B)
        return v

    # Utility function for the mass-spring-mass system
    def mass_spring_mass_vdot(self, z, kwargs):
        z = reshape_pt1(z)
        x1d3 = reshape_pt1(z[:, 3])
        A = self.k1 / self.m1 + \
            3 * self.k2 / self.m1 * self.mass_spring_mass_v(z, kwargs) ** 2
        vdot = reshape_pt1(x1d3 / A)
        return vdot

    # Flat transform (from x to z) for mass-spring-mass system
    def mass_spring_mass_xtoz(self, x, kwargs):
        x = reshape_pt1(x)
        z = x.clone()
        z[:, 2] = self.k1 / self.m1 * (x[:, 2] - x[:, 0]) + \
                  self.k2 / self.m1 * (x[:, 2] - x[:, 0]) ** 3
        z[:, 3] = self.k1 / self.m1 * (x[:, 3] - x[:, 1]) + \
                  3 * self.k2 / self.m1 * (x[:, 3] - x[:, 1]) * (
                          x[:, 2] - x[:, 0]) ** 2
        return reshape_pt1(z)

    # Inverse transform (from z to x) for mass-spring-mass system
    def mass_spring_mass_ztox(self, z, kwargs):
        z = reshape_pt1(z)
        x = z.clone()
        x[:, 2] = reshape_pt1_tonormal(
            self.mass_spring_mass_v(z, kwargs)) + z[:, 0]
        x[:, 3] = reshape_pt1_tonormal(
            self.mass_spring_mass_vdot(z, kwargs)) + z[:, 1]
        return reshape_pt1(x)


# Standard form of the continuous time reverse Duffing oscillator, with u(t)
class Reverse_Duffing:

    def __init__(self, device, kwargs):
        self.device = device

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.reverse_duffing_dynamics_z(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def reverse_duffing_dynamics_x(self, t, x, u, t0, init_control,
                                   process_noise_var, kwargs,
                                   impose_init_control=False):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_dim1_tonormal(u(t, kwargs, t0, init_control,
                                    impose_init_control))
        xdot = torch.empty_like(x)
        xdot[:, 0] = torch.pow(x[:, 1], 3)
        xdot[:, 1] = - x[:, 0] + reshape_pt1(u)
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    # Canonical form of the continuous time reverse Duffing oscillator, with u(t)
    def reverse_duffing_dynamics_z(self, t, x, u, t0, init_control,
                                   process_noise_var, kwargs,
                                   impose_init_control=False):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1_tonormal(u(t, kwargs, t0, init_control,
                                   impose_init_control))
        xdot = torch.tensor([x[:, 1], 3 * torch.pow(
            torch.abs(x[:, 1]), 2. / 3) * (u - x[:, 0])], device=device)
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    # Flat transform (from x to z) for reverse Duffing
    def reverse_duffing_xtoz(self, x):
        x = reshape_pt1(x)
        z = x.clone()
        z[:, 1] = torch.pow(x[:, 1], 3)
        return reshape_pt1(z)

    # Inverse transform (from z to x) for reverse Duffing
    def reverse_duffing_ztox(self, z):
        z = reshape_pt1(z)
        x = z.clone()
        x[:, 1] = torch.sign(z[:, 1]) * torch.pow(torch.abs(z[:, 1]), 1. / 3)
        return reshape_pt1(x)

    # True prior mean of only velocity (for backPhi with HGO and true prior)
    def reverse_duffing_dynamics_z_justvelocity_true(self, x, u, prior_kwargs):
        x = reshape_pt1(x)
        u = reshape_pt1_tonormal(u)
        vdot = 3 * torch.pow(torch.abs(x[:, 1]), 2. / 3) * (u - x[:, 0])
        return reshape_pt1(vdot)


# Dynamics of a building during an earthquake, with control law u(t),
# continuous time
# https://odr.chalmers.se/bitstream/20.500.12380/256887/1/256887.pdf
class Earthquake_building:

    def __init__(self, device, kwargs):
        self.device = device
        self.k = kwargs.get('k')
        self.m = kwargs.get('m')
        self.A = torch.tensor([[0., 1., 0., 0.], [0., 0., 0., 0.],
                               [0., 0., 0., 1.], [0., 0., 0., 0.]],
                              device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.earthquake_building_dynamics(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def earthquake_building_dynamics(self, t, x, u, t0, init_control,
                                     process_noise_var, kwargs,
                                     impose_init_control=False):
        device = x.device
        self.A = self.A.to(device)
        x = reshape_pt1(x)
        u = reshape_dim1_tonormal(
            u(t, kwargs, t0, init_control, impose_init_control))
        xdot = torch.matmul(self.A, x.t()).t()
        xdot[:, 1] += self.k / self.m * (x[:, 2] - 2 * x[:, 0]) - \
                      reshape_pt1_tonormal(u)
        xdot[:, 3] += self.k / self.m * (x[:, 0] - x[:, 2]) - \
                      reshape_pt1_tonormal(u)
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    def earthquake_building_dynamics_xu(self, x, u):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        xdot = torch.zeros_like(x)
        xdot[..., 0] = x[..., 1]
        xdot[..., 1] += self.k / self.m * (x[..., 2] - 2 * x[..., 0]) - \
                        u[..., 0]
        xdot[..., 2] = x[..., 3]
        xdot[..., 3] += self.k / self.m * (x[..., 0] - x[..., 2]) - u[..., 0]
        return xdot

# Same as Earthquake_building except the input created by the earthquake is
# considered as a time-dependent nonlinear perturbation instead of a control
# input, so extended state to take time = x5 into account in the state with
# tdot = 1. Constant control input actually contains parameters of
# perturbation to allow for different earthquakes on same model easily
class Earthquake_building_timedep:

    def __init__(self, device, kwargs):
        self.device = device
        self.k = kwargs.get('k')
        self.m = kwargs.get('m')
        self.A = torch.tensor([[0., 1., 0., 0., 0.], [0., 0., 0., 0., 0.],
                               [0., 0., 0., 1., 0.], [0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0.]],
                              device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.earthquake_building_dynamics(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def earthquake_building_dynamics(self, t, x, u, t0, init_control,
                                     process_noise_var, kwargs,
                                     impose_init_control=False):
        device = x.device
        self.A = self.A.to(device)
        x = reshape_pt1(x)
        u = reshape_dim1_tonormal(
            u(t, kwargs, t0, init_control, impose_init_control))
        gamma = u[:, 0]
        omega = u[:, 1]
        xdot = torch.matmul(self.A, x.t()).t()
        xdot[:, 1] += self.k / self.m * (x[:, 2] - 2 * x[:, 0]) - \
                      gamma * torch.cos(omega * x[:, 4])
        xdot[:, 3] += self.k / self.m * (x[:, 0] - x[:, 2]) - \
                      gamma * torch.cos(omega * x[:, 4])
        xdot[:, 4] = torch.ones_like(xdot[:, 4])
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    def earthquake_building_dynamics_xu(self, x, u):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        gamma = u[..., 0]
        omega = u[..., 1]
        xdot = torch.zeros_like(x)
        xdot[..., 0] = x[..., 1]
        xdot[..., 1] += self.k / self.m * (x[..., 2] - 2 * x[..., 0]) - \
                        gamma * torch.cos(omega * x[..., 4])
        xdot[..., 2] = x[..., 3]
        xdot[..., 3] += self.k / self.m * (x[..., 0] - x[..., 2]) - \
                        gamma * torch.cos(omega * x[..., 4])
        xdot[..., 4] = torch.ones_like(xdot[..., 4])
        return xdot

# Same as Earthquake_building except the input created by the earthquake is
# generated by a harmonic oscillator. Constant control input actually
# contains parameters of perturbation to allow for different earthquakes on
# same model easily
class Earthquake_building_extended:

    def __init__(self, device, kwargs):
        self.device = device
        self.k = kwargs.get('k')
        self.m = kwargs.get('m')
        self.t0 = kwargs.get('t0')
        # cheat to overwrite x5(t0) and x6(t0) with constant control
        self.A = torch.tensor([
            [0., 1., 0., 0., 0., 0.],
            [-2 * self.k / self.m, 0., self.k / self.m, 0., -1., 0.],
            [0., 0., 0., 1., 0., 0.],
            [self.k / self.m, 0., - self.k / self.m, 0., -1., 0.],
            [0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0.]], device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.earthquake_building_dynamics(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def earthquake_building_dynamics(self, t, x, u, t0, init_control,
                                     process_noise_var, kwargs,
                                     impose_init_control=False):
        device = x.device
        self.A = self.A.to(device)
        x = reshape_pt1(x)
        u = reshape_dim1_tonormal(
            u(t, kwargs, t0, init_control, impose_init_control))
        omega = u[:, 1]
        xdot = torch.matmul(self.A, x.t()).t()
        xdot[:, 5] = - omega ** 2 * x[:, 4]
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    def earthquake_building_dynamics_xu(self, x, u):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        omega = u[..., 1]
        xdot = torch.zeros_like(x)
        xdot[..., 0] = x[..., 1]
        xdot[..., 1] += self.k / self.m * (x[..., 2] - 2 * x[..., 0]) - \
                        x[..., 4]
        xdot[..., 2] = x[..., 3]
        xdot[..., 3] += self.k / self.m * (x[..., 0] - x[..., 2]) -\
                        x[..., 4]
        xdot[..., 4] = x[..., 5]
        xdot[..., 5] = - omega ** 2 * x[..., 4]
        return xdot


# ODE version of FitzHugh-Nagumo model of the evolution of the
# electromagnetic potential through a membrane subject to a stimulus
# https://en.wikipedia.org/wiki/FitzHugh–Nagumo_model
# Optimal control for estimation in partially observed elliptic and
# hypoelliptic linear stochastic differential equations, Q. Clairon, A. Samson
class FitzHugh_Nagumo_ODE:

    def __init__(self, device, kwargs):
        self.device = device
        self.eps = kwargs.get('eps')
        self.gamma = kwargs.get('gamma')
        self.beta = kwargs.get('beta')
        self.A = torch.tensor([[1 / self.eps, - 1 / self.eps],
                               [self.gamma, -1.]], device=self.device)

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.FitzHugh_Nagumo_dynamics(
            t, x, u, t0, init_control, process_noise_var, kwargs,
            impose_init_control)

    def FitzHugh_Nagumo_dynamics(self, t, x, u, t0, init_control,
                                     process_noise_var, kwargs,
                                     impose_init_control=False):
        device = x.device
        self.A = self.A.to(device)
        x = reshape_pt1(x)
        u = reshape_dim1_tonormal(
            u(t, kwargs, t0, init_control, impose_init_control))
        xdot = torch.matmul(self.A, x.t()).t()
        xdot[:, 0] += - 1 / self.eps * torch.pow(x[:, 0], 3) + u
        xdot[:, 1] += self.beta
        if process_noise_var != 0:
            xdot += torch.normal(0, np.sqrt(process_noise_var), size=xdot.shape,
                                 device=device)
        return xdot

    def FitzHugh_Nagumo_dynamics_xu(self, x, u):
        device = x.device
        x = reshape_pt1(x)
        u = reshape_pt1(u)
        xdot = torch.zeros_like(x)
        xdot[..., 0] += 1 / self.eps * (x[..., 0] - x[..., 1] -
                                        torch.pow(x[..., 0], 3)) + u[..., 0]
        xdot[..., 1] += self.gamma * x[..., 0] - x[..., 1] + self.beta * \
                        torch.ones_like(x[..., 1])
        return xdot
