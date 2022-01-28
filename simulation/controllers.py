from functools import partial

import numpy as np
import scipy.signal as signal
import torch

from utils.utils import reshape_pt1, reshape_dim1


# Possible controllers

# Cosinusoidal control law, imposing initial value
def cos_controller_02D(t, kwargs, t0, init_control, impose_init=False):
    device = t.device
    gamma = kwargs.get('gamma')
    omega = kwargs.get('omega')
    if np.isscalar(t):
        if impose_init and t == t0:
            u = torch.as_tensor(reshape_pt1(init_control), device=device)
        else:
            u = torch.tensor([[0, gamma * torch.cos(omega * t)]], device=device)
    else:
        if len(t.shape) == 0:
            if impose_init and t.item() == t0:
                u = torch.as_tensor(reshape_pt1(init_control), device=device)
            else:
                u = torch.tensor([[0, gamma * torch.cos((omega * t).float())]],
                                 device=device)
        else:
            u = reshape_pt1(torch.cat((
                reshape_dim1(torch.zeros(len(t), device=device)), reshape_dim1(
                    gamma * torch.cos((omega * t)))), dim=1))
            if impose_init and t[0].item() == t0:
                u[0] = torch.as_tensor(reshape_pt1(init_control), device=device)
    return u


# Sinusoidal control law, imposing initial value
def sin_controller_02D(t, kwargs, t0, init_control, impose_init=False):
    device = t.device
    gamma = kwargs.get('gamma')
    omega = kwargs.get('omega')
    if np.isscalar(t):
        if impose_init and t == t0:
            u = torch.as_tensor(reshape_pt1(init_control), device=device)
        else:
            u = torch.tensor([[0, gamma * torch.sin(omega * t)]], device=device)
    else:
        if len(t.shape) == 0:
            if impose_init and t.item() == t0:
                u = torch.as_tensor(reshape_pt1(init_control), device=device)
            else:
                u = torch.tensor([[0, gamma * torch.sin((omega * t).float())]],
                                 device=device)
        else:
            u = reshape_pt1(torch.cat((
                reshape_dim1(torch.zeros(len(t), device=device)), reshape_dim1(
                    gamma * torch.sin((omega * t)))), dim=1))
            if impose_init and t[0].item() == t0:
                u[0] = torch.as_tensor(reshape_pt1(init_control), device=device)
    return u


# Cosinusoidal control law, imposing initial value
def cos_controller_1D(t, kwargs, t0, init_control, impose_init=False):
    device = t.device
    gamma = kwargs.get('gamma')
    omega = kwargs.get('omega')
    if np.isscalar(t):
        if impose_init and t == t0:
            u = torch.as_tensor(reshape_pt1(init_control), device=device)
        else:
            u = torch.tensor([[gamma * torch.cos((omega * t).float())]],
                             device=device)
    else:
        if len(t.shape) == 0:
            if impose_init and t.item() == t0:
                u = torch.as_tensor(reshape_pt1(init_control), device=device)
            else:
                u = torch.tensor([[gamma * torch.cos(omega * t)]],
                                 device=device)
        else:
            u = reshape_dim1(gamma * torch.cos((omega * t).float()))
            if impose_init and t[0].item() == t0:
                u[0] = torch.as_tensor(reshape_pt1(init_control), device=device)
    return u


# Sinusoidal control law, imposing initial value
def sin_controller_1D(t, kwargs, t0, init_control, impose_init=False):
    device = t.device
    gamma = kwargs.get('gamma')
    omega = kwargs.get('omega')
    if np.isscalar(t):
        if impose_init and t == t0:
            u = torch.as_tensor(reshape_pt1(init_control), device=device)
        else:
            u = torch.tensor([[gamma * torch.sin((omega * t).float())]],
                             device=device)
    else:
        if len(t.shape) == 0:
            if impose_init and t.item() == t0:
                u = torch.as_tensor(reshape_pt1(init_control), device=device)
            else:
                u = torch.tensor([[gamma * torch.sin(omega * t)]],
                                 device=device)
        else:
            u = reshape_dim1(gamma * torch.sin((omega * t).float()))
            if impose_init and t[0].item() == t0:
                u[0] = torch.as_tensor(reshape_pt1(init_control), device=device)
    return u


# Chirp control law, imposing initial value
def chirp_controller(t, kwargs, t0, init_control, impose_init=False):
    device = t.device
    gamma = kwargs.get('gamma')
    f0 = kwargs.get('f0')
    f1 = kwargs.get('f1')
    t1 = kwargs.get('t1')
    nb_cycles = int(torch.floor(torch.min(t) / t1))
    t = t - nb_cycles * t1
    if np.isscalar(t):
        if impose_init and t == t0:
            u = torch.as_tensor(reshape_pt1(init_control), device=device)
        else:
            u = torch.tensor(
                [[0, signal.chirp(t, f0=f0, f1=f1, t1=t1, method='linear')]],
                device=device)
    else:
        u = reshape_pt1(torch.cat((
            reshape_dim1(torch.zeros(len(t), device=device)),
            torch.as_tensor(reshape_dim1(signal.chirp(
                t, f0=f0, f1=f1, t1=t1, method='linear')), device=device)),
            dim=1))  # TODO write own linear chirp (super slow)!
        if impose_init and t[0] == t0:
            u[0] = torch.as_tensor(reshape_pt1(init_control), device=device)
    return gamma * u


# Linear chirp control law, imposing initial value
def linear_chirp_controller(t, kwargs, t0, init_control, impose_init=False):
    device = t.device
    a = kwargs.get('a')
    b = kwargs.get('b')
    scale = kwargs.get('scale')
    if np.isscalar(t):
        if impose_init and t == t0:
            u = torch.as_tensor(reshape_pt1(init_control), device=device)
        else:
            u = torch.tensor([[torch.sin(
                2 * np.pi * t * (a + b * t))]], device=device)
    else:
        if len(t.shape) == 0:
            if impose_init and t.item() == t0:
                u = torch.as_tensor(reshape_pt1(init_control), device=device)
            else:
                u = torch.tensor(
                    [[torch.sin(2 * np.pi * t * (a + b * t))]],
                    device=device)
        else:
            u = reshape_dim1(torch.sin(2 * np.pi * t * (a + b * t)))
            if impose_init and t[0].item() == t0:
                u[0] = torch.as_tensor(reshape_pt1(init_control), device=device)
    return scale * u


# Fake control law just returning zeros (for when one needs to be defined for
# simulation but actually autonomous system)
def null_controller(t, kwargs, t0, init_control, impose_init=False):
    device = t.device
    init_control = reshape_pt1(init_control)
    if len(t.shape) == 0:
        t = t.view(1, )
    u = reshape_pt1(torch.zeros((len(t), init_control.shape[1]), device=device))
    return u


# Constant control law
def constant_controller(t, kwargs, t0, init_control, impose_init=False):
    device = t.device
    control_value = reshape_pt1(kwargs.get('control_value'))
    if len(t.shape) == 0:
        t = t.view(1, )
    u = reshape_pt1(torch.ones((len(t), init_control.shape[1]),
                               device=device) * control_value)
    return u


# Random control law
def random_controller(t, kwargs, t0, init_control, impose_init=False):
    device = t.device
    init_control = reshape_pt1(init_control)
    if len(t.shape) == 0:
        t = t.view(1, )
    u = reshape_pt1(torch.rand((len(t), init_control.shape[1]), device=device))
    return u


# Random control law in bounds (different signature, only for internal use)
def random_controller_bounds(umin, umax, t, kwargs, t0, init_control,
                             impose_init=False):
    return random_controller(t, kwargs, t0, init_control, impose_init) * \
           (umax - umin) + umin


# Array of control inputs using list of controllers. If called normally,
# returns tensor of inputs of shape (nb_controllers, N, nu), if called with
# an index, returns input of that controller of shape (N, nu), if called with
# several indices, returns tensor of inputs of shape (nb_indices, N, nu)
class Control_from_list:
    def __init__(self, controller_list, init_control):
        """
        Takes a list of callable controller functions. When called (as a
        whole or with indices),  returns the outputs of each controller of
        these indices with each initial value.

        :param controller_list: list of controller functions
        :type controller_list: list
        param init_control: initial values to impose for each controller
        :type init_control: torch.tensor, shape (nb_controllers, 1, nu)

        :attr controller_functions: list of functions of each controller
        :rtype:  [Callable]
        """
        self.controller_functions = controller_list
        self.init_control = init_control
        # Preallocate memory for the most classic cases to speed up
        # TODO Not compatible with computational graph to remember values u(t)!!
        # self.memory_len1 = torch.empty(len(self.controller_functions), 1,
        #                                self.init_control.shape[2],
        #                                device=init_control.device)
        # self.memory_len1_difftraj10 = torch.empty(10, 1,
        #                                           self.init_control.shape[2],
        #                                           device=init_control.device)
        # No other way to speed this up? AVoid for loop over functions?
        # https://stackoverflow.com/questions/52740724/how-to-efficiently-evaluate-a-whole-python-list-of-functions-element-wise

    def __call__(self, t, kwargs, t0, u0, impose_init=False):
        t = torch.as_tensor(t)
        device = t.device
        # if len(t.shape) == 0:
        #     diff_u = self.memory_len1
        # else:
        #     diff_u = torch.empty(len(self.controller_functions), len(t),
        #                          self.init_control.shape[2], device=device)
        if len(t.shape) == 0:
            length = 1
        else:
            length = len(t)
        diff_u = torch.empty(len(self.controller_functions), length,
                             self.init_control.shape[2], device=device)
        for i in range(len(diff_u)):  # TODO in parallel!
            diff_u[i] = self.controller_functions[i](t, kwargs.get(
                'controller_args')[i], t0, u0[i], impose_init)
        return diff_u

    def call_subset(self, idx, t, kwargs, t0, u0, impose_init=False):
        # Call a subset of self.controller_functions as above
        t = torch.as_tensor(t)
        device = t.device
        # if len(t.shape) == 0:
        #     if len(idx) == 10:
        #         diff_u = self.memory_len1_difftraj10
        #     else:
        #         diff_u = self.memory_len1[:len(idx)]
        # else:
        #     length = len(t)
        #     diff_u = torch.empty(len(idx), length, u0.shape[2], device=device)
        if len(t.shape) == 0:
            length = 1
        else:
            length = len(t)
        diff_u = torch.empty(len(idx), length, u0.shape[2], device=device)
        for i in range(len(idx)):  # TODO in parallel!
            index = int(idx[i])
            diff_u[i] = torch.unsqueeze(self[index](t, kwargs, t0, u0,
                                                    impose_init), 0)
        return diff_u

    def __getitem__(self, idx):
        if isinstance(idx, int):
            def idx_function(t, kwargs, t0, init_control, impose_init=False):
                return self.controller_functions[idx](
                    t, kwargs.get('controller_args')[idx], t0, init_control[
                        idx], impose_init)
        else:
            def idx_function(t, kwargs, t0, init_control, impose_init=False):
                return self.call_subset(idx, t, kwargs, t0, init_control,
                                        impose_init)

        return idx_function


# Array of control inputs using dictionary of controllers. If called normally,
# returns tensor of inputs of shape (nb_controllers, N, nu), if called with
# an index, returns input of that controller of shape (N, nu), if called with
# several indices, returns tensor of inputs of shape (nb_indices, N, nu)
class Control_from_dict(Control_from_list):
    def __init__(self, controller_dict, init_control, constrain_u=[]):
        """
        Takes a dict of controller functions and nb of times used, creates a
        list of callable functions. When called as a whole or with indices,
        returns the outputs of each controller of these indices with each
        initial value.

        :param controller_dict: dict of form {'controller a': 3}
        :type controller_dict: dict
        param init_control: initial values to impose for each controller
        :type init_control: torch.tensor, shape (nb_controllers, 1, nu)
        param constrain_u: min and max values of control
        :type constrain_u: list

        :attr controller_functions: list of functions of each controller in
        controller_dict
        :rtype:  [Callable]
        """
        self.controller_dict = controller_dict
        self.init_control = init_control
        self.constrain_u = constrain_u
        self.controller_functions = []
        self.umin = np.min(constrain_u)
        self.umax = np.max(constrain_u)
        total_nb = 0

        def common_controller(_controller, _t, _kwargs, _t0, _u0, _impose_u0):
            return _controller(_t, _kwargs, _t0, _u0, _impose_u0)

        for control, current_nb in controller_dict.items():
            current_iter = 0
            while current_iter < current_nb:
                if control == 'random':
                    controller = partial(random_controller_bounds,
                                         self.umin.copy(), self.umax.copy())
                elif control == 'sin_controller_1D':
                    controller = sin_controller_1D
                elif control == 'sin_controller_02D':
                    controller = sin_controller_02D
                elif control == 'cos_controller_1D':
                    controller = cos_controller_1D
                elif control == 'cos_controller_02D':
                    controller = cos_controller_02D
                elif control == 'null_controller':
                    controller = null_controller
                elif control == 'linear_chirp_controller':
                    controller = linear_chirp_controller
                elif control == 'constant_controller':
                    controller = constant_controller
                else:
                    raise ValueError(
                        'Controller is not defined. Available options are '
                        'random, sin_controller_1D, sin_controller_02D, '
                        'null_controller, linear_chirp_controller.')
                current_u = partial(common_controller, controller)
                self.controller_functions.append(current_u)
                current_iter += 1
                total_nb += 1
        super().__init__(self.controller_functions, self.init_control)
