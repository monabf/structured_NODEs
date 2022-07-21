import torch
import torch.nn as nn

from utils.config import Config
from utils.utils import reshape_pt1


# from torchdyn.core import DEFunc  # actually standalone, changed!

# Helper class: holds dyn_NODE function that outputs xdot by NODE dynamics model
# Same as torchdyn.models.defunc but time-dependent dynamics model that
# allows for inputs (basically self.m(s, x) instead of self.m(x))

class DEFunc_time(nn.Module):
    """
    Differential Equation Function Wrapper. Handles auxiliary tasks for
    NeuralDEs: depth concatenation, higher order dynamics and forward
    propagated integral losses.

    :param model: neural network parametrizing the vector field
    :type model: nn.Module
    :param order: order of the differential equation
    :type order: int
    """

    def __init__(self, model, controller, t0, init_control, config: Config,
                 sensitivity='autograd', intloss=None, order=1,
                 force_control=False):
        super().__init__()
        self.m, self.nfe, = model, 0.
        self.order, self.intloss, self.sensitivity = order, intloss, sensitivity
        self.controller = controller
        self.t0 = t0
        self.init_control = init_control
        self.config = config
        if force_control:
            self.no_control = False
        else:
            self.no_control = self.config.no_control
        # For difftraj: given a list of controllers, only simulate those that
        # correspond to a given minibatch i.e. have the indices _idx_batch
        self._idx_batch = None

    def reset_idx(self):
        self._idx_batch = None

    def set_idx(self, idx):
        self._idx_batch = idx

    def dyn_NODE(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False, verbose=False):
        # Actual dynamics model defined by the NODE = submodel(xt, ut)
        x = reshape_pt1(x)
        if not self.no_control:# and not self.config.no_control:  # new
        # if not self.config.no_control:  # old
            u = reshape_pt1(u(t, kwargs, t0, init_control, impose_init_control))
            x = torch.cat((x, u), dim=-1)
        xdot = self.m(x)
        return xdot

    def func_NODE(self, t, x):
        # Calls dynamics of NODE from t, x using dyn_NODE
        if self._idx_batch is None:
            return self.dyn_NODE(t, x, self.controller, self.t0,
                                 self.init_control, 0., self.config)
        else:
            return self.dyn_NODE(t, x, self.controller[self._idx_batch],
                                 self.t0, self.init_control, 0., self.config)

    def forward(self, s, x):
        self.nfe += 1
        # set `s` depth-variable to DepthCat modules
        for _, module in self.m.named_modules():
            if hasattr(module, 's'):
                module.s = s

        # if-else to handle autograd training with integral loss propagated in x[:, 0]
        if (not self.intloss is None) and self.sensitivity == 'autograd':
            x_dyn = x[:, 1:]
            dlds = self.intloss(s, x_dyn)
            if len(dlds.shape) == 1: dlds = dlds[:, None]
            if self.order > 1:
                x_dyn = self.horder_forward(s, x_dyn)
            else:
                x_dyn = self.func_NODE(s, x_dyn)
            self.dxds = x_dyn
            return torch.cat([dlds, x_dyn], 1).to(x_dyn)

        # regular forward
        else:
            if self.order > 1:
                x = self.horder_forward(s, x)
            else:
                x = self.func_NODE(s, x)
            self.dxds = x
            return x

    def horder_forward(self, s, x):
        # NOTE: higher-order in CNF is handled at the CNF level, to refactor
        x_new = []
        size_order = x.size(1) // self.order
        for i in range(1, self.order):
            x_new += [x[:, size_order * i:size_order * (i + 1)]]
        x_new += [self.func_NODE(s, x)]
        return torch.cat(x_new, 1).to(x)
