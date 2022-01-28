import torch
import torch.nn as nn
from torchdiffeq import odeint
# from torchdyn.numerics import Adjoint  # actually standalone, changed!


# Subclass to learn ODEs with NNs. Optimization problem to train the NN on one
# or several trajectories of solutions: minimize a loss on these
# trajectories by enforcing the ODE as a constraint and computing the loss
# gradient analytically using the forward or adjoint sensitivity method.

# This is basically the Adjoint from torchdyn, except it returns the whole
# forward trajectory xN -> x0 and not just the last step, and the whole backward
# trajectory with one boundary condition for each time interval xt+1 -> xt.

def flatten(iterable):
    return torch.cat([el.contiguous().flatten() for el in iterable])

# class My_Adjoint(Adjoint):
class My_Adjoint(nn.Module):
    """
    Default Adjoint class: forward returns whole traj x0 -> xN, backward is
    computed for whole xN -> x0 sequentially
    """

    def __getstate__(self):
        """Custom pickling, did not work by default."""
        return {'func': self.func, 'flat_params': self.flat_params}

    def __setstate__(self, state):
        """Restore state from the custom unpickled state values."""
        self.func = state['func']
        self.flat_params = state['flat_params']

    def __init__(self, intloss=None):
        super().__init__()

        self.intloss = intloss
        self.autograd_func = self._define_autograd_adjoint()

    def adjoint_dynamics(self, s, adjoint_state):
        """ Define the vector field of the augmented adjoint dynamics to be then integrated **backward**. An `Adjoint` object is istantiated into the `NeuralDE` if the adjoint method for back-propagation was selected.

        :param s: current depth
        :type s: float
        :param adjoint_state: tuple of four tensors constituting the *augmented adjoint state* to be integrated: `h` (hidden state of the neural ODE), `λ` (Lagrange multiplier), `μ` (loss gradient state), `s_adj` (adjoint state of the integration depth)
        :type adjoint_state: tuple of tensors
        """
        h, λ, μ, s_adj = adjoint_state[0:4]
        with torch.set_grad_enabled(True):
            s = s.to(h.device).requires_grad_(True)
            h = h.requires_grad_(True)
            f = self.func(s, h)
            dλds = \
                torch.autograd.grad(f, h, -λ, allow_unused=True,
                                    retain_graph=True)[0]
            # dμds is a tuple! of all self.f_params groups
            dμds = torch.autograd.grad(f, self.f_params, -λ, allow_unused=True,
                                       retain_graph=True)
            if not self.intloss is None:
                g = self.intloss(s, h)
                dgdh = torch.autograd.grad(g.sum(), h, allow_unused=True,
                                           retain_graph=True)[0]
                dλds = dλds - dgdh
        ds_adjds = torch.tensor(0.).to(self.s_span)

        # `None` safety check necessary for cert. applications e.g. Stable with bias on out layer
        dμds = torch.cat(
            [el.flatten() if el is not None else torch.zeros_like(p) for el, p
             in zip(dμds, self.f_params)]).to(dλds)

        return (f, dλds, dμds, ds_adjds)

    def _init_adjoint_state(self, sol, *grad_output):
        # Init: lambda to gradient loss % x (autograd), mu to 0
        λ0 = grad_output[-1][0]
        s_adj0 = torch.tensor(0.).to(self.s_span)
        μ0 = torch.zeros_like(self.flat_params)
        return (sol, λ0, μ0, s_adj0)

    def _define_autograd_adjoint(self):
        class autograd_adjoint(torch.autograd.Function):
            @staticmethod
            def forward(ctx, h0, flat_params, s_span):
                # Forward simu on whole traj x0 -> xN
                sol = odeint(self.func, h0, self.s_span, rtol=self.rtol,
                             atol=self.atol, method=self.method,
                             options=self.options)
                ctx.save_for_backward(self.s_span, self.flat_params, sol)
                return sol

            @staticmethod
            def backward(ctx, *grad_output):
                # Adjoint simu of backward pass on each xt+1 -> xt, returns
                # adjoint states concatenated on each t -> t-1 to tN -> t0
                # grad_output = gradient of loss % output x at each t_i
                s, flat_params, sol = ctx.saved_tensors
                self.f_params = tuple(self.func.parameters())
                for i in range(1, len(self.s_span)):
                    backward_span = torch.tensor(
                        [self.s_span[-i], self.s_span[-i - 1]])
                    if i == 1:
                        adj0 = self._init_adjoint_state(
                            sol[-i], (grad_output[0][-i],))
                    else:
                        adj0 = self._init_adjoint_state(
                            sol[-i], (grad_output[0][-i] + λ[-1],))
                    adj_sol = odeint(self.adjoint_dynamics, adj0, backward_span,
                                     rtol=self.rtol, atol=self.atol,
                                     method=self.method, options=self.options)
                    if i == 1:
                        λ = adj_sol[1]
                        μ = adj_sol[2]
                    else:
                        λ = torch.cat((
                            λ, torch.unsqueeze(adj_sol[1][-1], dim=0)), dim=0)
                        μ = torch.cat((
                            μ, torch.unsqueeze(adj_sol[2][-1], dim=0)), dim=0)
                    # No term corresponding to init state model: should be
                    # added for true adjoint gradient of loss % recog params
                    # but needs changes of code structure...
                return (λ, μ, None)

        return autograd_adjoint

    def forward(self, func, h0, s_span, rtol=1e-4, atol=1e-4, method='dopri5',
                options={}):
        if not isinstance(func, nn.Module):
            raise ValueError('func is required to be an instance of nn.Module.')
        self.flat_params = flatten(func.parameters())
        self.s_span = s_span
        self.func = func
        self.method, self.options = method, options
        self.atol, self.rtol = atol, rtol
        h0 = h0.requires_grad_(True)
        sol = self.autograd_func.apply(h0, self.flat_params, self.s_span)
        return sol


class My_Adjoint_difftraj(My_Adjoint):
    """
    Adjoint but on different trajectories of given size, i.e. forward on each
    x0 -> xN, and backward on each xN -> x0: can be run in parallel!
    """

    def __init__(self, intloss=None):
        super().__init__(intloss=intloss)

    def _define_autograd_adjoint(self):
        class autograd_adjoint(torch.autograd.Function):
            @staticmethod
            def forward(ctx, h0, flat_params, s_span):
                sol = odeint(self.func, h0, self.s_span, rtol=self.rtol,
                             atol=self.atol, method=self.method,
                             options=self.options)
                ctx.save_for_backward(self.s_span, self.flat_params, sol)
                return sol

            @staticmethod
            def backward(ctx, *grad_output):
                # Adjoint simu of backward pass on each xt+1 -> xt on each
                # difftraj, returns adjoint states concatenated on each
                # t -> t-1 to tN -> t0 per difftraj
                # grad_output = gradient of loss % output x at each t_i
                s, flat_params, sol = ctx.saved_tensors
                self.f_params = tuple(self.func.parameters())
                for i in range(1, len(self.s_span)):
                    backward_span = torch.tensor(
                        [self.s_span[-i], self.s_span[-i - 1]])
                    if i == 1:
                        adj0 = self._init_adjoint_state(
                            sol[-i], (grad_output[0][-i],))
                    else:
                        adj0 = self._init_adjoint_state(
                            sol[-i], (grad_output[0][-i] + λ[-1],))
                    adj_sol = odeint(self.adjoint_dynamics, adj0, backward_span,
                                     rtol=self.rtol, atol=self.atol,
                                     method=self.method, options=self.options)
                    if i == 1:
                        λ = adj_sol[1]
                        μ = adj_sol[2]
                    else:
                        λ = torch.cat((
                            λ, torch.unsqueeze(adj_sol[1][-1], dim=0)), dim=0)
                        μ = torch.cat((
                            μ, torch.unsqueeze(adj_sol[2][-1], dim=0)), dim=0)
                return (λ, μ, None)

        return autograd_adjoint
