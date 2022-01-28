import logging
import sys
import torch
from prettytable import PrettyTable

# Set double precision by default
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


# Different classes of useful objects defined in PyTorch, often to replace
# standard numpy objects

# Replaces sklearn StandardScaler()
# https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576
class StandardScaler:
    def __init__(self, X=None, mean=None, var=None):
        if X is not None:
            self._mean = torch.mean(X, dim=0)
            self._var = torch.var(X, dim=0, unbiased=False)
            self.n_samples_seen_ = len(X)
        elif mean is not None:
            self._mean = mean
            self._var = var
        # If var = 0., i.e. values all same, make it 1 so unchanged!
        idx = torch.nonzero(self._var == 0.)
        self._var[idx] += 1.
        self._scale = torch.sqrt(self._var)

    def fit(self, X):
        self._mean = torch.mean(X, dim=0)
        self._var = torch.var(X, dim=0, unbiased=False)
        # If var = 0., i.e. values all same, make it 1 so unchanged!
        idx = torch.nonzero(self._var == 0.)
        self._var[idx] += 1.
        self._scale = torch.sqrt(self._var)
        self.n_samples_seen_ = len(X)

    def transform(self, X):
        if torch.is_tensor(X):
            return (X - self._mean) / self._scale
        else:
            return (X - self._mean.numpy()) / self._scale.numpy()

    def inverse_transform(self, X):
        if torch.is_tensor(X):
            return self._scale * X + self._mean
        else:
            return self._scale.numpy() * X + self._mean.numpy()

    def set_scaler(self, mean, var):
        self._mean = mean
        self._var = var
        # If var = 0., i.e. values all same, make it 1 so unchanged!
        idx = torch.nonzero(self._var == 0.)
        self._var[idx] += 1.
        self._scale = torch.sqrt(self._var)

    def __str__(self):
        return f'Standard scaler of mean {self._mean} and var {self._var}\n'

    def __repr__(self):
        return f'Standard scaler of mean {self._mean} and var {self._var}\n'


# For simple early stopping since not implemented in pytorch
# https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=10, threshold=1e-4, verbose=False):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param threshold: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if not self.best_loss:
            self.best_loss = val_loss
        elif abs(self.best_loss - val_loss) > self.threshold:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logging.info(
                    f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                logging.info(f"Early stop at {val_loss} after {self.counter} "
                             f"epochs")
                self.early_stop = True


def print_parameters(model, sformat='{:60s} {:40s} {:20s} {:70s} {:40s} {:40s}',
                     file=sys.stdout):
    # https://docs.gpytorch.ai/en/v1.1.1/_modules/gpytorch/module.html?highlight=named_parameters_and_constraints#
    ''' Print model parameters
    '''
    print('\nParameter list:', file=file)
    print(sformat.format('Name', 'True Value', 'Type', 'Size', 'Constraint',
                         'Prior'), file=file)
    print(sformat.format('-' * 40, '-' * 60, '-' * 30, '-' * 15, '-' * 30,
                         '-' * 60), file=file)
    pretty = lambda list_: [f"{element:.4e}" for element in list_.flatten()]
    for name, param, constraint in model.named_parameters_and_constraints():
        if any(k in name for k in ('inducing', 'variational')):
            continue
        has_prior = False
        for namep, prior, closure, inv_closure, _ in model.named_priors():
            if name.rsplit('.', 1)[0] == namep.rsplit('.', 1)[0]:
                has_prior = True
                print(sformat.format(
                    name,
                    ' '.join(pretty(param if constraint is None else
                                    constraint.transform(param))),
                    str(type(param.data)),
                    str(list(param.size())),
                    str(constraint),
                    str((namep, str(prior))),
                ), file=file)
                break
        if not has_prior:
            print(sformat.format(
                name,
                ' '.join(pretty(param if constraint is None else
                                constraint.transform(param))),
                str(type(param.data)),
                str(list(param.size())),
                str(constraint),
                'None',
            ), file=file)
    print('\n', file=file)


# Get all parameters of a NN into a flattened tensor and count them
# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def get_parameters(model, verbose=False):
    table = PrettyTable(["Modules", "Shape", "Number"])
    params = []
    # params = torch.zeros((0, ), requires_grad=True)
    nb_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if verbose:
                print('No gradient required: ', name)
            continue
        nb = parameter.numel()
        table.add_row([name, parameter.shape, nb])
        params.append(parameter)
        # params = torch.cat((params, parameter.data.flatten()))
        nb_params += nb
    if verbose:
        print(table)
        print(f"Total Trainable Params: {nb_params}")
    return nb_params, tuple(params)
