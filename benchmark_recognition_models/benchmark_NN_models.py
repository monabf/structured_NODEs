import torch
import torch.nn as nn


# Classes of NN models used to learn recognition models on benchmark systems

# Normalize data in forward function: all that goes into the NN is normalized
# then denormalized. This step is taken into account in grad of output of NN
# so grads are still all good!


# Simple MLP model with one hidden layer. Can pass StandardScaler to
# normalize in and output in forward function.
class MLP1(nn.Module):

    def __init__(self, n_in, n_h, n_out, activation=nn.Tanh, scaler_X=None,
                 scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        # Layers: input * activation, hidden * activation, output
        super(MLP1, self).__init__()
        self.hidden1 = nn.Linear(n_in, n_h)
        # initialize weights using "He Init" if ReLU after, "Xavier" otherwise
        # nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.hidden1.weight)
        self.act1 = activation()
        self.hidden2 = nn.Linear(n_h, n_h)
        # nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.hidden2.weight)
        self.act2 = activation()
        self.hidden3 = nn.Linear(n_h, n_out)
        nn.init.xavier_uniform_(self.hidden3.weight)

    def set_scalers(self, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

    def forward(self, x):
        # Compute output through all layers. Normalize in, denormalize out
        if self.scaler_X:
            x = self.scaler_X.transform(x)
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.hidden3(x)
        if self.scaler_Y:
            x = self.scaler_Y.inverse_transform(x)
        return x


# Simple MLP model with two hidden layers. Can pass StandardScaler to
# normalize in and output in forward function.
class MLP2(nn.Module):

    def __init__(self, n_in, n_h1, n_h2, n_out, activation=nn.Tanh,
                 scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        # Layers: input * activation, hidden * activation, output
        super(MLP2, self).__init__()
        self.hidden1 = nn.Linear(n_in, n_h1)
        # initialize weights using "He Init" if ReLU after, "Xavier" otherwise
        nn.init.xavier_uniform_(self.hidden1.weight)
        self.act1 = activation()
        self.hidden2 = nn.Linear(n_h1, n_h2)
        nn.init.xavier_uniform_(self.hidden2.weight)
        self.act2 = activation()
        self.hidden3 = nn.Linear(n_h2, n_h2)
        nn.init.xavier_uniform_(self.hidden3.weight)
        self.act3 = activation()
        self.hidden4 = nn.Linear(n_h2, n_out)
        nn.init.xavier_uniform_(self.hidden4.weight)

    def set_scalers(self, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

    def forward(self, x):
        # Compute output through all layers. Normalize in, denormalize out
        if self.scaler_X:
            x = self.scaler_X.transform(x)
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.hidden4(x)
        if self.scaler_Y:
            x = self.scaler_Y.inverse_transform(x)
        return x


# Concatenate output of a nn.Module with a 1 * constant and a 0:
# Returns (model(in[idx11:idx12]), in[idx21:idx22], in[idx31:idx32]) from in
# Used for extended earthquake models: outputs (x1-4(0) from MLP2, gamma, 0)
class Earthquake_extended_recog(nn.Module):

    def __init__(self, model, config, n2, n3, idx11, idx12, idx21, idx22,
                 idx31, idx32, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        super(Earthquake_extended_recog, self).__init__()
        self.model = model
        self.config = config
        self.n2 = n2
        self.n3 = n3
        self.idx11 = idx11
        self.idx12 = idx12
        self.idx21 = idx21
        self.idx22 = idx22
        self.idx31 = idx31
        self.idx32 = idx32

    def set_scalers(self, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        # Avoid normalizing the last outputs
        y_mean = self.scaler_Y._mean.clone()
        y_mean[-(self.n2 + self.n3):] *= 0.
        y_var = self.scaler_Y._var.clone()
        y_var[-(self.n2 + self.n3):] /= y_var[-(self.n2 + self.n3):]
        self.scaler_Y.set_scaler(y_mean, y_var)

    def forward(self, x):
        # First compute (1 * in[idx21:idx22], 0) in right shape
        # Then normalize in, compute model(in), denormalize out, and concatenate
        if 'KKL' in self.config.init_state_obs_method and \
                not 'back' in self.config.init_state_obs_method:
            x2_res = torch.ones(tuple(list(x.shape[:-1]) + [self.n2]),
                                device=x.device)
            x3_res = torch.ones(tuple(list(x.shape[:-1]) + [self.n3]),
                                device=x.device)
            gamma = x[..., self.idx21:self.idx22].view(x2_res.shape)
            omega = x[..., self.idx31:self.idx32].view(x3_res.shape)
            x2_res *= gamma * torch.cos(
                omega * self.config.init_state_obs_T * self.config.dt)
            x3_res *= - gamma * omega * torch.sin(
                omega * self.config.init_state_obs_T * self.config.dt)
        else:
            x2_res = torch.ones(tuple(list(x.shape[:-1]) + [self.n2]),
                                device=x.device)
            x2_res *= x[..., self.idx21:self.idx22].view(x2_res.shape)
            x3_res = torch.zeros(tuple(list(x.shape[:-1]) + [self.n3]),
                                 device=x.device)
        if self.scaler_X:
            x = self.scaler_X.transform(x)
        x1_res = self.model(x[..., self.idx11:self.idx12])  # outputs x(0)
        x_res = torch.cat((x1_res, x2_res, x3_res), dim=-1)
        if self.scaler_Y:
            x_res = self.scaler_Y.inverse_transform(x_res)
        return x_res

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