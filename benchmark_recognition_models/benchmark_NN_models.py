import torch
import torch.nn as nn


# Classes of NN models used to learn recognition models on benchmark systems
# These can then be reused as a model attribute in a LearnODE (sub)class
# https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
# Good practice for training NN with ReLU:
# https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
# On activation functions:
# https://mlfromscratch.com/activation-functions-explained/#/
# Use batch normalization?


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


# Simple RNN model (based on pytorch) with n hidden layers. Can pass
# StandardScaler to normalize in and output in forward function.
# GRU: https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be
# RNN vs ResNet: https://cs.stackexchange.com/questions/63541/difference-between-residual-neural-net-and-recurrent-neural-net
class RNNn(nn.Module):
    def __init__(self, n_in, n_out, h0=None, n_hl=1, RNN=torch.nn.RNN,
                 RNN_args={}, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        super(RNNn, self).__init__()
        self.h0 = h0
        self.model = RNN(
            input_size=n_in, hidden_size=n_out, num_layers=n_hl,
            batch_first=True, *RNN_args)

    def set_scalers(self, scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

    def forward(self, x):
        # Compute seq of hidden variables given input seq and hidden init h0
        # Normalize in, denormalize out
        # Input shape (N,1,T,p+nu), output shape (N,1,n)
        xin = torch.squeeze(x, 1)
        if self.scaler_X:
            xin = self.scaler_X.transform(xin)
        if self.h0 is None:
            hn, xout = self.model(xin)
        else:
            hn, xout = self.model(xin, self.h0)
        if self.scaler_Y:
            xout = self.scaler_Y.inverse_transform(xout)
        if len(x.shape) > 3:
            return torch.transpose(xout, 0, 1)
        else:
            return torch.squeeze(xout, 0)

# Simple MLP model that takes as input the output of a previous model (for
# example an RNN)
class MLP2_xin(nn.Module):
    def __init__(self, n_in, n_h1, n_h2, n_out, model_in,
                 activation=nn.Tanh(), init=None, init_args={},
                 scaler_X=None, scaler_Y=None):
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        # Layers: input * activation, hidden * activation, output
        super(MLP2_xin, self).__init__()
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
        self.model_in = model_in

    def set_scalers(self, scaler_X=None, scaler_Y=None):
        self.model_in.set_scalers(scaler_X=scaler_X)
        self.scaler_Y = scaler_Y
        self.scaler_X = scaler_X  # only for saving, not using

    def forward(self, x):
        # Compute output through all layers. Normalize in, denormalize out
        x = self.model_in(x)
        # if self.scaler_X:
        #     x = self.scaler_X.transform(x)
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.hidden4(x)
        if self.scaler_Y:
            x = self.scaler_Y.inverse_transform(x)
        return x

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

# Simple MLP model that takes as input the output of a previous model (for
# example an RNN)
class MLPn_xin(MLPn):
    def __init__(self, num_hl, n_in, n_hl, n_out, model_in,
                 activation=nn.Tanh(), init=None, init_args={},
                 scaler_X=None, scaler_Y=None):
        super(MLPn_xin, self).__init__(
            num_hl=num_hl, n_in=n_in, n_hl=n_hl, n_out=n_out,
            activation=activation, init=init, init_args=init_args,
            scaler_X=scaler_X, scaler_Y=scaler_Y)
        self.model_in = model_in

    def set_scalers(self, scaler_X=None, scaler_Y=None):
        self.model_in.set_scalers(scaler_X=scaler_X)
        self.scaler_Y = scaler_Y
        self.scaler_X = scaler_X  # only for saving, not using

    def forward(self, x):
        # Compute output through all layers. Normalize in, denormalize out
        x = self.model_in(x)
        # if self.scaler_X:
        #     x = self.scaler_X.transform(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        if self.scaler_Y:
            x = self.scaler_Y.inverse_transform(x)
        return x
