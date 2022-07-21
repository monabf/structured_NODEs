import torch.nn as nn
import torch

# Classes of NN models used to learn dynamics of harmonic oscillator
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

# Simple RNN model (based on pytorch) with n hidden layers. Can pass
# StandardScaler to normalize in and output in forward function.
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
# Simple MLP model with n hidden layers. Can pass StandardScaler to
# normalize in and output in forward function.
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


