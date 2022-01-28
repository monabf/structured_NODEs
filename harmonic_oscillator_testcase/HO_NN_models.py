import torch.nn as nn


# Classes of NN models used to learn dynamics of harmonic oscillator

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


