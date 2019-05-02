from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

class Autoencoder(nn.Module):
    '''simplest auto-encoder ever'''
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

class AutoencoderConinBinar(nn.Module):
    '''simplest auto-encoder ever'''
    def __init__(self, input_bin_dim, input_cont_dim, hidden_dim):
        super(AutoencoderConinBinar, self).__init__()
        print('autoencoder with:', input_bin_dim, ' binary features and ', input_cont_dim, ' continuous features')
        self.fc1 = nn.Linear(input_bin_dim+input_cont_dim, hidden_dim)
        self.fcBin = nn.Linear(hidden_dim, input_bin_dim)
        self.fcCont = nn.Linear(hidden_dim, input_cont_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x1, x2 = self.sigmoid(self.fcBin(x)), self.fcCont(x)
        return x1, x2


class VariationalAutoencoder(nn.Module):
    """
    Variational Auto Encoder for Imputation
    
    Parameters
    ----------
    input_bin_dim : int
        Number of columns that have binary data.
    input_cont_dim : int
        Number of columns that have continuous data.
    hidden_dim : int, default 400
        Number of hidden layers.
    latent_dim : int, default 20
        Number of latent dimensions.
    """
    def __init__(self, input_bin_dim, input_cont_dim, hidden_dim=400, latent_dim=20):
        super(VariationalAutoencoder, self).__init__()
        print('variational autoencoder with:', input_bin_dim, ' binary features and ', input_cont_dim, ' continuous features')
        print('using ', hidden_dim, ' hidden dimensions and ', latent_dim, ' latent dimensions')
        # encoder
        self.latent_dim = latent_dim
        self.input_dim = input_bin_dim + input_cont_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        # rectified linear unit layer from hidden_dim to hidden_dim
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)  # mu layer
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)  # logvariance layer
        # decoder
        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x: Variable) -> (Variable, Variable):
        """
        Input vector x -> fully connected 1 -> ReLU -> (fully connected 21, fully connected 22)

        Parameters
        ----------
        x : torch.autograd.Variable, shape (batch_size, input_bin_dim + input_cont_dim)
            Single batch of data.

        Returns
        -------
        mu : torch.autograd.Variable, shape (x.shape[0], latent_dim, )
            Latent means.
        logvar : torch.autograd.Variable, shape (x.shape[0], latent_dim, )
            Latent log variances.
        """

        h1 = self.relu(self.fc1(x))  # type: Variable
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        """
        THE REPARAMETERIZATION IDEA:

        For each training sample (we get 128 batched at a time)

        - take the current learned mu, stddev for each of the latent_dim
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians

        Parameters
        ----------
        mu : torch.autograd.Variable, shape (x.shape[0], latent_dim, )
            Latent means.
        logvar : torch.autograd.Variable, shape (x.shape[0], latent_dim, )
            Latent log variances.

        Returns
        -------
        out : torch.autograd.Variable, shape (x.shape[0], latent_dim, )
            If training, this is a random sample from the normal distribution,
            during inference, it's the mean.
        """

        if self.training:
            # convert variance to standard deviation
            std = logvar.mul(0.5).exp_()  # type: Variable
            # draw elements from N(0,1)
            eps = Variable(std.data.new(std.size()).normal_())
            # sample from a normal distribution
            return eps.mul(std).add_(mu)

        else:
            return mu

    def decode(self, z: Variable) -> Variable:
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(impute_x: Variable, orig_x: Variable, mu: Variable, logvar: Variable, input_bin_dim: int, input_cont_dim: int) -> Variable:
    """
    Loss function from Appendix B of https://arxiv.org/abs/1312.6114.
    This is combination of how well the input and output agree as well as the KL-Divergence.

    Parameters
    ----------
    impute_x : torch.autograd.Variable
        Imputed version of orig_x.
    orig_x : torch.autograd.Variable
        Original data.
    mu : torch.autograd.Variable
        Latent means.
    logvar : torch.autograd.Variable
        Latent log variances.
    input_dim_bin : int
        Number of binary features.
    input_cont_dim : int
        Number of continuous features.

    Returns
    -------
    loss : torch.autograd.Variable
        Loss function from Appendix B of https://arxiv.org/abs/1312.6114.
    """

    input_dim = input_bin_dim + input_cont_dim
    bce = F.binary_cross_entropy(impute_x, orig_x.view(-1, input_dim))
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld /= 1 * input_dim # batch is size 1

    return bce + kld