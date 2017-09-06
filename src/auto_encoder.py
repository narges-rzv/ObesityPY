from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms



class Autoencoder(nn.Module):
    '''simplest auto-encoder ever'''
    def __init__(self, inputdim, hiddendim):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(inputdim, hiddendim)
        self.fc2 = nn.Linear(hiddendim, inputdim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

class AutoencoderConinBinar(nn.Module):
    '''simplest auto-encoder ever'''
    def __init__(self, inputdimbin, inputdimcont, hiddendim):
        super(AutoencoderConinBinar, self).__init__()
        print('autoencoder with:', inputdimbin, ' binary features and ', inputdimcont, ' continuous features')
        self.fc1 = nn.Linear(inputdimbin+inputdimcont, hiddendim)
        self.fcBin = nn.Linear(hiddendim, inputdimbin)
        self.fcCont = nn.Linear(hiddendim, inputdimcont)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x1, x2 = self.sigmoid(self.fcBin(x)), self.fcCont(x)
        return x1, x2