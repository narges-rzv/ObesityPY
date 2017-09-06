from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class TemporalCNN(nn.Module):
    def __init__(self, indims, timedim, ksize, kcnt, batchsize=1):
        super(TemporalCNN, self).__init__()
        self.ksize = ksize
        self.kcnt = kcnt
        self.indims = indims
        self.hiddims = kcnt*indims*(timedim - ksize +1)
        print('temporal cnn model initiated with :',self.hiddims, ' internal variables')
        self.conv1 = nn.Conv2d(1, kcnt, kernel_size=(1,ksize)).double()
        self.fc1 = nn.Linear(self.hiddims, 1).double()
        print(self)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, self.hiddims)
        x = self.fc1(x)
        return x


# model = Net()
# if args.cuda:
#     model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# def train(epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.data[0]))

# def test():
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
#         pred = output.data.max(1)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data).cpu().sum()

#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))


# for epoch in range(1, args.epochs + 1):
#     train(epoch)
#     test()
