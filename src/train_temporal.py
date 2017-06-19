from __future__ import print_function
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
import pickle
import torch
import torch.nn as nn 
import torch.nn.functional as functionalnn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data(TimeIn=[0,18*12], Timeout=[0, 18*12], outcomeIx=0):
    #(data, data_percentile, datakeys, datagenders)
    (d, dp, dk, dg) = pickle.load(open('timeseries_data20170615-173456.pkl','rb'))
    vitals = ['BMI', 'HC', 'Ht', 'Wt']
    #dimension of data is N x |vitals| x |time=18*12mnths|

    print ('total num of ppl with any of the vitals measured at age 0-24months:', ((d[:,:,TimeIn[0]:TimeIn[1]].sum(axis=2)>0).sum(axis=1)>0).sum())
    print ('total num of ppl with BMI measured at age 4-6:', ((d[:, outcomeIx, Timeout[0]:Timeout[1]].sum(axis=1)>0)).sum() )
    print ('total num of ppl with both of above consitions:', (((d[:, outcomeIx, Timeout[0]:Timeout[1]].sum(axis=1)>0)) & ((d[:,:, TimeIn[0]:TimeIn[1]].sum(axis=2)>0).sum(axis=1)>0)).sum() ) 

    ix_selected_cohort = (((d[:, 0, Timeout[0]:Timeout[1]].sum(axis=1)>0)) & ((d[:,:, TimeIn[0]:TimeIn[1]].sum(axis=2)>0).sum(axis=1)>0))
    dInput = d[ix_selected_cohort,:,TimeIn[0]:TimeIn[1]]
    dOutput = d[ix_selected_cohort, outcomeIx, Timeout[0]:Timeout[1]]
    dInputPerc = dp[ix_selected_cohort,:,TimeIn[0]:TimeIn[1]]
    dOutputPerc = dp[ix_selected_cohort, outcomeIx, Timeout[0]:Timeout[1]]
    dkselected = np.array(dk)[ix_selected_cohort]
    dgselected = np.array(dg)[ix_selected_cohort]

    print('input shape:',dInput.shape, 'output shape:', dOutput.shape)
    return dInput, dOutput, dkselected, dgselected

def split_train_valid_test(dInput, dOutput, dkselected=None, dgselected=None, ratioTest=0.25, ratioValid = 0.50):
    import random
    random.seed(0)
    assert dInput.shape[0] == dOutput.shape[0]
    ix = list(range(0,dInput.shape[0]))
    random.shuffle(ix)
    ix_test = ix[0:int(len(ix)*ratioTest)]
    ix_valid = ix[int(len(ix)*ratioTest):int(len(ix)*ratioValid)]
    ix_train = ix[int(len(ix)*ratioTest):]

    dInTrain, dOutTrain = dInput[ix_train,:,:], dOutput[ix_train,:]
    dInValid, dOutValid = dInput[ix_valid,:,:], dOutput[ix_valid,:]
    dInTest, dOutTest = dInput[ix_test,:,:], dOutput[ix_test,:]
    return dInTrain,dOutTrain, dInValid, dOutValid, dInTest, dOutTest


def augment_date(batchInput):
    error = np.random.normal(0,0.1,batchInput.numpy().shape)
    error[(batchInput.numpy()==0)] = 0
    batchInput += torch.from_numpy(error).float()

def build_train_lstm(dIn, dOut, dInValid, dOutValid, dInTest,dOutTest, hidden_dim=51, dropout=False, batch_size=32, num_layers=2, gap=0):
    # set ramdom seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    print('input data:', dIn.shape, dIn.__class__)
    dInputTransposed = dIn.transpose((2,0,1)).copy()
    dOutputTransposed = dOut.transpose().max(axis=0)
    print('output data:', dOut.shape, dOut.__class__)
    dInValidtrans, dOutValidtrans = dInValid.transpose((2,0,1)).copy(), dOutValid.transpose().max(axis=0)
    dInTesttrans, dOutTesttrans = dInTest.transpose((2,0,1)).copy(), dOutTest.transpose().max(axis=0)
    seq_size = dIn.shape[2] - gap
    input_dim = dIn.shape[1]
    target_dim = 1
    totalbatches = int(dIn.shape[0]/batch_size)


    class LSTMPredictor(nn.Module):
        def __init__(self, hidden_dim, input_dim, num_layers, dout, bfirst, tagset_size, minibatch_size, time_dim):
            super(LSTMPredictor, self).__init__()
            self.hidden_dim = hidden_dim
            self.minibatch_size = minibatch_size
            self.time_dim = time_dim
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dout, batch_first=bfirst)

            # The linear layer that maps from hidden state space to tag space
            self.hidden2output = nn.Linear(hidden_dim * time_dim, tagset_size)
            self.hidden = self.init_hidden(minibatch_size)
        def init_hidden(self,minibatch_size):
            # Before we've done anything, we dont have any hidden state.
            # Refer to the Pytorch documentation to see exactly
            # why they have this dimensionality.
            # The axes semantics are (num_layers, minibatch_size, hidden_dim)
            return (autograd.Variable(torch.zeros(num_layers, self.minibatch_size, self.hidden_dim)),
                    autograd.Variable(torch.zeros(num_layers, self.minibatch_size, self.hidden_dim)))
        def forward(self, input):        
            lstm_out, self.hidden = self.lstm(input)
            # print(lstm_out.size())
            lstm_out_trans = (torch.transpose(lstm_out.contiguous(), 0, 1))
            # print(lstm_out_trans.size())
            net_out1 = self.hidden2output(lstm_out_trans.contiguous().view(self.minibatch_size,self.hidden_dim*self.time_dim))
            # print(net_out1.size())
            # net_softmaxout = functionalnn.log_softmax(net_out)
            net_finalout = net_out1
            return net_finalout

    model = LSTMPredictor(hidden_dim, input_dim, num_layers, dropout, True, target_dim, batch_size, seq_size)
    #loss_function = nn.NLLLoss()
    loss_function = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # output_lstm = model(Variable(batchInput))
    # print(output_lstm)

    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        #shuffle the examples
        total_loss = 0
        total_cnt = 0
        ix_shuffle = range(0,dInputTransposed.shape[1])
        dInputTransposed_shuffled = dInputTransposed[:,ix_shuffle,:]
        dOutputTransposed_shuffled = dOutputTransposed[ix_shuffle]

        for batchIx in range(0, totalbatches):
            batchInput = torch.from_numpy(dInputTransposed_shuffled[0:seq_size, (batchIx*batch_size):(batchIx*batch_size) + batch_size, 0:input_dim]).float()
            augment_date(batchInput)
            batchTarget = torch.from_numpy(dOutputTransposed[(batchIx*batch_size):(batchIx*batch_size) + batch_size]).float()
            model.zero_grad()
            model.hidden = model.init_hidden(batch_size)

            predictions = model(Variable(batchInput))
            loss = loss_function(predictions, Variable(batchTarget))
            total_loss += loss.data.numpy()[0]
            total_cnt += 1
            loss.backward()
            optimizer.step()

        if (epoch % 10) == 0 :
            valid_loss = 0
            total_cnt_valid = 0
            for ixvalidBatch in range(0,int(len(dOutValid)/batch_size)):
                validBatchIn = torch.from_numpy(dInValidtrans[0:seq_size, (ixvalidBatch*batch_size):(ixvalidBatch*batch_size) + batch_size, :]).float()
                validbatchOut = torch.from_numpy(dOutValidtrans[(ixvalidBatch*batch_size):(ixvalidBatch*batch_size) + batch_size]).float()
                validPred = model(Variable(validBatchIn))
                loss = loss_function(validPred, Variable(validbatchOut))
                valid_loss += loss.data.numpy()[0]
                total_cnt_valid += 1
            print('average Valid mse loss at epoch:', epoch, ' is:',valid_loss/total_cnt_valid)


        print('average Train mse loss at epoch:', epoch, ' is:',total_loss/total_cnt)


