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
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pandas

def load_data(TimeIn=[0,18*12], Timeout=[0, 18*12], outcomeIx=0):
    #(data, data_percentile, datakeys, datagenders)
    (d, dp, dk, dg, dethn, drace) = pickle.load(open('timeseries_data20170620-174720.pkl','rb'))
    vitals = ['BMI', 'HC', 'Ht', 'Wt']
    #dimension of data is N x |vitals| x |time=18*12mnths|

    gender_streched = np.repeat(np.array(dg).reshape(len(dg),1,1), d.shape[2], axis=2)
    ethn_dummy, race_dummy = np.array(pandas.get_dummies(dethn)), np.array(pandas.get_dummies(drace))
    ethn_streched = np.repeat(np.array(ethn_dummy).reshape(len(dethn), ethn_dummy.shape[1],1), d.shape[2], axis=2)
    race_streched = np.repeat(np.array(race_dummy).reshape(len(drace), race_dummy.shape[1],1), d.shape[2], axis=2)

    d = np.concatenate([d, gender_streched, ethn_streched, race_streched], axis=1)

    print ('total num of ppl with any of the vitals measured at age 0-24months:', ((d[:,0:len(vitals),TimeIn[0]:TimeIn[1]].sum(axis=2)>0).sum(axis=1)>0).sum())
    print ('total num of ppl with BMI measured at age 4-6:', ((d[:, outcomeIx, Timeout[0]:Timeout[1]].sum(axis=1)>0)).sum() )
    print ('total num of ppl with both of above consitions:', (((d[:, outcomeIx, Timeout[0]:Timeout[1]].sum(axis=1)>0)) & ((d[:,0:len(vitals), TimeIn[0]:TimeIn[1]].sum(axis=2)>0).sum(axis=1)>0)).sum() ) 

    ix_selected_cohort = (((d[:, outcomeIx, Timeout[0]:Timeout[1]].sum(axis=1)>0)) & ((d[:, 0:len(vitals), TimeIn[0]:TimeIn[1]].sum(axis=2)>0).sum(axis=1)>0))
    dInput = d[ix_selected_cohort,:,TimeIn[0]:TimeIn[1]]
    dOutput = d[ix_selected_cohort, outcomeIx, Timeout[0]:Timeout[1]]
    dInputPerc = dp[ix_selected_cohort,:,TimeIn[0]:TimeIn[1]]
    dOutputPerc = dp[ix_selected_cohort, outcomeIx, Timeout[0]:Timeout[1]]
    dkselected = np.array(dk)[ix_selected_cohort]
    dgselected = np.array(dg)[ix_selected_cohort]
    dethenselected = np.array(dethn)[ix_selected_cohort]
    draceselected = np.array(drace)[ix_selected_cohort]

    print('input shape:',dInput.shape, 'output shape:', dOutput.shape)
    return dInput, dOutput, dkselected, dgselected, dethenselected, draceselected

def split_train_valid_test(dInput, dOutput, dkselected=None, dgselected=None, ratioTest=0.25, ratioValid = 0.50):
    import random
    random.seed(0)
    assert dInput.shape[0] == dOutput.shape[0]
    ix = list(range(0,dInput.shape[0]))
    random.shuffle(ix)
    ix_test = ix[0:int(len(ix)*ratioTest)]
    ix_valid = ix[int(len(ix)*ratioTest):int(len(ix)*ratioValid)]
    ix_train = ix[int(len(ix)*ratioValid):]

    dInTrain, dOutTrain = dInput[ix_train,:,:], dOutput[ix_train,:]
    dInValid, dOutValid = dInput[ix_valid,:,:], dOutput[ix_valid,:]
    dInTest, dOutTest = dInput[ix_test,:,:], dOutput[ix_test,:]
    return dInTrain,dOutTrain, dInValid, dOutValid, dInTest, dOutTest

def augment_date(batchInput):
    error = np.random.normal(0, 0.1, batchInput.numpy().shape)
    error[(batchInput.numpy()==0)] = 0
    batchInput += torch.from_numpy(error).float()

def impute(batchInputNumpy):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(20, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel)
    for dataix in range(0, batchInputNumpy.shape[1]):
        for dim in range(0, 4):
            X = np.nonzero(batchInputNumpy[:,dataix, dim])[0]
            if len(X) == 0:
                continue
            y = batchInputNumpy[:,dataix, dim][X]
            y_normed = (y - y.mean())
            gp.fit(X.reshape(-1, 1), y_normed.reshape(-1, 1))
            xpred = np.array(list(range(0, batchInputNumpy.shape[0]))).reshape(-1, 1)
            batchInputNumpy[:,dataix, dim] = gp.predict(xpred).ravel() + y.mean()
            # return(X,y,gp.predict(xpred).ravel() + y.mean() )
 

class LSTMPredictor(nn.Module):
    def __init__(self, hidden_dim, input_dim, num_layers, dout, bfirst, tagset_size, minibatch_size, time_dim, bidirectional):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.minibatch_size = minibatch_size
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dout, batch_first=bfirst, bidirectional=bidirectional)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2output = nn.Linear(hidden_dim * time_dim * self.directions, tagset_size)
        self.hidden = self.init_hidden(minibatch_size)

    def init_hidden(self,minibatch_size): # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(self.num_layers * self.directions, self.minibatch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.num_layers * self.directions, self.minibatch_size, self.hidden_dim)))
    
    def forward(self, input):        
        lstm_out, self.hidden = self.lstm(input)
        lstm_out_trans = (torch.transpose(lstm_out.contiguous(), 0, 1))
        net_out1 = self.hidden2output(lstm_out_trans.contiguous().view(self.minibatch_size,self.hidden_dim*self.time_dim*self.directions))
        # net_softmaxout = functionalnn.log_softmax(net_out)
        return net_out1

def build_train_lstm(dIn, dOut, dInValid, dOutValid, dInTest,dOutTest, model_file='obesityat520170620-105723.pyth_lstm', 
    hidden_dim=64, dropout=0.2, batch_size=16, num_layers=1, gap=0, bidirectional=False):
    # set ramdom seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    print('input data:', dIn.shape, dIn.__class__)
    dInputTransposed = dIn.transpose((0,2,1)).copy()
    dOutputTransposed = dOut.transpose().max(axis=0)
    print('output data:', dOut.shape, dOut.__class__)
    dInValidtrans, dOutValidtrans = dInValid.transpose((0,2,1)).copy(), dOutValid.transpose().max(axis=0)
    dInTesttrans, dOutTesttrans = dInTest.transpose((0,2,1)).copy(), dOutTest.transpose().max(axis=0)
    seq_size = dIn.shape[2] - gap
    input_dim = dIn.shape[1]
    target_dim = 1
    totalbatches = int(dIn.shape[0]/batch_size)

    model = LSTMPredictor(hidden_dim, input_dim, num_layers, dropout, True, target_dim, batch_size, seq_size, bidirectional)
    loss_function = nn.MSELoss() #if classification nn.NLLLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.05)
    
    try:
        model = torch.load(model_file)
        print('loaded model:', model_file)
        skip_train = True
    except:
        skip_train = False

    for epoch in range(3000):
        if skip_train == True:
            break
        total_loss = 0
        total_cnt = 0
        ix_shuffle = range(0,dInputTransposed.shape[0])
        dInputTransposed_shuffled = dInputTransposed[ix_shuffle,:,:]
        dOutputTransposed_shuffled = dOutputTransposed[ix_shuffle]
        for batchIx in range(0, totalbatches):
            batchInputNumpy = dInputTransposed_shuffled[(batchIx*batch_size):(batchIx*batch_size) + batch_size, 0:seq_size, 0:input_dim]
            #impute(batchInputNumpy)
            batchInput = torch.from_numpy(batchInputNumpy).float()
            # augment_date(batchInput)
            batchTarget = torch.from_numpy(dOutputTransposed_shuffled[(batchIx*batch_size):(batchIx*batch_size) + batch_size]).float()
            model.zero_grad()
            model.hidden = model.init_hidden(batch_size)
            predictions = model(Variable(batchInput))
            loss = loss_function(predictions, Variable(batchTarget))
            total_loss += loss.data.numpy()[0]
            total_cnt += 1
            loss.backward()
            optimizer.step()

        print('average Train mse loss at epoch:', epoch, ' is:',total_loss/total_cnt)
        if (epoch % 10) == 0 :
            valid_loss = 0
            total_cnt_valid = 0
            for ixvalidBatch in range(0,int(len(dOutValid)/batch_size)):
                validBatchIn = torch.from_numpy(dInValidtrans[(ixvalidBatch*batch_size):(ixvalidBatch*batch_size) + batch_size, 0:seq_size, :]).float()
                validbatchOut = torch.from_numpy(dOutValidtrans[(ixvalidBatch*batch_size):(ixvalidBatch*batch_size) + batch_size]).float()
                validPred = model(Variable(validBatchIn))
                loss = loss_function(validPred, Variable(validbatchOut))
                valid_loss += loss.data.numpy()[0]
                total_cnt_valid += 1
                timestr = time.strftime("%Y%m%d-%H%M%S")
                torch.save(model, 'obesityat5'+timestr+'.pyth_lstm')
            print('   average Valid mse loss at epoch:', epoch, ' is:',valid_loss/total_cnt_valid)

    test_pred_all = np.zeros((dOutTesttrans.shape[0]),dtype=float)
    test_loss = 0
    total_cnt_test = 0
    for ixtestBatch in range(0,int(len(dOutTest)/batch_size)):
        testBatchIn = torch.from_numpy(dInTesttrans[(ixtestBatch*batch_size):(ixtestBatch*batch_size) + batch_size, 0:seq_size, :]).float()
        testBatchOut = torch.from_numpy(dOutTesttrans[(ixtestBatch*batch_size):(ixtestBatch*batch_size) + batch_size]).float()
        testPred = model(Variable(testBatchIn))
        test_pred_all[(ixtestBatch*batch_size):(ixtestBatch*batch_size) + batch_size] = testPred.data.numpy().ravel().copy()
        loss = loss_function(testPred, Variable(testBatchOut))
        test_loss += loss.data.numpy()[0]
        total_cnt_test += 1
        
    print('average Test mse loss at epoch:', epoch, ' is:',test_loss/total_cnt_test)
    return test_pred_all, dOutTesttrans

