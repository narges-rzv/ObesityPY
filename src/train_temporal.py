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

def split_train_valid_test(dInput, dOutput, dkselected, dgselected):
    pass


def build_train_lstm(dIn, dOut, dkselected, dgselected, hidden_dim=51, dropout=False, batch_size=8, num_layers=2, gap=1):
    # set ramdom seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    print('input data:', dIn.shape, dIn.__class__)
    dInputTransposed = dIn.transpose((2,0,1)).copy()
    dOutputTransposed = dOut.transpose().copy()
    print('output data:', dOut.shape, dOut.__class__)
    seq_size = dIn.shape[2] - gap
    input_dim = dIn.shape[1]
    batchInput = torch.from_numpy(dInputTransposed[0:seq_size,0:batch_size,0:input_dim]).float()
    batchTarget = torch.from_numpy(dOutputTransposed[gap:seq_size+gap,0:batch_size]).float()
    target_dim = 1
    # lstm_mod = nn.LSTM(input_dim, hidden_dim, num_layers, dropout = dropout, batch_first=True)
    # output_lstm, hidden_lstm = lstm_mod(Variable(batchInput))

    class LSTMPredictor(nn.Module):
        def __init__(self, hidden_dim, input_dim, num_layers, dout, bfirst, tagset_size):
            super(LSTMPredictor, self).__init__()
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dout, batch_first=bfirst)

            # The linear layer that maps from hidden state space to tag space
            self.hidden2output = nn.Linear(hidden_dim, tagset_size)
            self.hidden = self.init_hidden()

        def init_hidden(self):
            # Before we've done anything, we dont have any hidden state.
            # Refer to the Pytorch documentation to see exactly
            # why they have this dimensionality.
            # The axes semantics are (num_layers, minibatch_size, hidden_dim)
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

        def forward(self, input):        
            lstm_out, self.hidden = self.lstm(input)
            print(len(batchInput))
            net_out = self.hidden2output((lstm_out.contiguous().view(len(batchInput), -1)))

            net_softmaxout = functionalnn.log_softmax(net_out)
            net_finalout = net_out

            return net_finalout

    model = LSTMPredictor(hidden_dim, input_dim, num_layers, dropout, True, target_dim)
    #loss_function = nn.NLLLoss()
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    output_lstm = model(Variable(batchInput))
    print(output_lstm)

    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Variables of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    # See what the scores are after training
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    #  for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)