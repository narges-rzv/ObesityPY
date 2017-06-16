import tensorflow as tf
import numpy as np
import pickle


def load_data(TimeIn=[0,2*12], Timeout=[int(4.5*12), int(5.5*12)], outcomeIx=0):
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
	return dInput, dOutput, dkselected, dgselecte

def split_train_valid_test(dInput, dOutput, dkselected, dgselected):
	
def build_train_lstm_model(batch_size=32, num_steps=24, lstm_size=64, dInput, dOutput, dkselected, dgselected):
	x = tf.placeholder(tf.int32, [batch_size, num_steps])
	lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
	#stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * number_of_layers,state_is_tuple=False)
	initial_state = state = tf.zeros([batch_size, lstm.state_size])
	for i in range(num_steps):
		output, state = lstm(x[:, i], state) 
		#output, state = stacked_lstm(words[:, i], state)
	final_state = state

	numpy_state = initial_state.eval()
	total_loss = 0.0
	for i in range(0, dInput.shape[0]):
		numpy_state, current_loss = session.run([final_state, loss],feed_dict={initial_state: numpy_state, x: dInput[i]})
		total_loss += current_loss
	




