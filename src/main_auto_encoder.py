from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

def loat_data():
	import pickle
	outgirls24 = pickle.load(open('outgirls_24_big.pkl', 'rb'))
	(model, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, auc_test, r2test, feature_headers, centroids, hnew, standardDevCentroids, cnt_clusters, distances, muxnew, stdxnew, mrnstrain, mrnstest) = outgirls24
	non_zero_ix = (xtrain.sum(axis=0) + xtest.sum(axis=0) > 0)
	xtrain = xtrain[:, non_zero_ix]
	xtest = xtest[:, non_zero_ix]
	feature_headers = list(np.array(feature_headers)[non_zero_ix])
	xtrainNorm, mu, sg, bin_ix, unobserved = normalize(xtrain)
	xtestNorm, mutest, sgtest, bin_ix_test, unobserved_test = normalize(xtest, mu=mu, std=sg, bin_ix=bin_ix)

	cont_ix = (np.array(bin_ix) == False)
	xtrain_bin = xtrain[:,bin_ix]

	xtrainNorm_cont = xtrainNorm[:,cont_ix]
	xtrainNorm_bin = xtrainNorm[:,bin_ix]
	unobserved_bin = unobserved[:, bin_ix]
	unobserved_cont = unobserved[:,cont_ix]

	xtestNorm_cont = xtestNorm[:,cont_ix]
	xtestNorm_bin = xtestNorm[:,bin_ix]
	unobservedtest_bin = unobserved_test[:, bin_ix]
	unobservedtest_cont = unobserved_test[:,cont_ix]

	print(xtrainNorm.shape, xtestNorm.shape, xtestNorm_bin.shape, xtestNorm_cont.shape)
	
	return(outgirls24, feature_headers, xtrain, xtest, non_zero_ix, cont_ix, xtrain_bin, xtrainNorm, mu, sg, bin_ix, unobserved, xtrainNorm_cont, xtestNorm, mutest, sgtest, bin_ix_test, unobserved_test, xtrainNorm_cont, xtrainNorm_bin,unobserved_bin, unobserved_cont, xtestNorm_cont, xtestNorm_cont, unobservedtest_bin, unobservedtest_cont)


def normalize(x, filter_percentile_more_than_percent=5, mu=[], std=[], bin_ix=[]):
    unobserved = (x == 0)*1.0
    if len(bin_ix) == 0:
        bin_ix = ( x.min(axis=0) == 0 ) & ( x.max(axis=0) == 1)
    xcop = x * 1.0
    xcop[xcop==0] = np.nan
    if len(mu) == 0:
        mu = np.nanmean(xcop, axis=0)
        mu[bin_ix] = 0.0
        mu[np.isnan(mu)] = 0.0
    if len(std) == 0:
        std = np.nanstd(xcop, axis=0)
        std[std==0]=1.0
        std[bin_ix]=1.0
        std[np.isnan(std)]=1.0

    normed_x = (x != 0) * ((x - mu)/ std*1.0)
    normed_x[abs(normed_x)>filter_percentile_more_than_percent] = 0
    return normed_x, mu, std, bin_ix, unobserved

def train_auto_encoder_binary_cont(outgirls24, feature_headers, xtrain, xtest, non_zero_ix, cont_ix, xtrain_bin, xtrainNorm, mu, sg, bin_ix, unobserved, xtrainNorm_cont, xtestNorm, mutest, sgtest, bin_ix_test, unobserved_test, xtrainNorm_cont, xtrainNorm_bin,unobserved_bin, unobserved_cont, xtestNorm_cont, xtestNorm_cont, unobservedtest_bin, unobservedtest_cont):
	auto2 = AutoencoderConinBinar(xtrainNorm_bin.shape[1], xtrainNorm_cont.shape[1], 100)
	optimizer = optim.SGD(auto2.parameters(), lr=0.5)
	np.random.seed(0)
	lossfuncBin = nn.BCELoss()
	lossfunccont = nn.MSELoss()
	for epoch in range(1, 200):
	    auto2.train()
	    for ix in range(len(xtrainNorm_bin)):
	        databin = Variable(torch.from_numpy(xtrainNorm_bin[ix]).float())
	        datacont = Variable(torch.from_numpy(xtrainNorm_cont[ix]).float())
	        databoth = Variable(torch.from_numpy(np.hstack([xtrainNorm_bin[ix], xtrainNorm_cont[ix]]))).float()
	        optimizer.zero_grad()
	        xtrainoutBin, xtrainoutCont = auto2(databoth)
	        loss = lossfuncBin(xtrainoutBin, databin) + lossfunccont(xtrainoutCont, datacont)
	        loss_list.append(loss)
	        loss.backward()
	        optimizer.step()
	        
	auto2.eval()
	xtrainout = np.zeros(xtrainNorm.shape)
	for ix in range(len(xtrainNorm_bin)):
	    databoth = Variable(torch.from_numpy(np.hstack([xtrainNorm_bin[ix], xtrainNorm_cont[ix]]))).float()
	    outbin, outcont = auto2(databoth)
	    xtrainout[ix,bin_ix] = outbin.data.numpy()
	    xtrainout[ix,cont_ix] = outcont.data.numpy()

	xtestout = np.zeros(xtestNorm.shape)
	for ix in range(len(xtestNorm_bin)):
	    databoth = Variable(torch.from_numpy(np.hstack([xtestNorm_bin[ix], xtestNorm_cont[ix]]))).float()
	    outbin, outcont = auto2(databoth)
	    xtestout[ix,bin_ix] = outbin.data.numpy()
	    xtestout[ix,cont_ix] = outcont.data.numpy()

	plt.imshow(xtrainout, aspect='auto', interpolation='nearest'); plt.colorbar()
	plt.show()
	plt.imshow(xtrainNorm, aspect='auto', interpolation='nearest'); plt.colorbar()
	plt.show()


	plt.imshow(xtestout, aspect='auto', interpolation='nearest'); plt.colorbar()
	plt.show()
	plt.imshow(xtestNorm, aspect='auto', interpolation='nearest'); plt.colorbar()
	plt.show()

	return (outgirls24, feature_headers, xtrain, xtest, non_zero_ix, cont_ix, xtrain_bin, xtrainNorm, mu, sg, bin_ix, unobserved, xtrainNorm_cont, xtestNorm, mutest, sgtest, bin_ix_test, unobserved_test, xtrainNorm_cont, xtrainNorm_bin,unobserved_bin, unobserved_cont, xtestNorm_cont, xtestNorm_cont, unobservedtest_bin, unobservedtest_cont, xtestout, xtrainout)


def train_auto_encoder_binary_cont_with_mask(outgirls24, feature_headers, xtrain, xtest, non_zero_ix, cont_ix, xtrain_bin, xtrainNorm, mu, sg, bin_ix, unobserved, xtrainNorm_cont, xtestNorm, mutest, sgtest, bin_ix_test, unobserved_test, xtrainNorm_cont, xtrainNorm_bin,unobserved_bin, unobserved_cont, xtestNorm_cont, xtestNorm_cont, unobservedtest_bin, unobservedtest_cont):




def run():
	(outgirls24, feature_headers, xtrain, xtest, non_zero_ix, cont_ix, xtrain_bin, xtrainNorm, mu, sg, bin_ix, unobserved, xtrainNorm_cont, xtestNorm, mutest, sgtest, bin_ix_test, unobserved_test, xtrainNorm_cont, xtrainNorm_bin,unobserved_bin, unobserved_cont, xtestNorm_cont, xtestNorm_cont, unobservedtest_bin, unobservedtest_cont) = load_data()
