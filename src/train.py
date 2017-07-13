import config as config_file
import pandas as pd
import pickle 
import re
import matplotlib.pylab as plt
import time
from datetime import timedelta
from dateutil import parser
import numpy as np
import outcome_def_pediatric_obesity
import build_features

def filter_training_set_forLinear(x, y, ylabel, headers, filterSTR='', percentile=False):
	print(x.shape, len(headers))
	if filterSTR.__class__ == list:
		index_finder_filterstr = np.zeros(len(headers))
		for fstr in filterSTR:
			# print(index_finder_filterstr + np.array([h.startswith(fstr) for h in headers]))
			index_finder_filterstr_tmp = np.array([h.startswith(fstr) for h in headers])
			if index_finder_filterstr_tmp.sum() > 1:
				print('alert: filter returned more than one feature:', fstr)
				index_finder_filterstr_tmp = np.array([h == fstr for h in headers])
			index_finder_filterstr = index_finder_filterstr + index_finder_filterstr_tmp
		index_finder_filterstr = (index_finder_filterstr > 0)
	else:
		index_finder_filterstr = np.array([h.startswith(filterSTR) for h in headers])
	index_finder_maternal = np.array([h.startswith('Maternal') for h in headers])
	index_finder_diagnosis = np.array([h.startswith('Diagnosis') for h in headers])
	index_finder_vital = np.array([h.startswith('Vital_latest') for h in headers])

	if index_finder_filterstr.sum() > 1 and filterSTR.__class__ != list:
		print('instead of *startswith*',filterSTR,'...trying *equals to*', filterSTR)
		index_finder_filterstr = np.array([h == filterSTR for h in headers])

	if filterSTR != '' and percentile == False:
		ix = (y>0) & (y < 50) & ((x[:,index_finder_filterstr].sum(axis=1) >= len(filterSTR)).ravel()) #& (x[:, index_finder_diagnosis].sum(axis=1).ravel()>0) #& ((x[:,index_finder_maternal].sum(axis=1).ravel()>0) )
	elif percentile == False:
		ix = (y>0) & (y < 50) #& (x[:,index_finder_diagnosis].sum(axis=1).ravel()>0) #& ((x[:,index_finder_maternal].sum(axis=1).ravel()>0) ) #
		print(ix.sum())
		
	if (percentile == True) & (filterSTR != ''):
		ix = (x[:,index_finder_filterstr].ravel() == True) #& ((x[:,index_finder_diagnosis].sum(axis=1).ravel()>0) ) 
	elif percentile == True:
		ix = (x[:,index_finder_filterstr].ravel() >= False) #& ((x[:,index_finder_diagnosis].sum(axis=1).ravel()>0) ) # no filter
	print(str(ix.sum()) + ' patients selected..')
	return ix, x[ix,:], y[ix], ylabel[ix]

def filter_training_set_forLogit(x, y, ylabel, headers, filterSTR=''):
	index_finder_filterstr = np.array([h.startswith(filterSTR) for h in headers])
	index_finder_maternal = np.array([h.startswith('Maternal') for h in headers])
	if index_finder_filterstr.sum() > 1:
		# print('filter should be set to *startswith*',filterSTR,'...trying as *equals*')
		index_finder_filterstr = np.array([h == filterSTR for h in headers])

	if filterSTR != '':
		ix = (x[:,index_finder_filterstr].ravel() == True)
	else:
		ix = (x[:,index_finder_filterstr].ravel() >= False) # no filter
	return ix, x[ix,:], y[ix], ylabel[ix]

def train_regression(x, y, ylabel, percentile, modelType):
	import sklearn
	if modelType == 'lasso':
		import sklearn.linear_model
		from sklearn.linear_model import Lasso
	if modelType == 'mlp':
		from sklearn.neural_network import MLPRegressor
	if modelType == 'randomforest':
		from sklearn.ensemble import RandomForestRegressor
	N = x.shape[0]
	ixlist = np.arange(0,N)	
	import random
	from sklearn import metrics

	random.seed(2)
	random.shuffle(ixlist)
	xtrain = x[ixlist[0:int(N*2/3)], :]
	ytrain = y[ixlist[0:int(N*2/3)]]
	xtest = x[ixlist[int(N*2/3):],:]
	ytest =  y[ixlist[int(N*2/3):]]
	ytestlabel = ylabel[ixlist[int(N*2/3):]]
	ytrainlabel = ylabel[ixlist[0:int(N*2/3)]]	

	best_alpha = -1
	best_score = -10000
	if modelType == 'lasso':
		hyperparamlist = [0.001, 0.005, 0.01, 0.1]
	if modelType == 'mlp':
		hyperparamlist = [(10,), (50,), (10,10), (50,10), (100,)]
	if modelType == 'randomforest':
		hyperparamlist = [10, 50, 100]

	for alpha_i in hyperparamlist:
		if modelType == 'lasso':
			clf = Lasso(alpha=alpha_i)
		if modelType == 'mlp':
			clf = MLPRegressor(hidden_layer_sizes=alpha_i, solver="lbfgs",verbose=True)
		if modelType == 'randomforest':
			clf = RandomForestRegressor(random_state=0, n_estimators=alpha_i)
		clf.fit(xtrain, ytrain)	
		auc_test = metrics.roc_auc_score(ytestlabel, clf.predict(xtest))
		# print('CV AUC for alpha:', alpha_i, 'is:', auc_test)
		if auc_test > best_score:
			best_score = auc_test #np.sqrt(((clf.predict(xtest)-ytest)**2).mean())
			best_alpha = alpha_i
	
	print('best lasso alpha via CV:', best_alpha)
	
	if modelType == 'lasso':
		clf = Lasso(alpha=best_alpha)
	if modelType == 'mlp':
		clf = MLPRegressor(hidden_layer_sizes=best_alpha,solver="lbfgs",verbose=True)
	if modelType == 'randomforest':
		clf = RandomForestRegressor(random_state=0, n_estimators=best_alpha)

	clf.fit(xtrain,ytrain)
	
	# print('R^2 score train:',clf.score(xtrain,ytrain))
	# print('RMSE score train:', np.sqrt(((clf.predict(xtrain)-ytrain)**2).mean()))
	fpr, tpr, thresholds = metrics.roc_curve(ytrainlabel, clf.predict(xtrain))
	print('AUC train:',metrics.auc(fpr, tpr))
	fpr, tpr, thresholds = metrics.roc_curve(ytestlabel, clf.predict(xtest))
	# print('R^2 score test:',clf.score(xtest,ytest))
	# print('RMSE score test:',np.sqrt(((clf.predict(xtest)-ytest)**2).mean()))
	print('AUC test:',metrics.auc(fpr, tpr))

	return (clf, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel)

def train_logistic_regression(x, y):
	import sklearn
	import sklearn.linear_model
	from sklearn.linear_model import LogisticRegression
	from sklearn import metrics
	N = x.shape[0]
	ixlist = np.arange(0,N)	
	import random
	random.seed(2)
	random.shuffle(ixlist)
	xtrain = x[ixlist[0:int(N*2/3)], :]
	ytrain = y[ixlist[0:int(N*2/3)]]
	xtest = x[ixlist[int(N*2/3):],:]
	ytest=  y[ixlist[int(N*2/3):]]
	best_alpha = -1
	best_score = 10000
	for alpha_i in [ 0.01, 0.1]:
		clf = LogisticRegression(penalty='l1', C=alpha_i, class_weight='balanced')
		clf.fit(xtrain, ytrain)	
		fpr, tpr, thresholds = metrics.roc_curve(ytest, clf.predict_proba(xtest)[:,1])
		if metrics.auc(fpr, tpr) < best_score:
			best_score = metrics.auc(fpr, tpr)
			best_alpha = alpha_i
	print('best logit C via CV:', best_alpha)
	clf = LogisticRegression(penalty='l1', C=best_alpha, class_weight='balanced')
	clf.fit(xtrain,ytrain)
	fpr, tpr, thresholds = metrics.roc_curve(ytrain, clf.predict_proba(xtrain)[:,1])
	print('AUC train:',metrics.auc(fpr, tpr))
	fpr, tpr, thresholds = metrics.roc_curve(ytest, clf.predict_proba(xtest)[:,1])
	print('AUC test:',metrics.auc(fpr, tpr))
	return (clf, xtrain, ytrain, xtest, ytest)

def normalize(x):
	bin_ix = ( x.min(axis=0) == 0 ) & ( x.max(axis=0) == 1)
	xcop = x * 1.0
	xcop[xcop==0] = np.nan
	mu = np.nanmean(xcop, axis=0)
	mu[bin_ix] = 0.0
	mu[np.isnan(mu)] = 0.0
	std = np.nanstd(xcop, axis=0)
	std[std==0]=1.0
	std[bin_ix]=1.0
	std[np.isnan(std)]=1.0
	return (x != 0) * ((x - mu)/ std*1.0)  

def train_linear_model_for_bmi(data_dic, data_dic_mom, agex_low, agex_high, months_from, months_to, modelType='lasso', percentile=False, filterSTR=''): #filterSTR='Gender:0 male'
	x1, y1, y1label, feature_headers = build_features.call_build_function(data_dic,data_dic_mom, agex_low, agex_high, months_from, months_to, percentile)
	ix, x2, y2, y2label = filter_training_set_forLinear(x1, y1, y1label, feature_headers, filterSTR, percentile)
	x2 = normalize(x2)

	print ('Predicting BMI at age:'+str(agex_low)+ '-'+str(agex_high)+ ' from data in ages:'+ str(months_from)+'-'+str(months_to*-1) + '')
	if filterSTR != '':
		print ('filtering patients with: ' , filterSTR)

	print ('total size',ix.sum())
	if (ix.sum() < 100):
		print('Not enough subjects. Next.')
		return (filterSTR, [])
	(model, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel) = train_regression(x2, y2, y2label, percentile, modelType)
	
	if modelType == 'lasso':
		model_weights = model.coef_
	if modelType == 'randomforest':
		model_weights = model.feature_importances_
	if modelType == 'mlp':
		print ('you need to implement gradient to get top weights. ')
		return (filterSTR, [])

	sorted_ix = np.argsort(-1* abs(model_weights))
	weights = model_weights[sorted_ix]
	factors = np.array(feature_headers)[sorted_ix]
	x2_reordered = x2[:,sorted_ix]
	print('total variables', x2.sum(axis=0).shape, ' and total subjects:', x2.shape[0])
	occurances = x2.sum(axis=0)[sorted_ix]
	zip_weights = {}
	sig_headers = []
	for i in range(0, (abs(model_weights)>0).sum()):
		tp = ((y2label > 0) & (x2_reordered[:,i].ravel() > 0)).sum()*1.0
		tn = ((y2label == 0) & (x2_reordered[:,i].ravel() == 0)).sum()*1.0
		fp = ((y2label > 0) & (x2_reordered[:,i].ravel() == 0)).sum()*1.0
		fn = ((y2label == 0) & (x2_reordered[:,i].ravel() > 0)).sum()*1.0
		if fp*fn*tp*tn == 0:
			oratio = np.nan
			low_OR = np.nan
			high_OR = np.nan
		else:
			oratio = tp*tn/(fp*fn)
			se = np.sqrt(1/tp + 1/fp + 1/tn + 1/fn)
			low_OR = np.exp(np.log(oratio) - 1.96 * se)
			high_OR = np.exp(np.log(oratio) + 1.96 * se)
		if low_OR >1 or high_OR < 1:
			print("weight: {0:4.3f} | {1} | occ: {2} | OR: {3:4.3f} [{4:4.3f} {5:4.3f}]".format(weights[i], factors[i], occurances[i], oratio, low_OR, high_OR))
			sig_headers.append(factors[i])

	return (filterSTR, sig_headers)
		# print('weight:' + str(weights[i]) + ' | ' + factors[i] + ' | occ:' + str(occurances[i])) + ' | OR:' + str(oratio) + ' [' + str(low_OR) + ' ' + str(high_OR) +']' 
	# 	if factors[i].startswith('Zipcode'):
	# 		zip_weights[factors[i].split(':')[1]] = weights[i]
	# return zip_weights
	#return (model, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, feature_headers)

def train_chain(data_dic, data_dic_mom, agex_low, agex_high, months_from, months_to, modelType='lasso', percentile=False, filterSTR=''):
	notDone = True
	while notDone == True:
		print('')
		(flist, sig_headers) = train_linear_model_for_bmi(data_dic, data_dic_mom, agex_low, agex_high, months_from, months_to, modelType, percentile, filterSTR)
		for s in sig_headers:
			if s in flist:
				continue
			flist_copy = flist.copy()
			flist_copy.append(s)
			if len(flist_copy) > 4 :
				return
			train_chain(data_dic, data_dic_mom, agex_low, agex_high, months_from, months_to, modelType, percentile, flist_copy)
		return


def train_logit_model_for_bmi(data_dic,data_dic_mom, agex_low, agex_high, months_from, months_to, percentile=False, filterSTR=''): #filterSTR='Gender:0 male'
	x1, y1, y1label, feature_headers = call_build_function(data_dic, data_dic_mom, agex_low, agex_high, months_from, months_to, percentile)
	ix, x2, y2, y2label= filter_training_set_forLogit(x1, y1, y1label, feature_headers, filterSTR)
	x2 = normalize(x2)

	print ('Predicting Obesity at age:'+str(agex_low)+ '-'+str(agex_high)+ ' from data in ages:'+ str(months_from)+'-'+str(months_to*-1) + '')
	if filterSTR != '':
		print ('filtering patients with: ' + filterSTR)

	print ('total size',ix.sum(), 'total positives', y2label.sum())
	if (ix.sum() < 100):
		print('Not enough subjects. Next.')
		return (filterSTR, [])
	(model, xtrain, ytrain, xtest, ytest) = train_logistic_regression(x2, y2label)
	print('size of model coef', model.coef_.ravel().shape)
	sorted_ix = np.argsort(-1* abs(model.coef_.ravel()))
	weights = model.coef_.ravel()[sorted_ix]
	factors = np.array(feature_headers)[sorted_ix]
	print('total variables', x2.sum(axis=0).shape, ' and total subjects:', x2.shape[0])
	occurances = x2.sum(axis=0)[sorted_ix]
	zip_weights = {}
	for i in range(0, (abs(model.coef_.ravel())>0).sum()):
		print('weight:' + str(weights[i])+' | ' + factors[i] + ' | occ:' + str(occurances[i]))
	# 	if factors[i].startswith('Zipcode'):
	# 		zip_weights[factors[i].split(':')[1]] = weights[i]
	# return zip_weights
