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
import random
from sklearn import metrics


def filter_training_set_forLinear(x, y, ylabel, headers, filterSTR='', percentile=False):
	print('x shape:', x.shape, 'num features:',len(headers))
	index_finder_anyvital = np.array([h.startswith('Vital') for h in headers])
	index_finder_maternal = np.array([h.startswith('Maternal') for h in headers])
	if filterSTR.__class__ == list:
		index_finder_filterstr = np.zeros(len(headers))
		for fstr in filterSTR:
			# print(index_finder_filterstr + np.array([h.startswith(fstr) for h in headers]))
			index_finder_filterstr_tmp = np.array([h.startswith(fstr) for h in headers])
			if index_finder_filterstr_tmp.sum() > 1:
				print('alert: filter returned more than one feature:', fstr)
				index_finder_filterstr_tmp = np.array([h == fstr for h in headers])
				print('set filter to h==', fstr)
			index_finder_filterstr = index_finder_filterstr + index_finder_filterstr_tmp
		index_finder_filterstr = (index_finder_filterstr > 0)
	else:
		index_finder_filterstr = np.array([h.startswith(filterSTR) for h in headers])

	if index_finder_filterstr.sum() > 1 and filterSTR.__class__ != list:
		print('instead of *startswith*',filterSTR,'...trying *equals to*', filterSTR)
		index_finder_filterstr = np.array([h == filterSTR for h in headers])

	if filterSTR != '' and percentile == False:
		ix = (y>10) & (y < 40) & (((x[:,index_finder_filterstr]!=0).sum(axis=1) >= len(filterSTR)).ravel()) & ((x[:,index_finder_anyvital] != 0).sum(axis=1) >= 1) & ((x[:,index_finder_maternal] != 0).sum(axis=1) >= 1)
	elif percentile == False:
		ix = (y>10) & (y < 40) 
		print(ix.sum())
		
	if (percentile == True) & (filterSTR != ''):
		ix = (x[:,index_finder_filterstr].ravel() == True) 
	elif percentile == True:
		ix = (x[:,index_finder_filterstr].ravel() >= False) 
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

def train_regression(x, y, ylabel, percentile, modelType, feature_headers):
	import sklearn
	if modelType == 'lasso':
		import sklearn.linear_model
		from sklearn.linear_model import Lasso
	if modelType == 'mlp':
		from sklearn.neural_network import MLPRegressor
	if modelType == 'randomforest':
		from sklearn.ensemble import RandomForestRegressor
	if modelType == 'temporalCNN':
		import cnn

	N = x.shape[0]
	ixlist = np.arange(0,N)	
	

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
		hyperparamlist = [(2000,10), (2000,50), (2000,100)]
	if modelType == 'temporalCNN':
		hyperparamlist = [(0.1)]

	for alpha_i in hyperparamlist:
		if modelType == 'lasso':
			clf = Lasso(alpha=alpha_i)
		if modelType == 'mlp':
			clf = MLPRegressor(hidden_layer_sizes=alpha_i, solver="lbfgs", verbose=True)
		if modelType == 'randomforest':
			clf = RandomForestRegressor(random_state=0, n_estimators=alpha_i[0], min_samples_split=alpha_i[1])
		if modelType == 'temporalCNN':
			# xcnndataTrain, xcnndataTest = xtrain.reshape(, xtest # need to be of size |vitals| x |time| x 
			clf = cnn.TemporalCNN(5, 8, 8, 64, 1)
			return (clf, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, 0, 0)
		clf.fit(xtrain, ytrain)	
		auc_test = metrics.roc_auc_score(ytestlabel, clf.predict(xtest))
		print('CV AUC for alpha:', alpha_i, 'is:', auc_test)
		if auc_test > best_score:
			best_score = auc_test #np.sqrt(((clf.predict(xtest)-ytest)**2).mean())
			best_alpha = alpha_i
	
	print('best alpha via CV:', best_alpha)
	
	if modelType == 'lasso':
		clf = Lasso(alpha=best_alpha)
	if modelType == 'mlp':
		clf = MLPRegressor(hidden_layer_sizes=best_alpha,solver="lbfgs",verbose=True)
	if modelType == 'randomforest':
		clf = RandomForestRegressor(random_state=0, n_estimators=best_alpha[0], min_samples_split=best_alpha[1])

	clf.fit(xtrain,ytrain)
	
	# print('R^2 score train:',clf.score(xtrain,ytrain))
	# print('RMSE score train: {0:4.3f}'.format(np.sqrt(((clf.predict(xtrain)-ytrain)**2).mean())))
	fpr, tpr, thresholds = metrics.roc_curve(ytrainlabel, clf.predict(xtrain))
	print('AUC train: {0:4.3f}'.format(metrics.auc(fpr, tpr)) + ' Explained Variance Score Train: {0:4.3f}'.format(metrics.explained_variance_score(ytrain, clf.predict(xtrain)))) #http://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score
	fpr, tpr, thresholds = metrics.roc_curve(ytestlabel, clf.predict(xtest))
	auc_test = metrics.auc(fpr, tpr); r2test = clf.score(xtest,ytest)
	# print('R^2 score test:',clf.score(xtest,ytest))
	# print('RMSE score test: {0:4.3f}'.format(np.sqrt(((clf.predict(xtest)-ytest)**2).mean())))
	print('AUC test: {0:4.3f}'.format(metrics.auc(fpr, tpr))+' Explained Variance Score Test: {0:4.3f}'.format(metrics.explained_variance_score(ytest, clf.predict(xtest))))
	return (clf, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, auc_test, r2test)

def normalize(x, filter_percentile_more_than_percent=5):
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
	normed_x = (x != 0) * ((x - mu)/ std*1.0)
	normed_x[abs(normed_x)>filter_percentile_more_than_percent] = 0
	return normed_x

def variable_subset(x, varsubset, h):
	print('subsetting variables that are only:', varsubset)
	hix = np.array([hi.split(':')[0].strip() in varsubset or hi in varsubset for hi in h])
	print('from ', x.shape[1] ,' variables to ', sum(hix))
	x = x[:, hix]
	h = np.array(h)[hix]
	print(h, x)
	return x, h

def add_temporal_features(x2, feature_headers, num_clusters, num_iters, y2, y2label, distType='eucledian'):
	#frst make sure only vital values are in x2.
	if feature_headers.__class__ == list:
		feature_headers = np.array(feature_headers)
	header_vital_ix = [h.startswith('Vital') for h in feature_headers]
	headers_vital = feature_headers[header_vital_ix]
	x2_vitals = x2[:, header_vital_ix]
	import timeseries
	xnew, hnew = timeseries.load_temporal_data(x2_vitals, headers_vital, y2, y2label)
	centroids, assignments, trendArray, standardDevCentroids, cnt_clusters = timeseries.k_means_clust(xnew, num_clusters, num_iters, hnew, distType)
	trend_headers = ['Trend:'+str(i)+' -occ:'+str(cnt_clusters[i]) for i in range(0, len(centroids))]
	return np.hstack([x2, trendArray]), np.hstack([feature_headers , np.array(trend_headers)]), centroids, hnew, standardDevCentroids, cnt_clusters

def train_regression_model_for_bmi(data_dic, data_dic_mom, agex_low, agex_high, months_from, months_to, modelType='lasso', percentile=False, filterSTR=['Gender:1'], variablesubset=['Vital'], num_clusters=20, num_iters=100, distType='euclidean'): #filterSTR='Gender:0 male'
	x1, y1, y1label, feature_headers = build_features.call_build_function(data_dic,data_dic_mom, agex_low, agex_high, months_from, months_to, percentile)
	ix, x2, y2, y2label = filter_training_set_forLinear(x1, y1, y1label, feature_headers, filterSTR, percentile)
	x2, feature_headers, centroids, hnew, standardDevCentroids, cnt_clusters = add_temporal_features(x2, feature_headers, num_clusters, num_iters, y2, y2label, distType)
	x2 = normalize(x2)
	if len(variablesubset) != 0:
		x2, feature_headers = variable_subset(x2, variablesubset, feature_headers)
	print('output is: average:{0:4.3f}'.format(y2.mean()), ' min:', y2.min(), ' max:', y2.max())
	print ('normalizing output.'); y2 = (y2-y2.mean())/y2.std()

	print ('Predicting BMI at age:'+str(agex_low)+ ' to '+str(agex_high)+ 'years, from data in ages:'+ str(months_from)+'-'+str(months_to) + ' months')
	if filterSTR != '':
		print ('filtering patients with: ' , filterSTR)

	print ('total size',ix.sum())
	if (ix.sum() < 150):
		print('Not enough subjects. Next.')
		return (filterSTR, [])
	if modelType == 'lasso':
		iters = 5
		model_weights_array = np.zeros((iters, x2.shape[1]), dtype=float)
		auc_test_list=np.zeros((iters), dtype=float); r2testlist = np.zeros((iters), dtype=float);
		for iteration in range(0,iters):
			randix = list(range(0, x2.shape[0]))
			random.shuffle(randix)
			randix = randix[0:int(len(randix)*0.9)]
			datax = x2[randix,:]; datay=y2[randix]; dataylabel = y2label[randix]
			(model, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, auc_test, r2test) = train_regression(datax, datay, dataylabel, percentile, modelType, feature_headers)
			model_weights_array[iteration, :] = model.coef_
			auc_test_list[iteration] = auc_test; r2testlist[iteration] = r2test

		model_weights = model_weights_array.mean(axis=0)
		model_weights_std = model_weights_array.std(axis=0)
		model_weights_conf_term = (1.96/np.sqrt(iters)) * model_weights_std
		test_auc_mean = auc_test_list.mean()
		test_auc_mean_ste = (1.96/np.sqrt(iters)) * auc_test_list.std()
		r2test_mean = r2testlist.mean()
		r2test_ste = (1.96/np.sqrt(iters)) * r2testlist.std()
	else:
		(model, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, auc_test, r2test) = train_regression(x2, y2, y2label, percentile, modelType, feature_headers)	
		model_weights_conf_term = np.zeros((x2.shape[1]), dtype=float)
		test_auc_mean = auc_test; r2test_mean= r2test;
		test_auc_mean_ste = 0; r2test_ste=0

	if modelType == 'randomforest':
		model_weights = model.feature_importances_
		model_weights_conf_term = model_weights*0

	if modelType == 'mlp':
		print ('you need to implement gradient to get top weights. ')
		return (filterSTR, [])

	sorted_ix = np.argsort(-1* abs(model_weights))
	weights = model_weights[sorted_ix]
	terms_sorted = model_weights_conf_term[sorted_ix]

	factors = np.array(feature_headers)[sorted_ix]
	x2_reordered = x2[:,sorted_ix]
	xtest_reordered = xtest[:, sorted_ix]
	print('total variables', x2.sum(axis=0).shape, ' and total subjects:', x2.shape[0])
	print('->AUC test: {0:4.3f} [{1:4.3f} {2:4.3f}]'.format(test_auc_mean, test_auc_mean - test_auc_mean_ste, test_auc_mean + test_auc_mean_ste))
	print('->Explained Variance (R2) test: {0:4.3f} [{1:4.3f} {2:4.3f}]'.format(r2test_mean, r2test_mean - r2test_ste,  r2test_mean + r2test_ste))

	occurances = (x2 != 0).sum(axis=0)[sorted_ix]
	zip_weights = {}
	sig_headers = []
	feature_categories = {}
	for i in range(0, (abs(model_weights)>0).sum()):
		fpr, tpr, thresholds = metrics.roc_curve(ytestlabel, xtest_reordered[:,i].ravel())
		feature_auc_indiv = metrics.auc(fpr, tpr)

		tp = ((y2label > 0) & (x2_reordered[:,i].ravel() > 0)).sum()*1.0
		tn = ((y2label == 0) & (x2_reordered[:,i].ravel() <= 0)).sum()*1.0
		fp = ((y2label > 0) & (x2_reordered[:,i].ravel() <= 0)).sum()*1.0
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
		try:
			feature_categories[factors[i].split(':')[0]] += weights[i]
		except:
			feature_categories[factors[i].split(':')[0]] = weights[i]

		star = ' '	
		if (low_OR > 1 or high_OR < 1): #or (weights[i]+terms_sorted[i]) < 0 or (weights[i]-terms_sorted[i]) > 0
			sig_headers.append(factors[i])
			star = '*'
		print("{8} {3} | coef {0:4.3f} [{1:4.3f} {2:4.3f}] | OR_adj {9:4.3f} [{10:4.3f} {11:4.3f}] | occ: {4} | OR_unadj: {5:4.3f} [{6:4.3f} {7:4.3f}] | indiv AUC: {12:4.3f}".format(weights[i], weights[i]-terms_sorted[i], weights[i]+terms_sorted[i], factors[i], occurances[i], oratio, low_OR, high_OR, star, np.exp(weights[i]), np.exp(weights[i]-terms_sorted[i]), np.exp(weights[i]+terms_sorted[i]), feature_auc_indiv))

	for k in feature_categories:
		print (k, ":", feature_categories[k])
	return (filterSTR, sig_headers,  centroids, hnew, standardDevCentroids, cnt_clusters) 
	
def train_chain(data_dic, data_dic_mom, agex_low, agex_high, months_from, months_to, modelType='lasso', percentile=False, filterSTR=''):
	notDone = True
	while notDone == True:
		print('')
		(flist, sig_headers, centroids, hnew) = train_regression_model_for_bmi(data_dic, data_dic_mom, agex_low, agex_high, months_from, months_to, modelType, percentile, filterSTR)
		for s in sig_headers:
			if s in flist or s.startswith('Vital:'):
				continue
			flist_copy = flist.copy()
			flist_copy.append(s)
			if len(flist_copy) > 4 :
				return
			train_chain(data_dic, data_dic_mom, agex_low, agex_high, months_from, months_to, modelType, percentile, flist_copy)
		return
