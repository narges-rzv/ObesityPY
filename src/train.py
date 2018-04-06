import os
import re
import time
import pickle
import random
import zscore
import matplotlib
matplotlib.use('TkAgg')
import build_features
import numpy as np
import pandas as pd
import config as config_file
import matplotlib.pylab as plt
import outcome_def_pediatric_obesity
from sklearn import metrics
from scipy.stats import norm
from sklearn.preprocessing import Imputer
from dateutil import parser
from datetime import timedelta
from dateutil.relativedelta import relativedelta
random.seed(2)

g_wfl = np.loadtxt(config_file.wght4leng_girl)
b_wfl = np.loadtxt(config_file.wght4leng_boy)

def filter_training_set_forLinear(x, y, ylabel, headers, filterSTR=[], percentile=False, mrns=[], filterSTRThresh=[], print_out=True):
    if filterSTR.__class__ == list:
        pass
    else:
        filterSTR = [filterSTR]

    if len(filterSTRThresh) != len(filterSTR):
        filterSTRThresh = []

    if len(filterSTRThresh) == 0 :
        filterSTRThresh = [0.5]*len(filterSTR) #make it binary, as before.

    if print_out:
        print('Original cohort size is:', x.shape[0], 'num features:',len(headers))
    else:
        print_statements = 'Original cohort size: {0:,d}, number of features: {1:,d}\n'.format(x.shape[0], len(headers))

    index_finder_anyvital = np.array([h.startswith('Vital') for h in headers])
    index_finder_maternal = np.array([h.startswith('Maternal') for h in headers])

    index_finder_filterstr = np.zeros(len(headers))

    for i, fstr in enumerate(filterSTR):
        # print(index_finder_filterstr + np.array([h.startswith(fstr) for h in headers]))
        index_finder_filterstr_tmp = np.array([h.startswith(fstr) for h in headers])
        if index_finder_filterstr_tmp.sum() > 1:
            if print_out:
                print('alert: filter returned more than one feature:', fstr)
                index_finder_filterstr_tmp = np.array([h == fstr for h in headers])
                print('set filter to h==', fstr)
            else:
                print_statements += 'alert: filter returned more than one feature: ' + str(fstr) + '\n'
                index_finder_filterstr_tmp = np.array([h == fstr for h in headers])
                print_statements += 'set filter to h==' + str(fstr) + '\n'
        index_finder_filterstr = index_finder_filterstr + index_finder_filterstr_tmp
        if print_out:
            print('total number of people who have: ', np.array(headers)[index_finder_filterstr_tmp], ' is:', ( x[:,index_finder_filterstr_tmp].ravel() > filterSTRThresh[i] ).sum() )
        else:
            print_statements += 'total number of people who have: '+str(np.array(headers)[index_finder_filterstr_tmp])+' is: {0:,d}\n'.format((x[:,index_finder_filterstr_tmp].ravel() > filterSTRThresh[i]).sum())

    index_finder_filterstr = (index_finder_filterstr > 0)

    # if index_finder_filterstr.sum() > 1 and filterSTR.__class__ != list:
    #     print('instead of *startswith*',filterSTR,'...trying *equals to*', filterSTR)
    #     index_finder_filterstr = np.array([h == filterSTR for h in headers])

    # import pdb
    # pdb.set_trace()
    if (len(filterSTR) != 0) and (percentile == False):
        ix = (y > 10) & (y < 40) & (((x[:,index_finder_filterstr] > np.array(filterSTRThresh)).sum(axis=1) >= index_finder_filterstr.sum()).ravel()) & ((x[:,index_finder_maternal] != 0).sum(axis=1) >= 1)
    if print_out:
        print('total number of people who have a BMI measured:', sum((y > 10) & (y < 40)))
        print('total number of people who have all filtered variables:', (((x[:,index_finder_filterstr] > np.array(filterSTRThresh)).sum(axis=1) >= index_finder_filterstr.sum()).ravel()).sum())
        print('total number of people who have maternal data available:', ((x[:,index_finder_maternal] != 0).sum(axis=1) > 0).sum() )
        print('intersection of the three above is:', sum(ix))
        print(str(ix.sum()) + ' patients selected..')
        return ix, x[ix,:], y[ix], ylabel[ix], mrns[ix]

    # elif percentile == False:
    #     ix = (y > 10) & (y < 40) & ((x[:,index_finder_anyvital] != 0).sum(axis=1) >= 1)
    #     print(ix.sum())

    # if (percentile == True) & (len(filterSTR) != 0):
    #     ix = (x[:,index_finder_filterstr].ravel() == True)
    # elif percentile == True:
    #     ix = (x[:,index_finder_filterstr].ravel() >= False)
    else:
        print_statements += 'total number of people who have a BMI measured: {0:,d}\n'.format(sum((y > 10) & (y < 40)))
        print_statements += 'total number of people who have all filtered variables: {0:,d}\n'.format((((x[:,index_finder_filterstr] > np.array(filterSTRThresh)).sum(axis=1) >= index_finder_filterstr.sum()).ravel()).sum())
        print_statements += 'total number of people who have maternal data available: {0:,d}\n'.format(((x[:,index_finder_maternal] != 0).sum(axis=1) > 0).sum())
        print_statements += 'intersection of the three above is: {0:,d}\n'.format(sum(ix))
        print_statements += '{0:,d} patients selected..\n\n'.format(ix.sum())
        return ix, x[ix,:], y[ix], ylabel[ix], mrns[ix], print_statements

def train_regression(x, y, ylabel, percentile, modelType, feature_headers, mrns):
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
    if modelType == 'gradientboost':
        from sklearn.ensemble import GradientBoostingRegressor
    if modelType == 'lars':
        from sklearn import linear_model
    N = x.shape[0]
    ixlist = np.arange(0,N)


    random.shuffle(ixlist)
    ix_train = ixlist[0:int(N*2/3)]
    ix_test = ixlist[int(N*2/3):]
    xtrain = x[ix_train]
    ytrain = y[ix_train]
    xtest = x[ix_test]
    ytest =  y[ix_test]
    ytestlabel = ylabel[ix_test]
    ytrainlabel = ylabel[ix_train]
    mrnstrain = mrns[ix_train]
    mrnstest = mrns[ix_test]

    best_alpha = -1
    best_score = -10000
    if modelType == 'lasso':
        hyperparamlist = [0.001, 0.005, 0.01, 0.1] #[alpha]
    if modelType == 'mlp':
        hyperparamlist = [(10,), (50,), (10,10), (50,10), (100,)] #[hidden_layer_sizes]
    if modelType == 'randomforest':
        hyperparamlist = [(est,minSplit,minLeaf) for est in [3000] for minSplit in [2] for minLeaf in (1,2,5,7)] #(2000,2), (2000,4), (2000,10) #[n_estimators, min_samples_split, min_samples_leaf]
    if modelType == 'temporalCNN':
        hyperparamlist = [(0.1)]
    if modelType == 'gradientboost':
        hyperparamlist = [(1500, 4, 2, 0.01,'lad'), (2500, 4, 2, 0.01,'lad'), (3500, 4, 2, 0.01,'lad')] #[n_estimators, max_depth, min_samples_split, learning_rate, loss]
    if modelType == 'lars':
        hyperparamlist = [0.001, 0.01, 0.1]

    for alpha_i in hyperparamlist:
        if modelType == 'lasso':
            clf = Lasso(alpha=alpha_i)
        if modelType == 'mlp':
            clf = MLPRegressor(hidden_layer_sizes=alpha_i, solver="lbfgs", verbose=True)
        if modelType == 'randomforest':
            clf = RandomForestRegressor(random_state=0, n_estimators=alpha_i[0], min_samples_split=alpha_i[1], min_samples_leaf=alpha_i[2], n_jobs=-1)
        if modelType == 'gradientboost':
            clf = GradientBoostingRegressor(n_estimators=alpha_i[0], max_depth=alpha_i[1], min_samples_split=alpha_i[2], learning_rate=alpha_i[3], loss=alpha_i[4])
        if modelType == 'lars':
            clf = linear_model.LassoLars(alpha=alpha_i)
        # if modelType == 'temporalCNN':
            # xcnndataTrain, xcnndataTest = xtrain.reshape(, xtest # need to be of size |vitals| x |time| x
            # clf = cnn.TemporalCNN(5, 8, 8, 64, 1)
            # return (clf, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, 0, 0)
        clf.fit(xtrain, ytrain)
        auc_test = metrics.explained_variance_score(ytest, clf.predict(xtest)) #roc_auc_score(ytestlabel, clf.predict(xtest))
        print('CV R^2 for alpha:', alpha_i, 'is:', auc_test)
        if auc_test > best_score:
            best_score = auc_test #np.sqrt(((clf.predict(xtest)-ytest)**2).mean())
            best_alpha = alpha_i

    print('best alpha via CV:', best_alpha)

    if modelType == 'lasso':
        clf = Lasso(alpha=best_alpha)
    if modelType == 'mlp':
        clf = MLPRegressor(hidden_layer_sizes=best_alpha,solver="lbfgs", verbose=True)
    if modelType == 'randomforest':
        clf = RandomForestRegressor(random_state=0, n_estimators=best_alpha[0], min_samples_split=best_alpha[1], min_samples_leaf=best_alpha[2], n_jobs=-1)
    if modelType == 'gradientboost':
        clf = GradientBoostingRegressor(n_estimators=best_alpha[0], max_depth=best_alpha[1], min_samples_split=best_alpha[2], learning_rate=best_alpha[3], loss=best_alpha[4])
    if modelType == 'lars':
        clf = linear_model.LassoLars(alpha=best_alpha)

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
    return (clf, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, auc_test, r2test, mrnstrain, mrnstest, ix_train, ix_test)

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

def variable_subset(x, varsubset, h, print_out=True):
    if not print:
        print_statements = 'subsetting variables that are only: ' + str(varsubset) + '\n'
        hix = np.array([hi.split(':')[0].strip() in varsubset or hi in varsubset for hi in h])
        print_statements += 'from {0:,d} variables to {1:,.2f}\n'.format(x.shape[1], sum(hix))
        x = x[:, hix]
        h = np.array(h)[hix]
        # print(h, x)
        return x, h, print_statements
    else:
        print('subsetting variables that are only:', varsubset)
        hix = np.array([hi.split(':')[0].strip() in varsubset or hi in varsubset for hi in h])
        print('from ', x.shape[1] ,' variables to ', sum(hix))
        x = x[:, hix]
        h = np.array(h)[hix]
        # print(h, x)
        return x, h

def add_temporal_features(x2, feature_headers, num_clusters, num_iters, y2, y2label, dist_type='eucledian', cross_valid=True, mux=None, stdx=None, do_impute=False, subset=[]):
    if isinstance(feature_headers, list):
        feature_headers = np.array(feature_headers)
    header_vital_ix = np.array([h.startswith('Vital') for h in feature_headers])
    headers_vital = feature_headers[header_vital_ix]
    x2_vitals = x2[:, header_vital_ix]
    mu_vital = mux[header_vital_ix]
    std_vital = stdx[header_vital_ix]
    import timeseries
    xnew, hnew, muxnew, stdxnew = timeseries.construct_temporal_data(x2_vitals, headers_vital, y2, y2label, mu_vital, std_vital, subset)
    centroids, assignments, trendArray, standardDevCentroids, cnt_clusters, distances = timeseries.k_means_clust(xnew, num_clusters, num_iters, hnew, distType=dist_type, cross_valid=cross_valid)
    trendArray[trendArray!=0] = 1
    trend_headers = ['Trend:'+str(i)+' -occ:'+str(cnt_clusters[i]) for i in range(0, len(centroids))]
    return np.hstack([x2, trendArray]), np.hstack([feature_headers , np.array(trend_headers)]), centroids, hnew, standardDevCentroids, cnt_clusters, distances, muxnew, stdxnew

def filter_correlations_via(corr_headers, corr_matrix, corr_vars_exclude, print_out=True):
    ix_header = np.ones((len(corr_headers)), dtype=bool)
    for ind, item in enumerate(corr_headers):
        if (item in corr_vars_exclude) or sum([item.startswith(ii) for ii in corr_vars_exclude]) > 0 :
            ix_header[ind] = False
    if print_out:
        print(ix_header.sum())
        return corr_headers[ix_header], corr_matrix[:,ix_header], ix_header
    else:
        print_statements = 'filtered correlated features to: {0:,d}'.format(ix_header.sum())
        return corr_headers[ix_header], corr_matrix[:,ix_header], ix_header, print_statements

def autoencoder_impute(x, bin_ix, hidden_nodes=100):
    try:
        import auto_encoder
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.autograd import Variable
    except:
        print('imputation requires pytorch. please install and make sure you can import it')
        raise
    cont_ix = (np.array(bin_ix) == False)
    non_zero_ix = (x.sum(axis=0) != 0)
    old_shape = x.shape
    bin_ix = np.array(bin_ix)[non_zero_ix].tolist()
    cont_ix = np.array(cont_ix)[non_zero_ix].tolist()
    x = x[:, non_zero_ix]
    x_cont = x[:,cont_ix]
    x_bin = x[:,bin_ix]
    print(sum(bin_ix), sum(cont_ix), hidden_nodes)

    autoencoder = auto_encoder.AutoencoderConinBinar(x_bin.shape[1], x_cont.shape[1], hidden_nodes)
    optimizer = optim.SGD(autoencoder.parameters(), lr=0.5)
    np.random.seed(0)
    lossfuncBin = nn.BCELoss()
    lossfunccont = nn.MSELoss()
    loss_list = []
    for epoch in range(1, 200):
        autoencoder.train()
        for ix in range(len(x)):
            databin = Variable(torch.from_numpy(x_bin[ix]).float())
            datacont = Variable(torch.from_numpy(x_cont[ix]).float())
            databoth = Variable(torch.from_numpy(np.hstack([x_bin[ix], x_cont[ix]]))).float()
            optimizer.zero_grad()
            xoutBin, xoutCont = autoencoder(databoth)
            loss = lossfuncBin(xoutBin, databin) + lossfunccont(xoutCont, datacont)
            loss_list.append(loss)
            loss.backward()
            optimizer.step()

    autoencoder.eval()
    xout = np.zeros(x.shape)
    for ix in range(len(x)):
        databoth = Variable(torch.from_numpy(np.hstack([x_bin[ix], x_cont[ix]]))).float()
        outbin, outcont = autoencoder(databoth)
        xout[ix,bin_ix] = outbin.data.numpy()
        xout[ix,cont_ix] = outcont.data.numpy()

    xfinal = np.zeros(old_shape)
    xfinal[:,non_zero_ix] = xout
    return xfinal

def filter_min_occurrences(x2,feature_headers, min_occur=0, print_out=True):
    """
    Filter columns that have less than min_occur ocurrences
    """

    feature_filter = (np.count_nonzero(x2, axis=0) >= min_occur)
    feature_headers = np.array(feature_headers)[feature_filter].tolist()
    x2 = x2[:,feature_filter]
    if print_out:
        print('{0:,d} features filtered with number of occurrences less than {1:,d}'.format(feature_filter.sum(), min_occur))
        return x2, feature_headers
    else:
        '{0:,d} features filtered with number of occurrences less than {1:,d}\n'.format(feature_filter.sum(), min_occur)
        return x2, feature_headers, statements

def run_lasso_single(args):
    xtrain, xtest, ytrain, ytest, ytrainlabel, ytestlabel, hyperparamlist = args
    for alpha_i in hyperparamlist:
        clf = Lasso(alpha=alpha_i)
        clf.fit(xtrain, ytrain)
        auc_test = metrics.explained_variance_score(ytest, clf.predict(xtest)) #roc_auc_score(ytestlabel, clf.predict(xtest))
        if auc_test > best_score:
            best_score = auc_test #np.sqrt(((clf.predict(xtest)-ytest)**2).mean())
            best_alpha = alpha_i
    clf = Lasso(alpha=best_alpha)
    clf.fit(xtrain,ytrain)
    return clf.coef_

def lasso_filter(x, y, ylabel, feature_headers, print_out=True):
    """
    Filter any columns that have zeroed out feature weights
    """
    if 'multiprocessing' not in sys.modules:
        import multiprocessing
        from multiprocessing import Pool
    if 'Lasso' not in sys.modules:
        from sklearn.linear_model import Lasso
    N = x.shape[0]
    ixlist = np.arange(0,N)
    random.shuffle(ixlist)
    ix_train = ixlist[0:int(N*0.8)]
    ix_test = ixlist[int(N*0.8):]
    xtrain = x[ix_train]
    ytrain = y[ix_train]
    xtest = x[ix_test]
    ytest =  y[ix_test]
    ytestlabel = ylabel[ix_test]
    ytrainlabel = ylabel[ix_train]

    best_alpha = -1
    best_score = -10000
    hyperparamlist = [0.001, 0.005, 0.01, 0.1] #[alpha]
    arguments = []
    n_train = len(ix_train)
    for it in range(10):
        ix_filt = random.shuffle(ix_train)[0:int(n_train*0.9)]
        tr = ix_filt[0:int(len(ix_filt)*0.7)]
        te = ix_filt[int(len(ix_filt)*0.7):]
        arguments.append(xtrain[tr],xtrain[te],ytrain[tr],ytest[te],ytrainlabel[tr],ytestlabel[te], hyperparamlist)

    node_count = int(multiprocessing.cpu_count()*0.8)
    if len(arguments) > node_count:
        num_args = len(arguments)
        nums = math.ceil(float(num_args)/node_count)
        outputs = []
        for i in range(nums):
            sub_args = args[i*node_count:(i+1)*node_count] if i < nums-1 else args[i*node_count:]
            print('Running batch {0:d} of {1:d}'.format(i+1, nums))
            with Pool(node_count) as p:
                output = p.map(run_lasso_single, sub_args)
            for out in output:
                outputs.append(out)
    else:
        with Pool(node_count) as p:
            print('processing', node_count, 'parallel jobs to create geocoded patient dictionary')
            outputs = p.map(run_lasso_single, arguments)

    model_weights = np.array(model_weights_array).mean(axis=0)
    model_weights_std = model_weights_array.std(axis=0)
    model_weights_conf_term = (1.96/np.sqrt(iters)) * model_weights_std

    # print('R^2 score train:',clf.score(xtrain,ytrain))
    # print('RMSE score train: {0:4.3f}'.format(np.sqrt(((clf.predict(xtrain)-ytrain)**2).mean())))
    fpr, tpr, thresholds = metrics.roc_curve(ytrainlabel, clf.predict(xtrain))
    print('AUC train: {0:4.3f}'.format(metrics.auc(fpr, tpr)) + ' Explained Variance Score Train: {0:4.3f}'.format(metrics.explained_variance_score(ytrain, clf.predict(xtrain)))) #http://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score
    fpr, tpr, thresholds = metrics.roc_curve(ytestlabel, clf.predict(xtest))
    auc_test = metrics.auc(fpr, tpr); r2test = clf.score(xtest,ytest)
    # print('R^2 score test:',clf.score(xtest,ytest))
    # print('RMSE score test: {0:4.3f}'.format(np.sqrt(((clf.predict(xtest)-ytest)**2).mean())))
    print('AUC test: {0:4.3f}'.format(metrics.auc(fpr, tpr))+' Explained Variance Score Test: {0:4.3f}'.format(metrics.explained_variance_score(ytest, clf.predict(xtest))))



    feature_headers = np.array(feature_headers)[feature_filter].tolist()

def prepare_data_for_analysis(data_dic, data_dic_mom, data_dic_hist_moms, lat_lon_dic, env_dic, x1, y1, y1label, feature_headers, mrns, agex_low, agex_high, months_from, months_to, outcome='obese', percentile=False, filterSTR=['Gender:1'],  filterSTRThresh=[0.5], variablesubset=['Vital'],variable_exclude=['Trend'], num_clusters=16, num_iters=100, dist_type='euclidean', corr_vars_exclude=['Vital'], do_impute=True, mrnForFilter=[], add_time=False, bin_ix=[], do_normalize=True, min_occur=0, binarize_diagnosis=True, get_char_tables=False, feature_info=True, subset=np.array([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]), delay_print=False): #filterSTR='Gender:0 male'
    """
    Transforms the data to be run for ML analyses. Returns x2, y2, y2label, mrns2, ix_filter, and feature_headers2.
    NOTE: use ix_filter to capture relavent data points from original array as the new dimensions will be differentself.

    #### PARAMETERS ####
    For the below features if not using set value to {}
    data_dic: data dictionary of newborns with each child's data as value for some provided key
    data_dic_mom: data dictionary of maternal data at birth with child mrn as key and data as value
    data_dic_hist_moms: historical maternal data dictionary with maternal mrn as the key and data as value
    lat_lon_dic: geocoded data dictionary of maternal addresses
    env_dic: aggregated census data
    x1: data array
    y1: data to be predicted
    y1label: obesity label for each child
    feature_headers: list of features that matches the column space of x1
    mrns: list of mrns that matches that corresponds to x1

    NOTE: the following four parameters must have matching values for creation of any precreated data sets (x1, y1, y1label, feature_headers, and mrns)
    agex_low: lower bound on child age a prediction should be made from
    agex_high: upper bound on child age a prediction should be made from
    months_from: lower bound on child age for prediction
    months_to: upper bound on child age for prediction

    outcome: default = 'obese'. obesity threshold for bmi/age percentile for outcome class.
        Source: https://www.cdc.gov/obesity/childhood/defining.html
        'overweight': 0.85 <= bmi percentile < 0.95
        'obese': 0.95 <= bmi percentile <= 1.0
        'extreme': 0.99 <= bmi percentile <= 1.0
        NOTE: only required if creating the data at this stage
    percentile: default False; filter to ensure certain types of features exist for each data point
    filterSTR: default ['Gender:1']; filter specific features to have vaules greater than 'filterSTRThresh' for each filterSTR
    filterSTRThresh: default [0.5]; filter out data points with values less than the provided amount for each filterSTR feature
    variablesubset: default []; use only specified list of feature(s) (can be exact match or feature category as long as each item is the start of a feature name)
    variable_exclude: not used
    num_clusters: default 16; number of kmeans clusters to use for timeseries data
    num_iters: default 100; number of iterations for kmeans clusters for timeseries data
    dist_type: default 'euclidean'; distance metric for kmeans clusters for timeseries data
    corr_vars_exclude: default ['Vital']; features to exclude from correlation results
    do_impute: default 'True'; impute missing data
    mrnForFilter: default []; filter data by mrn values
    add_time: default False; use timeseries analyses
    bin_ix: default []; list of binary features - will be determined if none provided
    do_normalize: default True; normalize the data
    min_occur: default 0; number of occurrences required for feature to be used
    lasso_selection: defautl False; use LASSO regression to determine the most important features
    binarize_diagnosis: default True; binarize any diagnosis features that are not already binary
    get_char_tables: defaut False; save the Table 1 and 2 output to file
    subset: default np.array([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]); used to determine timeseries subset
    delay_print: default False; print everything at the end -- created for when creating data using multiprocessing and don't want jumbled results
    """

    if any([len(x)==0 for x in (x1,y1,y1label,feature_headers,mrns)]):
        reporting = 'At least one required data not provided out of x1, y1, y1label, feature_headers, or mrns.\n'
        try:
            reporting += 'Creating data from the provided data dictionaries\n'
            x1, y1, y1label, feature_headers, mrns = build_feature2.call_build_function(data_dic, data_dic_mom, data_dic_hist_moms, lat_lon_dic, env_dic, agex_low, agex_high, months_from, months_to, percentile, prediction=outcome, mrnsForFilter=mrnForFilter)
            original_data = (x1, y1, y1label, feature_headers, mrns)
        except:
            reporting += 'Not all of the required data was provided. Exiting analysis.\n'
            print(reporting)
            return
    else:
        reporting = 'Using pre-prepared data\n'
    if not delay_print:
        print(reporting)

    if binarize_diagnosis:
        bin_ix = np.array([(h.startswith('Diagnosis:') or h.startswith('Maternal Diagnosis:') or h.startswith('Newborn Diagnosis:')) for h in feature_headers])
        reporting += str(bin_ix.sum()) + ' features are binary\n'
        x1[:,bin_ix] = (x1[:,bin_ix] > 0) * 1.0

    if delay_print:
        ix_filter, x2, y2, y2label, mrns, print_statements = filter_training_set_forLinear(x1, y1, y1label, feature_headers, filterSTR, percentile, mrns, filterSTRThresh, print_out=not delay_print)
        reporting += print_statements
    else:
        ix_filter, x2, y2, y2label, mrns = filter_training_set_forLinear(x1, y1, y1label, feature_headers, filterSTR, percentile, mrns, filterSTRThresh, print_out=not delay_print)
    if get_char_tables:
        print_charac_table(x2, y2, y2label, feature_headers)
        newdir = time.strftime("table_stats_%Y%m%d_")+str(months_from)+'to'+str(months_to)+'months_'+str(agex_low)+'to'+str(agex_high)+'years'
        if not os.path.exists(newdir):
            os.mkdir(newdir)
        get_stat_table(x2, y2, y2label, feature_headers, folder=newdir)


    if do_impute or do_normalize or add_time:
        x2, mux, stdx, bin_ix, unobserved  = normalize(x2, bin_ix=bin_ix)

    if do_impute:
        x2 = autoencoder_impute(x2, bin_ix)

    if add_time:
        x2, feature_headers, centroids, hnew, standardDevCentroids, cnt_clusters, distances, muxnew, stdxnew = add_temporal_features(x2, feature_headers, num_clusters, num_iters, y2, y2label, dist_type, True, mux, stdx, do_impute, subset)
    else:
        centroids, hnew, standardDevCentroids, cnt_clusters, distances, muxnew, stdxnew = ['NaN']*7

    corr_headers = np.array(feature_headers)
    corr_matrix = np.corrcoef(x2.transpose())
    if delay_print:
        corr_headers_filtered, corr_matrix_filtered, ix_corr_headers, print_statements = filter_correlations_via(corr_headers, corr_matrix, corr_vars_exclude, print_out=not delay_print)
        reporting += print_statements
        reporting += 'corr matrix is filtered to size: '+ str(corr_matrix_filtered.shape) + '\n'
    else:
        corr_headers_filtered, corr_matrix_filtered, ix_corr_headers = filter_correlations_via(corr_headers, corr_matrix, corr_vars_exclude, print_out=not delay_print)
        print('corr matrix is filtered to size: '+ str(corr_matrix_filtered.shape))

    if len(variablesubset) != 0:
        if delay_print:
            x2, feature_headers, print_statements, print_statements = variable_subset(x2, variablesubset, feature_headers, print_out=not delay_print)
            reporting += print_statements
        else:
            x2, feature_headers = variable_subset(x2, variablesubset, feature_headers, print_out=not delay_print)
    if min_occur > 0:
        if delay_print:
            x2, feature_headers, print_statements = filter_min_occurrences(x2, feature_headers, min_occur, print_out=not delay_print)
            reporting += print_statements
        else:
            x2, feature_headers = filter_min_occurrences(x2, feature_headers, min_occur, print_out=not delay_print)

    if delay_print:
        reporting += 'output is: average: {0:4.3f}, min: {1:4.3f}, max: {2:4.3f}\n'.format(y2.mean(), y2.min(), y2.max())
        reporting += 'total patients: {0:,d}, positive: {1:,.2f}, negative: {2:,.2f}\n'.format(y2.shape[0], y2label.sum(), y2.shape[0]-y2label.sum())
        reporting += 'normalizing output...\n'
    else:
        print('output is: average: {0:4.3f}, min: {1:4.3f}, max: {2:4.3f}'.format(y2.mean(), y2.min(), y2.max()))
        print('total patients: {0:,d}, positive: {1:,.2f}, negative: {2:,.2f}'.format(y2.shape[0], y2label.sum(), y2.shape[0]-y2label.sum()))
        print('normalizing output...')
    y2 = (y2-y2.mean())/y2.std()

    reporting += 'Predicting BMI at age: '+str(agex_low)+ ' to '+str(agex_high)+ ' years, from data in ages: '+ str(months_from)+' - '+str(months_to) + ' months\n'
    if filterSTR != '':
        reporting += 'filtering patients with: '+str(filterSTR)+'\n'

    reporting += 'total size: {0:,d}'.format(ix_filter.sum())
    print(reporting)
    if (ix_filter.sum() < 50):
        print('Not enough subjects. Next.')
        return (filterSTR, [])
    return x2, y2, y2label, mrns, ix_filter, feature_headers

def train_regression_model_for_bmi(data_dic, data_dic_mom, data_dic_hist_moms, lat_lon_dic, env_dic, x1, y1, y1label, feature_headers, mrns, agex_low, agex_high, months_from, months_to, outcome='obese', modelType='lasso', percentile=False, filterSTR=['Gender:1'],  filterSTRThresh=[0.5], variablesubset=['Vital'],variable_exclude=['Trend'], num_clusters=16, num_iters=100, dist_type='euclidean', corr_vars_exclude=['Vital'], return_data_for_error_analysis=False, return_data=False, return_data_transformed=False, return_train_test_data=False, do_impute=True, mrnForFilter=[], add_time=False, bin_ix=[], do_normalize=True, binarize_diagnosis=True, get_char_tables=False, feature_info=True, subset=np.array([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False])): #filterSTR='Gender:0 male'

    """
    Train regression model for predicting obesity outcome
    #### PARAMETERS ####
    For the below features if not using set value to {}
    data_dic: data dictionary of newborns with each child's data as value for some provided key
    data_dic_mom: data dictionary of maternal data at birth with child mrn as key and data as value
    data_dic_hist_moms: historical maternal data dictionary with maternal mrn as the key and data as value
    lat_lon_dic: geocoded data dictionary of maternal addresses
    env_dic: aggregated census data
    x1: data array
    y1: data to be predicted
    y1label: obesity label for each child
    feature_headers: list of features that matches the column space of x1
    mrns: list of mrns that matches that corresponds to x1

    NOTE: the following four parameters must have matching values for creation of any precreated data sets (x1, y1, y1label, feature_headers, and mrns)
    agex_low: lower bound on child age a prediction should be made from
    agex_high: upper bound on child age a prediction should be made from
    months_from: lower bound on child age for prediction
    months_to: upper bound on child age for prediction

    outcome: default = 'obese'. obesity threshold for bmi/age percentile for outcome class.
        Source: https://www.cdc.gov/obesity/childhood/defining.html
        'overweight': 0.85 <= bmi percentile < 0.95
        'obese': 0.95 <= bmi percentile <= 1.0
        'extreme': 0.99 <= bmi percentile <= 1.0
        NOTE: only required if creating the data at this stage
    modelType: default 'lasso'
        'lasso' - sklearn.linear_model.Lasso
        'mlp' - sklearn.neural_network.MLPRegressor
        'randomforest' -  sklearn.ensemble.RandomForestRegressor
        'temporalCNN' - cnn -- NOT IMPLEMENTED
        'gradientboost' - sklearn.ensemble.GradientBoostingRegressor
        'lars' - sklearn.linear_model
    percentile: default False; filter to ensure certain types of features exist for each data point
    filterSTR: default ['Gender:1']; filter specific features to have vaules greater than 'filterSTRThresh' for each filterSTR
    filterSTRThresh: default [0.5]; filter out data points with values less than the provided amount for each filterSTR feature
    variablesubset: default []; use only specified list of feature(s) (can be exact match or feature category as long as each item is the start of a feature name)
    variable_exclude: not used
    num_clusters: default 16; number of kmeans clusters to use for timeseries data
    num_iters: default 100; number of iterations for kmeans clusters for timeseries data
    dist_type: default 'euclidean'; distance metric for kmeans clusters for timeseries data
    corr_vars_exclude: default ['Vital']; features to exclude from correlation results
    return_data_for_error_analysis: default False; return last trained model with data to analyze model errors
    return_data: default False; return X, y, y_label, feature_headers, and mrns created in the data creation phase
        NOTE: this is not the imputed, normalized, binarized, etc. data. 'feature_headers' still returned otherwise.
    return_data_transformed: default False; if True and return_data==True the transformed data will be returned in place of the original, unaltered data set.
    return_train_test_data: default False; if True and return_data==TRue the train and test data used in the final analysis will be returned for error analysis
    do_impute: default 'True'; impute missing data
    mrnForFilter: default []; filter data by mrn values
    add_time: default False; use timeseries analyses
    bin_ix: default []; list of binary features - will be determined if none provided
    do_normalize: default True; normalize the data
    binarize_diagnosis: default True; binarize any diagnosis features that are not already binary
    get_char_tables: defaut False; save the Table 1 and 2 output to file
    feature_info: default True; output model feature characteristics post analysis
    subset: default np.array([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]); used to determine timeseries subset
    """

    if modelType == 'lasso' or modelType == 'randomforest' or modelType == 'gradientboost' or modelType == 'lars':
        iters = 10
        model_weights_array = np.zeros((iters, x2.shape[1]), dtype=float)
        auc_test_list=np.zeros((iters), dtype=float); r2testlist = np.zeros((iters), dtype=float);
        randix_track = np.zeros((int(x2.shape[0]*0.9), iters))
        ix_train_track = np.zeros((int(int(x2.shape[0]*0.9)*2/3), iters))
        ix_test_track = np.zeros((int(x2.shape[0]*0.9)-int(int(x2.shape[0]*0.9)*2/3), iters))
        for iteration in range(0, iters):
            randix = list(range(0, x2.shape[0]))
            random.shuffle(randix)
            randix = randix[0:int(len(randix)*0.9)]
            datax = x2[randix,:]; datay=y2[randix]; dataylabel = y2label[randix]; mrnx = mrns[randix]
            (model, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, auc_test, r2test, mrnstrain, mrnstest, ix_train, ix_test) = train_regression(datax, datay, dataylabel, percentile, modelType, feature_headers, mrnx)
            model_weights_array[iteration, :] = model.coef_ if ((modelType == 'lasso') or (modelType == 'lars')) else model.feature_importances_
            auc_test_list[iteration] = auc_test; r2testlist[iteration] = r2test

            randix_track[:,iteration] = randix
            ix_train_track[:,iteration] = ix_train
            ix_test_track[:,iteration] = ix_test


        model_weights = model_weights_array.mean(axis=0)
        model_weights_std = model_weights_array.std(axis=0)
        model_weights_conf_term = (1.96/np.sqrt(iters)) * model_weights_std
        test_auc_mean = auc_test_list.mean()
        test_auc_mean_ste = (1.96/np.sqrt(iters)) * auc_test_list.std()
        r2test_mean = r2testlist.mean()
        r2test_ste = (1.96/np.sqrt(iters)) * r2testlist.std()

        if return_data_for_error_analysis == True:
            print('->AUC test: {0:4.3f} 95% CI: [{1:4.3f} , {2:4.3f}]'.format(test_auc_mean, test_auc_mean - test_auc_mean_ste, test_auc_mean + test_auc_mean_ste))
            print('->Explained Variance (R2) test: {0:4.3f} 95% CI: [{1:4.3f} , {2:4.3f}]'.format(r2test_mean, r2test_mean - r2test_ste,  r2test_mean + r2test_ste))
            print('lets analyse this')
            return (model, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, auc_test, r2test, feature_headers, centroids, hnew, standardDevCentroids, cnt_clusters, distances, muxnew, stdxnew, mrnstrain, mrnstest, mrns)
    else:
        (model, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, auc_test, r2test, mrnstrain, mrnstest, ix_train, ix_test) = train_regression(x2, y2, y2label, percentile, modelType, feature_headers, mrnx)
        model_weights_conf_term = np.zeros((x2.shape[1]), dtype=float)
        test_auc_mean = auc_test; r2test_mean= r2test;
        test_auc_mean_ste = 0; r2test_ste=0

        print('->AUC test: {0:4.3f} 95% CI: [{1:4.3f} , {2:4.3f}]'.format(test_auc_mean, test_auc_mean - test_auc_mean_ste, test_auc_mean + test_auc_mean_ste))
        print('->Explained Variance (R2) test: {0:4.3f} 95% CI: [{1:4.3f} , {2:4.3f}]'.format(r2test_mean, r2test_mean - r2test_ste,  r2test_mean + r2test_ste))
        if return_data_for_error_analysis == True:
            print('lets analyse this')
            return (model, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, auc_test, r2test, feature_headers, centroids, hnew, standardDevCentroids, cnt_clusters, distances, muxnew, stdxnew, mrnstrain, mrnstest, mrns)

    if modelType == 'mlp':
        print ('you need to implement gradient to get top weights. ')
        return (filterSTR, [])

    sorted_ix = np.argsort(-1* abs(model_weights))
    weights = model_weights[sorted_ix]
    terms_sorted = model_weights_conf_term[sorted_ix]

    factors = np.array(feature_headers)[sorted_ix]
    x2_reordered = x2[:,sorted_ix]
    xtest_reordered = xtest[:, sorted_ix]

    ytestpred = model.predict(xtest)
    fpr, tpr, thresholds = metrics.roc_curve(ytestlabel, ytestpred)
    operating_Thresholds = []
    operating_levels = [0, 0.0001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    ix_level = 0

    for ix, thr in enumerate(thresholds):
        if fpr[ix] >= operating_levels[ix_level]:
            operating_Thresholds.append(thr)
            ix_level += 1
            if ix_level == len(operating_levels):
                break

    operating_Thresholds = thresholds
    report_metrics = 'Test set metrics:\n'
    prec_list = []
    recall_list = []
    spec_list = []
    for t in operating_Thresholds:
        tp = ((ytestlabel > 0) & (ytestpred.ravel() > t)).sum()*1.0
        tn = ((ytestlabel == 0) & (ytestpred.ravel() <= t)).sum()*1.0
        fn = ((ytestlabel > 0) & (ytestpred.ravel() <= t)).sum()*1.0
        fp = ((ytestlabel == 0) & (ytestpred.ravel() > t)).sum()*1.0

        sens = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
        f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) != 0 else 0.0

        report_metrics += '@threshold:{0:4.3f}, sens:{1:4.3f}, spec:{2:4.3f}, ppv:{3:4.3f}, acc:{4:4.3f}, f1:{5:4.3f} total+:{6:4.3f}\n'.format(t, sens, spec, ppv, acc, f1, tp+fp)
        prec_list.append(ppv)
        recall_list.append(sens)
        spec_list.append(spec)

    print('total variables', x2.sum(axis=0).shape, ' and total subjects:', x2.shape[0])
    print('->AUC test: {0:4.3f} 95% CI: [{1:4.3f} , {2:4.3f}]'.format(test_auc_mean, test_auc_mean - test_auc_mean_ste, test_auc_mean + test_auc_mean_ste))
    print('->Explained Variance (R2) test: {0:4.3f} 95% CI: [{1:4.3f} , {2:4.3f}]'.format(r2test_mean, r2test_mean - r2test_ste,  r2test_mean + r2test_ste))
    print(report_metrics)

    occurences = (x2 != 0).sum(axis=0)[sorted_ix]
    zip_weights = {}
    sig_headers = []
    feature_categories = {}
    for i in range(0, (abs(model_weights)>0).sum()):
        fpr, tpr, thresholds = metrics.roc_curve(ytestlabel, xtest_reordered[:,i].ravel())
        feature_auc_indiv = metrics.auc(fpr, tpr)
        corrs = corr_matrix_filtered[sorted_ix[i],:].ravel()
        top_corr_ix = np.argsort(-1*abs(corrs))
        corr_string = 'Correlated most with:\n'+'    '.join( [str(corr_headers_filtered[top_corr_ix[j]])+ ':' + "{0:4.3f}\n".format(corrs[top_corr_ix[j]]) for j in range(0,10)]  )

        tp = ((y2label > 0) & (x2_reordered[:,i].ravel() > 0)).sum()*1.0
        tn = ((y2label == 0) & (x2_reordered[:,i].ravel() <= 0)).sum()*1.0
        fn = ((y2label > 0) & (x2_reordered[:,i].ravel() <= 0)).sum()*1.0
        fp = ((y2label == 0) & (x2_reordered[:,i].ravel() > 0)).sum()*1.0

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
        if feature_info:
            print("{8} {3} | coef {0:4.3f} 95% CI: [{1:4.3f} , {2:4.3f}] | OR_adj {9:4.3f} [{10:4.3f} {11:4.3f}] | occ: {4} | OR_unadj: {5:4.3f} [{6:4.3f} {7:4.3f}] | indivs AUC:{12:4.3f}".format(weights[i], weights[i]-terms_sorted[i], weights[i]+terms_sorted[i], factors[i], occurences[i], oratio, low_OR, high_OR, star, np.exp(weights[i]), np.exp(weights[i]-terms_sorted[i]), np.exp(weights[i]+terms_sorted[i]), feature_auc_indiv))
            print(corr_string)

    for k in feature_categories:
        print (k, ":", feature_categories[k])

    if return_data:
        if return_data_transformed and return_train_test_data:
            return (model, x2, y2, y2label, ix_filter, randix_track, ix_train_track, ix_test_track, feature_headers, xtrain, ytrain, ytrainlabel, mrnstrain, xtest, ytest, ytestlabel, mrnstest, filterSTR, sig_headers, centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew, mrns, prec_list, recall_list, spec_list, test_auc_mean, test_auc_mean_ste, r2test_mean, r2test_ste)
        elif return_data_transformed and not return_train_test_data:
            return (model, x2, y2, y2label, ix_filter, randix_track, ix_train_track, ix_test_track, feature_headers, filterSTR, sig_headers, centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew, mrns, prec_list, recall_list, spec_list, test_auc_mean, test_auc_mean_ste, r2test_mean, r2test_ste)
        elif not return_data_transformed and return_train_test_data:
            return (model, xtrain, ytrain, ytrainlabel, mrnstrain, xtest, ytest, ytestlabel, mrnstest, feature_headers, filterSTR, sig_headers, centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew, mrns, prec_list, recall_list, spec_list, test_auc_mean, test_auc_mean_ste, r2test_mean, r2test_ste)
        else:
            return (model, original_data[0], original_data[1], original_data[2], original_data[3], original_data[4], filterSTR, sig_headers, centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew, mrns, prec_list, recall_list, spec_list, test_auc_mean, test_auc_mean_ste, r2test_mean, r2test_ste)
    else:
        return (feature_headers, filterSTR, sig_headers, centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew, mrns, prec_list, recall_list, spec_list, test_auc_mean, test_auc_mean_ste, r2test_mean, r2test_ste)

def analyse_errors(model, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, auc_test, r2test, feature_headers, centroids, hnew, standardDevCentroids, cnt_clusters, distances, muxnew, stdxnew, mrnstrain, mrnstest):
    pred = model.predict(xtrain)
    print('AUC train is:', metrics.roc_auc_score(y_score=pred, y_true=ytrainlabel))
    print('high risks are')
    # ix_sorted = np.argsort(-1*(ytrain))
    # plt.plot(pred[ix_sorted])
    # plt.plot(ytrain[ix_sorted])
    # plt.plot(ytrainlabel[ix_sorted])

    predtest = model.predict(xtest)
    predtrain = model.predict(xtrain)
    ix_sorted_test = np.argsort(-1*(predtest))[0:int(len(predtest)/3)]
    ix_sorted_train = np.argsort(-1*(predtrain))[0:int(len(predtrain)/3)]

    # return list(mrnstest[ix_sorted_test])+list(mrnstrain[ix_sorted_train])
    # plt.plot(predtest[ix_sorted_test])
    # plt.plot(ytest[ix_sorted_test])
    # plt.plot(ytestlabel[ix_sorted_test])

    try:
        weights = np.array(model.coef_).ravel()
    except:
        weights = np.array(model.feature_importances_).ravel()

    weights_sorted_ix = np.argsort(-1*abs(weights))
    for m in ix_sorted_test[0:4]:
        print('member', m, 'mrn:', mrnstest[m], ', predicted as', predtest[m], ' but should have been:', ytest[m], ' with label:', ytestlabel[m])
        for i in weights_sorted_ix:
            if xtest[m,:][i] == 0 :
                continue
            print('feature', feature_headers[i], ' with weight/importance', weights[i], 'is:', xtest[m,:][i])
        print('----')


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

def prec_rec_curve(recall_list, precision_list, titles_list, title, show=True, save=False):
    """
    Precision Recall Curves for multiple analyses to compare results
    #### PARAMETERS ####
    recall_list: list of lists of recall values
    precision_list: list of lists of precision values
    titles_list: list of legend labels for each model
    title: title of plot and filename if saving
    save: binary to indicate if plot should be saved
    """
    plt.figure(figsize=(10,10))
    for ix in range(len(precision_list)):
        plt.plot(recall_list[ix], precision_list[ix], label=titles_list[ix])
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.legend(fontsize = 8)
    plt.axis('equal')
    plt.title(title)
    plt.grid()
    if save:
        plt.savefig(title+'.png', dpi=300)
    if show:
        plt.show()
    return

def ROC_curve(recall_list, specificity_list, titles_list, title, show=True, save=False, dpi=96):
    """
    Receiver Operator Curves for multiple analyses to compare results
    #### PARAMETERS ####
    recall_list: list of lists of recall values
    specificity_list: list of lists of specificity values
    titles_list: list of legend labels for each model
    title: title of plot and filename if saving
    save: binary to indicate if plot should be saved
    dpi: default = 96; dpi setting for saving output
    """
    plt.figure(figsize=(10,10))
    for ix in range(len(prec_total)):
        plt.plot(1- np.array(spec_total[ix]), np.array(recall_total[ix]), label=titles_total[ix])

    plt.legend(fontsize = 8)
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.axis('equal')
    plt.title(title)
    plt.grid()
    if save:
        plt.savefig(title+'.png', dpi=dpi)
    if show:
        plt.show()
    return

def plot_growth_curve(data, mrn, key, readings='us', save_name=None, dpi=96, hide_mrn=False):
    """
    data: data dictionary containing ht/wt data
    mrn: default = None; use if defining patient to plot by mrn
    key: key to use for accessing the patient's information in the data dictionary. If mrn is provided, then
        this field is not used.
    readings: default = 'us'
        'us': ht/wt in inches/pounds
        'metric': ht/wt in centimeters/kilograms
    save_name: default = None; provide a <path/file_name.png> to save the plot.
    dpi: default = 96; dpi setting for saving output
    hide_mrn: default = False; place the mrn in the title of the plot. Use False if sharing outside
    """

    cols = ['Length','L','M','S','P01','P1','P3','P5','P10','P15','P25','P50','P75','P85','P90','P95','P97','P99','P999']
    cols_to_use = ['P01','P1','P3','P5','P10','P25','P50','P75','P90','P95','P97','P99','P999']
    lower = ['P01','P1','P3','P5','P10','P25']
    upper = ['P75','P90','P95','P97','P99','P999']

    # Get the data dictionary key if mrn is provided
    if mrn != None:
        for k in data.keys():
            if data[k]['mrn'] == mrn:
                key = k
                break
    # Get the mrn if the data dictionary key is provided
    else:
        mrn = data[key]['mrn']

    cutoff =  data[key]['bdate'] + relativedelta(years=+2)
    gender = data[key]['gender']
    wfl = b_wfl if gender == 0 else g_wfl

    # Create the height weight points for plotting
    if readings == 'us':
        ht_wt = {l[0]:[l[1]*2.54, 0] for l in data[key]['vitals']['Ht'] if l[0] < cutoff}
    elif readings == 'metric':
        ht_wt = {l[0]:[l[1], 0] for l in data[key]['vitals']['Ht'] if l[0] < cutoff}
    else:
        raise ValueError('"readings" need to be "metric" or "us"!')
    for w in data[key]['vitals']['Wt']:
        if w[0] >= cutoff:
            continue
        try:
            ht_wt[w[0]][1] = w[1]*0.4535924 if readings == 'us' else w[1]
        except:
            ht_wt[w[0]] = [0, w[1]*0.4535924] if readings == 'us' else [0, w[1]]

    # Plot the WHO weight for length growth curves
    plt.figure(figsize=(8,8))
    for i in [cols.index(x) for x in cols_to_use]:
        if i in [cols.index(x) for x in lower]:
            c = 'cornflowerblue'
        elif i == cols.index('P50'):
            c = 'black'
        elif i in [cols.index(x) for x in upper]:
            c = 'crimson'
        plt.plot(wfl[:,0], wfl[:,i], linewidth=0.8, color=c)
        plt.text(wfl[:,0][-1], wfl[:,i][-1], cols[i])

    if gender == 0:
        if hide_mrn:
            plt.title('Boys Weight for Length Z Score - Sample Patient', fontsize=16)
        else:
            plt.title('Boys Weight for Length Z Score - Patient '+str(mrn), fontsize=16)
    else:
        if hide_mrn:
            plt.title('Girls Weight for Length Z Score - Sample Patient', fontsize=16)
        else:
            plt.title('Girls Weight for Length Z Score - Patient '+str(mrn), fontsize=16)

    hts = []
    wts = []
    for k in sorted(ht_wt):
        vals = ht_wt[k]
        if any(v==0 for v in vals):
            continue
        hts.append(vals[0])
        wts.append(vals[1])

    plt.plot(hts, wts, linestyle='--', color='gray', alpha=0.7)
    for k in sorted(ht_wt):
        vals = ht_wt[k]
        if any(v==0 for v in vals):
            continue
        diff = relativedelta(k, data[key]['bdate'])
        lab = '{0:2.1f} months, Z Score: {1:1.1f}'.format((diff.years*12)+diff.months+(diff.days/30), zscore.zscore_wfl(gender, vals[0], vals[1]))
        plt.scatter(vals[0], vals[1], marker='+', s=60, label=lab)

    plt.legend(fontsize=10)
    plt.xlabel('Length (in cm)', fontsize=14)
    plt.ylabel('Weight (in kg)', fontsize=14)
    plt.xlim((45,114))
    plt.ylim((0,25))
    plt.xticks(np.arange(45,115,5))
    plt.yticks(np.arange(0,27.5,2.5))
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    if save_name != None:
        plt.savefig(save_name, dpi=dpi)
    plt.show()

def print_charac_table(x2, y2, y2label, headers, table_features=['Diagnosis:', 'Maternal Diagnosis:', 'Maternal-birthplace:', 'Maternal-marriageStatus:', 'Maternal-nationality:', 'Maternal-race:', 'Maternal-ethnicity:', 'Maternal-Language:', 'Vital:', 'Gender']):
    """
    Computing and printing to standard output, a characteristics table including various factors. Does not return anything.
    #### PARAMETERS ####
    x2: numpy matrix of size N x D where D is the number of features and N is number of samples. Can contain integer and float variables.
    y2: numpy matrix of size N where N is number of samples. Contains float outcomes
    y2label: numpy matrix of size N where N is number of samples. Contains Bool or binary outcomes.
    headers: list of length D, of D string description for each column in x2
    table_features: list of types of features that we are interested in reporting in the characteristics table.
    """
    y2pos_ix = (y2label > 0)
    print(' Characteristics Table')
    print(' Variable | Total N | Total Average(SD) | Pos N | Pos Average (SD) | Neg N | Neg Average (SD) | Odds Ratio (low, high) | Relative Risk | p-value for OR')
    for ix, h in enumerate(headers):
        if any( [h.startswith(charfeats) for charfeats in table_features] ):
            if (x2[:,ix] != 0).sum() > 5:
                bin_indicator = x2[:,ix].max()==1 and x2[:,ix].min()==0

                ix_total = (x2[:,ix] != 0)
                ix_total_pos = (y2label > 0) & (x2[:,ix] != 0)
                ix_total_neg = (y2label == 0) & (x2[:,ix] != 0)
                De = sum((y2label > 0) & (x2[:,ix] != 0)) * 1.0
                He = sum((y2label == 0) & (x2[:,ix] != 0)) * 1.0
                Dn = sum((y2label > 0) & (x2[:,ix] == 0)) * 1.0
                Hn = sum((y2label == 0) & (x2[:,ix] == 0)) * 1.0
                OR = (De/He)/(Dn/Hn)
                OR_sterror = np.sqrt(1/De + 1/He + 1/Dn + 1/Hn)
                OR_low, OR_high = np.exp(np.log(OR) - 1.96*OR_sterror), np.exp(np.log(OR) + 1.96*OR_sterror)
                RR = (De/(De+He))/(Dn/(Dn+Hn))

                md = x2[ix_total_pos,:][:,ix].mean() - x2[ix_total_neg,:][:,ix].mean()
                se = np.sqrt( np.var(x2[ix_total_pos,:][:,ix]) / len(x2[ix_total_pos,:][:,ix]) + np.var(x2[ix_total_neg,:][:,ix])/len(x2[ix_total_neg,:][:,ix]))
                lcl, ucl = md-2*se, md+2*se
                z = md/se

                pvalue = 2 * norm.cdf(-1*(np.abs(np.log(OR))/OR_sterror)) if bin_indicator else 2 * norm.cdf(-np.abs(z))

                print(h + ' | {0:4.0f} | {1:4.3f} ({2:4.3f}) | {3:4.0f} | {4:4.3f} ({5:4.3f}) | {6:4.0f} | {7:4.3f} ({8:4.3f}) | {9:4.3f} ({10:4.3f}, {11:4.3f}) | {12:4.3f} | '. format(\
                ix_total.sum(), x2[ix_total,:][:,ix].mean(), x2[ix_total,:][:,ix].std(), \
                ix_total_pos.sum(), x2[ix_total_pos,:][:,ix].mean() if not(bin_indicator) else 0, x2[ix_total_pos,:][:,ix].std() if not(bin_indicator) else 0,
                ix_total_neg.sum(), x2[ix_total_neg,:][:,ix].mean() if not(bin_indicator) else 0,  x2[ix_total_neg,:][:,ix].std() if not(bin_indicator) else 0,
                OR if bin_indicator else 0, OR_low if bin_indicator else 0, OR_high if bin_indicator else 0,
                RR if bin_indicator else 0,
                ) + str(pvalue))

def get_stat_table(x, y, ylabel, headers, folder=time.strftime("table_stats_%Y%m%d_%H:%M")):
    """
    Essentially the same as "print_charac_table", but saves tables required for reporting to two csv files called
    'summary_stats_t1.csv' and summary_stats_t2.csv in the designated folder.
    #### PARAMETERS ####
    x: numpy matrix of size N x D where D is the number of features and N is number of samples. Can contain integer and float variables.
    y: numpy matrix of size N where N is number of samples. Contains float outcomes
    ylabel: numpy matrix of size N where N is number of samples. Contains Bool or binary outcomes.
    headers: list of length D, of D string description for each column in x2
    folder: default = time.strftime("table_stats_%Y%m%d_%H:%M/"). Folder where the two csv files will be saved.
    """
    if type(headers) == np.ndarray:
        headers = headers.tolist()
    elif type(headers) == tuple:
        headers = list(headers)
    elif type(headers) != list:
        raise ValueError('"headers" should be a list!')

    y = y.ravel()
    ylabel = ylabel.ravel()

    headers1 = ['Variable','Total N','Pos N','Neg N','Odds Ratio', 'Odds Ratio Low', 'Odds Ratio High','Relative Risk','p-value for OR']
    categories = ['Maternal Ethnicity','Maternal Race','Maternal Marriage Status','Maternal Birthplace', 'Maternal Diagnosis','Infant Diagnosis']
    features1 = {
        'Gender':{
            'Male':['Gender:0 male'],
            'Female':['Gender:1 female']
        },
        'Maternal Ethnicity':{
            'Not Hispanic/Latio':['Maternal-ethnicity:NOT HISPANIC/LATINO'],
            'Hispanic/Latino':['Maternal-ethnicity:HISPANIC/LATINO'],
            'Other':['Maternal-ethnicity:OTHER']
        },
        'Maternal Race':{
            'Asian':['Maternal-race:ASIAN'],
            'Unknown/No Response':['Maternal-race:UNKNOWN/NO RESPONSE'],
            'Multiracial':['Maternal-race:MULTIRACIAL'],
            'Caucasian/White':['Maternal-race:CAUCASIAN/WHITE'],
            'African Amer/Black':['Maternal-race:AFRICAN AMER/BLAC']
         },
         'Maternal Marriage Status':{
            'Divorced':['Maternal-marriageStatus:DIVORCED'],
            'Partnered':['Maternal-marriageStatus:PARTNERED'],
            'Married':['Maternal-marriageStatus:MARRIED'],
            'Single':['Maternal-marriageStatus:SINGLE']
         },
         'Maternal Birthplace':{
            'China':['Maternal-birthplace:CHINA'],
            'Dominican Republic':['Maternal-birthplace:DOMINICAN REPUBLIC'],
            'Puerto Rico':['Maternal-birthplace:PUERTO RICO'],
            'Trinidad':['Maternal-birthplace:TRINIDAD'],
            'Elsalvador':['Maternal-birthplace:ELSALVADOR'],
            'Ecuador':['Maternal-birthplace:ECUADOR'],
            'United States':['Maternal-birthplace:UNITED STATES'],
            'Guatemala':['Maternal-birthplace:GUATEMALA'],
            'Honduras':['Maternal-birthplace:HONDURAS'],
            'Grenada':['Maternal-birthplace:GRENADA'],
            'Peru':['Maternal-birthplace:PERU'],
            'Haiti':['Maternal-birthplace:HAITI'],
            'Mexico':['Maternal-birthplace:MEXICO']
         },
         'Maternal Diagnosis':{
             'Nutritional diagnosis':['Maternal Diagnosis:9ccs52:Nutrit defic','Maternal Diagnosis:10ccs52:Nutrit defic'],
             'Diabetes Mellitus in pregnancy':['Maternal Diagnosis:9ccs186:DM in preg','Maternal Diagnosis:10ccs186:DM in preg'],
             'Diabetes Mellitus without complications':[['Maternal Diagnosis:9ccs186:DM in preg','Maternal Diagnosis:10ccs186:DM in preg'],
                ['Maternal Diagnosis:9ccs195:Ot compl bir','Maternal Diagnosis:10ccs195:Ot compl bir']],
             'Hypertension in pregnancy':['Maternal Diagnosis:9ccs183:HTN in preg','Maternal Diagnosis:10ccs183:HTN in preg'],
             'Complications at birth':['Maternal Diagnosis:9ccs195:Ot compl bir','Maternal Diagnosis:10ccs195:Ot compl bir'],
             'OB-related perin trauma':['Maternal Diagnosis:9ccs193:OB-related perin trauma','Maternal Diagnosis:10ccs193:OB-related perin trauma'],
             'Pelvic obstruction':['Maternal Diagnosis:9ccs188:Pelvic obstr','Maternal Diagnosis:10ccs188:Pelvic obstr']
         },
         'Infant Diagnosis':{'Nutritional diagnosis':['Diagnosis:9ccs52:Nutrit defic','Diagnosis:10ccs52:Nutrit defic'],
             'Epilepsy/convulsions':['Diagnosis:9ccs83:Epilepsy/cnv','Diagnosis:10ccs83:Epilepsy/cnv'],
             'Liver Diseases':['Diagnosis:9ccs151:Oth liver dx','Diagnosis:10ccs151:Oth liver dx'],
             'Skin Diseases':['Diagnosis:9ccs198:Ot infl skin','Diagnosis:10ccs198:Ot infl skin'],
             'Kidney Diseases':['Diagnosis:9ccs161:Ot dx kidney','Diagnosis:10ccs161:Ot dx kidney'],
             'Circulatory Diseases':['Diagnosis:9ccs117:Ot circul dx','Diagnosis:10ccs117:Ot circul dx'],
             'Pneumonia':['Diagnosis:9ccs129:Asp pneumon','Diagnosis:10ccs129:Asp pneumon']
                             }
    }
    headers2 = ['Variable','Total N','Total Average', 'Total SD','Pos N','Pos Average', 'Pos SD','Neg N','Neg Average', 'Neg SD','p-value']
    features2 = ['Vital: Wt for Len Percentile-avg19to24','Vital: BMI-avg19to24','Vital: Wt for Len Percentile-avg16to19','Vital: BMI-avg16to19']

    df1 = []
    with np.errstate(divide='ignore', invalid='ignore'):
        for k in features1.keys():
            row = [k] + ['']*(len(headers1)-1)
            df1.append(row)
            for kk in features1[k].keys():
                if len(features1[k][kk]) == 1:
                    col_ix = headers.index(features1[k][kk][0])
                    bin_indicator = x[:,col_ix].max()==1 and x[:,col_ix].min()==0
                    ix_total = (x[:,col_ix] != 0).sum()
                    ix_total_pos = (ylabel > 0) & (x[:,col_ix] != 0)
                    ix_total_neg = (ylabel == 0) & (x[:,col_ix] != 0)
                    De = sum((ylabel > 0) & (x[:,col_ix] != 0)) * 1.0
                    He = sum((ylabel == 0) & (x[:,col_ix] != 0)) * 1.0
                    Dn = sum((ylabel > 0) & (x[:,col_ix] == 0)) * 1.0
                    Hn = sum((ylabel == 0) & (x[:,col_ix] == 0)) * 1.0
                    OR = (De/He)/(Dn/Hn)
                    OR_sterror = np.sqrt(1/De + 1/He + 1/Dn + 1/Hn)
                    OR_low, OR_high = np.exp(np.log(OR) - 1.96*OR_sterror), np.exp(np.log(OR) + 1.96*OR_sterror)
                    RR = (De/(De+He))/(Dn/(Dn+Hn))

                    md = x[ix_total_pos,:][:,col_ix].mean() - x[ix_total_neg,:][:,col_ix].mean()
                    se = np.sqrt(np.var(x[ix_total_pos,:][:,col_ix]) / len(x[ix_total_pos,:][:,col_ix]) + np.var(x[ix_total_neg,:][:,col_ix])/len(x[ix_total_neg,:][:,col_ix]))
                    lcl, ucl = md-2*se, md+2*se
                    z = md/se

                    pvalue = 2 * norm.cdf(-1*(np.abs(np.log(OR))/OR_sterror)) if bin_indicator else 2 * norm.cdf(-np.abs(z))
                    row = [kk, ix_total.sum(), ix_total_pos.sum(), ix_total_neg.sum(), OR, OR_low, OR_high, RR, pvalue]
                    df1.append(row)
                else:
                    if type(features1[k][kk][0]) != list:
                        xx = np.zeros((x.shape[0]))
                        bin_indicator = all(((x[:,headers.index(f)].max()==1) & (x[:,headers.index(f)].min()==0)) or (x[:,headers.index(f)].std()==0) for f in features1[k][kk])
                        for f in features1[k][kk]:
                            xx += x[:,headers.index(f)]
                        if bin_indicator:
                            xx[xx > 1] = 1
                    else:
                        # should be two lists. first: features that should exist. second: features that should not exist
                        xx_diag = np.zeros((x.shape[0]))
                        xx_comp = np.zeros((x.shape[0]))
                        bin_indicator = all(((x[:,headers.index(f)].max()==1) & (x[:,headers.index(f)].min()==0)) or (x[:,headers.index(f)].std()==0) for f in features1[k][kk][0])
                        for f in features1[k][kk][0]:
                            col_ix = headers.index(f)
                            xx_diag += x[:,col_ix]
                        for f in features1[k][kk][1]:
                            xx_comp += x[:,headers.index(f)]
                        xx = (xx_diag != 0) & (xx_comp == 0)
                        if bin_indicator:
                            xx[xx > 1] = 1
                    ix_total = xx
                    ix_total_pos = (ylabel > 0) & (xx != 0)
                    ix_total_neg = (ylabel == 0) & (xx != 0)
                    De = sum((ylabel > 0) & (xx != 0)) * 1.0
                    He = sum((ylabel == 0) & (xx != 0)) * 1.0
                    Dn = sum((ylabel > 0) & (xx == 0)) * 1.0
                    Hn = sum((ylabel == 0) & (xx == 0)) * 1.0
                    OR = (De/He)/(Dn/Hn)
                    OR_sterror = np.sqrt(1/De + 1/He + 1/Dn + 1/Hn)
                    OR_low, OR_high = np.exp(np.log(OR) - 1.96*OR_sterror), np.exp(np.log(OR) + 1.96*OR_sterror)
                    RR = (De/(De+He))/(Dn/(Dn+Hn))

                    md = x[ix_total_pos,:][:,col_ix].mean() - x[ix_total_neg,:][:,col_ix].mean()
                    se = np.sqrt( np.var(x[ix_total_pos,:][:,col_ix]) / len(x[ix_total_pos,:][:,col_ix]) + np.var(x[ix_total_neg,:][:,col_ix])/len(x[ix_total_neg,:][:,col_ix]))
                    lcl, ucl = md-2*se, md+2*se
                    z = md/se

                    pvalue = 2 * norm.cdf(-1*(np.abs(np.log(OR))/OR_sterror)) if bin_indicator else 2 * norm.cdf(-np.abs(z))
                    row = [kk, ix_total.sum(), ix_total_pos.sum(), ix_total_neg.sum(), OR, OR_low, OR_high, RR, pvalue]
                    df1.append(row)

    df2 = []
    with np.errstate(divide='ignore', invalid='ignore'):
        for f in features2:
            col_ix = headers.index(f)
            bin_indicator = x[:,col_ix].max()==1 and x[:,col_ix].min()==0
            ix_total = (x[:,col_ix] != 0)
            ix_total_pos = (ylabel > 0) & (x[:,col_ix] != 0)
            ix_total_neg = (ylabel == 0) & (x[:,col_ix] != 0)
            De = sum((ylabel > 0) & (x[:,col_ix] != 0)) * 1.0
            He = sum((ylabel == 0) & (x[:,col_ix] != 0)) * 1.0
            Dn = sum((ylabel > 0) & (x[:,col_ix] == 0)) * 1.0
            Hn = sum((ylabel == 0) & (x[:,col_ix] == 0)) * 1.0
            OR = (De/He)/(Dn/Hn)
            OR_sterror = np.sqrt(1/De + 1/He + 1/Dn + 1/Hn)
            OR_low, OR_high = np.exp(np.log(OR) - 1.96*OR_sterror), np.exp(np.log(OR) + 1.96*OR_sterror)
            RR = (De/(De+He))/(Dn/(Dn+Hn))

            md = x[ix_total_pos,:][:,col_ix].mean() - x[ix_total_neg,:][:,col_ix].mean()
            se = np.sqrt( np.var(x[ix_total_pos,:][:,col_ix]) / len(x[ix_total_pos,:][:,col_ix]) + np.var(x[ix_total_neg,:][:,col_ix])/len(x[ix_total_neg,:][:,col_ix]))
            lcl, ucl = md-2*se, md+2*se
            z = md/se

            pvalue = 2 * norm.cdf(-1*(np.abs(np.log(OR))/OR_sterror)) if bin_indicator else 2 * norm.cdf(-np.abs(z))
            row = [f, ix_total.sum(), x[ix_total, col_ix].mean(), x[ix_total, col_ix].std(),
                   ix_total_pos.sum(), x[ix_total_pos, col_ix].mean(), x[ix_total_pos, col_ix].std(),
                   ix_total_neg.sum(), x[ix_total_neg, col_ix].mean(), x[ix_total_neg, col_ix].std(), pvalue]
            df2.append(row)

    df1 = pd.DataFrame(df1, columns=headers1)
    df1.replace(to_replace=np.nan, value=0, inplace=True)
    df2 = pd.DataFrame(df2, columns=headers2)
    df2.replace(to_replace=np.nan, value=0, inplace=True)

    if not os.path.isdir(folder):
        os.mkdir(folder)
    df1.to_csv(folder+'/summary_stats_t1.csv')
    df2.to_csv(folder+'/summary_stats_t2.csv')


if __name__=='__main__':
    d1 = pickle.load(open('patientdata_20170823.pkl', 'rb'))
    d1mom = pickle.load(open('patient_mother_data_20170724.pkl', 'rb'))
    mrnsboys = pickle.load(open('mrnsboys.pkl','rb'))
    mrnsgirls = pickle.load(open('mrnsgirl.pkl','rb'))

    (filterSTR, sig_headers,  centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew, mrnsdummy) = train_regression_model_for_bmi(d1, d1mom, 4.5, 5.5, 0, 24, filterSTR=['Gender:1'], variablesubset=[], num_clusters=16, num_iters=100, dist_type='euclidean', modelType='lasso', return_data_for_error_analysis=False, add_time=True, mrnForFilter=mrnsgirl)

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print('saving (filterSTR, sig_headers,  centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew, mrnsdummy) at out_'+timestr+'.pkl')

    pickle.dump(file=open('out_'+timestr+'.pkl', 'wb'), obj=(filterSTR, sig_headers,  centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew, mrnsdummy), protocol=-1)
