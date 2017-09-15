import config as config_file
import pandas as pd
import pickle 
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
import time
from datetime import timedelta
from dateutil import parser
import numpy as np
import outcome_def_pediatric_obesity
import build_features
import random
from sklearn import metrics
from sklearn.preprocessing import Imputer

def filter_training_set_forLinear(x, y, ylabel, headers, filterSTR='', percentile=False, mrns=[]):
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
        ix = (y > 10) & (y < 40) & (((x[:,index_finder_filterstr]!=0).sum(axis=1) >= len(filterSTR)).ravel()) & ((x[:,index_finder_anyvital] != 0).sum(axis=1) >= 1) #& ((x[:,index_finder_maternal] != 0).sum(axis=1) >= 1)

    elif percentile == False:
        ix = (y > 10) & (y < 40) & ((x[:,index_finder_anyvital] != 0).sum(axis=1) >= 1)
        print(ix.sum())
        
    if (percentile == True) & (filterSTR != ''):
        ix = (x[:,index_finder_filterstr].ravel() == True) 
    elif percentile == True:
        ix = (x[:,index_finder_filterstr].ravel() >= False) 
    print(str(ix.sum()) + ' patients selected..')
    return ix, x[ix,:], y[ix], ylabel[ix], mrns[ix]

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
    

    random.seed(2)
    random.shuffle(ixlist)
    xtrain = x[ixlist[0:int(N*2/3)], :]
    ytrain = y[ixlist[0:int(N*2/3)]]
    xtest = x[ixlist[int(N*2/3):],:]
    ytest =  y[ixlist[int(N*2/3):]]
    ytestlabel = ylabel[ixlist[int(N*2/3):]]
    ytrainlabel = ylabel[ixlist[0:int(N*2/3)]]    
    mrnstrain = mrns[ixlist[0:int(N*2/3)]]
    mrnstest = mrns[ixlist[int(N*2/3):]]

    best_alpha = -1
    best_score = -10000
    if modelType == 'lasso':
        hyperparamlist = [0.001, 0.005, 0.01, 0.1]
    if modelType == 'mlp':
        hyperparamlist = [(10,), (50,), (10,10), (50,10), (100,)]
    if modelType == 'randomforest':
        hyperparamlist = [(2000,2), (2000,4), (2000,10)]
    if modelType == 'temporalCNN':
        hyperparamlist = [(0.1)]
    if modelType == 'gradientboost':
        hyperparamlist = [(1500, 4, 2, 0.01,'lad'), (2500, 4, 2, 0.01,'lad')]
    if modelType == 'lars':
        hyperparamlist = [0.001, 0.01, 0.1]

    for alpha_i in hyperparamlist:
        if modelType == 'lasso':
            clf = Lasso(alpha=alpha_i)
        if modelType == 'mlp':
            clf = MLPRegressor(hidden_layer_sizes=alpha_i, solver="lbfgs", verbose=True)
        if modelType == 'randomforest':
            clf = RandomForestRegressor(random_state=0, n_estimators=alpha_i[0], min_samples_split=alpha_i[1])
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
        clf = RandomForestRegressor(random_state=0, n_estimators=best_alpha[0], min_samples_split=best_alpha[1])
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
    return (clf, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, auc_test, r2test, mrnstrain, mrnstest)

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

def variable_subset(x, varsubset, h):
    print('subsetting variables that are only:', varsubset)
    hix = np.array([hi.split(':')[0].strip() in varsubset or hi in varsubset for hi in h])
    print('from ', x.shape[1] ,' variables to ', sum(hix))
    x = x[:, hix]
    h = np.array(h)[hix]
    # print(h, x)
    return x, h

def add_temporal_features(x2, feature_headers, num_clusters, num_iters, y2, y2label,
        dist_type='eucledian', cross_valid=True, mux=None, stdx=None, do_impute=False):
    if isinstance(feature_headers, list):
        feature_headers = np.array(feature_headers)
    header_vital_ix = np.array([h.startswith('Vital') for h in feature_headers])
    headers_vital = feature_headers[header_vital_ix]
    x2_vitals = x2[:, header_vital_ix]
    mu_vital = mux[header_vital_ix]
    std_vital = stdx[header_vital_ix]
    import timeseries
    xnew, hnew, muxnew, stdxnew = timeseries.construct_temporal_data(x2_vitals, headers_vital, y2, y2label, mu_vital, std_vital)
    centroids, assignments, trendArray, standardDevCentroids, cnt_clusters, distances = timeseries.k_means_clust(xnew, num_clusters, num_iters, hnew, dist_type=dist_type, cross_valid=cross_valid)
    trendArray[trendArray!=0] = 1
    trend_headers = ['Trend:'+str(i)+' -occ:'+str(cnt_clusters[i]) for i in range(0, len(centroids))]
    return np.hstack([x2, trendArray]), np.hstack([feature_headers , np.array(trend_headers)]), centroids, hnew, standardDevCentroids, cnt_clusters, distances, muxnew, stdxnew

def filter_correlations_via(corr_headers, corr_matrix, corr_vars_exclude):
    ix_header = np.ones((len(corr_headers)), dtype=bool)
    for ind, item in enumerate(corr_headers):
        if (item in corr_vars_exclude) or sum([item.startswith(ii) for ii in corr_vars_exclude]) > 0 :
            ix_header[ind] = False
    print(ix_header.sum())
    return corr_headers[ix_header], corr_matrix[:,ix_header], ix_header

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

def train_regression_model_for_bmi(data_dic, data_dic_mom, agex_low, agex_high, months_from, months_to, modelType='lasso', percentile=False, filterSTR=['Gender:1'], variablesubset=['Vital'],variable_exclude=['Trend'], num_clusters=16, num_iters=100, dist_type='euclidean', corr_vars_exclude=['Vital'], return_data_for_error_analysis=False, do_impute=True, mrnForFilter=[], add_time=False, bin_ix=[], do_normalize=True, binarize_diagnosis=True): #filterSTR='Gender:0 male'
    
    x1, y1, y1label, feature_headers, mrns = build_features.call_build_function(data_dic,data_dic_mom, agex_low, agex_high, months_from, months_to, percentile, mrnsForFilter=mrnForFilter)
    if binarize_diagnosis:
        bin_ix = True - np.array([(h.startswith('Vital') or h.startswith('MatDeliveryAge')) for h in feature_headers])
        print(bin_ix.sum(), 'features are binary')
        x1[:,bin_ix] = (x1[:,bin_ix] > 0) * 1.0
    
    ix, x2, y2, y2label, mrns = filter_training_set_forLinear(x1, y1, y1label, feature_headers, filterSTR, percentile, mrns)
    
    if do_impute or do_normalize or add_time:
        x2, mux, stdx, bin_ix, unobserved  = normalize(x2, bin_ix=bin_ix)

    if do_impute:
        x2 = autoencoder_impute(x2, bin_ix)

    if add_time:
        x2, feature_headers, centroids, hnew, standardDevCentroids, cnt_clusters, distances, muxnew, stdxnew = add_temporal_features(x2, feature_headers, num_clusters, num_iters, y2, y2label, dist_type, True, mux, stdx, do_impute)
    else:
        centroids, hnew, standardDevCentroids, cnt_clusters, distances, muxnew, stdxnew = ['NaN']*7

    corr_headers = np.array(feature_headers)
    corr_matrix = np.corrcoef(x2.transpose())
    corr_headers_filtered, corr_matrix_filtered, ix_corr_headers = filter_correlations_via(corr_headers, corr_matrix, corr_vars_exclude)
    print('corr matrix is filtered to size', corr_matrix_filtered.shape)

    if len(variablesubset) != 0:
        x2, feature_headers = variable_subset(x2, variablesubset, feature_headers)

    print ('output is: average:{0:4.3f}'.format(y2.mean()), ' min:', y2.min(), ' max:', y2.max())
    print ('normalizing output.'); y2 = (y2-y2.mean())/y2.std()

    print ('Predicting BMI at age:'+str(agex_low)+ ' to '+str(agex_high)+ 'years, from data in ages:'+ str(months_from)+'-'+str(months_to) + ' months')
    if filterSTR != '':
        print ('filtering patients with: ' , filterSTR)

    print ('total size',ix.sum())
    if (ix.sum() < 50):
        print('Not enough subjects. Next.')
        return (filterSTR, [])

    if modelType == 'lasso' or modelType == 'randomforest' or modelType == 'gradientboost' or modelType == 'lars':
        iters = 10
        model_weights_array = np.zeros((iters, x2.shape[1]), dtype=float)
        auc_test_list=np.zeros((iters), dtype=float); r2testlist = np.zeros((iters), dtype=float);
        for iteration in range(0,iters):
            randix = list(range(0, x2.shape[0]))
            random.shuffle(randix)
            randix = randix[0:int(len(randix)*0.9)]
            datax = x2[randix,:]; datay=y2[randix]; dataylabel = y2label[randix]; mrnx = mrns[randix]
            (model, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, auc_test, r2test, mrnstrain, mrnstest) = train_regression(datax, datay, dataylabel, percentile, modelType, feature_headers, mrnx)
            model_weights_array[iteration, :] = model.coef_ if ((modelType == 'lasso') or (modelType == 'lars')) else model.feature_importances_
            auc_test_list[iteration] = auc_test; r2testlist[iteration] = r2test
            
        model_weights = model_weights_array.mean(axis=0)
        model_weights_std = model_weights_array.std(axis=0)
        model_weights_conf_term = (1.96/np.sqrt(iters)) * model_weights_std
        test_auc_mean = auc_test_list.mean()
        test_auc_mean_ste = (1.96/np.sqrt(iters)) * auc_test_list.std()
        r2test_mean = r2testlist.mean()
        r2test_ste = (1.96/np.sqrt(iters)) * r2testlist.std()

        if return_data_for_error_analysis == True:
            print('lets analyse this')
            return (model, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, auc_test, r2test, feature_headers, centroids, hnew, standardDevCentroids, cnt_clusters, distances, muxnew, stdxnew, mrnstrain, mrnstest, mrns)

    else:
        (model, xtrain, ytrain, xtest, ytest, ytestlabel, ytrainlabel, auc_test, r2test, mrnstrain, mrnstest) = train_regression(x2, y2, y2label, percentile, modelType, feature_headers, mrnx)    
        model_weights_conf_term = np.zeros((x2.shape[1]), dtype=float)
        test_auc_mean = auc_test; r2test_mean= r2test;
        test_auc_mean_ste = 0; r2test_ste=0

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
    for t in operating_Thresholds:
        tp = ((ytestlabel > 0) & (ytestpred.ravel() > t)).sum()*1.0
        tn = ((ytestlabel == 0) & (ytestpred.ravel() <= t)).sum()*1.0
        fn = ((ytestlabel > 0) & (ytestpred.ravel() <= t)).sum()*1.0
        fp = ((ytestlabel == 0) & (ytestpred.ravel() > t)).sum()*1.0

        sens = tp / (tp + fn)
        spec = tn / (tn + fp) 
        ppv = tp / (tp + fp)
        acc = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2*tp / (2*tp + fp + fn)

        report_metrics += '@threshold:{0:4.3f}, sens:{1:4.3f}, spec:{2:4.3f}, ppv:{3:4.3f}, acc:{4:4.3f}, f1:{5:4.3f} \n'.format(t, sens, spec, ppv, acc, f1)
    
    print('total variables', x2.sum(axis=0).shape, ' and total subjects:', x2.shape[0])
    print('->AUC test: {0:4.3f} [{1:4.3f} {2:4.3f}]'.format(test_auc_mean, test_auc_mean - test_auc_mean_ste, test_auc_mean + test_auc_mean_ste))
    print('->Explained Variance (R2) test: {0:4.3f} [{1:4.3f} {2:4.3f}]'.format(r2test_mean, r2test_mean - r2test_ste,  r2test_mean + r2test_ste))
    print(report_metrics)

    occurances = (x2 != 0).sum(axis=0)[sorted_ix]
    zip_weights = {}
    sig_headers = []
    feature_categories = {}
    for i in range(0, (abs(model_weights)>0).sum()):
        fpr, tpr, thresholds = metrics.roc_curve(ytestlabel, xtest_reordered[:,i].ravel())
        feature_auc_indiv = metrics.auc(fpr, tpr)
        corrs = corr_matrix_filtered[sorted_ix[i],:].ravel()
        top_corr_ix = np.argsort(-1*corrs)
        corr_string = 'Correlated most with:\n'+'    '.join( [str(corr_headers_filtered[top_corr_ix[j]])+ ':' + "{0:4.3f}\n".format(corrs[top_corr_ix[j]]) for j in range(0,10)]  ) 

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
        print("{8} {3} | coef {0:4.3f} [{1:4.3f} {2:4.3f}] | OR_adj {9:4.3f} [{10:4.3f} {11:4.3f}] | occ: {4} | OR_unadj: {5:4.3f} [{6:4.3f} {7:4.3f}] | indivs AUC:{12:4.3f}".format(weights[i], weights[i]-terms_sorted[i], weights[i]+terms_sorted[i], factors[i], occurances[i], oratio, low_OR, high_OR, star, np.exp(weights[i]), np.exp(weights[i]-terms_sorted[i]), np.exp(weights[i]+terms_sorted[i]), feature_auc_indiv))
        print(corr_string)

    for k in feature_categories:
        print (k, ":", feature_categories[k])
    return (filterSTR, sig_headers,  centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew, mrns) 

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

    return list(mrnstest[ix_sorted_test])+list(mrnstrain[ix_sorted_train])
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
