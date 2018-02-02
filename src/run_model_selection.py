import pickle
import train
import build_features
import numpy as np
import time

#Get the data
d1 = pickle.load(open('../python objects/patientdata_20170823.pkl', 'rb'))
d1mom = pickle.load(open('../python objects/patient_mother_data_20170724.pkl', 'rb'))
lat_lon_dic = pickle.load(open('../python objects/lat_lon_data_20170920.pkl', 'rb'))
env_dic= pickle.load(open('../python objects/census_data_20170920.pkl', 'rb'))
d1mom_hist = pickle.load(open('../../data/full_lutheran_mother_data.pkl', 'rb'))

#Create the overall data sets
x1_no_maternal,y1,y1label,feature_headers,mrns = build_features.call_build_function(d1, d1mom, d1mom_hist, lat_lon_dic, env_dic, 4.5, 5.5,  0, 24, False)
x1_maternal,y1,y1label,feature_headers,mrns = build_features.call_build_function(d1, d1mom, d1mom_hist, lat_lon_dic, env_dic, 4.5, 5.5,  0, 24, False)

timestr = time.strftime("%Y%m%d-%H%M%S")
# pickle.dump(x1_no_maternal, open('x_no_maternal_'+timestr+'.pkl', 'wb'))
# pickle.dump(x1_maternal, open('x_maternal_'+timestr+'.pkl', 'wb'))
np.savetxt('x_no_maternal_'+timestr+'.txt', x1_no_maternal, delimiter='\t')
np.savetxt('x_w_maternal_'+timestr+'.txt', x1_maternal, delimiter='\t')
pickle.dump(y1, open('y_'+timestr+'.pkl', 'wb'))
pickle.dump(y1label, open('y_label_'+timestr+'.pkl', 'wb'))
pickle.dump(feature_headers, open('feature_headers_'+timestr+'.pkl', 'wb'))
pickle.dump(mrns, open('mrns_'+timestr+'.pkl', 'wb'))


#Run all the models
prec_total = []
recall_total = []
spec_total = []
titles_total = []
model_list = []
auc_list = []
r2_list = []

filter_str = ['Vital: Wt-avg0to1', 'Vital: Wt-avg5to7', 'Vital: Wt-avg10to13', 'Vital: Wt-avg19to24']

for modeltype_ix in ['lasso','randomforest']:
    for ix, gender in enumerate(['boys', 'girls']):
        for mother_hist_ix in ['w/o Lutheran', 'w/ Lutheran']:

            x1 = x1_no_maternal if mother_hist_ix == 'w/o Lutheran' else x1_maternal
            filterstr_ix = 'Gender:0' if ix == 0 else 'Gender:1'
            gender_clean = 'boys' if ix==0 else 'girls'

            print(modeltype_ix + '\t' + filterstr_ix + '\t' + mother_hist_ix)
            (model, x, y, ylabel, features, mrn_list,
            filterSTR, sig_headers, centroids, hnew, standardDevCentroids, cnt_clusters, muxnew,stdxnew,
            mrn_filtered_list, prec_list, recall_list, spec_list, test_auc_mean, test_auc_mean_ste, r2test_mean, r2test_ste) = \
                    train.train_regression_model_for_bmi({}, {}, {}, {}, {}, x1, y1, y1label, feature_headers, mrns, 4.5, 5.5, 0, 24,
                            filterSTR=[filterstr_ix]+filter_str,
                            variablesubset=[],
                            modelType=modeltype_ix,
                            return_data_for_error_analysis=False,
                            return_data=True,
                            return_data_transformed=True,
                            do_impute=False)

            titles_total.append(gender_clean + ' ' + mother_hist_ix + '- model: ' + modeltype_ix + ' AUC: {0:4.3f} [{1:4.3f}, {2:4.3f}]' .format(test_auc_mean, test_auc_mean - test_auc_mean_ste, test_auc_mean + test_auc_mean_ste))
            model_list.append(model)
            prec_total.append(prec_list)
            recall_total.append(recall_list)
            spec_total.append(spec_list)
            auc_list.append([test_auc_mean, test_auc_mean - test_auc_mean_ste, test_auc_mean + test_auc_mean_ste])
            r2_list.append([r2test_mean, r2test_mean - r2test_ste, r2test_mean + r2test_ste])

        # 'Vital: BMI-latest' only model
        print(modeltype_ix + '\t' + filterstr_ix + '\t' + mother_hist_ix + ' Vital: BMI-latest')
        (model, x, y, ylabel, features, mrn_list,
        filterSTR, sig_headers,  centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew,
        mrn_filtered_list, prec_list, recall_list, spec_list, test_auc_mean, test_auc_mean_ste, r2test_mean, r2test_ste) = \
                train.train_regression_model_for_bmi({}, {}, {}, {}, {}, x1, y1, y1label, feature_headers, mrns, 4.5, 5.5,  0, 24,
                        filterSTR=[filterstr_ix]+filter_str,
                        variablesubset=['Vital: BMI-latest'],
                        modelType=modeltype_ix,
                        return_data_for_error_analysis=False,
                        return_data=True,
                        return_data_transformed=True,
                        do_impute=False)

        titles_total.append(gender_clean + ' ' + mother_hist_ix + '- model: ' + modeltype_ix + ' - ' + 'Vital: BMI-latest' + ' AUC: {0:4.3f} [{1:4.3f}, {2:4.3f}]' .format(test_auc_mean, test_auc_mean - test_auc_mean_ste, test_auc_mean + test_auc_mean_ste))
        model_list.append(model)
        prec_total.append(prec_list)
        recall_total.append(recall_list)
        spec_total.append(spec_list)
        auc_list.append([test_auc_mean, test_auc_mean - test_auc_mean_ste, test_auc_mean + test_auc_mean_ste])
        r2_list.append([r2test_mean, r2test_mean - r2test_ste, r2test_mean + r2test_ste])

        # 'Vital: BMI-avg19to24' only model
        print(modeltype_ix + '\t' + filterstr_ix + '\t' + mother_hist_ix + ' Vital: BMI-avg19to24')
        (model, x, y, ylabel, features, mrn_list,
        filterSTR, sig_headers,  centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew,
        mrn_filtered_list, prec_list, recall_list, spec_list, test_auc_mean, test_auc_mean_ste, r2test_mean, r2test_ste) = \
                train.train_regression_model_for_bmi({}, {}, {}, {}, {}, x1, y1, y1label, feature_headers, mrns, 4.5, 5.5,  0, 24,
                        filterSTR=[filterstr_ix]+filter_str,
                        variablesubset=['Vital: BMI-avg19to24'],
                        modelType=modeltype_ix,
                        return_data_for_error_analysis=False,
                        return_data=True,
                        return_data_transformed=True,
                        do_impute=False)
        titles_total.append(gender_clean + ' ' + mother_hist_ix + '- model: ' + modeltype_ix + ' - ' + 'Vital: BMI-avg19to24' + ' AUC: {0:4.3f} [{1:4.3f}, {2:4.3f}]' .format(test_auc_mean, test_auc_mean - test_auc_mean_ste, test_auc_mean + test_auc_mean_ste))
        model_list.append(model)
        prec_total.append(prec_list)
        recall_total.append(recall_list)
        spec_total.append(spec_list)
        auc_list.append([test_auc_mean, test_auc_mean - test_auc_mean_ste, test_auc_mean + test_auc_mean_ste])
        r2_list.append([r2test_mean, r2test_mean - r2test_ste, r2test_mean + r2test_ste])

        # Vital: Wt-latest
        print(modeltype_ix + '\t' + filterstr_ix + '\t' + mother_hist_ix + ' Vital: Wt-latest')
        (model, x, y, ylabel, features, mrn_list,
        filterSTR, sig_headers, centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew,
        mrn_filtered_list, prec_list, recall_list, spec_list, test_auc_mean, test_auc_mean_ste, r2test_mean, r2test_ste) = \
                train.train_regression_model_for_bmi({}, {}, {}, {}, {}, x1, y1, y1label, feature_headers, mrns, 4.5, 5.5,  0, 24,
                        filterSTR=[filterstr_ix]+filter_str,
                        variablesubset=['Vital: Wt-latest'],
                        modelType=modeltype_ix,
                        return_data_for_error_analysis=False,
                        return_data=True,
                        return_data_transformed=True,
                        do_impute=False)
        titles_total.append(gender_clean + ' ' + mother_hist_ix + '- model: ' + modeltype_ix + ' - ' + 'Vital: Wt-latest' + ' AUC: {0:4.3f} [{1:4.3f}, {2:4.3f}]' .format(test_auc_mean, test_auc_mean - test_auc_mean_ste, test_auc_mean + test_auc_mean_ste))
        model_list.append(model)
        prec_total.append(prec_list)
        recall_total.append(recall_list)
        spec_total.append(spec_list)
        auc_list.append([test_auc_mean, test_auc_mean - test_auc_mean_ste, test_auc_mean + test_auc_mean_ste])
        r2_list.append([r2test_mean, r2test_mean - r2test_ste, r2test_mean + r2test_ste])

        # Vital: Wt-avg19to24
        print(modeltype_ix + '\t' + filterstr_ix + '\t' + mother_hist_ix + ' Vital: Wt-avg19to24')
        (model, x, y, ylabel, features, mrn_list,
        filterSTR, sig_headers,  centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew,
        mrn_filtered_list, prec_list, recall_list, spec_list, test_auc_mean, test_auc_mean_ste, r2test_mean, r2test_ste) = \
                train.train_regression_model_for_bmi({}, {}, {}, {}, {}, x1, y1, y1label, feature_headers, mrns, 4.5, 5.5,  0, 24,
                        filterSTR=[filterstr_ix]+filter_str,
                        variablesubset=['Vital: Wt-avg19to24'],
                        modelType=modeltype_ix,
                        return_data_for_error_analysis=False,
                        return_data=True,
                        return_data_transformed=True,
                        do_impute=False)

        titles_total.append(gender_clean + ' ' + mother_hist_ix + '- model: ' + modeltype_ix + ' - ' + 'Vital: Wt-avg19to24' + ' AUC: {0:4.3f} [{1:4.3f}, {2:4.3f}]' .format(test_auc_mean, test_auc_mean - test_auc_mean_ste, test_auc_mean + test_auc_mean_ste))
        model_list.append(model)
        prec_total.append(prec_list)
        recall_total.append(recall_list)
        spec_total.append(spec_list)
        auc_list.append([test_auc_mean, test_auc_mean - test_auc_mean_ste, test_auc_mean + test_auc_mean_ste])
        r2_list.append([r2test_mean, r2test_mean - r2test_ste, r2test_mean + r2test_ste])

# Save the outputs
pickle.dump(titles_total, open('titles_total_'+timestr+'.pkl', 'wb'))
pickle.dump(model_list, open('model_list_'+timestr+'.pkl', 'wb'))
pickle.dump(prec_total, open('prec_total_'+timestr+'.pkl', 'wb'))
pickle.dump(recall_total, open('recall_total_'+timestr+'.pkl', 'wb'))
pickle.dump(spec_total, open('spec_total_'+timestr+'.pkl', 'wb'))
pickle.dump(auc_list, open('auc_list_'+timestr+'.pkl', 'wb'))
pickle.dump(r2_list, open('r2_list_'+timestr+'.pkl', 'wb'))

# get the training metric plots
title1='Precision vs. Recall Curves: Obesity at 5 years from 24 months'
title2='ROC Curve: Obesity at 5 years from 24 months'
train.prec_rec_curve(recall_total, prec_total, titles_total, title1, show=False, save=True)
train.ROC_curve(recall_total, spec_total, titles_total, title2, show=False, save=True)
