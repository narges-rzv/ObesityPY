import pickle
import train
import build_features
import numpy as np
import time
import os

#Get the data
d1 = pickle.load(open('../python objects/patientdata_20170823.pkl', 'rb'))
d1mom = pickle.load(open('../python objects/patient_mother_data_20170724.pkl', 'rb'))
lat_lon_dic = pickle.load(open('../python objects/lat_lon_data_20170920.pkl', 'rb'))
env_dic= pickle.load(open('../python objects/census_data_20170920.pkl', 'rb'))
d1mom_hist = pickle.load(open('../python objects/full_lutheran_mother_data.pkl', 'rb'))

# Instantiate the lists
prec_total = []
recall_total = []
spec_total = []
titles_total = []
model_list = []
auc_list = []
r2_list = []

agex_low = 4.5
agex_high = 5.5
months_from = 0
months_to in 24

timestr = time.strftime("%Y%m%d")
newdir='../outputs_'+str(months_to)+'months_analyses'+timestr
os.mkdir(newdir)

#Create the overall data sets
x1_no_maternal,y1,y1label,feature_headers,mrns = build_features.call_build_function(d1, d1mom, {}, lat_lon_dic, env_dic, agex_low, agex_high, months_from, months_to, False)
x1_maternal,y1,y1label,feature_headers,mrns = build_features.call_build_function(d1, d1mom, d1mom_hist, lat_lon_dic, env_dic, agex_low, agex_high, months_from, months_to, False)

np.savez_compressed(newdir+'/no_maternal_'+str(months_to)+'months', x=x1_no_maternal, y=y1, ylabel=y1label, mrns=mrns, features=np.array(feature_headers))
np.savez_compressed(newdir+'/maternal_'+str(months_to)+'months', x=x1_maternal, y=y1, ylabel=y1label, mrns=mrns, features=np.array(feature_headers))

#Run all the models
filter_str = ['Vital: Wt-avg0to1','Vital: Wt-avg1to3','Vital: Wt-avg10to13','Vital: Wt-avg19to24','Vital: BMI-avg0to1','Vital: BMI-avg1to3','Vital: BMI-avg10to13','Vital: BMI-avg19to24']

print('\nPredicting obesity at 5 years from %d months' %(months_to))
for modeltype_ix in ['lasso','randomforest', 'gradientboost']:
    for ix, gender in enumerate(['boys', 'girls']):
        for mother_hist_ix in ['no_maternal', 'maternal']:
            x1 = x1_no_maternal if mother_hist_ix == 'no_maternal' else x1_maternal
            filterstr_ix = 'Gender:0' if ix == 0 else 'Gender:1'
            gender_clean = 'boys' if ix==0 else 'girls'

            print('\n' + modeltype_ix + '\t' + filterstr_ix + '\t' + mother_hist_ix + '@ ' + str(months_to) + ' months')
            (model, x2, y2, y2label, ix_filter, randix, ix_train, ix_test, feature_headers2, xtrain, ytrain, ytrainlabel, mrnstrain, xtest, ytest, ytestlabel, mrnstest,
            filterSTR, sig_headers, centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew, mrns2,
            prec_list, recall_list, spec_list, test_auc_mean, test_auc_mean_ste, r2test_mean, r2test_ste) = \
                    train.train_regression_model_for_bmi({}, {}, {}, {}, {}, x1, y1, y1label, feature_headers, mrns, agex_low, agex_high, months_from, months_to,
                            filterSTR=filter_str + [filterstr_ix],
                            variablesubset=[],
                            modelType=modeltype_ix,
                            return_data_for_error_analysis=False,
                            return_data=True,
                            return_data_transformed=True,
                            return_train_test_data=True,
                            do_impute=False)

            np.savez_compressed(newdir+'/train_data_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,str(months_to),'months']), x=xtrain, mrns=mrnstrain, features=np.array(feature_headers2), y=ytrain, ylabel=ytrainlabel)
            np.savez_compressed(newdir+'/test_data_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,str(months_to),'months']), x=xtest, mrns=mrnstest, features=np.array(feature_headers2), y=ytest, ylabel=ytestlabel)
            np.savez_compressed(newdir+'/data_transformed_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,str(months_to),'months']), x=x2, mrns=mrns2, features=np.array(feature_headers2), y=y2, ylabel=y2label, ix2=ix_filter, modeling_ix=randix, train_ix=ix_train, test_ix=ix_test)

            titles_total.append(gender_clean + ' ' + mother_hist_ix + ' - model: ' + modeltype_ix + '@ ' + str(months_to) + 'months')
            model_list.append(model)
            prec_total.append(prec_list)
            recall_total.append(recall_list)
            spec_total.append(spec_list)
            auc_list.append([test_auc_mean, test_auc_mean_ste])
            r2_list.append([r2test_mean, r2test_ste])

            # without vitals and maternal historical data
            print(modeltype_ix + '\t' + filterstr_ix + '\t' + mother_hist_ix + '@ ' + str(months_to) + ' months w/o vitals and maternal historical data')
            (model, x2, y2, y2label, ix_filter, randix, ix_train, ix_test, feature_headers2, xtrain, ytrain, ytrainlabel, mrnstrain, xtest, ytest, ytestlabel, mrnstest,
            filterSTR, sig_headers, centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew, mrns2,
            prec_list, recall_list, spec_list, test_auc_mean, test_auc_mean_ste, r2test_mean, r2test_ste) = \
                    train.train_regression_model_for_bmi({}, {}, {}, {}, {}, x1, y1, y1label, feature_headers, mrns, agex_low, agex_high, months_from, months_to,
                            filterSTR=[filterstr_ix],
                            variablesubset=['Census','Diagnosis','Ethnicity','Gender','Lab','MatDeliveryAge',
                                'Medication','Newborn Diagnosis','Prim_Insur','Race','Second_Insur','Zipcode'],
                            modelType=modeltype_ix,
                            return_data_for_error_analysis=False,
                            return_data=True,
                            return_data_transformed=True,
                            return_train_test_data=True,
                            do_impute=False)

            np.savez_compressed(newdir+'/train_data_comb_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'no_vitals_no_mat',str(months_to),'months']), x=xtrain, mrns=mrnstrain, features=np.array(feature_headers2), y=ytrain, ylabel=ytrainlabel)
            np.savez_compressed(newdir+'/testdata_comb_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'no_vitals_no_mat',str(months_to),'months']), x=xtest, mrns=mrnstest, features=np.array(feature_headers2), y=ytest, ylabel=ytestlabel)
            np.savez_compressed(newdir+'/data_transformed_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'no_vitals_no_mat',str(months_to),'months']), x=x2, mrns=mrns2, features=np.array(feature_headers2), y=y2, ylabel=y2label, ix2=ix_filter, modeling_ix=randix, train_ix=ix_train, test_ix=ix_test)

            titles_total.append(gender_clean + ' ' + mother_hist_ix + ' w/o vitals and maternal' + ' - model: ' + modeltype_ix + '@ ' + str(months_to) + 'months')
            model_list.append(model)
            prec_total.append(prec_list)
            recall_total.append(recall_list)
            spec_total.append(spec_list)
            auc_list.append([test_auc_mean, test_auc_mean_ste])
            r2_list.append([r2test_mean, r2test_ste])

            # without exclusion criteria (except gender)
            print(modeltype_ix + '\t' + filterstr_ix + '\t' + mother_hist_ix + '@ ' + str(months_to) + ' months no exclusion')
            (model, x2, y2, y2label, ix_filter, randix, ix_train, ix_test, feature_headers2, xtrain, ytrain, ytrainlabel, mrnstrain, xtest, ytest, ytestlabel, mrnstest,
            filterSTR, sig_headers, centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew, mrns2,
            prec_list, recall_list, spec_list, test_auc_mean, test_auc_mean_ste, r2test_mean, r2test_ste) = \
                    train.train_regression_model_for_bmi({}, {}, {}, {}, {}, x1, y1, y1label, feature_headers, mrns, agex_low, agex_high, months_from, months_to,
                            filterSTR=[filterstr_ix],
                            variablesubset=[],
                            modelType=modeltype_ix,
                            return_data_for_error_analysis=False,
                            return_data=True,
                            return_data_transformed=True,
                            return_train_test_data=True,
                            do_impute=False)

            np.savez_compressed(newdir+'/train_data_comb_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'no_exclusion',str(months_to),'months']), x=xtrain, mrns=mrnstrain, features=np.array(feature_headers2), y=ytrain, ylabel=ytrainlabel)
            np.savez_compressed(newdir+'/testdata_comb_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'no_exclusion',str(months_to),'months']), x=xtest, mrns=mrnstest, features=np.array(feature_headers2), y=ytest, ylabel=ytestlabel)
            np.savez_compressed(newdir+'/data_transformed_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'no_exclusion',str(months_to),'months']), x=x2, mrns=mrns2, features=np.array(feature_headers2), y=y2, ylabel=y2label, ix2=ix_filter, modeling_ix=randix, train_ix=ix_train, test_ix=ix_test)

            titles_total.append(gender_clean + ' ' + mother_hist_ix + ' w/o exclusions' + ' - model: ' + modeltype_ix + '@ ' + str(months_to) + 'months')
            model_list.append(model)
            prec_total.append(prec_list)
            recall_total.append(recall_list)
            spec_total.append(spec_list)
            auc_list.append([test_auc_mean, test_auc_mean_ste])
            r2_list.append([r2test_mean, r2test_ste])

        # 'Vital: BMI-latest' only model
        print(modeltype_ix + '\t' + filterstr_ix + '\t' + ' Vital: BMI-latest @ ' + str(months_to) + ' months')
        (model, x2, y2, y2label, ix_filter, randix, ix_train, ix_test, feature_headers2, xtrain, ytrain, ytrainlabel, mrnstrain, xtest, ytest, ytestlabel, mrnstest,
        filterSTR, sig_headers, centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew, mrns2,
        prec_list, recall_list, spec_list, test_auc_mean, test_auc_mean_ste, r2test_mean, r2test_ste) = \
                train.train_regression_model_for_bmi({}, {}, {}, {}, {}, x1, y1, y1label, feature_headers, mrns, agex_low, agex_high, months_from, months_to,
                        filterSTR=filter_str + [filterstr_ix],
                        variablesubset=['Vital: BMI-latest'],
                        modelType=modeltype_ix,
                        return_data_for_error_analysis=False,
                        return_data=True,
                        return_data_transformed=True,
                        return_train_test_data=True,
                        do_impute=False)

        np.savez_compressed(newdir+'/train_data_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'Vital: BMI-latest',str(months_to),'months']), x=xtrain, mrns=mrnstrain, features=np.array(feature_headers2), y=ytrain, ylabel=ytrainlabel)
        np.savez_compressed(newdir+'/test_data_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'Vital: BMI-latest',str(months_to),'months']), x=xtest, mrns=mrnstest, features=np.array(feature_headers2), y=ytest, ylabel=ytestlabel)
        np.savez_compressed(newdir+'/data_transformed_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'Vital: BMI-latest',str(months_to),'months']), x=x2, mrns=mrns2, features=np.array(feature_headers2), y=y2, ylabel=y2label, ix2=ix_filter, modeling_ix=randix, train_ix=ix_train, test_ix=ix_test)

        titles_total.append(gender_clean + ' - model: ' + modeltype_ix + ' - ' + 'Vital: BMI-latest @ ' + str(months_to) + 'months')
        model_list.append(model)
        prec_total.append(prec_list)
        recall_total.append(recall_list)
        spec_total.append(spec_list)
        auc_list.append([test_auc_mean, test_auc_mean_ste])
        r2_list.append([r2test_mean, r2test_ste])

        # 'Vital: Wt-latest' only model
        print(modeltype_ix + '\t' + filterstr_ix + '\t' + ' Vital: Wt-latest @ ' + str(months_to) + ' months')
        (model, x2, y2, y2label, ix_filter, randix, ix_train, ix_test, feature_headers2, xtrain, ytrain, ytrainlabel, mrnstrain, xtest, ytest, ytestlabel, mrnstest,
        filterSTR, sig_headers, centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew, mrns2,
        prec_list, recall_list, spec_list, test_auc_mean, test_auc_mean_ste, r2test_mean, r2test_ste) = \
                train.train_regression_model_for_bmi({}, {}, {}, {}, {}, x1, y1, y1label, feature_headers, mrns, agex_low, agex_high, months_from, months_to,
                        filterSTR=filter_str + [filterstr_ix],
                        variablesubset=['Vital: Wt-latest'],
                        modelType=modeltype_ix,
                        return_data_for_error_analysis=False,
                        return_data=True,
                        return_data_transformed=True,
                        return_train_test_data=True,
                        do_impute=False)

        np.savez_compressed(newdir+'/train_data_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'Vital: Wt-latest',str(months_to),'months']), x=xtrain, mrns=mrnstrain, features=np.array(feature_headers2), y=ytrain, ylabel=ytrainlabel)
        np.savez_compressed(newdir+'/test_data_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'Vital: Wt-latest',str(months_to),'months']), x=xtest, mrns=mrnstest, features=np.array(feature_headers2), y=ytest, ylabel=ytestlabel)
        np.savez_compressed(newdir+'/data_transformed_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'Vital: Wt-latest',str(months_to),'months']), x=x2, mrns=mrns2, features=np.array(feature_headers2), y=y2, ylabel=y2label, ix2=ix_filter, modeling_ix=randix, train_ix=ix_train, test_ix=ix_test)

        titles_total.append(gender_clean + ' - model: ' + modeltype_ix + ' - ' + 'Vital: Wt-latest @ ' + str(months_to) + 'months')
        model_list.append(model)
        prec_total.append(prec_list)
        recall_total.append(recall_list)
        spec_total.append(spec_list)
        auc_list.append([test_auc_mean, test_auc_mean_ste])
        r2_list.append([r2test_mean, r2test_ste])


# Save the outputs -- overwrite at each step in case of crash
pickle.dump(titles_total, open(newdir+'/titles_total.pkl', 'wb'))
pickle.dump(model_list, open(newdir+'/model_list.pkl', 'wb'))
pickle.dump(prec_total, open(newdir+'/prec_total.pkl', 'wb'))
pickle.dump(recall_total, open(newdir+'/recall_total.pkl', 'wb'))
pickle.dump(spec_total, open(newdir+'/spec_total.pkl', 'wb'))
pickle.dump(auc_list, open(newdir+'/auc_list.pkl', 'wb'))
pickle.dump(r2_list, open(newdir+'/r2_list.pkl', 'wb'))

# get the training metric plots
# title1='Precision vs. Recall Curves: Obesity at 5 years from 24 months'
# title2='ROC Curve: Obesity at 5 years from 24 months'
# train.prec_rec_curve(recall_total, prec_total, titles_total, title1, show=False, save=True)
# train.ROC_curve(recall_total, spec_total, titles_total, title2, show=False, save=True)
