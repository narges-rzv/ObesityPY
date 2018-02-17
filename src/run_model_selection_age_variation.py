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


timestr = time.strftime("%Y%m%d")
newdir='../outputs_age_analyses'+timestr
os.mkdir(newdir)

agex_low = 4.5
agex_high = 5.5
months_from = 0
for months_to in [6,12,18,24,36,48]:

    #Create the overall data sets
    x1_no_maternal,y1,y1label,feature_headers,mrns = build_features.call_build_function(d1, d1mom, {}, lat_lon_dic, env_dic, agex_low, agex_high, months_from, months_to, False)
    x1_maternal,y1,y1label,feature_headers,mrns = build_features.call_build_function(d1, d1mom, d1mom_hist, lat_lon_dic, env_dic, agex_low, agex_high, months_from, months_to, False)

    # pickle.dump(x1_no_maternal, open('x_no_maternal_'+timestr+'.pkl', 'wb'))
    # pickle.dump(x1_maternal, open('x_maternal_'+timestr+'.pkl', 'wb'))
    np.save(newdir+'/x_no_maternal_pred_at_'+str(months_to)+'months.npy', x1_no_maternal)
    np.save(newdir+'/x_w_maternal_pred_at_'+str(months_to)+'months.npy', x1_maternal)
    np.save(newdir+'/y_pred_at_'+str(months_to)+'months.npy', y1)
    np.save(newdir+'/y_label_pred_at_'+str(months_to)+'months.npy', y1label)
    pickle.dump(feature_headers, open(newdir+'/feature_headers_pred_at_'+str(months_to)+'months.pkl', 'wb'))
    pickle.dump(mrns, open(newdir+'/mrns_pred_at_'+str(months_to)+'months.pkl', 'wb'))

    #Run all the models
    # filter_str = ['Vital: Wt-latest', 'Vital: BMI-latest']
    if months_to == 6:
        filter_str = ['Vital: Wt-avg0to1','Vital: Wt-avg1to3','Vital: BMI-avg0to1','Vital: BMI-avg1to3']
    elif months_to == 12 or months_to == 18:
        filter_str = ['Vital: Wt-avg0to1','Vital: Wt-avg1to3','Vital: Wt-avg10to13','Vital: BMI-avg0to1','Vital: BMI-avg1to3','Vital: BMI-avg10to13']
    else:
        filter_str = ['Vital: Wt-avg0to1','Vital: Wt-avg1to3','Vital: Wt-avg10to13','Vital: Wt-avg19to24','Vital: BMI-avg0to1','Vital: BMI-avg1to3','Vital: BMI-avg10to13','Vital: BMI-avg19to24']

    print('\nPredicting obesity at 5 years from %d months' %(months_to))
    for modeltype_ix in ['lasso','randomforest', 'gradientboost']:
        for ix, gender in enumerate(['boys', 'girls']):
            for mother_hist_ix in ['no_maternal', 'maternal']:
                x1 = x1_no_maternal if mother_hist_ix == 'no_maternal' else x1_maternal
                filterstr_ix = 'Gender:0' if ix == 0 else 'Gender:1'
                gender_clean = 'boys' if ix==0 else 'girls'

                print('\n' + modeltype_ix + '\t' + filterstr_ix + '\t' + mother_hist_ix + '@ ' + str(months_to) + ' months')
                (model, x2, y2, y2label, feature_headers2, xtrain, ytrain, ytrainlabel, mrnstrain, xtest, ytest, ytestlabel, mrnstest,
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

                np.save(newdir+'/train_data_comb_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,str(months_to),'months.npy']), np.hstack((xtrain,mrnstrain.reshape(-1,1),ytrain.reshape(-1,1),ytrainlabel.reshape(-1,1))))
                np.save(newdir+'/test_data_comb_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,str(months_to),'months.npy']), np.hstack((xtest,mrnstest.reshape(-1,1),ytest.reshape(-1,1),ytestlabel.reshape(-1,1))))
                np.save(newdir+'/data_transformed_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,str(months_to)+'months.npy']), np.hstack((x2,mrns2.reshape(-1,1),y2.reshape(-1,1),y2label.reshape(-1,1))))
                pickle.dump(feature_headers2, open(newdir+'/feature_headers_pred_at_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,str(months_to)+'months_transformed.pkl']), 'wb'))

                titles_total.append(gender_clean + ' ' + mother_hist_ix + ' - model: ' + modeltype_ix + '@ ' + str(months_to) + 'months')
                model_list.append(model)
                prec_total.append(prec_list)
                recall_total.append(recall_list)
                spec_total.append(spec_list)
                auc_list.append([test_auc_mean, test_auc_mean_ste])
                r2_list.append([r2test_mean, r2test_ste])

                # without vitals and maternal historical data
                print(modeltype_ix + '\t' + filterstr_ix + '\t' + mother_hist_ix + '@ ' + str(months_to) + ' months w/o vitals and maternal historical data')
                (model, x2, y2, y2label, feature_headers2, xtrain, ytrain, ytrainlabel, mrnstrain, xtest, ytest, ytestlabel, mrnstest,
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

                np.save(newdir+'/train_data_comb_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'no_vitals_no_mat',str(months_to),'months.npy']), np.hstack((xtrain,mrnstrain.reshape(-1,1),ytrain.reshape(-1,1),ytrainlabel.reshape(-1,1))))
                np.save(newdir+'/test_data_comb_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'no_vitals_no_mat',str(months_to),'months.npy']), np.hstack((xtest,mrnstest.reshape(-1,1),ytest.reshape(-1,1),ytestlabel.reshape(-1,1))))
                np.save(newdir+'/data_transformed_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'no_vitals_no_mat',str(months_to)+'months.npy']), np.hstack((x2,mrns2.reshape(-1,1),y2.reshape(-1,1),y2label.reshape(-1,1))))
                pickle.dump(feature_headers2, open(newdir+'/feature_headers_pred_at_'+'_'.join([modeltype_ix,'no_vitals_no_mat',gender_clean,mother_hist_ix,str(months_to)+'months_transformed.pkl']), 'wb'))

                titles_total.append(gender_clean + ' ' + mother_hist_ix + ' w/o vitals and maternal' + ' - model: ' + modeltype_ix + '@ ' + str(months_to) + 'months')
                model_list.append(model)
                prec_total.append(prec_list)
                recall_total.append(recall_list)
                spec_total.append(spec_list)
                auc_list.append([test_auc_mean, test_auc_mean_ste])
                r2_list.append([r2test_mean, r2test_ste])

            # 'Vital: BMI-latest' only model
            print(modeltype_ix + '\t' + filterstr_ix + '\t' + ' Vital: BMI-latest @ ' + str(months_to) + ' months')
            (model, x2, y2, y2label, feature_headers2, xtrain, ytrain, ytrainlabel, mrnstrain, xtest, ytest, ytestlabel, mrnstest,
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

            np.save(newdir+'/train_data_comb_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'Vital: BMI-latest',str(months_to),'months.npy']), np.hstack((xtrain,mrnstrain.reshape(-1,1),ytrain.reshape(-1,1),ytrainlabel.reshape(-1,1))))
            np.save(newdir+'/test_data_comb_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'Vital: BMI-latest',str(months_to),'months.npy']), np.hstack((xtest,mrnstest.reshape(-1,1),ytest.reshape(-1,1),ytestlabel.reshape(-1,1))))
            np.save(newdir+'/data_transformed_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'Vital: BMI-latest',str(months_to)+'months.npy']), np.hstack((x2,mrns2.reshape(-1,1),y2.reshape(-1,1),y2label.reshape(-1,1))))
            pickle.dump(feature_headers2, open(newdir+'/feature_headers_pred_at_'+'_'.join([modeltype_ix,'Vital: BMI-latest',gender_clean,mother_hist_ix,str(months_to)+'months_transformed.pkl']), 'wb'))

            titles_total.append(gender_clean + ' - model: ' + modeltype_ix + ' - ' + 'Vital: BMI-latest @ ' + str(months_to) + 'months')
            model_list.append(model)
            prec_total.append(prec_list)
            recall_total.append(recall_list)
            spec_total.append(spec_list)
            auc_list.append([test_auc_mean, test_auc_mean_ste])
            r2_list.append([r2test_mean, r2test_ste])

            # 'Vital: Wt-latest' only model
            print(modeltype_ix + '\t' + filterstr_ix + '\t' + ' Vital: Wt-latest @ ' + str(months_to) + ' months')
            (model, x2, y2, y2label, feature_headers2, xtrain, ytrain, ytrainlabel, mrnstrain, xtest, ytest, ytestlabel, mrnstest,
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

            np.save(newdir+'/train_data_comb_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'Vital: Wt-latest',str(months_to),'months.npy']), np.hstack((xtrain,mrnstrain.reshape(-1,1),ytrain.reshape(-1,1),ytrainlabel.reshape(-1,1))))
            np.save(newdir+'/test_data_comb_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'Vital: Wt-latest',str(months_to),'months.npy']), np.hstack((xtest,mrnstest.reshape(-1,1),ytest.reshape(-1,1),ytestlabel.reshape(-1,1))))
            np.save(newdir+'/data_transformed_'+'_'.join([modeltype_ix,gender_clean,mother_hist_ix,'Vital: Wt-latest',str(months_to)+'months.npy']), np.hstack((x2,mrns2.reshape(-1,1),y2.reshape(-1,1),y2label.reshape(-1,1))))
            pickle.dump(feature_headers2, open(newdir+'/feature_headers_pred_at_'+'_'.join([modeltype_ix,'Vital: Wt-latest',gender_clean,mother_hist_ix,str(months_to)+'months_transformed.pkl']), 'wb'))

            titles_total.append(gender_clean + ' - model: ' + modeltype_ix + ' - ' + 'Vital: Wt-latest @ ' + str(months_to) + 'months')
            model_list.append(model)
            prec_total.append(prec_list)
            recall_total.append(recall_list)
            spec_total.append(spec_list)
            auc_list.append([test_auc_mean, test_auc_mean_ste])
            r2_list.append([r2test_mean, r2test_ste])


# Save the outputs
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
