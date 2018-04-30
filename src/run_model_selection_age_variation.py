import pickle
import train
import build_features
import numpy as np
import time
import os

#Get the data
d1 = pickle.load(open('../python objects/patientdata_20170823.pkl', 'rb'))
d1mom = pickle.load(open('../python objects/patient_mother_data_20170724.pkl', 'rb'))
lat_lon_dic = pickle.load(open('../python objects/lat_lon_data_20180305.pkl', 'rb'))
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
    x1,y1,y1label,feature_headers,mrns = build_features.call_build_function(d1, d1mom, d1mom_hist, lat_lon_dic, env_dic, agex_low, agex_high, months_from, months_to, False)

    np.savez_compressed(newdir+'/no_maternal_'+str(months_to)+'months', x=x1_no_maternal, y=y1, ylabel=y1label, mrns=mrns, features=np.array(feature_headers))
    np.savez_compressed(newdir+'/maternal_'+str(months_to)+'months', x=x1_maternal, y=y1, ylabel=y1label, mrns=mrns, features=np.array(feature_headers))

    #Run all the models
    # filter_str = ['Vital: Wt-latest', 'Vital: BMI-latest']
    if months_to == 6:
        filter_str = ['Vital: Wt-avg0to1','Vital: Wt-avg1to3','Vital: BMI-avg0to1','Vital: BMI-avg1to3']
    elif months_to == 12 or months_to == 18:
        filter_str = ['Vital: Wt-avg0to1','Vital: Wt-avg1to3','Vital: Wt-avg10to13','Vital: BMI-avg0to1','Vital: BMI-avg1to3','Vital: BMI-avg10to13']
    else:
        filter_str = ['Vital: Wt-avg0to1','Vital: Wt-avg1to3','Vital: Wt-avg10to13','Vital: Wt-avg19to24','Vital: BMI-avg0to1','Vital: BMI-avg1to3','Vital: BMI-avg10to13','Vital: BMI-avg19to24']

    # Get the table statistics for each of the prediction points
    ix_filter_boys, x2_boys, y2_boys, y2label_boys_obese, mrns_boys = train.filter_training_set_forLinear(x1, y1, y1label, feature_headers, filterSTR=filter_str+['Gender:0'], mrns=mrns, percentile=False)
    ix_filter_girls, x2_girls, y2_girls, y2label_girls_obese, mrns_girls = train.filter_training_set_forLinear(x1, y1, y1label, feature_headers, filterSTR=filter_str+['Gender:1'], mrns=mrns, percentile=False)
    x2 = np.vstack((x2_boys, x2_girls))
    y2 = np.vstack((y2_boys.reshape(-1,1), y2_girls.reshape(-1,1))).ravel()
    y2label_obese = np.vstack((y2label_boys_obese.reshape(-1,1), y2label_girls_obese.reshape(-1,1))).ravel()
    statsdir = time.strftime("table_stats_%Y%m%d_")+str(months_from)+'to'+str(months_to)+'months_'+str(agex_low)+'to'+str(agex_high)+'years'
    if not os.path.exists(statsdir):
        os.mkdir(statsdir)
    train.get_stat_table(x2, y2, y2label_obese, feature_headers, folder=statsdir)

    print('\nPredicting obesity at 5 years from %d months' %(months_to))
    for modeltype_ix in ['lasso','randomforest', 'gradientboost']:
        for ix, gender in enumerate(['boys', 'girls']):
            filterstr_ix = 'Gender:0' if ix == 0 else 'Gender:1'
            gender_clean = 'boys' if ix==0 else 'girls'

            print('\n' + gender_clean + ' - ' + modeltype_ix + ' @ ' + str(months_to) + ' months')
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

            # np.savez_compressed(newdir+'/train_data_'+'_'.join([modeltype_ix,gender_clean,str(months_to),'months']), x=xtrain, mrns=mrnstrain, features=np.array(feature_headers2), y=ytrain, ylabel=ytrainlabel)
            # np.savez_compressed(newdir+'/test_data_'+'_'.join([modeltype_ix,gender_clean,str(months_to),'months']), x=xtest, mrns=mrnstest, features=np.array(feature_headers2), y=ytest, ylabel=ytestlabel)
            np.savez_compressed(newdir+'/data_transformed_'+'_'.join([modeltype_ix,gender_clean,str(months_to),'months']), x=x2, mrns=mrns2, features=np.array(feature_headers2), y=y2, ylabel=y2label, ix2=ix_filter, modeling_ix=randix, train_ix=ix_train, test_ix=ix_test)

            titles_total.append(gender_clean + ' - ' + modeltype_ix + '@ ' + str(months_to) + 'months')
            model_list.append(model)
            prec_total.append(prec_list)
            recall_total.append(recall_list)
            spec_total.append(spec_list)
            auc_list.append([test_auc_mean, test_auc_mean_ste])
            r2_list.append([r2test_mean, r2test_ste])

            # without weight and bmi
            print('\n' + gender_clean + ' no weight and bmi - ' + modeltype_ix + ' @ ' + str(months_to) + ' months')
            (model, x2, y2, y2label, ix_filter, randix, ix_train, ix_test, feature_headers2, xtrain, ytrain, ytrainlabel, mrnstrain, xtest, ytest, ytestlabel, mrnstest,
            filterSTR, sig_headers, centroids, hnew, standardDevCentroids, cnt_clusters, muxnew, stdxnew, mrns2,
            prec_list, recall_list, spec_list, test_auc_mean, test_auc_mean_ste, r2test_mean, r2test_ste) = \
                    train.train_regression_model_for_bmi({}, {}, {}, {}, {}, x1, y1, y1label, feature_headers, mrns, agex_low, agex_high, months_from, months_to,
                            filterSTR=[filterstr_ix],
                            variablesubset=[fh for fh in feature_headers if not any([x in fh for x in ('Wt','BMI')])],
                            modelType=modeltype_ix,
                            return_data_for_error_analysis=False,
                            return_data=True,
                            return_data_transformed=True,
                            return_train_test_data=True,
                            do_impute=False)

            # np.savez_compressed(newdir+'/train_data_comb_'+'_'.join([modeltype_ix,gender_clean,'no_vitals_no_mat',str(months_to),'months']), x=xtrain, mrns=mrnstrain, features=np.array(feature_headers2), y=ytrain, ylabel=ytrainlabel)
            # np.savez_compressed(newdir+'/testdata_comb_'+'_'.join([modeltype_ix,gender_clean,'no_vitals_no_mat',str(months_to),'months']), x=xtest, mrns=mrnstest, features=np.array(feature_headers2), y=ytest, ylabel=ytestlabel)
            np.savez_compressed(newdir+'/data_transformed_'+'_'.join([modeltype_ix,gender_clean,'no_vitals_no_mat',str(months_to),'months']), x=x2, mrns=mrns2, features=np.array(feature_headers2), y=y2, ylabel=y2label, ix2=ix_filter, modeling_ix=randix, train_ix=ix_train, test_ix=ix_test)

            titles_total.append(gender_clean + ' no weight and bmi - ' + modeltype_ix + ' @ ' + str(months_to) + ' months')
            model_list.append(model)
            prec_total.append(prec_list)
            recall_total.append(recall_list)
            spec_total.append(spec_list)
            auc_list.append([test_auc_mean, test_auc_mean_ste])
            r2_list.append([r2test_mean, r2test_ste])

            # without exclusion criteria (except gender)
            print('\n' gender_clean + ' no exclusions - ' + modeltype_ix + ' @ ' + str(months_to) + ' months')
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

            # np.savez_compressed(newdir+'/train_data_comb_'+'_'.join([modeltype_ix,gender_clean,'no_exclusion',str(months_to),'months']), x=xtrain, mrns=mrnstrain, features=np.array(feature_headers2), y=ytrain, ylabel=ytrainlabel)
            # np.savez_compressed(newdir+'/testdata_comb_'+'_'.join([modeltype_ix,gender_clean,'no_exclusion',str(months_to),'months']), x=xtest, mrns=mrnstest, features=np.array(feature_headers2), y=ytest, ylabel=ytestlabel)
            np.savez_compressed(newdir+'/data_transformed_'+'_'.join([modeltype_ix,gender_clean,'no_exclusion',str(months_to),'months']), x=x2, mrns=mrns2, features=np.array(feature_headers2), y=y2, ylabel=y2label, ix2=ix_filter, modeling_ix=randix, train_ix=ix_train, test_ix=ix_test)

            titles_total.append(gender_clean + ' no exclusions - ' + modeltype_ix + '@ ' + str(months_to) + 'months')
            model_list.append(model)
            prec_total.append(prec_list)
            recall_total.append(recall_list)
            spec_total.append(spec_list)
            auc_list.append([test_auc_mean, test_auc_mean_ste])
            r2_list.append([r2test_mean, r2test_ste])

            # 'Vital: BMI-latest' only model
            print('\n' + gender_clean + ' Vital: BMI-latest - ' + modeltype_ix + ' @ ' + str(months_to) + ' months')
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

            # np.savez_compressed(newdir+'/train_data_'+'_'.join([modeltype_ix,gender_clean,'Vital_BMI_latest',str(months_to),'months']), x=xtrain, mrns=mrnstrain, features=np.array(feature_headers2), y=ytrain, ylabel=ytrainlabel)
            # np.savez_compressed(newdir+'/test_data_'+'_'.join([modeltype_ix,gender_clean,'Vital_BMI_latest',str(months_to),'months']), x=xtest, mrns=mrnstest, features=np.array(feature_headers2), y=ytest, ylabel=ytestlabel)
            np.savez_compressed(newdir+'/data_transformed_'+'_'.join([modeltype_ix,gender_clean,'Vital_BMI_latest',str(months_to),'months']), x=x2, mrns=mrns2, features=np.array(feature_headers2), y=y2, ylabel=y2label, ix2=ix_filter, modeling_ix=randix, train_ix=ix_train, test_ix=ix_test)

            titles_total.append(gender_clean + ' Vital: BMI-latest - ' + modeltype_ix + ' @ ' + str(months_to) + 'months')
            model_list.append(model)
            prec_total.append(prec_list)
            recall_total.append(recall_list)
            spec_total.append(spec_list)
            auc_list.append([test_auc_mean, test_auc_mean_ste])
            r2_list.append([r2test_mean, r2test_ste])

            # 'Vital: Wt-latest' only model
            print('\n' + gender_clean + ' Vital: Wt-latest - ' + modeltype_ix + ' @ ' + str(months_to) + ' months')
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

            # np.savez_compressed(newdir+'/train_data_'+'_'.join([modeltype_ix,gender_clean,'Vital_Wt_latest',str(months_to),'months']), x=xtrain, mrns=mrnstrain, features=np.array(feature_headers2), y=ytrain, ylabel=ytrainlabel)
            # np.savez_compressed(newdir+'/test_data_'+'_'.join([modeltype_ix,gender_clean,'Vital_Wt_latest',str(months_to),'months']), x=xtest, mrns=mrnstest, features=np.array(feature_headers2), y=ytest, ylabel=ytestlabel)
            np.savez_compressed(newdir+'/data_transformed_'+'_'.join([modeltype_ix,gender_clean,'Vital_Wt_latest',str(months_to),'months']), x=x2, mrns=mrns2, features=np.array(feature_headers2), y=y2, ylabel=y2label, ix2=ix_filter, modeling_ix=randix, train_ix=ix_train, test_ix=ix_test)

            titles_total.append(gender_clean + ' Vital: Wt-latest - ' + modeltype_ix + ' - ' + ' @ ' + str(months_to) + 'months')
            model_list.append(model)
            prec_total.append(prec_list)
            recall_total.append(recall_list)
            spec_total.append(spec_list)
            auc_list.append([test_auc_mean, test_auc_mean_ste])
            r2_list.append([r2test_mean, r2test_ste])


            # Save the outputs -- overwrite at each step in case of crash
            pickle.dump(titles_total, open(newdir+'/titles_total_'+str(months_to)+'_months.pkl', 'wb'))
            pickle.dump(model_list, open(newdir+'/model_list_'+str(months_to)+'_months.pkl', 'wb'))
            pickle.dump(prec_total, open(newdir+'/prec_total_'+str(months_to)+'_months.pkl', 'wb'))
            pickle.dump(recall_total, open(newdir+'/recall_total_'+str(months_to)+'_months.pkl', 'wb'))
            pickle.dump(spec_total, open(newdir+'/spec_total_'+str(months_to)+'_months.pkl', 'wb'))
            pickle.dump(auc_list, open(newdir+'/auc_list_'+str(months_to)+'_months.pkl', 'wb'))
            pickle.dump(r2_list, open(newdir+'/r2_list_'+str(months_to)+'_months.pkl', 'wb'))

# get the training metric plots
# title1='Precision vs. Recall Curves: Obesity at 5 years from 24 months'
# title2='ROC Curve: Obesity at 5 years from 24 months'
# train.prec_rec_curve(recall_total, prec_total, titles_total, title1, show=False, save=True)
# train.ROC_curve(recall_total, spec_total, titles_total, title2, show=False, save=True)
