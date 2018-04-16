import pickle
import train
import build_features
import numpy as np
import time
import os

d1 = pickle.load(open('../python objects/patientdata_20170823.pkl', 'rb'))
d1mom = pickle.load(open('../python objects/patient_mother_data_20170724.pkl', 'rb'))
lat_lon_dic = pickle.load(open('../python objects/lat_lon_data_20180305.pkl', 'rb'))
env_dic= pickle.load(open('../python objects/census_data_20170329.pkl', 'rb'))
d1mom_hist = pickle.load(open('../python objects/full_lutheran_mother_data.pkl', 'rb'))

timestr = time.strftime("%Y%m%d")
newdir='../outputs_age_analyses'+timestr
os.mkdir(newdir)




no_wt_bmi = [fh for fh in feature_headers if not any([x in fh for x in ('Wt','BMI')])]
no_census = [fh for fh in feature_headers if not any([x in fh.lower() for x in ('census','zipcode')])]

data = {'boys':{
    'no_min':{
        'full':{'data':[], 'variablesubset':[], 'filt_str':['Gender:0']},
        'no_census':{'data':[], 'variablesubset':no_census, 'filt_str':['Gender:0']},
        'no_wt_bmi':{'data':[], 'variablesubset':no_wt_bmi, 'filt_str':['Gender:0']},
        'wfl_latest':{'data':[], 'variablesubset':no_wt_bmi, 'filt_str':['Gender:0']},
        'wfl_19_24':{'data':[], 'variablesubset':[], 'filt_str':['Gender:0','Vital: Wt for Length ZScore-avg19to24']},
        'wfl_latest':{'data':[], 'variablesubset':[], 'filt_str':['Gender:0', 'Vital: Wt for Length ZScore-avg19to24']},
        'bmi_19_24':{'data':[], 'variablesubset':[], 'filt_str':['Gender:0','Vital: BMI-avg19to24']},
        'bmi_latest':{'data':[], 'variablesubset':[], 'filt_str':['Gender:0', 'Vital: BMI-latest']}
        },
    'min_5':{
        'full':{'data':[], 'variablesubset':[], 'filt_str':['Gender:0']},
        'no_census':{'data':[], 'variablesubset':no_census, 'filt_str':['Gender:0']},
        'no_wt_bmi':{'data':[], 'variablesubset':no_wt_bmi, 'filt_str':['Gender:0']}
        },
    'lasso_fts':{
        'full':{'data':[], 'variablesubset':[]}, 'filt_str':['Gender:0'],
        'no_census':{'data':[], 'variablesubset':no_census, 'filt_str':['Gender:0']},
        'no_wt_bmi':{'data':[], 'variablesubset':no_wt_bmi, 'filt_str':['Gender:0']}
        },
    },
    'girls':{
        'no_min':{
            'full':{'data':[], 'variablesubset':[], 'filt_str':['Gender:1']},
            'no_census':{'data':[], 'variablesubset':no_census, 'filt_str':['Gender:1']},
            'no_wt_bmi':{'data':[], 'variablesubset':no_wt_bmi, 'filt_str':['Gender:1']},
            'wfl_19_24':{'data':[], 'variablesubset':[], 'filt_str':['Gender:1','Vital: Wt for Length ZScore-avg19to24']},
            'wfl_latest':{'data':[], 'variablesubset':[], 'filt_str':['Gender:1', 'Vital: Wt for Length ZScore-avg19to24']},
            'bmi_19_24':{'data':[], 'variablesubset':[], 'filt_str':['Gender:1','Vital: BMI-avg19to24']},
            'bmi_latest':{'data':[], 'variablesubset':[], 'filt_str':['Gender:1', 'Vital: BMI-latest']}
            },
        'min_5':{
            'full':{'data':[], 'variablesubset':[], 'filt_str':['Gender:1'],
            'no_census':{'data':[], 'variablesubset':no_census, 'filt_str':['Gender:1']},
            'no_wt_bmi':{'data':[], 'variablesubset':no_wt_bmi, 'filt_str':['Gender:1']}
            },
        'lasso_fts':{
            'full':{'data':[], 'variablesubset':[], 'filt_str':['Gender:1']},
            'no_census':{'data':[], 'variablesubset':no_census, 'filt_str':['Gender:1']},
            'no_wt_bmi':{'data':[], 'variablesubset':no_wt_bmi, 'filt_str':['Gender:1']}
            }
        }
    }
}

for gender in [*data]:
    for subset in [*data[gender]]:
        min_occ = 5 if subset == 'min_5' else 0
        lasso_sel = True if subset == 'lasso_fts' else False
        for filt in [*data[gender][subset]]:
            x2, y2, y2label, mrns2, ix_filter, feature_headers2, corr_headers_filtered, corrs_matrix_filterd, ix_corr_headers = \
                train.prepare_data_for_analysis({}, {}, {}, {}, {},
                    x1, y1, y1label[:,label_ix['obese']], feature_headers, mrns,
                    agex_low, agex_high, months_from, months_to,
                    filterSTR=data[gender][subset][filt],
                    variablesubset=data[gender][subset][filt]['variablesubset'],
                    do_impute=False,
                    do_normalize=True,
                    min_occur=min_occ,
                    feature_info=False,
                    delay_print=False,
                    lasso_selection=lasso_sel)
        corr_headers_filtered = np.array(corr_headers_filtered) if type(corr_headers_filtered) == list else corr_headers_filtered
        ix_corr_headers = np.array(ix_corr_headers) if type(ix_corr_headers) == list else ix_corr_headers
        data[gender][subset][filt][data] = [x2, y2, y2label, mrns2, ix_filter, feature_headers2, corr_headers_filtered, corrs_matrix_filterd, ix_corr_headers]
        fname=newdir+'/'+'_'.join(['x2',gender,months_to,'months_obese',subset,filt])
        np.savez_compressed(fname, x2=x2, mrns2=mrns2, features2=np.array(feature_headers2), y2=y2, y2label=y2label, corr_mat=corrs_matrix_filterd, ix_corr_headers=ix_corr_headers, corr_headers_filtered=corr_headers_filtered)
        print('data saved to',fname)



title_list = []
auc_list = []
r2_list = []
exp_var_list = []
prec_list = []
results_list = []
features_list = []
for gender in [*data]:
    for subset in [*data[gender]]:
        for filt in [*data[gender][subset]]:
            x2, y2, y2label, mrns2, ix_filter, feature_headers2, corr_headers_filtered, corrs_matrix_filterd, ix_corr_headers = data[gender][subset][filt]['data']
            for model_type in ('lasso','randomforest','gradientboost'):
                print('Training',model_type,'model on',gender,subset,filt)
                title = ' '.join([gender.title(),'-',model_type.title(),subset,filt,'@'+str(months_to), 'obese'])
                (model_list, randix_track, ix_train_track, ix_val_track, test_ix, results_arr, results_cols,
                 feature_data, feature_data_cols, auc_val_mean, auc_val_mean_ste, var_val_mean, var_val_mean_ste, r2val_mean, r2val_mean_ste,
                 auc_test_mean, auc_test_mean_ste, var_test_mean, var_test_mean_ste, r2test_mean, r2test_ste) = \
                        train.train_regression_model_for_bmi_parallel(x2, y2_no, y2label, feature_headers2, mrns2,
                            corr_headers_filtered, corr_matrix_filtered, ix_corr_headers,
                            modelType=model_type,
                            percentile=False,
                            return_data_for_error_analysis=False,
                            feature_info=True)

                title_list.append(title)
                auc_list.append([auc_val_mean, auc_val_mean_ste, auc_test_mean, auc_test_mean_ste])
                r2_list.append([r2val_mean, r2val_mean_ste, r2test_mean, r2test_mean_ste])
                exp_var_list.append([exp_var_val_mean, exp_var_val_mean_ste, exp_var_test_mean, exp_var_test_mean_ste])
                results_list.append(results_arr)
                features_list.append(feature_data)
                fname = newdir+'_'.join(['/'+gender,str(months_to),'months_obese',model_type,'index',subset,filt])
                np.savez_compressed(fname, cv_ix=randix_track, train_ix=ix_train_track, val_ix=ix_val_track, test_ix=test_ix)
                print('Train/Validation/Test indices saved to:',fname)
                fname = newdir+'_'.join(['/'+gender,str(months_to),'months_obese',model_type,'results',subset,filt])
                np.savez_compressed(fname, results=results_arr, results_cols=results_cols, features=feature_data, feature_cols=feature_data_cols)
                print('Train/Validation/Test indices saved to:',fname)
                fname = newdir+'_'.join(['/'+gender,str(months_to),'months_obese',model_type,'models',subset,filt])
                pickle.dump(model_list, open(fname, 'wb'))
                print('Train/Validation/Test indices saved to:',fname)



pickle.dump(titles_list, open(newdir+'/titles_list_'+str(months_to)+'_months_obese.pkl', 'wb'))
pickle.dump(prec_list, open(newdir+'/prec_total_'+str(months_to)+'_months_obese.pkl', 'wb'))
pickle.dump(recall_total, open(newdir+'/recall_total_'+str(months_to)+'_months_obese.pkl', 'wb'))
pickle.dump(spec_total, open(newdir+'/spec_total_'+str(months_to)+'_months_obese.pkl', 'wb'))
pickle.dump(auc_list, open(newdir+'/auc_list_'+str(months_to)+'_months_obese.pkl', 'wb'))
pickle.dump(r2_list, open(newdir+'/r2_list_'+str(months_to)+'_months_obese.pkl', 'wb'))
pickle.dump(results_list, open(newdir+'/results_list'+str(months_to)+'_months_obese.pkl', 'wb'))
pickle.dump(features_list, open(newdir+'/features_list'+str(months_to)+'_months_obese.pkl', 'wb'))
