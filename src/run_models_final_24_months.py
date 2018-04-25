import train
import build_features

import os
import time
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

# Get the data
d1 = pickle.load(open('../python objects/patientdata_20170823.pkl', 'rb'))
d1mom = pickle.load(open('../python objects/patient_mother_data_20170724.pkl', 'rb'))
lat_lon_dic = pickle.load(open('../python objects/lat_lon_data_20180329.pkl', 'rb'))
env_dic= pickle.load(open('../python objects/census_data_20170920.pkl', 'rb'))
d1mom_hist = pickle.load(open('../python objects/full_lutheran_mother_data.pkl', 'rb'))

timestr = time.strftime("%Y%m%d")
newdir='../outputs_24_months_final_'+timestr
os.mkdir(newdir)

agex_low = 4.5
agex_high = 5.5
months_from = 0
months_to = 24
no_wt_bmi = ['Census', 'Lab','Number of Visits','MatDeliveryAge','Maternal','Diagnosis','Newborn Diagnosis']+[fh for fh in feature_headers if fh.startswith('Vital') and not any([x in fh for x in ('Wt','BMI')])]
no_census = ['Vital','Lab','Number of Visits','MatDeliveryAge','Maternal','Diagnosis','Newborn Diagnosis']
label_ix = {'underweight':0,'normal':1,'overweight':2,'obese':3,'class I severe obesity':4,'class II severe obesity':5}

# Create the ML-ready data and save
x1,y1,y1label,feature_headers,mrns = build_features.call_build_function(d1, d1mom, d1mom_hist, lat_lon_dic, env_dic, agex_low, agex_high, months_from, months_to, False, prediction='multi')
np.savez_compressed(newdir+'/raw_matrix_data_'+str(months_to)+'months', x=x1, y=y1, ylabel=y1label, mrns=mrns, features=np.array(feature_headers))

# Create a data dictionary for all of our data permutations for training models
data = {
    'boys':{
        'no_min':{
            'full':{'data':[], 'variablesubset':[], 'filt_str':['Gender:0']},
            'no_census':{'data':[], 'variablesubset':no_census, 'filt_str':['Gender:0']},
            'no_wt_bmi':{'data':[], 'variablesubset':no_wt_bmi, 'filt_str':['Gender:0']},
            'wfl_latest':{'data':[], 'variablesubset':no_wt_bmi, 'filt_str':['Gender:0']},
            'wfl_19_24':{'data':[], 'variablesubset':['Vital: Wt for Length ZScore-avg19to24'], 'filt_str':['Gender:0']},
            'wfl_latest':{'data':[], 'variablesubset':['Vital: Wt for Length ZScore-avg19to24'], 'filt_str':['Gender:0']},
            'bmi_19_24':{'data':[], 'variablesubset':['Vital: BMI-avg19to24'], 'filt_str':['Gender:0']},
            'bmi_latest':{'data':[], 'variablesubset':['Vital: BMI-latest'], 'filt_str':['Gender:0']}
            },
        'min_5':{
            'full':{'data':[], 'variablesubset':[], 'filt_str':['Gender:0']},
            'no_census':{'data':[], 'variablesubset':no_census, 'filt_str':['Gender:0']},
            'no_wt_bmi':{'data':[], 'variablesubset':no_wt_bmi, 'filt_str':['Gender:0']}
            },
        'lasso_fts':{
            'full':{'data':[], 'variablesubset':[], 'filt_str':['Gender:0']},
            'no_census':{'data':[], 'variablesubset':no_census, 'filt_str':['Gender:0']},
            'no_wt_bmi':{'data':[], 'variablesubset':no_wt_bmi, 'filt_str':['Gender:0']}
            },
    },
    'girls':{
        'no_min':{
            'full':{'data':[], 'variablesubset':[], 'filt_str':['Gender:1']},
            'no_census':{'data':[], 'variablesubset':no_census, 'filt_str':['Gender:1']},
            'no_wt_bmi':{'data':[], 'variablesubset':no_wt_bmi, 'filt_str':['Gender:1']},
            'wfl_19_24':{'data':[], 'variablesubset':['Vital: Wt for Length ZScore-avg19to24'], 'filt_str':['Gender:1']},
            'wfl_latest':{'data':[], 'variablesubset':['Vital: Wt for Length ZScore-avg19to24'], 'filt_str':['Gender:1']},
            'bmi_19_24':{'data':[], 'variablesubset':['Vital: BMI-avg19to24'], 'filt_str':['Gender:1']},
            'bmi_latest':{'data':[], 'variablesubset':['Vital: BMI-latest'], 'filt_str':['Gender:1']}
            },
        'min_5':{
            'full':{'data':[], 'variablesubset':[], 'filt_str':['Gender:1']},
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

# Create all of the data permutations for training
start = datetime.now() 
for gender in [*data]:
    for subset in [*data[gender]]:
        min_occ = 5 if subset == 'min_5' else 0
        lasso_sel = True if subset == 'lasso_fts' else False
        for filt in [*data[gender][subset]]:
            print(gender, subset, filt)
            x2, y2, y2label, mrns2, ix_filter, feature_headers2, corr_headers_filtered, corrs_matrix_filtered, ix_corr_headers = \
                train.prepare_data_for_analysis({}, {}, {}, {}, {},
                    x1, y1, y1label[:,label_ix['obese']], feature_headers, mrns,
                    agex_low, agex_high, months_from, months_to,
                    filterSTR=data[gender][subset][filt]['filt_str'],
                    variablesubset=data[gender][subset][filt]['variablesubset'],
                    do_impute=False,
                    do_normalize=True,
                    min_occur=min_occ,
                    feature_info=False,
                    delay_print=False,
                    lasso_selection=lasso_sel)
            corr_headers_filtered = np.array(corr_headers_filtered) if type(corr_headers_filtered) == list else corr_headers_filtered
            ix_corr_headers = np.array(ix_corr_headers) if type(ix_corr_headers) == list else ix_corr_headers
            data[gender][subset][filt]['data'] = [x2, y2, y2label, mrns2, ix_filter, feature_headers2, corr_headers_filtered, corrs_matrix_filtered, ix_corr_headers]
            fname=newdir+'/'+'_'.join(['x2',str(gender),str(months_to),'months_obese',str(subset),str(filt)])
            np.savez_compressed(fname, x2=x2, mrns2=mrns2, features2=np.array(feature_headers2), y2=y2, y2label=y2label, corr_mat=corrs_matrix_filtered, ix_corr_headers=ix_corr_headers, corr_headers_filtered=corr_headers_filtered)
            print('data saved to',fname)
            print('Time elapsed {}'.format(datetime.now() - start))
            print('-----------------------------------------------------\n')

# Output some summary information on the breakdown of the data distributions
for gender in [*data]:
    for subset in [*data[gender]]:
        for filt in [*data[gender][subset]]:
            x2, y2, y2label, mrns2, ix_filter, feature_headers2, corr_headers_filtered, corr_matrix_filtered, ix_corr_headers = data[gender][subset][filt]['data']
            print(gender, subset, filt, 'children:',x2.shape[0], 'features:',x2.shape[1],'pos:', y2label.sum())


# Boys test set creation
x2, y2, y2label, mrns2, ix_filter, feature_headers2, corr_headers_filtered, corr_matrix_filtered, ix_corr_headers = data['boys']['no_min']['full']['data']
N = x2.shape[0]
test_ix_boys = list(range(0,N))
random.shuffle(test_ix_boys)
test_ix_boys = test_ix_boys[:int(N*0.2)]

# Girls test set creation
x2, y2, y2label, mrns2, ix_filter, feature_headers2, corr_headers_filtered, corr_matrix_filtered, ix_corr_headers = data['girls']['no_min']['full']['data']
N = x2.shape[0]
test_ix_girls = list(range(0,N))
random.shuffle(test_ix_girls)
test_ix_girls = test_ix_girls[:int(N*0.2)]

# Run all the models
title_list = []
auc_list = []
r2_list = []
exp_var_list = []
prec_list = []
results_list = []
features_list = []
start = datetime.now()
for gender in [*data]:
    test_ix = test_ix_boys if gender == 'boys' else test_ix_girls
    for subset in [*data[gender]]:
        for filt in [*data[gender][subset]]:
            x2, y2, y2label, mrns2, ix_filter, feature_headers2, corr_headers_filtered, corr_matrix_filtered, ix_corr_headers = data[gender][subset][filt]['data']
            for model_type in ('lasso','randomforest','gradientboost'):
                analysis_start = datetime.now()
                print(gender, subset, filt, model_type)
                title = ' '.join([gender.title(),'-',model_type.title(),subset,filt,'@'+str(months_to), 'obese'])
                (model_list, randix_track, ix_train_track, ix_val_track, test_ix, results_arr, results_cols,
                 feature_data, feature_data_cols, auc_val_mean, auc_val_mean_ste, var_val_mean, var_val_mean_ste, r2val_mean, r2val_mean_ste,
                 auc_test_mean, auc_test_mean_ste, var_test_mean, var_test_mean_ste, r2test_mean, r2test_mean_ste) = \
                        train.train_regression_model_for_bmi_parallel(x2, y2, y2label, feature_headers2, mrns2,
                            corr_headers_filtered, corr_matrix_filtered, ix_corr_headers,
                            modelType=model_type,
                            test_ix=test_ix,
                            percentile=False,
                            return_data_for_error_analysis=False,
                            feature_info=True)

                title_list.append(title)
                auc_list.append([auc_val_mean, auc_val_mean_ste, auc_test_mean, auc_test_mean_ste])
                r2_list.append([r2val_mean, r2val_mean_ste, r2test_mean, r2test_mean_ste])
                exp_var_list.append([var_val_mean, var_val_mean_ste, var_test_mean, var_test_mean_ste])
                results_list.append(results_arr)
                features_list.append(feature_data)
                fname = newdir+'_'.join(['/'+gender,str(months_to),'months_obese',model_type,'index',subset,filt])
                fname = newdir+'_'.join(['/'+str(gender),str(months_to),'months_obese',str(model_type),'index',str(subset),str(filt)])
                np.savez_compressed(fname, cv_ix=randix_track, train_ix=ix_train_track, val_ix=ix_val_track, test_ix=test_ix)
                print('Train/Validation/Test indices saved to:',fname)
                fname = newdir+'_'.join(['/'+str(gender),str(months_to),'months_obese',str(model_type),'results',str(subset),str(filt)])
                np.savez_compressed(fname, results=results_arr, results_cols=results_cols, features=feature_data, feature_cols=feature_data_cols)
                print('Results saved to:',fname)
                fname = newdir+'_'.join(['/'+str(gender),str(months_to),'months_obese',str(model_type),'models',str(subset),str(filt)])+'.pkl'
                pickle.dump(model_list, open(fname, 'wb'))
                print('Models saved to:',fname)
                end_time = datetime.now()
                print('Time to run analysis {}'.format(end_time - analysis_start))
                print('Total Time elapsed {}'.format(end_time - start))
                print('-----------------------------------------------------\n')

# Save all the high-level results information for later use
pickle.dump(title_list, open(newdir+'/titles_list_'+str(months_to)+'_months_obese.pkl', 'wb'))
pickle.dump(auc_list, open(newdir+'/auc_list_'+str(months_to)+'_months_obese.pkl', 'wb'))
pickle.dump(r2_list, open(newdir+'/r2_list_'+str(months_to)+'_months_obese.pkl', 'wb'))
pickle.dump(exp_var_list, open(newdir+'/exp_var_list_'+str(months_to)+'_months_obese.pkl', 'wb'))
pickle.dump(results_list, open(newdir+'/results_list_'+str(months_to)+'_months_obese.pkl', 'wb'))
pickle.dump(features_list, open(newdir+'/features_list_'+str(months_to)+'_months_obese.pkl', 'wb'))

# Create an array of results for observing model performanace and save for later use
df = pd.DataFrame(np.hstack((np.array(title_list).reshape(-1,1),np.array(auc_list),np.array(r2_list),np.array(exp_var_list))), 
    columns=['title','AUC Validation Mean','AUC Validation Mean STE', 'AUC Test Mean', 'AUC Test STE',
      'R^2 Validation Mean','R^2 Validation Mean STE', 'R^2 Test Mean', 'R^2 Test STE',
      'Explained Variance Validation Mean','Explained Variance Validation Mean STE', 'Explained Variance Test Mean', 'Explained Variance Test STE'
    ])
for col in df.columns:
    if col != 'title':
        df[col] = df[col].astype(float)
df.to_csv(newdir+'/val_test_results_matrix_24_months.csv',index=False)

# Plot the ROC Curves for all the multi-feature models
fig = plt.figure(figsize=(16,24))
filters = [(g,f) for f in ['no_min','min_5','lasso_fts'] for g in ['Boys','Girls']]
style = {'Lasso':'-', 'Randomforest':'--', 'Gradientboost':':'}
for i, f in enumerate(filters):
    ax = plt.subplot(321+i)
    for ix, t in enumerate(title_list):
        if all(ff in t for ff in f) and all(ff not in t for ff in ('bmi_','wfl')):
            ls = t.split(' - ')[1].split(' ')[0]
            plt.plot(1 - results_list[ix][:,7], results_list[ix][:,6], linestyle=style[ls], label=t[:-9]+' AUC val: {0:4.3f}, AUC test: {1:4.3f}'.format(auc_list[ix][0], auc_list[ix][2]))
    
    plt.xlim((0,1))
    plt.ylim((0,1))
    lims = [np.min([plt.xlim(), plt.ylim()]), np.max([plt.xlim(), plt.ylim()])]
    plt.plot(lims, lims, 'k-', alpha=0.6, zorder=0, label='Random Decision')
    plt.legend(fontsize=10)
    plt.title(f[0]+' '+f[1]+' ROC Curve', fontsize=18)
    if i%2 == 0:
        plt.ylabel('Sensitivity (TPR)', fontsize=16)
    if i in range(len(filters)-2,len(filters)):
        plt.xlabel('1 - Specificity (FPR)', fontsize=16)
    plt.grid(True)
    plt.tight_layout()

fig.savefig(newdir+'/all_final_multi_feature_AUC_ROC_comparison.png', bbox_inches='tight', dpi=360)
plt.clf()
plt.close()

# Plot the ROC Curves for the single feature models
fig = plt.figure(figsize=(16,16))
filters = [(g,f) for f in ['bmi_','wfl'] for g in ['Boys','Girls']]
style = {'Lasso':'-', 'Randomforest':'--', 'Gradientboost':':'}
for i, f in enumerate(filters):
    ax = plt.subplot(221+i)
    for ix, t in enumerate(title_list):
        if all(ff in t for ff in f):
            ls = t.split(' - ')[1].split(' ')[0]
            plt.plot(1 - results_list[ix][:,7], results_list[ix][:,6], linestyle=style[ls], label=t[:-9]+' AUC val: {0:4.3f}, AUC test: {1:4.3f}'.format(auc_list[ix][0], auc_list[ix][2]))
    
    plt.xlim((0,1))
    plt.ylim((0,1))
    lims = [np.min([plt.xlim(), plt.ylim()]), np.max([plt.xlim(), plt.ylim()])]
    plt.plot(lims, lims, 'k-', alpha=0.6, zorder=0, label='Random Decision')
    plt.legend(fontsize=10)
    plt.title(f[0]+' '+f[1].strip('_')+' ROC Curve', fontsize=18)
    if i%2 == 0:
        plt.ylabel('Sensitivity (TPR)', fontsize=16)
    if i in range(len(filters)-2,len(filters)):
        plt.xlabel('1 - Specificity (FPR)', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    
fig.savefig(newdir+'/all_final_single_feature_AUC_ROC_comparison.png', bbox_inches='tight', dpi=360)
plt.clf()
plt.close()

# Plot the Precision Recall Curves for all the multi-feature models
fig = plt.figure(figsize=(16,24))
filters = [(g,f) for f in ['no_min','min_5','lasso_fts'] for g in ['Boys','Girls']]
style = {'Lasso':'-', 'Randomforest':'--', 'Gradientboost':':'}
for i, f in enumerate(filters):
    ax = plt.subplot(321+i)
    for ix, t in enumerate(title_list):
        if all(ff in t for ff in f) and all(ff not in t for ff in ('bmi_','wfl')):
            ls = t.split(' - ')[1].split(' ')[0]
            rec = results_list[ix][:,6]
            ppv = results_list[ix][:,5]
            index = 1 + np.max(list(set([ix for ix,x in enumerate(rec) if x == 0]) & set([ix for ix,x in enumerate(ppv) if x == 0])))
            plt.plot(rec[index:], ppv[index:], linestyle=style[ls], label=t[:-9]+' AUC val: {0:4.3f}, AUC test: {1:4.3f}'.format(auc_list[ix][0], auc_list[ix][2]))
    
    plt.legend(fontsize=10)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title(f[0]+' '+f[1]+' Precision Recall Curve', fontsize=18)
    if i == 0:
        plt.ylabel('Precision (PPV)', fontsize=14)
    plt.xlabel('Recall (Sensitivity)', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
fig.savefig(newdir+'/all_final_multi_feature_prec_rec_comparison.png', bbox_inches='tight', dpi=360)
plt.clf()
plt.close()

# Plot the Precision Recall Curves for all the single feature models
fig = plt.figure(figsize=(16,24))
filters = [(g,f) for f in ['bmi_','wfl'] for g in ['Boys','Girls']]
style = {'Lasso':'-', 'Randomforest':'--', 'Gradientboost':':'}
for i, f in enumerate(filters):
    ax = plt.subplot(321+i)
    for ix, t in enumerate(title_list):
        if all(ff in t for ff in f):
            ls = t.split(' - ')[1].split(' ')[0]
            rec = results_list[ix][:,6]
            ppv = results_list[ix][:,5]
            index = 1 + np.max(list(set([ix for ix,x in enumerate(rec) if x == 0]) & set([ix for ix,x in enumerate(ppv) if x == 0])))
            plt.plot(rec[index:], ppv[index:], linestyle=style[ls], label=t[:-9]+' AUC val: {0:4.3f}, AUC test: {1:4.3f}'.format(auc_list[ix][0], auc_list[ix][2]))
    
    plt.legend(fontsize=10)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title(f[0]+' '+f[1].strip('_')+' Precision Recall Curve', fontsize=18)
    if i == 0:
        plt.ylabel('Precision (PPV)', fontsize=14)
    plt.xlabel('Recall (Sensitivity)', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
fig.savefig(newdir+'/all_final_single_feature_prec_rec_comparison.png', bbox_inches='tight', dpi=360)
plt.clf()
plt.close()

# Plot the ROC Curves for the top performing multi-feature model compared to the single feature model
fig = plt.figure(figsize=(12,12))
# top_ix format: ['gender','age','index',title, auc, auc_low, auc_high,specificity,precision,recall]
for i,g in enumerate(['boys','girls']):
    plot_ix = [df[df.title.str.contains(g.title()) & ~df.title.str.contains('wfl') & ~df.title.str.contains('bmi_')].sort_values(by='AUC Test Mean', ascending=False).index[0],
        df[df.title.str.contains(g.title()) & (df.title.str.contains('wfl_19_24'))].sort_values(by='AUC Test Mean', ascending=False).index[0],
        df[df.title.str.contains(g.title()) & (df.title.str.contains('wfl_latest'))].sort_values(by='AUC Test Mean', ascending=False).index[0],
        df[df.title.str.contains(g.title()) & (df.title.str.contains('bmi_19_24'))].sort_values(by='AUC Test Mean', ascending=False).index[0],
        df[df.title.str.contains(g.title()) & (df.title.str.contains('bmi_latest'))].sort_values(by='AUC Test Mean', ascending=False).index[0]
        ]
    ax = plt.subplot(221+i)
    title = g.title() + ' ROC Curve'
    plt.title(title, fontsize=14)
    for ix in plot_ix:
        w = 1.5
        s = title_list[ix].split(' - ')[1]
        end = s.find('@')
        lab = s[:end-1] + ', AUC val: {0:0.3f}, AUC test: {1:0.3f}'.format(auc_list[ix][0], auc_list[ix][2])
        plt.plot(1 - results_list[ix][:,7], results_list[ix][:,6], linewidth=w, label=lab)
    plt.grid(True)
    plt.xlim((0,1))
    plt.ylim((0,1))
    lims = [np.min([plt.xlim(), plt.ylim()]), np.max([plt.xlim(), plt.ylim()])]
    plt.plot(lims, lims, 'k-', alpha=0.6, zorder=0, label='Random Decision')
    plt.legend(fontsize=10)
    if i == 0:
        plt.ylabel('Sensitivity (TPR)', fontsize=12)
    plt.xlabel('1 - Specificity (FPR)', fontsize=12)
    plt.tight_layout()
            
fig.savefig(newdir+'/best_models_AUC_ROC_comparison.png', bbox_inches='tight', dpi=360)
plt.clf()
plt.close()

# Plot the Precision Recall Curves for the best multi-feature model and the single feature models
fig = plt.figure(figsize=(12,12))
# top_ix format: ['gender','age','index',title, auc, auc_low, auc_high,specificity,precision,recall]
for i,g in enumerate(['boys','girls']):
    plot_ix = [df[df.title.str.contains(g.title()) & ~df.title.str.contains('wfl') & ~df.title.str.contains('bmi_')].sort_values(by='AUC Test Mean', ascending=False).index[0],
        df[df.title.str.contains(g.title()) & (df.title.str.contains('wfl_19_24'))].sort_values(by='AUC Test Mean', ascending=False).index[0],
        df[df.title.str.contains(g.title()) & (df.title.str.contains('wfl_latest'))].sort_values(by='AUC Test Mean', ascending=False).index[0],
        df[df.title.str.contains(g.title()) & (df.title.str.contains('bmi_19_24'))].sort_values(by='AUC Test Mean', ascending=False).index[0],
        df[df.title.str.contains(g.title()) & (df.title.str.contains('bmi_latest'))].sort_values(by='AUC Test Mean', ascending=False).index[0]
        ]
    ax = plt.subplot(221+i)
    title = g.title() + ' Precision Recall Curve'
    plt.title(title, fontsize=14)
    for ix in plot_ix:
        w = 1.5
        s = title_list[ix].split(' - ')[1]
        end = s.find('@')
        lab = s[:end-1] + ', AUC val: {0:0.3f}, AUC test: {1:0.3f}'.format(auc_list[ix][0], auc_list[ix][2])
        rec = results_list[ix][:,6]
        ppv = results_list[ix][:,5]
        index = 1 + np.max(list(set([ix for ix,x in enumerate(rec) if x == 0]) & set([ix for ix,x in enumerate(ppv) if x == 0])))
        plt.plot(rec[index:], ppv[index:], linewidth=w, label=lab)
    plt.grid(True)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.legend(fontsize=9, loc='lower left')
    if i == 0:
        plt.ylabel('Precision (PPV)', fontsize=12)
    plt.xlabel('Recall (TPR)', fontsize=12)
    plt.tight_layout()

fig.savefig(newdir+'/best_models_prec_rec_comparison.png', bbox_inches='tight', dpi=360)
plt.clf()
plt.close()