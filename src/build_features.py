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
from scipy import stats


def build_features_icd(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    for diag in patient_data['diags']:
        # print(diag , diag.replace('.','').strip(), feature_index[diag.replace('.','').strip()])
        for edatel in patient_data['diags'][diag]:
            edate = edatel[0]
            if edate >= reference_date_end or edate <= reference_date_start:
                continue
            try:
                res[feature_index[diag.replace('.','').strip()]] = True
            except KeyError:
                try:
                    res[feature_index[diag.replace('.','').strip()[0:-2]]] = True
                except KeyError:
                    pass #print('--->',diag.replace('.','').strip()[0:-1])    
            break
    return res

def build_features_lab(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=float)
    for key1 in patient_data['labs']:
        for edatel in patient_data['labs'][key1]:
            edate = edatel[0]
            if edate >= reference_date_end or edate <= reference_date_start:
                continue
            try:
                res[feature_index[key1.strip()]] = edatel[1]
            except KeyError:
                pass # print('key error lab:', key1) 
            break
    return res

def build_features_med(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    for key1 in patient_data['meds']:
        for edatel in patient_data['meds'][key1]:
            edate = edatel[0]
            if edate >= reference_date_end or edate <= reference_date_start:
                continue
            try:
                res[feature_index[key1.strip()]] = True
            except KeyError:
                pass # print ('key error', key1.strip())
            break
    return res

def build_features_gen(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    code = patient_data['gender']
    res[feature_index[int(code)]] = True
    return res

def build_features_vitalLatest(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=float)
    bdate = patient_data['bdate']
    for code in patient_data['vitals']:
        for (edate, vitalval) in patient_data['vitals'][code]:
            if edate >= reference_date_end or edate <= reference_date_start:
                continue
            try:
                res[feature_index[code.strip()]] = vitalval
            except:
                pass
    return res

def build_features_vitalAverage_0_3(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers, 0, 3)

def build_features_vitalAverage_3_6(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers, 3, 6)

def build_features_vitalAverage_6_9(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers, 6, 9)

def build_features_vitalAverage_9_12(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers, 9, 12)

def build_features_vitalAverage_12_15(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers, 12, 15)

def build_features_vitalAverage_15_18(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers, 15, 18)

def build_features_vitalAverage_18_21(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers, 18, 21)

def build_features_vitalAverage_21_24(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    return build_features_vitalAverage(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers, 21, 24)

def build_features_vitalAverage(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers, frommonth, tomonth):
    res = np.zeros(len(feature_headers), dtype=float)
    res_cnt = np.zeros(len(feature_headers), dtype=float)
    bdate = patient_data['bdate']
    for code in patient_data['vitals']:
        for (edate, vitalval) in patient_data['vitals'][code]:
            if edate >= reference_date_end or edate <= reference_date_start:
                continue
            try:
                age_at_vital = (edate - bdate).days / 30
                if (age_at_vital < frommonth) or (age_at_vital > tomonth) :
                    continue
                res[feature_index[code.strip()]] += vitalval
                res_cnt[feature_index[code.strip()]] += 1
                # print(code, age_at_vital, vitalval)
            except:
                pass
    res_cnt[(res_cnt==0)] = 1.0 
    res = res/res_cnt
    return res

def build_features_ethn(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    code = patient_data['ethnicity']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res

def build_features_mat_insurance1(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'insur1' not in maternal_data:
        # print (maternal_data)
        return res
    code = maternal_data['insur1']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res
def build_features_mat_insurance2(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'insur2' not in maternal_data:
        # print (maternal_data)
        return res
    code = maternal_data['insur2']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res

def build_features_race(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    code = patient_data['race']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res
def build_features_zipcd(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'zip' in patient_data:
        code = patient_data['zip'][0][1]
        if code in feature_index and pd.notnull(code):
            res[feature_index[code]] = True
    return res

def build_features_mat_icd(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'diags' not in maternal_data:
        return res
    for diag in maternal_data['diags']:
        # print(diag , diag.replace('.','').strip(), feature_index[diag.replace('.','').strip()])
        try:
            res[feature_index[diag.replace('.','').strip()]] = True
        except KeyError:
            try:
                res[feature_index[diag.replace('.','').strip()[0:-2]]] = True
            except KeyError:
                pass #print('--->',diag.replace('.','').strip()[0:-1])
    return res
def build_features_nb_icd(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'nbdiags' not in maternal_data:
        return res
    for diag in maternal_data['nbdiags']:
        try:
            res[feature_index[diag.replace('.','').strip()]] = True
        except KeyError:
            try:
                res[feature_index[diag.replace('.','').strip()[0:-2]]] = True
            except KeyError:
                pass #print('--->',diag.replace('.','').strip()[0:-1])    
    return res
def build_features_mat_race(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'race' not in maternal_data:
        return res
    code = maternal_data['race']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res

def build_features_mat_ethn(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'ethnicity' not in maternal_data:
        return res
    code = maternal_data['ethnicity']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res

def build_features_mat_lang(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'lang' not in maternal_data:
        return res
    code = maternal_data['lang']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res

def build_features_mat_natn(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'nationality' not in maternal_data:
        return res
    code = maternal_data['nationality']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res

def build_features_mat_marriage(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'marriage' not in maternal_data:
        return res
    code = maternal_data['marriage']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res
def build_features_mat_birthpl(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=bool)
    if 'birthplace' not in maternal_data:
        return res
    code = maternal_data['birthplace']
    if code in feature_index and pd.notnull(code):
        res[feature_index[code]] = True
    return res
def build_features_mat_agedel(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
    res = np.zeros(len(feature_headers), dtype=int)
    if 'agedeliv' not in maternal_data:
        return res
    age = int(maternal_data['agedeliv'])
    res[0] = age
    return res

def build_feature_matlang_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.BM_Language, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.BM_Language, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Maternal-Language:'+ i for i in codesNnames]
    return feature_index, feature_headers

def build_feature_matinsurance1_index():
    try:
        codesNnames1 = [l.strip().decode('utf-8') for l in open(config_file.BM_Prim_Ins, 'rb').readlines()]
    except:
        codesNnames1 = [l.strip().decode('latin-1') for l in open(config_file.BM_Prim_Ins, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames1):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Prim_Insur:'+ i for i in codesNnames1]
    return feature_index, feature_headers

def build_feature_matinsurance2_index():
    try:
        codesNnames2 = [l.strip().decode('utf-8') for l in open(config_file.BM_Second_Ins, 'rb').readlines()]
    except:
        codesNnames2 = [l.strip().decode('latin-1') for l in open(config_file.BM_Second_Ins, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames2):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Second_Insur:'+ i for i in codesNnames2]
    return feature_index, feature_headers

def build_feature_matethn_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.BM_EthnicityList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.BM_EthnicityList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Maternal-ethnicity:'+ i for i in codesNnames]
    return feature_index, feature_headers

def build_feature_matrace_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.BM_RaceList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.BM_RaceList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Maternal-race:'+ i for i in codesNnames]
    return feature_index, feature_headers
def build_feature_matnatn_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.BM_NationalityList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.BM_NationalityList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Maternal-nationality:'+ i for i in codesNnames]
    return feature_index, feature_headers

def build_feature_matmarriage_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.BM_Marital_StatusList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.BM_Marital_StatusList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Maternal-marriageStatus:'+ i for i in codesNnames]
    return feature_index, feature_headers
def build_feature_matbirthpl_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.BM_BirthPlace, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.BM_BirthPlace, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Maternal-birthplace:'+ i for i in codesNnames]
    return feature_index, feature_headers
def build_feature_agedeliv_index():
    feature_index={}
    feature_headers=['MatDeliveryAge']
    return feature_index, feature_headers

def build_feature_Mat_ICD_index():
    try:
        icd9 = [l.strip().decode("utf-8")  for l in open(config_file.icd9List, 'rb').readlines()]
    except:
        icd9 = [l.strip().decode('latin-1')  for l in open(config_file.icd9List, 'rb').readlines()]
    try:
        icd10 = [l.strip().decode("utf-8")  for l in open(config_file.icd10List, 'rb').readlines()]
    except:
        icd10 = [l.strip().decode('latin-1')   for l in open(config_file.icd10List, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, icd) in enumerate(icd9 + icd10):
        icd_codes = icd.split('|')[0].strip().split(' ')
        icd_codes_desc = icd.split('|')[1].strip()
        # print(icd_codes_desc)
        feature_headers.append('Maternal Diagnosis:'+icd_codes_desc)
        for icd_code in icd_codes:
            if icd_code in feature_index:
                feature_index[icd_code].append(ix)
                # print('warning - double icd in 9&10:', icd_code)
            else:
                feature_index[icd_code] = [ix]
    # feature_headers = ['Diagnosis:' + i for i in  (icd9 + icd10)]
    return feature_index, feature_headers
def build_feature_NB_ICD_index():
    try:
        icd9 = [l.strip().decode("utf-8")  for l in open(config_file.icd9List, 'rb').readlines()]
    except:
        icd9 = [l.strip().decode('latin-1')  for l in open(config_file.icd9List, 'rb').readlines()]
    try:
        icd10 = [l.strip().decode("utf-8")  for l in open(config_file.icd10List, 'rb').readlines()]
    except:
        icd10 = [l.strip().decode('latin-1')   for l in open(config_file.icd10List, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, icd) in enumerate(icd9 + icd10):
        icd_codes = icd.split('|')[0].strip().split(' ')
        icd_codes_desc = icd.split('|')[1].strip()
        # print(icd_codes_desc)
        feature_headers.append('Newborn Diagnosis:'+icd_codes_desc)
        for icd_code in icd_codes:
            if icd_code in feature_index:
                feature_index[icd_code].append(ix)
                # print('warning - double icd in 9&10:', icd_code)
            else:
                feature_index[icd_code] = [ix]
    # feature_headers = ['Diagnosis:' + i for i in  (icd9 + icd10)]
    return feature_index, feature_headers

def build_feature_gender_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.genderList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.genderList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = int(codeline.split(' ')[0].strip())
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Gender:'+ i for i in codesNnames]
    return feature_index, feature_headers
def build_feature_ethn_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.ethnicityList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.ethnicityList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Ethnicity:'+ i for i in codesNnames]
    return feature_index, feature_headers

def build_feature_vitallatest_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.vitalsList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.vitalsList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        codes = codeline.strip().split('#')[0]
        descr = codeline.strip().split('#')[1]
        for code in codes.split(' | '):
            if code in feature_index:
                feature_index[code.strip()].append(ix)
            else:
                feature_index[code.strip()] = [ix]
        feature_headers.append('Vital:'+ descr)
    return feature_index, feature_headers
def build_feature_zipcd_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.zipList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.zipList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Zipcode:'+ i for i in codesNnames]
    return feature_index, feature_headers
def build_feature_race_index():
    try:
        codesNnames = [l.strip().decode('utf-8') for l in open(config_file.raceList, 'rb').readlines()]
    except:
        codesNnames = [l.strip().decode('latin-1') for l in open(config_file.raceList, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, codeline) in enumerate(codesNnames):
        code = codeline.strip()
        if code in feature_index:
            feature_index[code].append(ix)
            # print('double!!', icd_code)
        else:
            feature_index[code] = [ix]
    feature_headers = ['Race:'+ i for i in codesNnames]
    return feature_index, feature_headers

def build_feature_lab_index():
    try:
        labsfile = [l.strip().decode("utf-8")  for l in open(config_file.labslist, 'rb').readlines()]
    except:
        labsfile = [l.strip().decode('latin-1')  for l in open(config_file.labslist, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, labcd) in enumerate(labsfile):
        lab_codes = labcd.split('|')[0].strip().split('#')
        lab_codes_desc = labcd.split('|')[1].strip()
        feature_headers.append('Lab:'+lab_codes_desc)
        for lab_code in lab_codes:
            if lab_code in feature_index:
                feature_index[lab_code].append(ix)
            else:
                feature_index[lab_code] = [ix]
    return feature_index, feature_headers

def build_feature_med_index():
    try:
        medsfile = [l.strip().decode("utf-8")  for l in open(config_file.medslist, 'rb').readlines()]
    except:
        medsfile = [l.strip().decode('latin-1')  for l in open(config_file.medslist, 'rb').readlines()]
    
    feature_index = {}
    feature_headers = []
    for (ix, medcd) in enumerate(medsfile):
        med_codes = medcd.split('|')[0].strip().split('#')
        med_codes_desc = medcd.split('|')[1].strip()
        feature_headers.append('Medication:'+med_codes_desc)
        for med_code in med_codes:
            if med_code in feature_index:
                feature_index[med_code].append(ix)
            else:
                feature_index[med_code] = [ix]
    return feature_index, feature_headers


def build_feature_ICD_index():
    try:
        icd9 = [l.strip().decode("utf-8")  for l in open(config_file.icd9List, 'rb').readlines()]
    except:
        icd9 = [l.strip().decode('latin-1')  for l in open(config_file.icd9List, 'rb').readlines()]
    try:
        icd10 = [l.strip().decode("utf-8")  for l in open(config_file.icd10List, 'rb').readlines()]
    except:
        icd10 = [l.strip().decode('latin-1')   for l in open(config_file.icd10List, 'rb').readlines()]
    feature_index = {}
    feature_headers = []
    for (ix, icd) in enumerate(icd9 + icd10):
        icd_codes = icd.split('|')[0].strip().split(' ')
        icd_codes_desc = icd.split('|')[1].strip()
        # print(icd_codes_desc)
        feature_headers.append('Diagnosis:' + icd_codes_desc)
        for icd_code in icd_codes:
            if icd_code in feature_index:
                feature_index[icd_code].append(ix)
                # print('warning - double icd in 9&10:', icd_code)
            else:
                feature_index[icd_code] = [ix]
    # feature_headers = ['Diagnosis:' + i for i in  (icd9 + icd10)]
    return feature_index, feature_headers

def call_build_function(data_dic, data_dic_moms, agex_low, agex_high, months_from, months_to, percentile, mrnsForFilter=[]):
    outcome = np.zeros(len(data_dic.keys()), dtype=float)
    outcomelabels = np.zeros(len(data_dic.keys()), dtype=float)
    feature_index_gen, feature_headers_gen = build_feature_gender_index()
    feature_index_icd, feature_headers_icd = build_feature_ICD_index()
    feature_index_lab, feature_headers_lab = build_feature_lab_index()
    feature_index_med, feature_headers_med = build_feature_med_index()
    feature_index_ethn, feature_headers_ethn = build_feature_ethn_index()
    feature_index_race, feature_headers_race = build_feature_race_index()
    feature_index_zipcd, feature_headers_zipcd = build_feature_zipcd_index()
    feature_index_vitalLatest, feature_headers_vitalsLatest = build_feature_vitallatest_index()
    
    feature_index_mat_ethn, feature_headers_mat_ethn = build_feature_matethn_index()
    feature_index_mat_race, feature_headers_mat_race = build_feature_matrace_index()
    feature_index_mat_marriage, feature_headers_mat_marriage = build_feature_matmarriage_index()
    feature_index_mat_natn, feature_headers_mat_natn = build_feature_matnatn_index()
    feature_index_mat_birthpl, feature_headers_mat_birthpl = build_feature_matbirthpl_index()
    feature_index_mat_lang, feature_headers_mat_lang = build_feature_matlang_index()
    feature_index_mat_agedeliv, feature_headers_age_deliv = build_feature_agedeliv_index()
    feature_index_mat_icd, feature_headers_mat_icd = build_feature_Mat_ICD_index()
    feature_index_nb_icd, feature_headers_nb_icd = build_feature_NB_ICD_index()
    feature_index_mat_insurance1, feature_headers_mat_insurance1 = build_feature_matinsurance1_index()
    feature_index_mat_insurance2, feature_headers_mat_insurance2 = build_feature_matinsurance2_index()
    
    funcs = [
        (build_features_icd, [ feature_index_icd, feature_headers_icd ]), #
        (build_features_lab, [ feature_index_lab, feature_headers_lab ]),
        (build_features_med, [ feature_index_med, feature_headers_med ]),
        (build_features_gen, [ feature_index_gen, feature_headers_gen ]), #
        (build_features_ethn, [ feature_index_ethn, feature_headers_ethn]),
        (build_features_race, [ feature_index_race, feature_headers_race]),
        (build_features_vitalLatest, [ feature_index_vitalLatest, [h+'-latest' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_0_3, [ feature_index_vitalLatest, [h+'-avg0to3' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_3_6, [ feature_index_vitalLatest, [h+'-avg3to6' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_6_9, [ feature_index_vitalLatest, [h+'-avg6to9' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_9_12, [ feature_index_vitalLatest, [h+'-avg9to12' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_12_15, [ feature_index_vitalLatest, [h+'-avg12to15' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_15_18, [ feature_index_vitalLatest, [h+'-avg15to18' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_18_21, [ feature_index_vitalLatest, [h+'-avg18to21' for h in feature_headers_vitalsLatest]]),
        (build_features_vitalAverage_21_24, [ feature_index_vitalLatest, [h+'-avg21to24' for h in feature_headers_vitalsLatest]]),
        # environment
        (build_features_zipcd, [ feature_index_zipcd, feature_headers_zipcd]),
        # maternal features
        (build_features_mat_icd, [ feature_index_mat_icd, feature_headers_mat_icd]), #
        (build_features_nb_icd, [ feature_index_nb_icd, feature_headers_nb_icd]),
        # (build_features_del_icd, [ feature_index_mat_deldiag, feature_headers_mat_deldiag ]),
        (build_features_mat_ethn, [ feature_index_mat_ethn, feature_headers_mat_ethn]), #
        (build_features_mat_insurance1, [ feature_index_mat_insurance1, feature_headers_mat_insurance1]), #
        (build_features_mat_insurance2, [ feature_index_mat_insurance2, feature_headers_mat_insurance2]),
        (build_features_mat_race, [ feature_index_mat_race, feature_headers_mat_race]),
        (build_features_mat_lang, [ feature_index_mat_lang, feature_headers_mat_lang]),
        (build_features_mat_natn, [ feature_index_mat_natn, feature_headers_mat_natn]),
        (build_features_mat_marriage, [ feature_index_mat_marriage, feature_headers_mat_marriage ]), #
        (build_features_mat_birthpl, [ feature_index_mat_birthpl, feature_headers_mat_birthpl]),
        (build_features_mat_agedel, [ feature_index_mat_agedeliv, feature_headers_age_deliv])
    ]

    features = np.zeros((len(data_dic.keys()), sum([len(f[1][1]) for f in funcs ]) ), dtype=float)
    mrns = [0]*len(data_dic.keys())

    headers = []
    for (pos, f ) in enumerate(funcs):
        headers += f[1][1]

    for (ix, k) in enumerate(data_dic):
        if (len(mrnsForFilter) > 0) & (str(data_dic[k]['mrn']) not in mrnsForFilter):
            continue
        flag=False
        bdate = data_dic[k]['bdate']
        mrns[ix] = data_dic[k]['mrn']
        if ('vitals' in data_dic[k]) and ('BMI' in data_dic[k]['vitals']):
            BMI_list = []
            BMI_outcome_list = []
            for (edate, bmi) in data_dic[k]['vitals']['BMI']:
                age = (edate - bdate).days / 365.0
                if (age >= agex_low) and (age< agex_high):
                    BMI_list.append(bmi)
                    BMI_outcome_list.append((outcome_def_pediatric_obesity.outcome(bmi, data_dic[k]['gender'], age) >= 0.95))
                    if (flag == False): #compute features once
                        if data_dic[k]['mrn'] in data_dic_moms:
                            maternal_data = data_dic_moms[data_dic[k]['mrn']]
                        else:
                            maternal_data = {}
                        ix_pos_start = 0
                        ix_pos_end = len(funcs[0][1][1])
                        for (pos, f) in enumerate(funcs):
                            func = f[0]
                            features[ix, ix_pos_start:ix_pos_end] = func(
                                data_dic[k], 
                                maternal_data,
                                bdate + timedelta(days=months_from*30), 
                                bdate + timedelta(days=months_to*30), 
                                *f[1])
                            ix_pos_start += len(f[1][1])
                            try:
                                ix_pos_end += len(funcs[pos+1][1][1])
                            except IndexError:
                                ix_pos_end = features.shape[1]
                        flag = True
            if (flag == True) and len(BMI_list)>1:
                # print(BMI_list)
                outcomelabels[ix] = stats.mode(np.array(BMI_outcome_list)).mode[0]
                outcome[ix] = np.array(BMI_list).mean()
                
    return features, outcome, outcomelabels, headers, np.array(mrns)