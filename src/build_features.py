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
				# age_at_vital = (edate - bdate).days / 365.0
				res[feature_index[code.strip()]] = vitalval
					# print(code, age_at_vital, vitalval)
			except:
				pass
				# print('error with', (code, edate, vitalval) )
	return res

def build_features_ethn(patient_data, maternal_data, reference_date_start, reference_date_end, feature_index, feature_headers):
	res = np.zeros(len(feature_headers), dtype=bool)
	code = patient_data['ethnicity']
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
	feature_headers=['MatDeliveryAge:']
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
		icd_code = icd.split(' ')[0].strip()
		if icd_code in feature_index:
			feature_index[icd_code].append(ix)
			# print('double!!', icd_code)
		else:
			feature_index[icd_code] = [ix]
	feature_headers = ['MAT_Diagnosis:' + i for i in  (icd9 + icd10)]
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
		icd_code = icd.split(' ')[0].strip()
		if icd_code in feature_index:
			feature_index[icd_code].append(ix)
			# print('double!!', icd_code)
		else:
			feature_index[icd_code] = [ix]
	feature_headers = ['NB_Diagnosis:' + i for i in  (icd9 + icd10)]
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
		code = codeline.strip()
		if code in feature_index:
			feature_index[code].append(ix)
		else:
			feature_index[code] = [ix]
	feature_headers = ['Vital_latest:'+ i for i in codesNnames]
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
		icd_code = icd.split(' ')[0].strip()
		if icd_code in feature_index:
			feature_index[icd_code].append(ix)
			# print('double!!', icd_code)
		else:
			feature_index[icd_code] = [ix]
	feature_headers = ['Diagnosis:' + i for i in  (icd9 + icd10)]
	return feature_index, feature_headers

def call_build_function(data_dic, data_dic_moms, agex_low, agex_high, months_from, months_to, percentile):
	outcome = np.zeros(len(data_dic.keys()), dtype=float)
	outcomelabels = np.zeros(len(data_dic.keys()), dtype=float)
	feature_index_gen, feature_headers_gen = build_feature_gender_index()
	feature_index_icd, feature_headers_icd = build_feature_ICD_index()
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
	# feature_index_mat_deldiag, feature_headers_mat_deldiag = build_feature_deldiag_index()

	
	funcs = [
		(build_features_icd, [ feature_index_icd, feature_headers_icd ]), #
		(build_features_gen, [ feature_index_gen, feature_headers_gen ]), #
		(build_features_ethn, [ feature_index_ethn, feature_headers_ethn]),
		(build_features_race, [ feature_index_race, feature_headers_race]),
		(build_features_vitalLatest, [ feature_index_vitalLatest, feature_headers_vitalsLatest]),
		# environment
		(build_features_zipcd, [ feature_index_zipcd, feature_headers_zipcd]),
		# maternal features
		# (build_features_mat_icd, [ feature_index_mat_icd, feature_headers_mat_icd]), #
		# (build_features_nb_icd, [ feature_index_nb_icd, feature_headers_nb_icd]),
		# (build_features_del_icd, [ feature_index_mat_deldiag, feature_headers_mat_deldiag ]),
		(build_features_mat_ethn, [ feature_index_mat_ethn, feature_headers_mat_ethn]), #
		(build_features_mat_race, [ feature_index_mat_race, feature_headers_mat_race]),
		(build_features_mat_lang, [ feature_index_mat_lang, feature_headers_mat_lang]),
		(build_features_mat_natn, [ feature_index_mat_natn, feature_headers_mat_natn]),
		(build_features_mat_marriage, [ feature_index_mat_marriage, feature_headers_mat_marriage ]), #
		(build_features_mat_birthpl, [ feature_index_mat_birthpl, feature_headers_mat_birthpl]),
		(build_features_mat_agedel, [ feature_index_mat_agedeliv, feature_headers_age_deliv])
	]

	features = np.zeros((len(data_dic.keys()), sum([len(f[1][1]) for f in funcs ]) ), dtype=bool)
	
	headers = []
	for (pos, f ) in enumerate(funcs):
		headers += f[1][1]

	for (ix, k) in enumerate(data_dic):
		flag=False
		bdate = data_dic[k]['bdate']
		if ('vitals' in data_dic[k]) and ('BMI' in data_dic[k]['vitals']):
			for (edate, bmi) in data_dic[k]['vitals']['BMI']:
				age = (edate - bdate).days / 365.0
				if (age >= agex_low) and (age< agex_high) and (flag == False):
					# print ('patient ', k, 'BMI at age:', age, 'is:', bmi)
					if data_dic[k]['mrn'] in data_dic_moms:
						maternal_data = data_dic_moms[data_dic[k]['mrn']]
					else:
						maternal_data = {}
					outcomelabels[ix] = (outcome_def_pediatric_obesity.percentile(bmi, data_dic[k]['gender'], age) >= 0.95)
					if percentile == True:
						outcome[ix] = outcome_def_pediatric_obesity.percentile(bmi, data_dic[k]['gender'], age)
					else:
						outcome[ix] = bmi
					ix_pos_start = 0
					ix_pos_end = len(funcs[0][1][1])
					for (pos, f) in enumerate(funcs):
						func = f[0]
						features[ix, ix_pos_start:ix_pos_end] = func(
							data_dic[k], 
							maternal_data,
							bdate + timedelta(days=months_from*30), #edate - timedelta(days=months_from*30),
							bdate + timedelta(days=months_to*30), #edate + timedelta(days=months_to*30), 
							*f[1])
						ix_pos_start += len(f[1][1])
						try:
							ix_pos_end += len(funcs[pos+1][1][1])
						except IndexError:
							ix_pos_end = features.shape[1]
					flag = True

				
	return features, outcome, outcomelabels, headers