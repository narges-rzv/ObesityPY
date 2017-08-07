import config as config_file
import pandas as pd
import pickle 
import re
import matplotlib.pylab as plt
import time
from datetime import timedelta
from dateutil import parser
import numpy as np

def load_csv_input():
	print('loading data:', config_file.input_csv)
	data1 = pd.read_csv(config_file.input_csv[0], delimiter=config_file.input_csv_delimiter)
	data2 = pd.read_csv(config_file.input_csv[1], delimiter=config_file.input_csv_delimiter)
	data = pd.concat([data1, data2])
	print('done')
	return (data)

def load_mom_csv_input():
	return pd.read_csv(config_file.mom_input_csv[0],delimiter=config_file.input_csv_delimiter)

def analyse_ages(data):
	birth = pd.to_datetime(data[config_file.input_csv_birth_colname])
	order = pd.to_datetime(data[config_file.input_csv_order_colname])
	diff = (order - birth).apply(lambda l: l.days)
	diffmx = (diff[diff>0]).as_matrix()
	#import matplotlib.pylab as plt
	#plt.hist(diffmax, bins=100)
	#plt.show
	return (diffmx, birth, order)

def parse_data(data):
	db = {}
	for ix, item in data.iterrows():
		try:
			mid = item[config_file.input_csv_mid_colname]
		except ValueError:
			pass
		try:
			mrn = item[config_file.input_csv_mrn_colname]
		except ValueError:
			pass	
		try:
			bdate = parser.parse(item[config_file.input_csv_birth_colname])
		except ValueError:
			pass
		try:
			odate = parser.parse(item[config_file.input_csv_order_colname])
		except ValueError:
			pass
		try:
			gender = item[config_file.input_csv_gender_colname].startswith('F')
		except:
			pass
		try:
			address = item[config_file.input_csv_addr_colname]
		except ValueError:
			pass
		try:	
			email = item[config_file.input_csv_email_colname]
		except ValueError:
			pass
		try:	
			zipcode = item[config_file.input_csv_zip_colname]
		except ValueError:
			pass
		try:	
			vitals = item[config_file.input_csv_vitals_colname]
		except ValueError:
			pass
		try:	
			vitals_dic = parse_vitals_dic(vitals)
		except ValueError:
			pass
		try:	
			diags = item[config_file.input_csv_diag_colname]
			diags_dic = parse_diag_dic(diags)
		except ValueError:
			pass
		try:	
			labs = item[config_file.input_csv_labs_colname]
		except ValueError:
			pass
		try:	
			lab_vals = item[config_file.input_csv_labres_colname]
			labs_dic = parse_labs_dic(labs, lab_vals)
		except ValueError:
			pass
		try:	
			meds = item[config_file.input_csv_med_colname]
			meds_dic = parse_medications(meds)
		except ValueError:
			pass
		try:	
			vaccines = item[config_file.input_csv_vac_colname]
		except ValueError:
			pass
		try:	
			ethnicity = item[config_file.input_csv_eth_colname]
		except ValueError:
			pass
		try:	
			race = item[config_file.input_csv_race_colname]
		except ValueError:
			pass

		if mid not in db:
			db[mid] = {}
			db[mid]['diags']={}
			db[mid]['vitals']={}
			db[mid]['labs']={}	
			db[mid]['meds']={}		

		db[mid]['bdate'] = bdate
		db[mid]['gender'] = gender
		db[mid]['ethnicity'] = ethnicity
		db[mid]['race'] = race
		db[mid]['mrn'] = mrn
		
		if 'odate' in db[mid]:
			db[mid]['odate'].append(odate)
		else:
			db[mid]['odate'] = [odate]

		if 'address' in db[mid]:
			db[mid]['address'].append([odate, address])
		else:
			db[mid]['address'] = [[odate, address]]
		
		if 'email' in db[mid]:	
			db[mid]['email'].append([odate, email])
		else:
			db[mid]['email'] = [[odate, email]]

		if 'zip' in db[mid]:
			db[mid]['zip'].append([odate, zipcode])
		else:
			db[mid]['zip'] = [[odate, zipcode]]

		for k in vitals_dic.keys():
			if k in db[mid]['vitals']:
				db[mid]['vitals'][k].append([odate, vitals_dic[k]])
			else:
				db[mid]['vitals'][k] = [[odate, vitals_dic[k]]]

		for k in diags_dic.keys():
			if k in db[mid]['diags']:
				db[mid]['diags'][k].append([odate, diags_dic[k]])
			else:
				db[mid]['diags'][k] = [[odate, diags_dic[k]]]

		for k in meds_dic.keys():
			if k in db[mid]['meds']:
				db[mid]['meds'][k].append([odate, meds_dic[k]])
			else:
				db[mid]['meds'][k] = [[odate, meds_dic[k]]]

		for k in labs_dic.keys():
			if k in db[mid]['labs']:
				db[mid]['labs'][k].append([odate, labs_dic[k]])
			else:
				db[mid]['labs'][k] = [[odate, labs_dic[k]]]
	return db
def parse_mother_data(data):
	db = {}
	for ix, item in data.iterrows():
		try:
			mid = item[config_file.input_csv_newborn_MRN]
		except ValueError:
			pass
		try:
			mom_mrn = item[config_file.input_csv_mothers_MRN]
		except ValueError:
			pass	
		try:
			agedeliv = int(item[config_file.input_csv_mothers_agedeliv])
		except ValueError:
			pass
		try:	
			diags = str(item[config_file.input_csv_mothers_diags])
			if pd.notnull(diags): diags_dic = diags.split(';')
		except ValueError:
			pass
		try:	
			nbdiags = str(	item[config_file.input_csv_mothers_NB_diags])
			if pd.notnull(nbdiags): nbdiags_dic = nbdiags.split(';')
		except ValueError:
			pass
		try:	
			deldiags = str(item[config_file.input_csv_mothers_deliv_diags])
			if pd.notnull(deldiags): deldiags_dic = deldiags.split(';')
		except ValueError:
			pass
		except AttributeError:
			print(deldiags)
		try:	
			ethnicity = item[config_file.input_csv_mothers_ethn]
		except ValueError:
			pass
		try:	
			race = item[config_file.input_csv_mothers_race]
		except ValueError:
			pass
		try:	
			nationality = item[config_file.input_csv_mothers_national]
		except ValueError:
			pass
		try:	
			marriage = item[config_file.input_csv_mothers_marriage]
		except ValueError:
			pass
		try:	
			birthplace = item[config_file.input_csv_mothers_birthplace]
		except ValueError:
			pass
		try:	
			lang = item[config_file.input_csv_mothers_lang]
		except ValueError:
			pass
		try:	
			insur1 = item[config_file.input_csv_mothers_insur1]
		except ValueError:
			pass
		try:	
			insur2 = item[config_file.input_csv_mothers_insur2]
		except ValueError:
			pass

		if mid not in db:
			db[mid] = {}
			db[mid]['diags']={}
			db[mid]['nbdiags']={}
			db[mid]['deldiags']={}

		db[mid]['mom_mrn'] = mom_mrn
		db[mid]['ethnicity'] = ethnicity
		db[mid]['race'] = race
		db[mid]['nationality'] = nationality
		db[mid]['marriage'] = marriage
		db[mid]['birthplace'] = birthplace
		db[mid]['lang'] = lang
		db[mid]['insur1'] = insur1
		db[mid]['insur2'] = insur2
		db[mid]['agedeliv'] = agedeliv

		for k in diags_dic:
			db[mid]['diags'][k]=True
		for k in nbdiags_dic:
			db[mid]['nbdiags'][k]=True
		for k in deldiags_dic:
			db[mid]['deldiags'][k]=True
	return db

def parse_vitals_dic(str1):
	vitals = {}
	ws = re.split('\|+ ', str1.strip())
	if len(ws) == 0:
		return vitals

	for w in ws:
		if w.strip() == '':
			continue

		ks = re.split('(-*\d*\.*\d+)', w.strip())
		if len(ks) < 2: 
			continue
		try:
			vitals[ks[0].strip(' ')] = float(ks[1])
		except ValueError:
			try:
				vitals[ks[0].strip(' ')] = [float(ks[1].split('/')[0]), float(ks[1].split('/')[1])]
			except:
				pass		
	return vitals

def parse_diag_dic(str1):
	diag = {}
	ws = re.split('\|+ ', str1.strip())
	if len(ws) == 0:
		return diag

	for w in ws:
		if w.strip() == '':
			continue
		ks = re.split(' ', w.strip())
		if len(ks[0]) > 6:
			#print(w)
			continue
		diag[ks[0]] = 1
	return diag

def parse_medications(str1):
	meds = {}
	ws = re.split('\|+ ', str1.strip())
	
	if len(ws) == 0:
		return meds

	for w in ws:
		if w.strip() == '':
			continue
		# ks = re.split(' ', w.strip())
		# print(w)
		meds[w.strip()] = 1

	return meds

def parse_labs_dic(str1, str2):
	d1 = {}
	ws1 = re.split('\|+ ', str1.strip()) #lab codes
	try:
		str2 = str2.lower().replace('positive', '1').replace('pos','1').replace('negative','-1').replace('neg','-1')
		ws2 = re.split('\|+ ', str2.strip()) #results
	except:
		# print ('str2', str2)
		return d1

	if len(ws1) == 0 or len(ws1) != len(ws2):
		return d1

	for i, w in enumerate(ws1):
		if w.strip() == '':
			continue
		ks = re.split(' ', w.strip())
		try:
			d1[w.strip()] = float(ws2[i])
		except ValueError:
			pass #print('value error in parsing', ws1[i], ws2[i])
	return d1

def run_builddata():
	(data) = load_csv_input()
	db = parse_data(data)
	return db

if __name__=='__main__':
	run_builddata()