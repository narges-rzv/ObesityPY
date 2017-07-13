input_csv = ["/mnt/d0/encrypted_data/data/All_patients_of_age_18_or_less_in_eCW_for_at_least_2_years_II.csv","/mnt/d0/encrypted_data/data/All_patients_of_age_18_or_less_in_eCW_for_at_least_2_years_II_sheet2.csv"]
input_csv_delimiter = ","

input_csv_mid_colname = 'patientid'
input_csv_mrn_colname = 'mrn'
input_csv_zip_colname = 'zip'
input_csv_email_colname = 'email'
input_csv_birth_colname = 'date_of_birth'
input_csv_order_colname = 'order_date'
input_csv_vitals_colname = 'vitals'
input_csv_vitals_colname_bmi = 'BMI'
input_csv_gender_colname = 'gender'
input_csv_addr_colname = 'address'
input_csv_diag_colname = 'diagnosis'
input_csv_labs_colname = 'labs'
input_csv_labres_colname = 'lab_results'
input_csv_med_colname = 'meds'
input_csv_vac_colname = 'vaccines'
input_csv_eth_colname = 'ethnicity'
input_csv_race_colname = 'race'
vital_keys = {'Temp', 'Ht', 'Wt', 'BMI', 'BP', 'HR', 'Oxygen', 'Pulse', 'Hearing', 'Vision', 'RR', 'PEF', 'Pre-gravid', 'Repeat', 'Pain', 'HC', 'Fundal', 'Education', 'Insulin', 'HIV', 'BMI Percentile', 'Ht Percentile', 'Wt Percentile', 'Wt Change', 'Oxygen sat', 'Pulse sitting', 'Vision (R) CC', 'Vision (L) CC', ''}

shelve_database_file = '/mnt/d0/encrypted_data/data/shelveDB.shlv'
icd9List = '../auxdata/icd9listccs.txt'
icd10List = '../auxdata/icd10listccs.txt'
genderList = '../auxdata/genderlist.txt'
ethnicityList = '../auxdata/ethnicityList.txt'
raceList = '../auxdata/raceList.txt'
zipList = '../auxdata/zipList.txt'
vitalsList = '../auxdata/vitals.txt'

CDC_BMI_Ref = '../auxdata/CDC_BMIs.txt'

#maternal info
input_csv_mothers = ['']
input_csv_mothers_delim = ','

input_csv_mothers_MRN = 'NB_MRN'
input_csv_mothers_agedeliv = 'BM_Age_at_Delivery' #int
input_csv_mothers_marriage = 'BM_Marital_Status'
input_csv_mothers_race = 'BM_Race'
input_csv_mothers_ethn = 'BM_Ethnicity'
input_csv_mothers_national = 'BM_Nationality'
input_csv_mothers_birthplace = 'BM_BirthPlace'
input_csv_mothers_lang = 'BM_Language'
input_csv_mothers_insur1 = 'BM_Prim_Ins'
input_csv_mothers_insur2 = 'BM_Second_Ins'
input_csv_mothers_diags = 'MAT_ACCT_DAIG_LIST'# (icd9/10 list separated by ';')
input_csv_mothers_NB_diags = 'NEW ACCT DIAG List'# (icd9/10 list separated by ';')
input_csv_mothers_deliv_diags = 'Delv DIAG'# (icd9/10 list separated by ';')


BM_Marital_StatusList = '../auxdata/BM_Marital_Status.txt'
BM_RaceList = '../auxdata/BM_Race.txt'
BM_EthnicityList = '../auxdata/BM_Ethnicity.txt'
BM_NationalityList = '../auxdata/BM_Nationality.txt'
BM_BirthPlace = '../auxdata/BM_BirthPlace.txt'
BM_Language = '../auxdata/BM_Language.txt'
BM_Prim_Ins = '../auxdata/BM_Prim_Ins.txt'
BM_Second_Ins = '../auxdata/BM_Second_Ins.txt'

