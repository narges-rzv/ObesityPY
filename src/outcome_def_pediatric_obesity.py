import config
import numpy as np
CDC_Ranges_95th_percentile = {}
CDC_Ranges_95th_percentile[0] = {}
CDC_Ranges_95th_percentile[1] = {}
p_levels  = [0.05,0.10,0.25,0.50,0.75,0.85,0.90,0.95,0.97,1]

def init():
	global CDC_Ranges_95th_percentile
	if CDC_Ranges_95th_percentile[0]=={}:
		load_CDC_refs()

def load_CDC_refs(inputfile='None'):
	if inputfile =='None'
		inputfile = config.CDC_BMI_Ref
	global CDC_Ranges_95th_percentile
	rawdata = np.loadtxt(config.CDC_BMI_Ref, delimiter=',')
	for i in range(0,rawdata.shape[0]):
		CDC_Ranges_95th_percentile[rawdata[i][0]][rawdata[i][1]] = [rawdata[i][5], rawdata[i][6], rawdata[i][7], rawdata[i][8], rawdata[i][9], rawdata[i][10], rawdata[i][11], rawdata[i][12], rawdata[i][13], rawdata[i][14]]

def outcome(bmi, gender, agex):
	global CDC_Ranges_95th_percentile
	age_range_low = agex - (agex%0.5)
	if bmi > CDC_Ranges_95th_percentile[gender][age_range_low][7]:
		return 1.0
	return 0.0

def linear_interpolation(bmi, x_1, x_2, y_1, y_2):
	# print(bmi, x_1, x_2, y_1, y_2)
	return y_1 + ((y_2 - y_1) * (bmi - x_1) / (x_2 - x_1))

def percentile(bmi, gender, agex):
	global CDC_Ranges_95th_percentile
	age_range_low = agex - (agex%0.5)

	for i, p_l in enumerate(p_levels):
		if i == len(p_levels) -1 and bmi >= CDC_Ranges_95th_percentile[gender][age_range_low][i]:
			return 0.97
		if bmi < CDC_Ranges_95th_percentile[gender][age_range_low][0]:
			return 0.05

		if (bmi >= CDC_Ranges_95th_percentile[gender][age_range_low][i]) & (bmi < CDC_Ranges_95th_percentile[gender][age_range_low][i+1]):
			return linear_interpolation(bmi, CDC_Ranges_95th_percentile[gender][age_range_low][i], CDC_Ranges_95th_percentile[gender][age_range_low][i+1], p_levels[i], p_levels[i+1])
			
	return 0.0

init()
print(CDC_Ranges_95th_percentile)