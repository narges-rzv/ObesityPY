import math
import numpy as np
import config

WHO_percentiles = {}
CDC_percentiles = {}

# load the WHO and CDC data
def init():
    global CDC_percentiles
    global WHO_percentiles
    if WHO_percentiles=={}:
    	load_WHO_refs()
    if CDC_percentiles=={}:
    	load_CDC_refs()

def load_WHO_refs(girls_inputfile='None', boys_inputfile='None'):
    if girls_inputfile == 'None':
        girls_inputfile = config.wght4leng_girl
    if boys_inputfile == 'None':
        boys_inputfile = config.wght4leng_girl
    global WHO_percentiles
    for ix, inputfile in enumerate([boys_inputfile, girls_inputfile]):
        # columns of inputfile: Length,L,M,S,P01,P1,P3,P5,P10,P15,P25,P50,P75,P85,P90,P95,P97,P99,P999
        rawdata = np.loadtxt(inputfile, delimiter='\t')
        WHO_percentiles[ix] = rawdata[:,1:]
    WHO_percentiles['length'] = rawdata[:,0].tolist()

def load_CDC_refs(girls_inputfile='None', boys_inputfile='None'):
    if girls_inputfile == 'None':
        girls_inputfile = config.bmi_girl
    if boys_inputfile == 'None':
        boys_inputfile = config.bmi_boy
    global CDC_percentiles
    for ix, inputfile in enumerate([boys_inputfile, girls_inputfile]):
        # columns of inputfile: Agemos,L,M,S,P3,P5,P10,P25,P50,P75,P85,P90,P95,P97
        rawdata = np.loadtxt(inputfile, delimiter=',')
        CDC_percentiles[ix] = rawdata[:,1:]
    CDC_percentiles['age'] = rawdata[:,0].tolist()

def linear_interpolation(val1, x_1, x_2, y_1, y_2):
    return y_1 + ((y_2 - y_1) * (val1 - x_1) / (x_2 - x_1))

def zscore_wfl(gender, length, weight, units='metric'):
    """
    Calculates the WHO weight for length Z-score from https://www.cdc.gov/nccdphp/dnpao/growthcharts/resources/sas.htm
    where Z = (((value / M)**L) – 1) / (S * L). In addition any Z-score with absolute value greater than 5 is
    forced to sign(Z) * 1
    NOTE: This should only be used for children under the age of 2 as BMI values cannot be accurately recorded until 2 years of age.

    #### PARAMETERS ####
    parameters should either be arrays or single items
    gender: 0 for male, 1 for female
    length: length/height
    weight: weight
    units: default = 'metric'.
        'metric': lengths/weights assumed to be in cm/kg respectively
        'usa': lengths/weights assumed to be in in/lb respectively
    """
    if units not in ('metric','usa'):
        raise ValueError('Invalid measurement systm. Must be "metric" or "usa".')

    global WHO_percentiles
    if units == 'usa':
        length *= 2.54 #inches to cm
        weight *= 0.4535924 #pounds to kg

    if all([type(x)==np.ndarray for x in (gender,weight,length)]):
        weight = weight.astype(float)
        length = length.astype(float)
        zscores = np.zeros(gender.reshape(-1,1).shape[0])
        L = np.zeros(gender.reshape(-1,1).shape[0])
        M = np.zeros(gender.reshape(-1,1).shape[0])
        S = np.zeros(gender.reshape(-1,1).shape[0])
        failed = 0
        for ix in range(zscores.shape[0]):
            if length[ix] < np.min(WHO_percentiles['length']) or length[ix] > np.max(WHO_percentiles['length']):
                continue

            if math.fmod(length[ix]*10, 1) == 0:
                ix_low = WHO_percentiles['length'].index(length[ix])
                L[ix] = WHO_percentiles[gender[ix]][ix_low,0]
                M[ix] = WHO_percentiles[gender[ix]][ix_low,1]
                S[ix] = WHO_percentiles[gender[ix]][ix_low,2]
            else:
                ix_low = WHO_percentiles['length'].index(int(length[ix]*10)/10)
                L[ix] = linear_interpolation(length[ix], WHO_percentiles['length'][ix_low], WHO_percentiles['length'][ix_low+1], WHO_percentiles[gender[ix]][ix_low,0], WHO_percentiles[gender[ix]][ix_low+1,0])
                M[ix] = linear_interpolation(length[ix], WHO_percentiles['length'][ix_low], WHO_percentiles['length'][ix_low+1], WHO_percentiles[gender[ix]][ix_low,1], WHO_percentiles[gender[ix]][ix_low+1,1])
                S[ix] = linear_interpolation(length[ix], WHO_percentiles['length'][ix_low], WHO_percentiles['length'][ix_low+1], WHO_percentiles[gender[ix]][ix_low,2], WHO_percentiles[gender[ix]][ix_low+1,2])

        with np.errstate(divide='ignore', invalid='ignore'):
            zscores = (((weight / M)**L) - 1.) / (S * L)
            zscores[(np.abs(zscores) > 5)] = np.sign(zscores[(np.abs(zscores) > 5)])
        return np.nan_to_num(zscores)
    else:
        if length < np.min(WHO_percentiles['length']) or length > np.max(WHO_percentiles['length']):
            return 0.0
        if math.fmod(length*10, 1) == 0:
            ix = WHO_percentiles['length'].index(length)
            L = WHO_percentiles[gender][ix,0]
            M = WHO_percentiles[gender][ix,1]
            S = WHO_percentiles[gender][ix,2]
        else:
            ix_low = WHO_percentiles['length'].index(int(length*10)/10)
            L = linear_interpolation(length, WHO_percentiles['length'][ix_low], WHO_percentiles['length'][ix_low+1], WHO_percentiles[gender][ix_low,0], WHO_percentiles[gender][ix_low+1,0])
            M = linear_interpolation(length, WHO_percentiles['length'][ix_low], WHO_percentiles['length'][ix_low+1], WHO_percentiles[gender][ix_low,1], WHO_percentiles[gender][ix_low+1,1])
            S = linear_interpolation(length, WHO_percentiles['length'][ix_low], WHO_percentiles['length'][ix_low+1], WHO_percentiles[gender][ix_low,2], WHO_percentiles[gender][ix_low+1,2])

        Z = (((weight / M)**L) - 1) / (S * L)
        if abs(Z) > 5:
            Z = np.sign(Z)
        return Z

def zscore_bmi(gender, age, bmi, unit='months'):
    """
    Calculates the CDC BMI Z-score from https://www.cdc.gov/nccdphp/dnpao/growthcharts/resources/sas.htm
    where Z = (((value / M)**L) – 1) / (S * L). In addition any Z-score with absolute value greater than 5 is
    forced to sign(Z) * 1
    NOTE: This should only be used for anyone between the ages of 2 and 20.

    #### PARAMETERS ####
    parameters should either be arrays or single items
    gender: 0 for male, 1 for female
    age: age in months between 24 and 240
    bmi: bmi
    unit: default = 'months'
        'months': age in months
        'years': age in years
    """
    global CDC_percentiles
    if unit not in ('years','months'):
        raise ValueError('Invalid input for unit. Must be "years" or "months".')
    if unit == 'years':
        age *= 12.0

    if all([type(x)==np.ndarray for x in (gender,age,bmi)]):
        bmi = bmi.astype(float)
        zscores = np.zeros(gender.reshape(-1,1).shape[0])
        L = np.zeros(gender.reshape(-1,1).shape[0])
        M = np.zeros(gender.reshape(-1,1).shape[0])
        S = np.zeros(gender.reshape(-1,1).shape[0])
        for ix in range(zscores.shape[0]):
            if (age[ix] < np.min(CDC_percentiles['age']) or age[ix] > np.max(CDC_percentiles['age'])) and unit == 'months':
                continue

            if math.fmod(age[ix], 1) == 0.5:
                ix_low = CDC_percentiles['age'].index(age[ix])
                L[ix] = CDC_percentiles[gender[ix]][ix_low,0]
                M[ix] = CDC_percentiles[gender[ix]][ix_low,1]
                S[ix] = CDC_percentiles[gender[ix]][ix_low,2]
                continue
            elif math.fmod(age[ix], 1) < 0.5:
                ix_low = CDC_percentiles['age'].index(age[ix] - math.fmod(age[ix], 1) - 0.5)
            else:
                ix_low = CDC_percentiles['age'].index(age[ix] - math.fmod(age[ix], 1) + 0.5)
            L[ix] = linear_interpolation(age[ix], CDC_percentiles['age'][ix_low], CDC_percentiles['age'][ix_low+1], CDC_percentiles[gender[ix]][ix_low,0], CDC_percentiles[gender[ix]][ix_low+1,0])
            M[ix] = linear_interpolation(age[ix], CDC_percentiles['age'][ix_low], CDC_percentiles['age'][ix_low+1], CDC_percentiles[gender[ix]][ix_low,1], CDC_percentiles[gender[ix]][ix_low+1,1])
            S[ix] = linear_interpolation(age[ix], CDC_percentiles['age'][ix_low], CDC_percentiles['age'][ix_low+1], CDC_percentiles[gender[ix]][ix_low,2], CDC_percentiles[gender[ix]][ix_low+1,2])
        with np.errstate(divide='ignore', invalid='ignore'):
            zscores = (((bmi / M)**L) - 1.) / (S * L)
            zscores[(np.abs(zscores) > 5)] = np.sign(zscores[(np.abs(zscores) > 5)])
        return np.nan_to_num(zscores)
    else:
        if age < np.min(CDC_percentiles['age']) or age > np.max(CDC_percentiles['age']):
            return 0.0

        if math.fmod(age, 1) == 0.5:
            ix_low = CDC_percentiles['age'].index(age)
            L = CDC_percentiles[gender][ix_low,0]
            M = CDC_percentiles[gender][ix_low,1]
            S = CDC_percentiles[gender][ix_low,2]
            Z = (((bmi / M)**L) - 1) / (S * L)
            if abs(Z) > 5:
                Z = np.sign(Z)
            return Z
        elif math.fmod(age, 1) < 0.5:
            ix_low = CDC_percentiles['age'].index(age - math.fmod(age, 1) - 0.5)
        else:
            ix_low = CDC_percentiles['age'].index(age - math.fmod(age, 1) + 0.5)
        L = linear_interpolation(age, CDC_percentiles['age'][ix_low], CDC_percentiles['age'][ix_low+1], CDC_percentiles[gender][ix_low,0], CDC_percentiles[gender][ix_low+1,0])
        M = linear_interpolation(age, CDC_percentiles['age'][ix_low], CDC_percentiles['age'][ix_low+1], CDC_percentiles[gender][ix_low,1], CDC_percentiles[gender][ix_low+1,1])
        S = linear_interpolation(age, CDC_percentiles['age'][ix_low], CDC_percentiles['age'][ix_low+1], CDC_percentiles[gender][ix_low,2], CDC_percentiles[gender][ix_low+1,2])
        Z = (((bmi / M)**L) - 1) / (S * L)
        if abs(Z) > 5:
            Z = np.sign(Z)
        return Z

init()
