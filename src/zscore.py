import math
import numpy as np

# load the WHO and CDC data

# weight-for-length headers
# Length,L,M,S,P01,P1,P3,P5,P10,P15,P25,P50,P75,P85,P90,P95,P97,P99,P999
g_wfl = np.loadtxt('../auxdata/WHO_wfl_girls_p_exp.txt', delimiter='\t')
b_wfl = np.loadtxt('../auxdata/WHO_wfl_boys_p_exp.txt', delimiter='\t')
wfl = {0:b_wfl[:,1:4], 1:g_wfl[:,1:4], 'length':g_wfl[:,0].tolist()}

# bmi headers & info
# 1 == male; 2 == female
# AGEMOS,L,M,S
g_bmi = np.loadtxt('../auxdata/CDC_bmi_girls_z.csv', delimiter=',')
b_bmi = np.loadtxt('../auxdata/CDC_bmi_boys_z.csv', delimiter=',')
bmi_dic = {0:b_bmi[:,1:], 1:g_bmi[:,1:], 'age':g_bmi[:,0].tolist()}

def linear_interpolation(val1, x_1, x_2, y_1, y_2):
    return y_1 + ((y_2 - y_1) * (val1 - x_1) / (x_2 - x_1))

def zscore_wfl(gender, length, weight):
    """
    Calculates the WHO weight for length Z-score from https://www.cdc.gov/nccdphp/dnpao/growthcharts/resources/sas.htm
    where Z = (((value / M)**L) – 1) / (S * L). In addition any Z-score with absolute value greater than 5 is
    forced to sign(Z) * 1
    NOTE: This should only be used for children under the age of 2 as BMI values cannot be accurately recorded until 2 years of age.

    #### PARAMETERS ####
    parameters should either be arrays or single items
    gender: 0 for male, 1 for female
    length: length/height in cm between 45 and 110
    weight: weight in kg
    """
    if all([type(x)==np.ndarray for x in (gender,weight,length)]):
        weight = weight.astype(float)
        length = length.astype(float)
        zscores = np.zeros(gender.reshape(-1,1).shape[0])
        L = np.zeros(gender.reshape(-1,1).shape[0])
        M = np.zeros(gender.reshape(-1,1).shape[0])
        S = np.zeros(gender.reshape(-1,1).shape[0])
        for ix in range(zscores.shape[0]):
            if length[ix] < 45 or length[ix] > 110:
                continue
            if math.fmod(length[ix]*10, 1) == 0:
                ix_low = wfl['length'].index(length[ix])
                L[ix] = wfl[gender[ix]][ix_low,0]
                M[ix] = wfl[gender[ix]][ix_low,1]
                S[ix] = wfl[gender[ix]][ix_low,2]
            else:
                ix_low = wfl['length'].index(int(length[ix]*10)/10)
                L[ix] = linear_interpolation(length[ix], wfl['length'][ix_low], wfl['length'][ix_low+1], wfl[gender[ix]][ix_low,0], wfl[gender[ix]][ix_low+1,0])
                M[ix] = linear_interpolation(length[ix], wfl['length'][ix_low], wfl['length'][ix_low+1], wfl[gender[ix]][ix_low,1], wfl[gender[ix]][ix_low+1,1])
                S[ix] = linear_interpolation(length[ix], wfl['length'][ix_low], wfl['length'][ix_low+1], wfl[gender[ix]][ix_low,2], wfl[gender[ix]][ix_low+1,2])

        zscores = (((weight / M)**L) - 1.) / (S * L)
        zscores[(np.abs(zscores) > 5)] = np.sign(zscores[(np.abs(zscores) > 5)])
        return np.nan_to_num(zscores)
    else:
        if length < 45 or length > 110:
            return 0
        if math.fmod(length*10, 1) == 0:
            ix = wfl['length'].index(length)
            L = wfl[gender][ix,0]
            M = wfl[gender][ix,1]
            S = wfl[gender][ix,2]
        else:
            ix_low = wfl['length'].index(int(length*10)/10)
            L = linear_interpolation(length, wfl['length'][ix_low], wfl['length'][ix_low+1], wfl[gender][ix_low,0], wfl[gender][ix_low+1,0])
            M = linear_interpolation(length, wfl['length'][ix_low], wfl['length'][ix_low+1], wfl[gender][ix_low,1], wfl[gender][ix_low+1,1])
            S = linear_interpolation(length, wfl['length'][ix_low], wfl['length'][ix_low+1], wfl[gender][ix_low,2], wfl[gender][ix_low+1,2])

        Z = (((weight / M)**L) - 1) / (S * L)
        if abs(Z) > 5:
            Z = np.sign(Z) * 1.
        return Z

def zscore_bmi(gender, age, bmi):
    """
    Calculates the CDC BMI Z-score from https://www.cdc.gov/nccdphp/dnpao/growthcharts/resources/sas.htm
    where Z = (((value / M)**L) – 1) / (S * L). In addition any Z-score with absolute value greater than 5 is
    forced to sign(Z) * 1
    NOTE: This should only be used for anyone between the ages of 2 and 20.

    #### PARAMETERS ####
    parameters should either be arrays or single items
    gender: 0 for male, 1 for female
    age: age in months between 23.5 and 120
    bmi: bmi
    """
    if all([type(x)==np.ndarray for x in (gender,age,bmi)]):
        bmi = bmi.astype(float)
        zscores = np.zeros(gender.reshape(-1,1).shape[0])
        L = np.zeros(gender.reshape(-1,1).shape[0])
        M = np.zeros(gender.reshape(-1,1).shape[0])
        S = np.zeros(gender.reshape(-1,1).shape[0])
        for ix in range(zscores.shape[0]):
            if age[ix] < 23.5 or age[ix] > 240:
                continue
            if math.fmod(age[ix], 1) == 0.5:
                ix_low = bmi_dic['age'].index(age[ix])
                L[ix] = bmi_dic[gender[ix]][ix_low,0]
                M[ix] = bmi_dic[gender[ix]][ix_low,1]
                S[ix] = bmi_dic[gender[ix]][ix_low,2]
                continue
            elif math.fmod(age[ix], 1) < 0.5:
                ix_low = bmi_dic['age'].index(age[ix] - math.fmod(age[ix], 1) - 0.5)
            else:
                ix_low = bmi_dic['age'].index(age[ix] - math.fmod(age[ix], 1) + 0.5)
            L[ix] = linear_interpolation(age[ix], bmi_dic['age'][ix_low], bmi_dic['age'][ix_low+1], bmi_dic[gender[ix]][ix_low,0], bmi_dic[gender[ix]][ix_low+1,0])
            M[ix] = linear_interpolation(age[ix], bmi_dic['age'][ix_low], bmi_dic['age'][ix_low+1], bmi_dic[gender[ix]][ix_low,1], bmi_dic[gender[ix]][ix_low+1,1])
            S[ix] = linear_interpolation(age[ix], bmi_dic['age'][ix_low], bmi_dic['age'][ix_low+1], bmi_dic[gender[ix]][ix_low,2], bmi_dic[gender[ix]][ix_low+1,2])
        zscores = (((bmi / M)**L) - 1.) / (S * L)
        zscores[(np.abs(zscores) > 5)] = np.sign(zscores[(np.abs(zscores) > 5)])
        return np.nan_to_num(zscores)
    else:
        if age < 23.5 or age > 240:
            return 0
        if math.fmod(age, 1) == 0.5:
            ix_low = bmi_dic['age'].index(age)
            L = bmi_dic[gender][ix_low,0]
            M = bmi_dic[gender][ix_low,1]
            S = bmi_dic[gender][ix_low,2]
            Z = (((bmi / M)**L) - 1) / (S * L)
            if abs(Z) > 5:
                Z = np.sign(Z) * 1.
            return Z
        elif math.fmod(age, 1) < 0.5:
            ix_low = bmi_dic['age'].index(age - math.fmod(age, 1) - 0.5)
        else:
            ix_low = bmi_dic['age'].index(age - math.fmod(age, 1) + 0.5)
        L = linear_interpolation(age, bmi_dic['age'][ix_low], bmi_dic['age'][ix_low+1], bmi_dic[gender][ix_low,0], bmi_dic[gender][ix_low+1,0])
        M = linear_interpolation(age, bmi_dic['age'][ix_low], bmi_dic['age'][ix_low+1], bmi_dic[gender][ix_low,1], bmi_dic[gender][ix_low+1,1])
        S = linear_interpolation(age, bmi_dic['age'][ix_low], bmi_dic['age'][ix_low+1], bmi_dic[gender][ix_low,2], bmi_dic[gender][ix_low+1,2])
        Z = (((bmi / M)**L) - 1) / (S * L)
        if abs(Z) > 5:
            Z = np.sign(Z) * 1.
        return Z
