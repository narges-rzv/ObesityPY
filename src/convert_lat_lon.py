import pickle
import numpy as np
import pandas as pd

def main():
    df = pd.read_csv('/Volumes/CPO/Environment data/DB_Geocoding_Updated_wBlocks_Cleaned_wgs84.csv')

    lat_lon_dic = {}
    dfv = df.values
    cols = df.columns.tolist()
    ix_mrn = cols.index('mrn')
    ix_block = cols.index('BLOCKCE10')
    ix_tract = cols.index('TRACTCE10')
    ix_city = cols.index('city')
    ix_county = cols.index('COUNTYFP10')
    ix_xc = cols.index('XCoordinate')
    ix_yc = cols.index('YCoordinate')
    ix_x = cols.index('X')
    ix_y = cols.index('Y')
    ix_zip = cols.index('zip')

    for ix in range(df.shape[0]):
        try:
            mrn = int(dfv[ix,ix_mrn])
        except:
            mrn = dfv[ix,ix_mrn]
        if mrn in lat_lon_dic.keys():
            raise ValueError('patient '+str(mrn)+' already exists in dictionary')

        lat_lon_dic[mrn] = {}
        lat_lon_dic[mrn]['censblock'] = dfv[ix,ix_block]
        if str(dfv[ix,ix_tract])[-2:] == '00' and len(str(dfv[ix,ix_tract])) > 3:
            lat_lon_dic[mrn]['centrac'] = int(str(dfv[ix,ix_tract])[:-2])
        else:
            lat_lon_dic[mrn]['centrac'] = dfv[ix,ix_tract]
        lat_lon_dic[mrn]['city'] = dfv[ix,ix_city]
        lat_lon_dic[mrn]['county'] = dfv[ix,ix_county]
        lat_lon_dic[mrn]['easting'] = dfv[ix,ix_xc]
        lat_lon_dic[mrn]['northing'] = dfv[ix,ix_yc]
        lat_lon_dic[mrn]['lat'] = dfv[ix,ix_y]
        lat_lon_dic[mrn]['lon'] = dfv[ix,ix_x]
        lat_lon_dic[mrn]['zip'] = dfv[ix,ix_zip]

    pickle.dump(lat_lon_dic, open('/Volumes/CPO/ObesityPY/python objects/lat_lon_data_20180305.pkl', 'wb'))

if __name__ == '__main__':
    main()
