import os

import csv
import time
import pickle
import argparse
import censusgeocode as cg

def create_loc_dict(headers, line, location, lat, lon):
    tract = int(location['TRACT'])
    county = int(location['COUNTY'])
    block = int(location['BLOCK'])
    geoid = location['GEOID'] # STATEFIPS (2) + COUNTYFIPS (3) + TRACT (6) + BLOCK (4) = 15 digit 2010 census identifier
    loc_dict = {
        'owner': line[headers['owner']],
        'owner_type': line[headers['owner type']],
        'use': line[headers['known use']],
        'area': line[headers['area acres']],
        'address': line[headers['address line1']],
        'bbl': line[headers['bbl']],
        'zip': line[headers['postal code']],
        'latitude': lat,
        'longitude': lon,
        'block': block,
        'geoid': geoid
    }
    return county, tract, loc_dict

def read_line(headers, i, line):
    """
    Reads the line from the csv and indefinitely tries to get the census geocoder
    to return a valid location. The geocoder seems to arbitrarily send an error
    message. My best guess is this is some weird connection issue.
    """
    lat = float(line[headers['latitude']])
    lon = float(line[headers['longitude']])
    print(i, lat, lon)
    # keep pinging the census geocoder until a valid result is sent
    try:
        location = cg.coordinates(x=lon, y=lat)['2010 Census Blocks'][0]
    except:
        location, lat, lon = read_line(headers, i, line)

    return location, lat, lon

def read_file(fname):
    locations = {}
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                headers = {el:ix for ix, el in enumerate(line)}
            else:
                line = [el.strip() for el in line]
                location, lat, lon  = read_line(headers, i, line)
                county, tract, loc_dict = create_loc_dict(headers, line, location, lat, lon)
                try:
                    locations[tract][county].append(loc_dict)
                except:
                    try:
                        locations[tract][county] = [loc_dict]
                    except:
                        locations[tract] = {county: [loc_dict]}

    return locations

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process ACS files from given folder paths')
    parser.add_argument('-f', '--filename', required=True, type=str, metavar='fname', dest='fname', help='filepath/filename to vacant lots data')
    parser.add_argument('-s', '--savepath', required=False, type=str, metavar='save_path', dest='save_path', help='path to save data', default='.')
    # parser.add_argument('-n', '--nodecount', required=False, type=int, metavar='node_count', dest='node_count', help='number of cpus to use for processing', default=math.ceil(mp.cpu_count() * 0.8))
    args = parser.parse_args()

    if args.save_path == '':
        args.save_path = '/'.join(el for el in args.filename.split('/')[:-1])

    cg.CensusGeocode(benchmark='Public_AR_Census2010', vintage='Census2010_Census2010')
    vacant_lots_dict = read_file(args.fname)
    sp = '/'.join((args.save_path, 'vacant_lots_dict_10092018.pkl'))
    pickle.dump(vacant_lots_dict, open(sp, 'wb'))
    print('vacant lots dictionary saved to {}'.format(sp))
