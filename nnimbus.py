#! /usr/bin/env python3

# ,----------------,
# | PYTHON IMPORTS |----------------------------------------------------------------------------------------------------
# '----------------'

import logging
import settings
import os, sys
import pandas as pd
import numpy as np

from decouple import config
from core.pre_process import PreProcess

# ,----------------------,
# | ENVIRONMENT SETTINGS |----------------------------------------------------------------------------------------------
# '----------------------'

# Path for the CSV's (input and output):
IN_CSV_LIST = config('IN_CSV_LIST', default='')
OUT_CSV_LIST = config('OUT_CSV_LIST', default='')

RAIN_CSV = config('RAIN_CSV', default='')
NORAIN_CSV = config('NORAIN_CSV', default='')

# Geographical coordinates for regional subset:
# The parameters follow the sctructure: latlim=[min, max] and lonlim=[min, max]
LAT_LIMIT=[-34.0, 6.0]
LON_LIMIT=[-75.0, -35.0]

# Minimal threshold of rain rate:
THRESHOLD_RAIN=0.1

# Identification of dtype for Dataframe in LoadCSV:
COLUMN_TYPES = {'numpixs': 'int64', 'lat': 'float64','lon': 'float64','sfccode': 'float64','T2m': 'float64',
                        'tcwv': 'float64','skint': 'float64','sfcprcp': 'float64','cnvprcp': 'float64',
                        '10V': 'float64','10H': 'float64','18V': 'float64','18H': 'float64',
                        '23V': 'float64','36V': 'float64','36H': 'float64','89V': 'float64',
                        '89H': 'float64','166V': 'float64','166H': 'float64','186V': 'float64',
                        '190V': 'float64','emis10V': 'float64', 'emis10H': 'float64','emis18V': 'float64',
                        'emis18H': 'float64','emis23V': 'float64','emis36V': 'float64','emis36H': 'float64',
                        'emis89V': 'float64','emis89H': 'float64', 'emis166V': 'float64', 'emis166H': 'float64',
                        'emis186V': 'float64','emis190V': 'float64'}

# ,------------------,
# | NNIMBUS SETTINGS |--------------------------------------------------------------------------------------------------
# '------------------'

pre_process = True
training = False
post_process = False
prediction = False
validation = False

# Setup timer function
def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))

# ,-----------,
# | RUN MODEL |---------------------------------------------------------------------------------------------------------
# '-----------'


def main():
    # setup information logs
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.DEBUG)

    logging.info(f'Starting NNIMBUS @ {}')
    pass

if __name__ == '__main__':
    main()
