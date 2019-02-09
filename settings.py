#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 15:31:35 2019

@author: rainfall
"""
from decouple import config

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

# Identification of dtype for Dataframe in LOadCSV:
COLUMN_TYPES = {'numpixs': 'int64', 'lat': 'float64','lon': 'float64','sfccode': 'float64','T2m': 'float64',
                        'tcwv': 'float64','skint': 'float64','sfcprcp': 'float64','cnvprcp': 'float64',
                        '10V': 'float64','10H': 'float64','18V': 'float64','18H': 'float64',
                        '23V': 'float64','36V': 'float64','36H': 'float64','89V': 'float64',
                        '89H': 'float64','166V': 'float64','166H': 'float64','186V': 'float64',
                        '190V': 'float64','emis10V': 'float64', 'emis10H': 'float64','emis18V': 'float64',
                        'emis18H': 'float64','emis23V': 'float64','emis36V': 'float64','emis36H': 'float64',
                        'emis89V': 'float64','emis89H': 'float64', 'emis166V': 'float64', 'emis166H': 'float64',
                        'emis186V': 'float64','emis190V': 'float64'}

# 