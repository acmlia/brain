#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 15:31:35 2019

@author: rainfall
"""

IN_CSV_LIST = '/media/DATA/tmp/datasets/original_global'
OUT_CSV_LIST = '/media/DATA/tmp/datasets/regional' 

RAIN_CSV = '/media/DATA/tmp/datasets/regional/rain'
NORAIN_CSV = '/media/DATA/tmp/datasets/regional/norain'

# Geographical coordinates for regional subset:
# The parameters follow the sctructure: latlim=[min, max] and lonlim=[min, max]
LAT_LIMIT=[-34.0, 6.0]
LON_LIMIT=[-75.0, -35.0]

# Minimal threshold of rain rate:
THRESHOLD_RAIN=0.1