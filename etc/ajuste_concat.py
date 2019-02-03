#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:26:02 2019

@author: liaamaral
"""

#------ 
# Load the main libraries
import os
import csv
import numpy as np
import pandas as pd
import logging

#------ 
# Data input and output paths:
pathin="/media/DATA/tmp/datasets/subsetDB/rain/" # Path of the rain dataset 
pathrain="/media/DATA/tmp/datasets/subsetDB/rain/" # Path of the rain dataset 
#pathnorain="/Volumes/lia_595gb/randel/python/dados/subsetDB/norain/" # Path of the non rain dataset

file = 'BR_yrly_rain.csv'
df = pd.read_csv(os.path.join(pathin, file), sep=',', decimal='.', encoding="utf8")

pos33=np.where(np.isnan(df.iloc[:,33]))
val34=df.iloc[:,34].iloc[pos33]
vec_correto=df.iloc[:,33].fillna(val34)
df["emis190V_OK"]=""
df["emis190V_OK"]=vec_correto

df2=df[['lat', 'lon', 'sfccode', 'T2m', 'tcwv', 'skint', 'sfcprcp',
     'cnvprcp', '10V', '10H', '18V', '18H', '23V', '36V', '36H', '89V',
     '89H', '166V', '166H', '186V', '190V', 'emis10V', 'emis10H', 'emis18V',
     'emis18H', 'emis23V', 'emis36V', 'emis36H', 'emis89V', 'emis89H',
     'emis166V', 'emis166H', 'emis186V',]].copy()

df2["emis190V"]=vec_correto
fname = 'BR_yrly_rain_OK.csv'

df2.to_csv(os.path.join(pathrain, fname),index=False,sep=",",decimal='.')
print("The file ", fname ," was genetared!")