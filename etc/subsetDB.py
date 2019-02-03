#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:50:39 2018

@author: liaamaral
"""
# Load the main libraries
import numpy as np
import pandas as pd 
import csv
import glob
import json
import os, sys


# Data input and output paths:
#pathin="/Volumes/lia_595gb/randel/DB_original/ascii/var2d/csv/"  # Path of original dataset in txt and csv formats
#pathout="/Volumes/lia_595gb/randel/python/dados/subsetDB/" # Path of dataset opened  and treated with python (subset)

pathin="/Volumes/lia_595gb/randel/python/dados/subsetDB/"  # Path of original dataset in txt and csv formats
pathout="/Volumes/lia_595gb/randel/python/dados/subsetDB/dezembro/" # Path of dataset opened  and treated with python (subset)


#ddict = {}
for file in os.listdir(pathin):
    if file.endswith(".csv"):
        name = os.path.splitext(file)[0]
        df = pd.read_csv(os.path.join(pathin, file), sep=',', header=5, skipinitialspace=True, decimal='.')
#        values=df2.values
#        lat=values[:,1]
#        lon=values[:,2]
#        t2m=values[:,4]
#        
        latmin=-34.0
        latmax=+6.0
        lonmin=-75.0
        lonmax=-35.0
        
        # Subset of interesting area:
        sub=np.where((df.lat<=latmax) & (df.lat>=latmin) & (df.lon<=lonmax) & (df.lon>=lonmin))
        
        col=[
             'lat', 'lon', 'sfccode', 'T2m', 'tcwv', 'skint', 'sfcprcp',
             'cnvprcp', '10V', '10H', '18V', '18H', '23V', '36V', '36H', '89V',
             '89H', '166V', '166H', '186V', '190V', 'emis10V', 'emis10H', 'emis18V',
             'emis18H', 'emis23V', 'emis36V', 'emis36H', 'emis89V', 'emis89H',
             'emis166V', 'emis166H', 'emis186V', 'emis190V'
             ]
        df2=pd.DataFrame(columns=col)
        df2=df.copy()
        df2=df2.iloc[sub] 
        df2.drop(['numpixs'],axis=1,inplace=True)
        lines=df2.index
       
        subname=name[0:15]
        fname="BR_var2d_"+subname+".csv"
       
        df2.to_csv(os.path.join(pathout, fname),index=False,sep=",",decimal='.')
        print("The file ", fname ," was genetared!")
        
