#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:37:24 2018

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

pathin="/media/DATA/tmp/datasets/subsetDB/"
pathrain="/media/DATA/tmp/datasets/subsetDB/rain/" # Path of the rain dataset rain
pathnorain="/media/DATA/tmp/datasets/subsetDB/norain/" # Path of the no rain dataset

#ddict = {}
for file in os.listdir(pathin):
    #if file.endswith(".csv"):
     if file.startswith(".", 0, len(file)): 
         name = os.path.splitext(file)[0]
         print("File name starts with point: ", name)
     else:
         name = os.path.splitext(file)[0]
         df = pd.read_csv(os.path.join(pathin, file), sep=',', decimal='.', encoding="utf8")

         # Verification of DB area:
         print("The min latitude ", df['lat'].min()  ," and the max latitude", df['lat'].max())
         print("The min longitude ", df['lon'].min() ," and the max longitude", df['lon'].max())

         # Rain/No Rain threshold(th) = 0.5 (test):
         th_rr = 1.0
         rain = np.where((df['sfcprcp']>=th_rr))
         norain = np.where((df['sfcprcp']<th_rr)) 
        
         # Creation of the output DB: 
         df2 = df.copy()
         df_rain = df2.iloc[rain] 
         df_norain = df2.iloc[norain] 
        
         # Giving the output file names:
         subname=name[18:24]
         rainDB="BR_v2d_rain_"+subname+".csv"
         norainDB="BR_v2d_norain_"+subname+".csv"
        
         # Saving the new output DB's (rain and no rain):
         df_rain.to_csv(os.path.join(pathrain, rainDB),index=False,sep=",",decimal='.')
         print("The file ", rainDB ," was genetared!")
         df_norain.to_csv(os.path.join(pathnorain, norainDB),index=False,sep=",",decimal='.')
         print("The file ", norainDB ," was genetared!")
