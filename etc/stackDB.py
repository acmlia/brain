#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 09:52:17 2018

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

#------ 
# Create the list of Dataframes, eliminating the files that start with ".":

frames = []
for file in os.listdir(pathin):
    if file.startswith(".", 0, len(file)): 
         name = os.path.splitext(file)[0]
         print("File name starts with point: ", name)
    else:
        logging.debug(file)
        df = pd.read_csv(os.path.join(pathin, file), sep=',', decimal='.', encoding="utf8")
        df.reset_index(drop=True, inplace=True)
        frames.append(df)
        logging.debug(frames)
        
#------        
# Concatenation of the monthly Dataframes into the yearly Dataframe:
        
try:
    DB_yrly_rain = pd.concat(frames, sort=False, ignore_index=True, verify_integrity=True)
except ValueError as e:
    print("ValueError:", e)     
    
# Repairing the additional column wrongly generated in concatenation:

if np.where(np.isfinite(DB_yrly_rain.iloc[:,34])):
    DB_yrly_rain["correto"]=DB_yrly_rain.iloc[:,34]
else:
    pos=np.where(isnan())
    DB_yrly_rain["correto"]=DB_yrly_rain.iloc[:,33]
    
#DB_yrly_norain = pd.concat(frames)

#------ 
# Giving the output file names:

DB_name="BR_yrly_rain.csv"
#DB_yrly_norain="BR_yrly_norain_.csv"

#------        
# Saving the new output DB's (rain and no rain):

#DB_yrly_rain.to_csv(os.path.join(pathrain, DB_name),index=False,sep=",",decimal='.')
#print("The file ", DB_yrly_rain ," was genetared!")
DB_yrly_rain.to_csv(os.path.join(pathrain, DB_name),index=False,sep=",",decimal='.')
print("The file ", DB_name ," was genetared!")
