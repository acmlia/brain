#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 16:06:59 2019

@author: rainfall
"""

import settings
import numpy as np
import pandas as pd
import os, sys


class BRain:
    '''
    This is my custom rain classifier embeded with an automatic screening method
    
    :param infile: sets the input file path (string)
    :param file: file name of the input CSV data (string)
    :param outfile: sets the output file of the network (string)
    '''
    def __init__(self):
        self.IN_CSV_LIST = settings.IN_CSV_LIST
        self.OUT_CSV_LIST = settings.OUT_CSV_LIST
        self.LAT_LIMIT = settings.LAT_LIMIT
        self.LON_LIMIT = settings.LON_LIMIT
        
        
    def LoadCSV(self, path, file):   
        if file.startswith(".", 0, len(file)): 
            filename = os.path.splitext(file)[0]
            print("File name starts with point: {} - Skipping...".format(filename))
        elif file.endswith(".csv"):
            try:
                dataframe = pd.read_csv(os.path.join(path, file), sep=",", decimal='.')
            except:
                print('Unexpected error:', sys.exc_info()[0])          
            return dataframe
           
    
    def ExtractRegion(self):
        for file in os.listdir(self.IN_CSV_LIST):
            print("Loading dataframe from file: ", file)
            dataframe = self.LoadCSV(self.IN_CSV_LIST, file)
            print("Extracting region from file '{}' using LAT limits: '{}' and LON limits: '{}'".format(file, self.LAT_LIMIT, self.LON_LIMIT))
            filename = os.path.splitext(file)[0]
            subset = np.where(
                    (dataframe['lat']<=self.LAT_LIMIT[1]) &
                    (dataframe['lat']>=self.LAT_LIMIT[0]) &
                    (dataframe['lon']<=self.LON_LIMIT[1]) &
                    (dataframe['lon']>=self.LON_LIMIT[0]))
            dataframe_regional=dataframe.copy()
            dataframe_regional=dataframe.iloc[subset]
            dataframe_regional.drop(['numpixs'], axis=1, inplace=True)
            subname=filename[9:15]
            fname="Regional_BR_"+subname+"_var2d.csv"
            dataframe_regional.to_csv(os.path.join(self.OUT_CSV_LIST, filename),index=False,sep=",",decimal='.')
            print("The file ", fname ," was genetared!")
        

    def PrintSettings(self):
        print(self.__dict__)


mybrain = BRain()
mybrain.PrintSettings()
mybrain.LoadCSV()
mybrain.ExtractRegion()
# essa é uma mensagem para eu do futuro. Git é muito maneiro!!!


