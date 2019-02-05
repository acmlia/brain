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
            print("File name starts with point: {} - Skipping...".format(file))
        elif file.endswith(".csv"):
            try:
#               dataframe = pd.DataFrame()
                dataframe = pd.read_csv(os.path.join(path, file), sep=',', header=5, skipinitialspace=True, decimal='.')
                print('Dataframe {} was loaded'.format(file))
            except:
                print('Unexpected error:', sys.exc_info()[0])
                
        return dataframe
           
    
    def ExtractRegion(self, dataframe):  
        print("Extracting region from dataframe using LAT limits: '{}' and LON limits: '{}'".format(
                self.LAT_LIMIT, 
                self.LON_LIMIT))
        
        subset = np.where(
                (dataframe['lat']<=self.LAT_LIMIT[1]) & 
                (dataframe['lat']>=self.LAT_LIMIT[0]) &
                (dataframe['lon']<=self.LON_LIMIT[1]) &
                (dataframe['lon']>=self.LON_LIMIT[0]))
        
        dataframe_regional=dataframe.copy()
        dataframe_regional=dataframe.iloc[subset]
        dataframe_regional.drop(['numpixs'], axis=1, inplace=True)
        print("Extraction completed!")
        return dataframe_regional
        

    def PrintSettings(self):
        print(self.__dict__)


    
mybrain = BRain()
meudataframe = mybrain.LoadCSV('/media/DATA/tmp/datasets/test/', 'CSU.LSWG.201410.bin.csv')
meudataframerecortado = mybrain.ExtractRegion(meudataframe)


