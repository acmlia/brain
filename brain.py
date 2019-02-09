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
        self.THRESHOLD_RAIN = settings.THRESHOLD_RAIN
        self.RAIN_CSV = settings.RAIN_CSV
        self.NORAIN_CSV = settings.NORAIN_CSV
        
        
    def LoadCSV(self, path, file):
        '''
        Load CSV files (original)
    
        :param path: sets the csv files path (string)
        :param file: file name or file list (string)
        :return:  dataframe (DataFrame)
        
        '''
        column_types = {'numpixs': 'int64', 'lat': 'float64','lon': 'float64','sfccode': 'float64','T2m': 'float64',
                        'tcwv': 'float64','skint': 'float64','sfcprcp': 'float64','cnvprcp': 'float64',
                        '10V': 'float64','10H': 'float64','18V': 'float64','18H': 'float64',
                        '23V': 'float64','36V': 'float64','36H': 'float64','89V': 'float64',
                        '89H': 'float64','166V': 'float64','166H': 'float64','186V': 'float64',
                        '190V': 'float64','emis10V': 'float64', 'emis10H': 'float64','emis18V': 'float64',
                        'emis18H': 'float64','emis23V': 'float64','emis36V': 'float64','emis36H': 'float64',
                        'emis89V': 'float64','emis89H': 'float64', 'emis166V': 'float64', 'emis166H': 'float64',
                        'emis186V': 'float64','emis190V': 'float64'}
        
        if file.startswith(".", 0, len(file)): 
            print("File name starts with point: {} - Skipping...".format(file))
        elif file.endswith(".csv"):
            try:
                dataframe = pd.DataFrame()
                dataframe = pd.read_csv(os.path.join(path, file), sep=',', header=5, skipinitialspace=True, decimal='.', dtype=column_types)
                print('Dataframe {} was loaded'.format(file))
            except:
                print('Unexpected error:', sys.exc_info()[0])
                
        return dataframe
           
    
    def ExtractRegion(self, dataframe):  
        '''
        Extract regional areas from the global dataset (original)
    
        :param dataframe: original global dataframe (DataFrame)
        :return:  dataframe_regional (DataFrame)
        
        '''
        
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
    
    
    def ThresholdRainNoRain(self, dataframe_regional):
        '''
        Defines the minimum threshold to consider in the Rain Dataset
    
        :param dataframe_regional: the regional dataset with all pixels (rain and no rain)(DataFrame)
        :return:  rain  and norain dataframes (DataFrame)
        
        '''
 
        # Rain/No Rain threshold(th):
        threshold_rain = self.THRESHOLD_RAIN
        rain_pixels = np.where((dataframe_regional['sfcprcp']>=threshold_rain))
        norain_pixels = np.where((dataframe_regional['sfcprcp']<threshold_rain)) 
        
        df_reg_copy = dataframe_regional.copy()
        dataframe_rain = df_reg_copy.iloc[rain_pixels] 
        dataframe_norain = df_reg_copy.iloc[norain_pixels]
        print("Dataframes Rain and NoRain created!")

        return dataframe_rain, dataframe_norain
 

    def PrintSettings(self):
        '''
        Shows the settings of the main parameters necessary to process the algorithm.
        
        '''
        
        print(self.__dict__)


    
mybrain = BRain()
for idx, elemento in enumerate(os.listdir(mybrain.IN_CSV_LIST)):
    print("posicao do loop: {} | elemento da pasta: {}".format(idx, elemento))
    dataframe_original = mybrain.LoadCSV(mybrain.IN_CSV_LIST, elemento)
    #-------------------------------------------------------------------------
    dataframe_regional = mybrain.ExtractRegion(dataframe_original)
    data=elemento[9:15]
    dataframe_reg_name="Regional_BR_"+data+"_var2d.csv"
    dataframe_regional.to_csv(os.path.join(mybrain.OUT_CSV_LIST, dataframe_reg_name),index=False,sep=",",decimal='.')
     #-------------------------------------------------------------------------
    dataframe_rain, dataframe_norain = mybrain.ThresholdRainNoRain(dataframe_regional)
    dataframe_rain_name="Regional_BR_rain_"+data+"_var2d.csv"
    dataframe_norain_name="Regional_BR_norain_"+data+"_var2d.csv"
    dataframe_rain.to_csv(os.path.join(mybrain.RAIN_CSV, dataframe_rain_name),index=False,sep=",",decimal='.')
    dataframe_norain.to_csv(os.path.join(mybrain.NORAIN_CSV, dataframe_norain_name),index=False,sep=",",decimal='.')

#    print("The file ", dataframe_rain ," was genetared!")
#    print("The file ", dataframe_norain ," was genetared!")
#    dataframe_norain.to_csv(os.path.join(pathnorain, norainDB),index=False,sep=",",decimal='.')
#    print("The file ", norainDB ," was genetared!")



