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
import logging
from PRE_PROCESS_TRAINING.PCA import HelloFunction



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
        self.COLUMN_TYPES = settings.COLUMN_TYPES
        
        
    def LoadCSV(self, path, file):
        '''
        Load CSV files (original)
    
        :param path: sets the csv files path (string)
        :param file: file name or file list (string)
        :return:  dataframe (DataFrame)
        
        '''
        
        if file.startswith(".", 0, len(file)): 
            print("File name starts with point: {} - Skipping...".format(file))
        elif file.endswith(".csv"):
            try:
                dataframe = pd.DataFrame()
                dataframe = pd.read_csv(os.path.join(path, file), sep=',', header=5, skipinitialspace=True, decimal='.', dtype=self.COLUMN_TYPES)
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
        
        
    def ConcatenationMonthlyDF(self, path, dataframe_name):
        '''
        Concatenate the monthly rain and norain dataframes into yearly dataframes.
        
        '''
        
        frames = []
        for idx, file in enumerate(os.listdir(path)):
            if file.startswith(".", 0, len(file)): 
                print("File name starts with point: ", file)
            else:
                logging.debug(file)
                print("posicao do loop: {} | elemento da pasta: {}".format(idx, file))
                df = pd.read_csv(os.path.join(path, file), sep=',', decimal='.', encoding="utf8")
                df.reset_index(drop=True, inplace=True)
                frames.append(df)
                logging.debug(frames)
                
        #------        
        # Concatenation of the monthly Dataframes into the yearly Dataframe:
         
        try:
            dataframe_yrly = pd.concat(frames, sort=False, ignore_index=True, verify_integrity=True)
        except ValueError as e:
            print("ValueError:", e)     
            
        #------
        # Repairing the additional column wrongly generated in concatenation:

        if np.where(np.isfinite(dataframe_yrly.iloc[:,34])):
            dataframe_yrly["correto"]=dataframe_yrly.iloc[:,34]
        else:
            #pos=np.where(isnan())
            dataframe_yrly["correto"]=dataframe_yrly.iloc[:,33]

       
        dataframe_yrly_name=dataframe_name
      
        #------        
        # Saving the new output DB's (rain and no rain):
        dataframe_yrly.to_csv(os.path.join(path, dataframe_yrly_name),index=False,sep=",",decimal='.')
        print("The file ", dataframe_yrly_name ," was genetared!")
        
        return dataframe_yrly
    
    def FitConcatenationDF(self, path, file):
        
        dataframe = pd.read_csv(os.path.join(path, file), sep=',', decimal='.', encoding="utf8")

        pos33=np.where(np.isnan(dataframe.iloc[:,33]))
        val34=dataframe.iloc[:,34].iloc[pos33]
        vec_correto=dataframe.iloc[:,33].fillna(val34)
        dataframe["emis190V_OK"]=""
        dataframe["emis190V_OK"]=vec_correto

        dataframe_copy_OK=dataframe[['lat', 'lon', 'sfccode', 'T2m', 'tcwv', 'skint', 'sfcprcp',
                'cnvprcp', '10V', '10H', '18V', '18H', '23V', '36V', '36H', '89V',
                '89H', '166V', '166H', '186V', '190V', 'emis10V', 'emis10H', 'emis18V',
                'emis18H', 'emis23V', 'emis36V', 'emis36H', 'emis89V', 'emis89H',
                'emis166V', 'emis166H', 'emis186V',]].copy()

        dataframe_copy_OK["emis190V"]=vec_correto
        file_name=os.path.splitext(file)[0]+"_OK.csv"
        dataframe_copy_OK.to_csv(os.path.join(path, file_name),index=False,sep=",",decimal='.')
        print("The file ", file_name ," was genetared!")
        
        return dataframe_copy_OK
        
    
mybrain = BRain()
#dataframe_yrly = mybrain.ConcatenationMonthlyDF(settings.RAIN_CSV, "Yearly_BR_rain_var2d.csv")
#dataframe_OK = mybrain.FitConcatenationDF(settings.RAIN_CSV, "Yearly_BR_rain_var2d.csv")
HelloFunction()
###  Loop for CREATION of the regional and rain and norain dataframes.
# You can change the INPUT/OUTPUT PATH depending on your need:

#------------------------------------------------------------------------------
#for idx, elemento in enumerate(os.listdir(mybrain.IN_CSV_LIST)):
#    print("posicao do loop: {} | elemento da pasta: {}".format(idx, elemento))
#    dataframe_original = mybrain.LoadCSV(mybrain.IN_CSV_LIST, elemento)
#    #-------------------------------------------------------------------------
#    dataframe_regional = mybrain.ExtractRegion(dataframe_original)
#    data=elemento[9:15]
#    dataframe_reg_name="Regional_BR_"+data+"_var2d.csv"
#    dataframe_regional.to_csv(os.path.join(mybrain.OUT_CSV_LIST, dataframe_reg_name),index=False,sep=",",decimal='.')
#     #-------------------------------------------------------------------------
#    dataframe_rain, dataframe_norain = mybrain.ThresholdRainNoRain(dataframe_regional)
#    dataframe_rain_name="Regional_BR_rain_"+data+"_var2d.csv"
#    dataframe_norain_name="Regional_BR_norain_"+data+"_var2d.csv"
#    dataframe_rain.to_csv(os.path.join(mybrain.RAIN_CSV, dataframe_rain_name),index=False,sep=",",decimal='.')
#    dataframe_norain.to_csv(os.path.join(mybrain.NORAIN_CSV, dataframe_norain_name),index=False,sep=",",decimal='.')

#    print("The file ", dataframe_rain ," was genetared!")
#    print("The file ", dataframe_norain ," was genetared!")
#    dataframe_norain.to_csv(os.path.join(pathnorain, norainDB),index=False,sep=",",decimal='.')
#    print("The file ", norainDB ," was genetared!")
#------------------------------------------------------------------------------

###  Loop for CONCATENATION of the rain and norain dataframes in Yearly Dataframes:
# You can change the INPUT/OUTPUT PATH depending on your need:

#------------------------------------------------------------------------------
