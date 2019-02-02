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
    def __init__(self, infile, file, outfile):
        self.infile = infile
        self.file = file
        self.outfile = outfile    
    
    
    def LoadDataFrame(self):   
        try:
            dataframe = pd.read_csv(os.path.join(self.infile, self.file), sep=",", decimal='.')
        except:
            print('Unexpected error:', sys.exc_info()[0])          
        return dataframe
        
mybrain = BRain(settings.INFILE,
                settings.FILE,
                settings.OUTFILE)

customdf = pd.DataFrame()


