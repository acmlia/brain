#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 23:04:36 2019

@author: rainfall
"""

from __future__ import absolute_import, division, print_function

import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
from src.meteoro_skills import CategoricalMetrics

import tensorflow as tf
from tensorflow import keras
from keras import backend
from tensorflow.keras import layers
from keras.layers import GaussianNoise
from keras.layers import GaussianDropout
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import model_from_yaml

print('TF version '+tf.__version__)

# ------------------------------------------------------------------------------

def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

# ------------------------------------------------------------------------------

class Prediction:
    """
    This module is intended to automate the TensorFlow Neural Network training.
    """
    PCA = PCA()
    seed = 0
    run_prefix = ''
    tver = ''
    vernick = ''
    file = ''
    path = ''
    fig_title = ''
    path_fig = ''
    mod_out_pth = ''
    mod_out_name = ''
    ymlv = ''
    ymlp = ''
    ymlf = ''


    def __init__(self, random_seed=0,
                 run_prefix='',
                 version='',
                 version_nickname='',
                 file_csv='',
                 path_csv='',
                 fig_title='',
                 figure_path='',
                 model_out_path='',
                 model_out_name='',
                 yaml_version='',
                 yaml_path='',
                 yaml_file=''):

        self.seed=random_seed
        self.run_prefix=run_prefix
        self.tver=version
        self.vernick=version_nickname
        self.file=file_csv
        self.path=path_csv
        self.path_fig=figure_path
        self.fig_title=run_prefix+version+version_nickname
        self.mod_out_pth=model_out_path
        self.mod_out_name=model_out_name
        self.ymlv=yaml_version
        self.ymlp=yaml_path
        self.ymlf=yaml_file
    # -------------------------------------------------------------------------
    # DROP DATA OUTSIDE INTERVAL
    # -------------------------------------------------------------------------
       
    @staticmethod
    def keep_interval(keepfrom: 0.0, keepto: 1.0, dataframe, target_col: str):
        keepinterval = np.where((dataframe[target_col] >= keepfrom) &
                                (dataframe[target_col] <= keepto))
        result = dataframe.iloc[keepinterval]
        return result

        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------ 

    def PredictScreening(self):

        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------ 
        ## load YAML and create model
        yaml_file = open(self.ymlp+'screening_'+self.ymlv+'.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        loaded_model.load_weights(self.ymlp+'screening_'+self.ymlv+'.h5')
        print("Loaded models yaml and h5 from disk!")
        #------------------------------------------------------------------------------
        #------------------------------------------------------------------------------
        
        # Fix random seed for reproducibility:
        np.random.seed(self.seed)

        # Load dataset:
        df = pd.read_csv(os.path.join(self.path, self.file), sep=',', decimal='.')
        x, y= df.loc[:,['36V', '89V', '166V', '190V']], df.loc[:,['TagRain']]
        
        x_arr = np.asanyarray(x)
        y_arr = np.asanyarray(y)
        y_true = np.ravel(y_arr)

        # Scaling the input paramaters:
#       scaler_min_max = MinMaxScaler()
        norm_sc = Normalizer()
        x_normalized= norm_sc.fit_transform(x_arr)

        # Split the dataset in test and train samples:
#        x_train, x_test, y_train, y_test = train_test_split(x_normalized,
#                                                            y_arr, test_size=0.10,
#                                                            random_state=101)

        # Doing prediction from the test dataset:
        y_pred = loaded_model.predict(x_normalized)
        y_pred = np.ravel(y_pred)

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # Appplying meteorological skills to verify the performance of the model, in this case, categorical scores:

        skills = CategoricalMetrics()
        print('>>>> DEBUG >>>>', y_true,'\n',y_pred)
        val_accuracy, val_bias, val_pod, val_pofd, val_far, val_csi, val_ph, val_ets, val_hss, val_hkd = skills.metrics(y_true, y_pred)
        
        #converting to text file
        print("converting arrays to text files")
        np.savetxt('file_numpy.txt', zip(val_accuracy, val_bias, val_pod,
                                         val_pofd, val_far, val_csi, val_ph,
                                         val_ets, val_hss, val_hkd),
                                         fmt="%5.2f")
        print("Text file saved!")

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        df['SCR'] = ""
        df['SCR'] = y_pred
        filename=self.file[22:58]
        filename = 'validation_TagRain_SCR_'+filename+'.csv'
        df.to_csv(os.path.join(self.path, filename), index=False, sep=",", decimal='.')

        return df

#    def PredictRetrieval():
#        
#    return