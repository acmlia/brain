#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:09:37 2019

@author: dvdgmf
"""

from __future__ import absolute_import, division, print_function
import os
import time
import glob
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import GaussianNoise
from keras.layers import GaussianDropout
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml

################################################################################
wpath = '/mnt/AC9AF51E9AF4E5AC/A1ML/phd_datasets/'
ymalf = 'redes_finais/final_reg_R1.yaml'
h5f = 'redes_finais/final_reg_R1.h5'
dados = 'teste_validation_195_SCR.csv'
## load YAML and create model
yaml_file = open(os.path.join(wpath, ymalf), 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights(os.path.join(wpath, h5f))
# ------------------------------------------------------------------------------

df_pred = pd.read_csv(os.path.join(wpath, dados), sep=',', decimal='.')



# ------------------------------------------------------------------------------
df_input = df_pred.loc[:, ['10V', '10H', '18V', '18H', '36V', '36H', '89V', '89H',
                           '166V', '166H', '183VH', 'sfccode', 'temp2m', 'tcwv', 'PCT36', 'PCT89', '89VH',
                           'lat_s1']]

colunas = ['10V', '10H', '18V', '18H', '36V', '36H', '89V', '89H',
           '166V', '166H', '183VH', 'sfccode', 'temp2m', 'tcwv', 'PCT36', 'PCT89', '89VH', 'lat_s1']
scaler = StandardScaler()
normed_input = scaler.fit_transform(df_input)
df_normed_input = pd.DataFrame(normed_input[:],
                               columns=colunas)
ancillary = df_normed_input.loc[:, ['183VH', 'sfccode', 'temp2m', 'tcwv', 'PCT36', 'PCT89', '89VH', 'lat_s1']]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Choosing the number of components:

TB1 = df_normed_input.loc[:, ['10V', '10H', '18V', '18H']]
TB2 = df_normed_input.loc[:, ['36V', '36H', '89V', '89H', '166V', '166H']]

# ------------------------------------------------------------------------------
# Verifying the number of components that most contribute:
pca = PCA()
pca1 = pca.fit(TB1)
# ---
pca_trans1 = PCA(n_components=2)
pca1 = pca_trans1.fit(TB1)
TB1_transformed = pca_trans1.transform(TB1)
print("original shape:   ", TB1.shape)
print("transformed shape:", TB1_transformed.shape)
# ------------------------------------------------------------------------------
pca = PCA()
pca2 = pca.fit(TB2)
pca_trans2 = PCA(n_components=2)
pca2 = pca_trans2.fit(TB2)
TB2_transformed = pca_trans2.transform(TB2)
print("original shape:   ", TB2.shape)
print("transformed shape:", TB2_transformed.shape)
# ------------------------------------------------------------------------------
# JOIN THE TREATED VARIABLES IN ONE SINGLE DATASET AGAIN:

PCA1 = pd.DataFrame()

PCA1 = pd.DataFrame(TB1_transformed[:],
                    columns=['pca1_1', 'pca_2'])
PCA2 = pd.DataFrame(TB2_transformed[:],
                    columns=['pca2_1', 'pca2_2'])

dataset = PCA1.join(PCA2, how='right')
dataset = dataset.join(ancillary, how='right')
dataset = dataset.join(df_pred.loc[:, ['sfcprcp_s1']], how='right')
dataset = dataset.join(df_pred.loc[:, ['SCR']], how='right')

dataset = dataset[dataset['sfcprcp_s1'] != -9999.0]
dataset = dataset[dataset['sfcprcp_s1'] > 0.1]

SCR = dataset.pop('SCR')
y_true = dataset.pop('sfcprcp_s1')

x_normed = dataset.values
y_pred = loaded_model.predict(x_normed).flatten()

# ------------------------------------------------------------------------------
#plt.figure()
#plt.scatter(y_true, y_pred)
#plt.xlabel('True Values [sfcprcp]')
#plt.ylabel('Predictions [sfcprcp]')
#plt.axis('equal')
#plt.axis('square')
#plt.xlim([0, plt.xlim()[1]])
#plt.ylim([0, plt.ylim()[1]])
#plt.plot([-100, 100], [-100, 100])
#fig_name = "test-stuff-delete-please/TESTE.png"
#plt.savefig(wpath + fig_name)
# ------------------------------------------------------------------------------


plt.hist2d(y_true, y_pred, cmin=1, bins=(50, 50), cmap=plt.cm.jet, range=np.array([(0, 60), (0, 60)]))
plt.axis('equal')
plt.axis('square')
plt.plot([0, 100], [0, 100], ls="--", c=".3")
plt.xlim([0, max(y_true)])
plt.ylim([0, max(y_true)])
plt.colorbar()



