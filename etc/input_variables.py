#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:57:11 2019

@author: liaamaral
"""
# --------------------------
# Import liraries:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import glob
import json
import os, sys
from sklearn.decomposition import PCA

# --------------------------
# Paths:

path="/media/DATA/tmp/datasets/subsetDB/rain/"
file="BR_yrly_rain_OK.csv" 
df = pd.read_csv(os.path.join(path, file), sep=",", decimal='.')

# --------------------------
# Input variables:


df['delta_HF']=df['186V']-df['190V']

df1=df[['10V','10H','18V','18H','23V','36V','36H']]

df2=df[['89V','89H','166V','166H','186V','190V']]

# --------------------------
# How to insert new column in the DataFrame:

idx = 0
new_col = df[rr] # can be a list, a Series, an array or a scalar   
df2.insert(loc=idx, column='rr', value=new_col)

# --------------------------
# Choosing the number of components:

pca_ver = PCA().fit(df2)
plt.plot(np.cumsum(pca_ver.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance');

# --------------------------
# Comparing the data before and after the transformation in components:

pca1 = PCA(n_components=2)
pca1.fit(df1)
df1_transformed = pca1.transform(df1)
print("original shape:   ", df1.shape)
print("transformed shape:", df1_transformed.shape)


pca2 = PCA(n_components=2)
pca2.fit(df2)
df2_transformed = pca2.transform(df2)
print("original shape:   ", df2.shape)
print("transformed shape:", df2_transformed.shape)


#pca.components_
#pca.explained_variance_
#pca.explained_variance_ratio_
#pca.singular_values_
#pca.mean_
#pca.noise_variance_
#pca.components_.shape


# --------------------------
#  Plotting heatmap to see the relation between the compenents and the input variables:
df1_comp1=pd.DataFrame(pca1.components_, columns=df1.columns)

plt.figure(figsize=(12,8))
sns.heatmap(df1_comp1, cmap='plasma')

df2_comp2=pd.DataFrame(pca2.components_, columns=df2.columns)

plt.figure(figsize=(12,8))
sns.heatmap(df2_comp2, cmap='plasma')

# --------------------------
#  Plot the first two principal components of each point to learn about the data:
plt.figure(figsize=(12,8))
plt.scatter(df1_transformed[:, 0], df1_transformed[:, 1],
            c=df['sfcprcp'], edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('PuRd', 10))
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar();

# --------------------------
#  Plot the first two principal components of each point to learn about the data:
plt.figure(figsize=(12,8))
plt.scatter(df2_transformed[:, 0], df2_transformed[:, 1],
            c=df['sfcprcp'], edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('PuRd', 10))
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar();

# --------------------------
# Plot of correlation between the FIRST component and the RAIN:
plt.figure(figsize=(12,8))
plt.scatter(df['sfcprcp'], df1_transformed[:, 0],edgecolor='none', alpha=0.5,
            cmap= 'green')
plt.xlabel('Rain Rate (mm/h)')
plt.ylabel('Component 1')

# --------------------------
# Plot of correlation between the SECOND component and the RAIN:
plt.figure(figsize=(12,8))
plt.scatter(df['sfcprcp'], df1_transformed[:, 1],edgecolor='none', alpha=0.5,
            cmap= 'red')
plt.xlabel('Rain Rate (mm/h)')
plt.ylabel('Component 2')

# --------------------------
# Plot of correlation between the FIRST component and the RAIN:
plt.figure(figsize=(12,8))
plt.scatter(df['sfcprcp'], df2_transformed[:, 0],edgecolor='none', alpha=0.5,
            cmap= 'red')
plt.xlabel('Rain Rate (mm/h)')
plt.ylabel('Component 1')

# --------------------------
# Plot of correlation between the SECOND component and the RAIN:
plt.figure(figsize=(12,8))
plt.scatter(df['sfcprcp'], df2_transformed[:, 1],edgecolor='none', alpha=0.5,
            cmap= 'red')
plt.xlabel('Rain Rate (mm/h)')
plt.ylabel('Component 2')

# --------------------------
# Joing the input vars in ONE dataset:

df_NN = pd.DataFrame()

frames=[]
frames.append(df_pca1)
frames.append(df_pca2)

df_NN=pd.concat(frames, axis=1, sort=False, ignore_index=True, verify_integrity=True)
df_NN = pd.DataFrame()

# How to insert new column in the DataFrame:
idx = 4
new_col = df['delta_HF'] # can be a list, a Series, an array or a scalar   
df_NN.insert(loc=idx, column='delta_HF', value=new_col)

# Putting the different input variables in the same scale:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_NN)
scaled_df_NN = scaler.transform(df_NN)

###############################################################################
###############################################################################
# --------------------------
# Input variables:


x=scaled_df_NN
#x=df[['36V','36H','89V','89H','166V','166H','190V']].values
y=df[['sfcprcp']].astype(int)
y=np.ravel(y)

# --------------------------
# Split in training and test datasets:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

# --------------------------
# Creating the instance for the NN:

redeNeural=MLPClassifier(verbose=True, hidden_layer_sizes = (2,), activation='logistic', solver='adam', max_iter=100)
redeNeural.fit(x_train,y_train)
pred_redeNeural=redeNeural.predict(x_test)

# --------------------------
# Print the classification from metrics from NN:

print(classification_report(y_test, pred_redeNeural))
print('\n')
print(confusion_matrix(y_test, pred_redeNeural))

# --------------------------
# Finding the best parameters by using the GridSearchCV:
parameters = {'solver': ['adam', 'sgd'], 'max_iter': [100,300,500,700,1000], 'activation': ['logistic', 'tanh'], 'hidden_layer_sizes':np.arange(4, 9)}
grid = GridSearchCV(redeNeural, parameters, n_jobs=-1, cv=5, verbose=5)
grid.fit(x_train, y_train)
pred_grid=grid.predict(x_test)

# --------------------------
# Print the classification from metrics from NN:
print(classification_report(y_test, pred_grid))
print('\n')
print(confusion_matrix(y_test, pred_grid))

print("-----------------Features--------------------")
print("Best score: %0.4f" % grid.best_score_)
print("Using the following parameters:")
print(grid.best_params_)
print("---------------------------------------------")

# --------------------------


