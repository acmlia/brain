#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:10:25 2018

@author: liaamaral
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import glob
import json
import os, sys
get_ipython().run_line_magic('matplotlib', 'inline')

pathin="/Volumes/lia_595gb/randel/python/dados/recorte/"
pathout="/Volumes/lia_595gb/randel/python/dados/recorte/figures/"

pathin


for file in os.listdir(pathin):
    if file.endswith(".csv"):
        name = os.path.splitext(file)[0]
        df = pd.read_csv(os.path.join(pathin, file), sep=',', decimal='.')

df.head()

cols=df.columns

cols

tch=cols[8:21]
rr=cols[6]
lch=cols[8:15]
hch=cols[15:21]

tch


TB=df[tch]

TB.head()


df2= df[tch].copy() 


df2.head()


# How to insert new column in the DataFrame:

idx = 0
new_col = df[rr] # can be a list, a Series, an array or a scalar   
df2.insert(loc=idx, column='rr', value=new_col)


df2.head()

from sklearn.decomposition import PCA


# Choosing the number of components:

pca_ver = PCA().fit(TB)
plt.plot(np.cumsum(pca_ver.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance');

# Comparing the data before and after the transformation in components:

pca = PCA(n_components=2)
pca.fit(TB)
TB_transformed = pca.transform(TB)
print("original shape:   ", TB.shape)
print("transformed shape:", TB_transformed.shape)

#  Plot the first two principal components of each point to learn about the data:
plt.figure(figsize=(12,8))
plt.scatter(TB_transformed[:, 0], TB_transformed[:, 1],
            c=df[rr], edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('YlGn', 10))
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar();

pca.components_

pca.explained_variance_

pca.explained_variance_ratio_

pca.singular_values_

pca.mean_

pca.noise_variance_

pca.components_.shape


tch


df_comp1=pd.DataFrame(pca.components_, columns=tch)


df_comp1

plt.figure(figsize=(12,8))
sns.heatmap(df_comp1, cmap='plasma')

TB_transformed.shape

print('NumPy covariance matrix: \n%s' %np.cov(TB_transformed.T))

# Plot of correlation between the first component and the rain:
plt.figure(figsize=(12,8))
plt.scatter(df[rr], TB_transformed[:, 0],edgecolor='none', alpha=0.5,
            cmap= 'blue')
plt.xlabel('Rain Rate (mm/h)')
plt.ylabel('Component 1')

# Plot of correlation between the first component and the rain:
plt.figure(figsize=(12,8))
plt.scatter(df[rr], TB_transformed[:, 1], c='red', edgecolor='none', alpha=0.5)
plt.xlabel('Rain Rate (mm/h)')
plt.ylabel('Component 2')

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(TB)

scaled_TB = scaler.transform(TB)

scaled_TB

pca.fit(scaled_TB)

TB_sc_transformed=pca.transform(scaled_TB)
TB_sc_transformed.shape

# Plot of correlation between the first component and the rain:
plt.figure(figsize=(12,8))
plt.scatter(df[rr], TB_sc_transformed[:, 0],edgecolor='none', alpha=0.5,
            cmap= 'green')
plt.xlabel('Rain Rate (mm/h)')
plt.ylabel('Component 1')


# Plot of correlation between the first component and the rain:
plt.figure(figsize=(12,8))
plt.scatter(df[rr], TB_sc_transformed[:, 1], c='red', edgecolor='none', alpha=0.5)
plt.xlabel('Rain Rate (mm/h)')
plt.ylabel('Component 2')


#  Plot the first two principal components of each point to learn about the data:
plt.figure(figsize=(12,8))
plt.scatter(TB_sc_transformed[:, 0], TB_sc_transformed[:, 1],
            c=df[rr], edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('YlGn', 10))
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar();


pca4 = PCA(n_components=4)
pca4.fit(TB)
TB_transformed4 = pca4.transform(TB)
print("original shape:   ", TB.shape)
print("transformed shape:", TB_transformed4.shape)

#  Plot the first two principal components of each point to learn about the data:
plt.figure(figsize=(12,8))
plt.scatter(TB_transformed4[:, 0], TB_transformed4[:, 1],
            c=df[rr], edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('YlGn', 10))
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('PCA n_components = 4')
plt.colorbar();

pca4.components_

pca4.explained_variance_

pca4.explained_variance_ratio_





