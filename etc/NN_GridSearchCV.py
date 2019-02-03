#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:57:11 2019

@author: liaamaral
"""
# --------------------------
# Import liraries:

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler   
import numpy as np
import pandas as pd
import csv
import os, sys

# --------------------------
# Paths:

path="/media/DATA/tmp/datasets/subsetDB/rain/"
file="BR_yrly_rain.csv" 
df = pd.read_csv(os.path.join(path, file), sep=",", decimal='.')

# --------------------------
# Input variables:

TBl=df[['10V','10H','18V','18H','36V','36H']].values
TBh=df[['89V','89H','166V','166H','186','190V']].values
#x=df[['36V','36H','89V','89H','166V','166H','190V']].values
y=df[['sfcprcp']].astype(int)
y=np.ravel(y)


# --------------------------
# PRE - PROCESSING:

scaler = StandardScaler()
scaler.fit()

# --------------------------
# Split in training and test datasets:

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)

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
print(classification_report(y_test, pred_redeNeural))
print('\n')
print(confusion_matrix(y_test, pred_redeNeural))

print("-----------------Features--------------------")
print("Best score: %0.4f" % grid.best_score_)
print("Using the following parameters:")
print(grid.best_params_)
print("---------------------------------------------")
