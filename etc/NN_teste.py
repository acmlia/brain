#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:50:01 2019

@author: liaamaral
"""
# ---------------------------
# Loading the main libraries:
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import csv
import os, sys

# --------------------------
# Paths:

path="/Volumes/lia_595gb/randel/python/dados/subsetDB/rain/"
file="BR_yrly_rain_OK.csv" 
df = pd.read_csv(os.path.join(path, file), sep=",", decimal='.')

# --------------------------
# Input variables:

entradas=df[['36V','36H','89V','89H','166V','166H','190V']].values
saidas=df[['sfcprcp']].astype(int)
saidas=np.ravel(saidas)

redeNeural=MLPClassifier(verbose=True, hidden_layer_sizes = (2,), activation='logistic', solver='adam', max_iter=200)
redeNeural.fit(entradas,saidas)
redeNeural.predict([[270, 235, 210, 256]])

redeNeural.score(entradas,saidas)

# ----------------------------

def optimizerNN(inputs,outputs,verbose,hidden_layers_list,activation,solver,max_iter,tol):
    print("inicializando iterador com {} valores de layers".format(len(hidden_layers_list)))
    listavalidacao = []
    for i in hidden_layers_list:
        print("rodada = {}".format(i))
        rodada = MLPClassifier(verbose=verbose, 
                               hidden_layer_sizes = hidden_layers_list[i-1], 
                               activation=activation, 
                               solver=solver, 
                               max_iter=max_iter,
                               tol=tol)
        print("ajustando modelo da iteração #{}".format(i))
        rodada.fit(inputs,outputs)
        print("validando rede da iteração #{}".format(i))
        validacao = rodada.score(inputs,outputs)
        listavalidacao.append(validacao)
    
    return listavalidacao

entradas=df[['36V','36H','89V','89H','166V','166H','190V']].values
saidas=df[['sfcprcp']].astype(int)
saidas=np.ravel(saidas)
listacamadas = [1,2,3]

resultado = optimizerNN(entradas, saidas, "True", listacamadas, "logistic", "adam", 200, 0.0001)