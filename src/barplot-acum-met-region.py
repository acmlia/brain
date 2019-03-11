#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 23:52:37 2019

@author: dvdgmf
"""
import pandas as pd
import os

path = '/mnt/AC9AF51E9AF4E5AC/AMIGOS/mozao/CSV-meteo-regions/'

metreg = ['R1','R2','R3','R4','R5']

legenda = ['SEP',
           'OCT',
           'NOV',
           'DEC',
           'JAN',
           'FEB',
           'MAR',
           'APR',
           'MAY',
           'JUN',
           'JUL',
           'AUG']

def acumuladores(dataframes):
    acumulados = []
    for df in dataframes:
        if df['sfcprcp'].sum() > 0:
            acumulados.append(df['sfcprcp'].sum()/30)
        else:
            acumulados.append(0)
    return acumulados


for reg in metreg:
    todo_region_frames = []
    for idx, file in enumerate(os.listdir(path+reg)):
        #title = file.split('_')[1]+' '+file.split('_')[3]
        if file.endswith(".csv"):
            print("Reading file: ", file)
            df = pd.read_csv(os.path.join(path, reg, file), sep=',', decimal='.', encoding="utf8")
            df.reset_index(drop=True, inplace=True)
            todo_region_frames.append(df)
    acm_df = acumuladores(todo_region_frames)
    plotable_df = pd.DataFrame({'acumulado':acm_df,'legenda':legenda})
    ax = plotable_df.plot.bar(x='legenda', y='acumulado', rot=0)
    ax.set_ylabel("Precipitation (mm)")
    ax.set_xlabel("Months from 09/2014 to 08/2015")
    ax.set_title(reg)
    fig = ax.get_figure()
    fig.savefig('/mnt/AC9AF51E9AF4E5AC/AMIGOS/mozao/'
                'script-histogramas-R1-R5/plots/'+reg+'.pdf')
