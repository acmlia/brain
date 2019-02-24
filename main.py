#! /usr/bin/env python3

import logging
import settings
import os, sys
from src.preprocess import Preprocess


def main():
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.DEBUG)

    IN_CSV_LIST = settings.IN_CSV_LIST
    OUT_CSV_LIST = settings.OUT_CSV_LIST
    RAIN_CSV = settings.RAIN_CSV
    NORAIN_CSV = settings.NORAIN_CSV
    LAT_LIMIT = settings.LAT_LIMIT
    LON_LIMIT = settings.LON_LIMIT
    THRESHOLD_RAIN = settings.THRESHOLD_RAIN
    COLUMN_TYPES = settings.COLUMN_TYPES

    # ,---------------------,
    # | Code starts here :) |
    # '---------------------'

    minharede = Preprocess(
        IN_CSV_LIST,
        OUT_CSV_LIST,
        RAIN_CSV,
        NORAIN_CSV,
        LAT_LIMIT,
        LON_LIMIT,
        THRESHOLD_RAIN,
        COLUMN_TYPES)

    # Print the network initial settings
    minharede.print_settings()
    
    ## Loop for CREATION of the regional and rain and norain dataframes.
    # You can change the INPUT/OUTPUT PATH depending on your need:

    # ------------------------------------------------------------------------------
    # for elemento in os.listdir(minharede.IN_CSV_LIST):
    #     print('elemento da pasta: {}'.format(elemento))
    #     dataframe_regional = minharede.LoadCSV(minharede.IN_CSV_LIST, elemento)
    #     data = elemento[9:16]
    #     dataframe_rain, dataframe_norain = minharede.ThresholdRainNoRain(dataframe_regional)
    #     dataframe_rain_name = "br_clip_rain_"+data+"_var2d.csv"
    #     dataframe_norain_name = "br_clip_norain_"+data+"_var2d.csv"
    #     dataframe_rain.to_csv(os.path.join(minharede.RAIN_CSV, dataframe_rain_name), index=False, sep=",", decimal='.')
    #     dataframe_norain.to_csv(os.path.join(minharede.NORAIN_CSV, dataframe_norain_name), index=False, sep=",", decimal='.')
    
    # minharede.ConcatenationMonthlyDF(RAIN_CSV, "yearly_br_rain_var2d.csv")
    # minharede.ConcatenationMonthlyDF(NORAIN_CSV, "yearly_br_norain_var2d.csv")
    minharede.testOGR()

if __name__ == '__main__':
    main()
