#! /usr/bin/env python3

import logging
import settings
import os, sys
import pandas as pd
import numpy as np

from src.training import Training
from src.validation import Validation
from src.graphics_builder import GraphicsBuilder
from src.preprocess import Preprocess
from src.pretraining import PreTraining


def main() -> object:
    """
    :rtype: object
    """
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.DEBUG)

    # IN_CSV_LIST = settings.IN_CSV_LIST
    # OUT_CSV_LIST = settings.OUT_CSV_LIST
    # RAIN_CSV = settings.RAIN_CSV
    # NORAIN_CSV = settings.NORAIN_CSV
    # LAT_LIMIT = settings.LAT_LIMIT
    # LON_LIMIT = settings.LON_LIMIT
    # THRESHOLD_RAIN = settings.THRESHOLD_RAIN
    # COLUMN_TYPES = settings.COLUMN_TYPES

    # ,---------------------,
    # | Code starts here :) |
    # '---------------------'

    git_dir = '/media/DATA/tmp/git-repositories/'

#    Retrieval = Training(random_seed=7,
#                   run_prefix='tf_regr_',
#                   version='R9',
#                   version_nickname='_underc1_hot_0956_',
#                   csv_entry='yearly_br_underc1_hot_0956.csv',
#                   csv_path='/home/david//DATA/tmp/datasets/brazil/brazil_qgis/csv/',
#                   figure_path=git_dir+'/redes_finais/figures/',
#                   model_out_path=git_dir+'/redes_finais/output_models/',
#                   model_out_name='tf_reg_R9')

    Screening = Training(random_seed=7,
                   csv_entry='yearly_br_underc1_hot_0956.csv',
                   csv_path='/home/david/DATA/',
                   model_out_path=git_dir+'/redes_finais/screening/',
                   model_out_name='screening_final')

#------------------------------------------------------------------------------
# VALIDATION CONFIGURATIONS:
#------------------------------------------------------------------------------
#    hdf5 = Validation(path_hdf5='/media/DATA/tmp/git-repositories/validation/HDF5/20181123/')

#    vldar0 = Validation(yaml_version='R9',
#                        yaml_path=git_dir+'redes_finais/output_models/',
#                        path_csv='/media/DATA/tmp/git-repositories/redes_finais/validation/csv/',
#                        file_csv='teste_manual_195_SCR.csv',
#                        path_fig=git_dir+'redes_finais/validation/figures_vld/')
#------------------------------------------------------------------------------
# CALL THE PROGRAMS:
#------------------------------------------------------------------------------
    Screening.autoExecClass()
#    vldar0.autoVld()
#    hdf5.read_hdf5_1CGMI()
#    hdf5.read_hdf5_2AGPROF()
#    hdf5.read_hdf5_2BCMB()

    # rede_pp = Preprocess(
    #     IN_CSV_LIST,
    #     OUT_CSV_LIST,
    #     RAIN_CSV,
    #     NORAIN_CSV,
    #     LAT_LIMIT,
    #     LON_LIMIT,
    #     THRESHOLD_RAIN,
    #     COLUMN_TYPES)
    #
    # rede_pt = PreTraining(IN_CSV_LIST,
    #                       OUT_CSV_LIST,
    #                       RAIN_CSV,
    #                       NORAIN_CSV,
    #                       COLUMN_TYPES)

    # Print the network initial settings
    # minharede.print_settings()

    # Loop for CREATION of the regional and rain and norain dataframes.
    # You can change the INPUT/OUTPUT PATH depending on your need:

    # ------------------------------------------------------------------------------
    #     dataframe = rede_pt.AdditionalInputVariables(rede_pt.IN_CSV_LIST, elemento)

    #     dataframe_regional = minharede.LoadCSV(minharede.IN_CSV_LIST, elemento)
    #     data = elemento[9:16]
    #     dataframe_rain, dataframe_norain = minharede.ThresholdRainNoRain(dataframe_regional)
    #     dataframe_rain_name = "br_clip_rain_"+data+"_var2d.csv"
    #     dataframe_norain_name = "br_clip_norain_"+data+"_var2d.csv"
    #     dataframe_rain.to_csv(os.path.join(minharede.RAIN_CSV, dataframe_rain_name), index=False, sep=",", decimal='.')
    #     dataframe_norain.to_csv(os.path.join(minharede.NORAIN_CSV, dataframe_norain_name), index=False, sep=",", decimal='.')

    # df = rede_pp.ConcatenationMonthlyDF(IN_CSV_LIST, "yearly_clip_br_var2d.csv")
    # df_OK = rede_pt.AdditionalInputVariables(IN_CSV_LIST, "yearly_clip_br_var2d.csv")
    # Loop for graphics generation:
    # for elemento in os.listdir(rede_pt.IN_CSV_LIST):
    #     print('elemento da pasta: {}'.format(elemento))
    #     df = rede_pp.LoadCSV(rede_pp.IN_CSV_LIST, elemento)
    #     month = elemento[9:16]
    #     region = elemento[22:24]
    #     filename = 'scatter_'+month+'_'+region+'.hmtl'
    #     gb = GraphicsBuilder()
    #     gb.scatter_plotter(dataframe=df, xvalue='sfcprcp', xtitle='Precipitation (mm/h)', yvalue='PCT89',
    #                             ytitle='PCT89 (K)', chart_title='Scatter Precipitation x PCT89 - '+month+' - '+region,
    #                             output_file_name=os.path.join(OUT_CSV_LIST, filename))

       # minharede.ConcatenationMonthlyDF(NORAIN_CSV, "yearly_br_norain_var2d.csv")
    # rede_pp.read_shapefile(path_file='/media/DATA/tmp/datasets/regional/qgis/regional_2014-09_clip.shp')
    # ------------------------------------------------------------------------------
    # df = rede_pp.LoadCSV(rede_pp.IN_CSV_LIST, 'yearly_br_rain_var2d_OK.csv')
    # df_final = rede_pp.TagRainNoRain(df)
    # df_name = 'yearly_br_rain_var2d_OK_tag.csv'
    # path = rede_pp.OUT_CSV_LIST
    # df_final.to_csv(os.path.join(path, df_name), index=False, sep=",", decimal='.')
    # print("The file ", df_name, " was saved!")

    # ------------------------------------------------------------------------------
    # Append regional df's:
    #    regional_frames = []
    #    path = rede_pp.IN_CSV_LIST+'yearly/'
    #    for file in os.listdir(path):
    #        if os.path.isfile(path+file):
    #            df = rede_pp.LoadCSV(path, file)
    #            regional_frames.append(df)
    #    print(len(regional_frames))
    #    gb = GraphicsBuilder()
    #    gb.multiple_scatter(regional_frames)


    # ------------------------------------------------------------------------------
    # Loop for graphics generation:
    # for elemento in os.listdir(rede_pt.IN_CSV_LIST):
    #     print('elemento da pasta: {}'.format(elemento))
    #     df = rede_pp.LoadCSV(rede_pp.IN_CSV_LIST, elemento)
    #     month = elemento[9:16]
    #     region = elemento[22:24]
    #     filename = 'scatter_'+month+'_'+region+'.hmtl'
    #     gb = GraphicsBuilder()
    #     gb.scatter_plotter(dataframe=df, xvalue='sfcprcp', xtitle='Precipitation (mm/h)', yvalue='PCT89',
    #                             ytitle='PCT89 (K)', chart_title='Scatter Precipitation x PCT89 - '+month+' - '+region,
    #                             output_file_name=os.path.join(OUT_CSV_LIST, filename))
     # ------------------------------------------------------------------------------
    # Plotting Box Plot by month and region:
    # classe = ['C1', 'C2', 'C3', 'C4']
    # for c in classe:
    #     path = rede_pp.IN_CSV_LIST
    #     for file in os.listdir(path):
    #         if os.path.isfile(path+file):
    #             print('elemento da pasta: {}'.format(file))
    #             df = rede_pp.LoadCSV(path, file)
    #             idx_pxl_classe, size_pxl_classe = rede_pp.SelectionByClasse(df, c)
    #             #df_rain, df_norain, size_norain, size_rain = rede_pp.ThresholdRainNoRain(df)
    #             #month = file[9:16]
    #             region = file[12:14]
    #             filename = 'boxplot_yearly_'+region+c+'.hmtl'
    #             #filename = 'boxplot_'+month+'_'+region+'.hmtl'
    #             gb = GraphicsBuilder()
    #             gb.boxplot_per_classe(df_rain=df_rain, df_norain=df_norain, yvalue1=df_norain['89V'], ytitle1='Pixels - no rain',
    #                           yvalue2=df_rain['89V'], ytitle2='Pixels - rain', size_norain=size_norain, size_rain=size_rain,
    #                           classe=c, chart_title='Box Plot 89V - Yearly - '+region+c,
    #                           output_file_name=os.path.join(path, filename))
#   # ------------------------------------------------------------------------------
    # Plotting Box Plot by month and region:
#    path = rede_pp.IN_CSV_LIST
#    for file in os.listdir(path):
#        if os.path.isfile(path+file):
#            print('elemento da pasta: {}'.format(file))
#            df = rede_pp.LoadCSV(path, file)
#            classe = ['C1', 'C2', 'C3', 'C4']
#            idx1= np.where((df['CLASSE'] == classe[0]))
#            idx2= np.where((df['CLASSE'] == classe[1]))
#            idx3= np.where((df['CLASSE'] == classe[2]))
#            idx4= np.where((df['CLASSE'] == classe[3]))
#            size_idx1 = str(len(idx1[0]))
#            size_idx2 = str(len(idx2[0]))
#            size_idx3 = str(len(idx3[0]))
#            size_idx4 = str(len(idx4[0]))
#            region = file[12:14]
#            filename = 'boxplot_yearly_PCT89_'+region+'_by_classes.html'
#            #filename = 'boxplot_'+month+'_'+region+'.hmtl'
#            gb = GraphicsBuilder()
#            gb.boxplot_per_classe(dataframe=df, yvalue1=df['PCT89'].iloc[idx1], yvalue2=df['PCT89'].iloc[idx2],
#                                  yvalue3=df['PCT89'].iloc[idx3], yvalue4=df['PCT89'].iloc[idx4],
#                                  ytitle1='Pixels - C1', ytitle2='Pixels - C2', ytitle3='Pixels - C3',
#                                  ytitle4='Pixels - C4', size_idx1=size_idx1, size_idx2=size_idx2,
#                                  size_idx3=size_idx3, size_idx4=size_idx4,
#                                  chart_title='Box Plot PCT89 - Yearly - '+region+ ' by classes',
#                                  output_file_name=os.path.join(OUT_CSV_LIST, filename))
  # ------------------------------------------------------------------------------    #
    # regioes = ['R1', 'R2', 'R3', 'R4', 'R5']
    # for r in regioes:
    #     frames = []

    # path = '/media/DATA/tmp/datasets/regionais/meteo_regions/csv_regions/yearly/'
    # #path = '/media/DATA/tmp/datasets/regionais/meteo_regions/csv_regions/' + r + '/'
    # pathout = '/media/DATA/tmp/datasets/regionais/meteo_regions/csv_regions/TAG/yearly/'

    #path = '/media/DATA/tmp/datasets/regionais/meteo_regions/csv_regions/yearly/'
    #path = '/media/DATA/tmp/datasets/regionais/meteo_regions/csv_regions/' + r + '/'
    #pathout = '/media/DATA/tmp/datasets/regionais/meteo_regions/csv_regions/TAG/yearly/'

    # for file in os.listdir(path):
    #     df = rede_pp.LoadCSV(path, file)
    #     df_final = rede_pp.TagRainNoRain(df)
    #     df_name = os.path.splitext(file)[0] + "_TAG.csv"
    #     df_final.to_csv(os.path.join(pathout, df_name), index=False, sep=",", decimal='.')
    #     print("The file ", df_name, " was saved!")

    # ------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
