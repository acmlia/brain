#! /usr/bin/env python3

import logging
import settings
import os, sys
import pandas as pd

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

    git_dir = '/mnt/AC9AF51E9AF4E5AC/repos'

    ar0 = Training(random_seed=7,
                   run_prefix='tf_regr_',
                   version='R1',
                   version_nickname='_undc1_0956_',
                   csv_entry='yearly_br_underc1_hot_0956.csv',
                   csv_path='/mnt/AC9AF51E9AF4E5AC/A1ML/phd_datasets/',
                   figure_path=git_dir+'/jobs/tf_regression_figures/figures_vld/',
                   model_out_path=git_dir+'/jobs/output_models/',
                   model_out_name='tf_reg_R1')

    vldar0 = Validation(yaml_version='R1',
                        yaml_path=git_dir+'/jobs/output_models/',
                        yaml_file='tf_reg_R1',
                        path_csv='/mnt/AC9AF51E9AF4E5AC/A1ML/phd_datasets/',
                        file_csv='teste_validation_195_SCR.csv',
                        path_fig=git_dir+'/jobs/tf_regression_figures/figures_vld/',
                        version_tag='',
                        figure_title='')
    ar0.autoExec()

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
#    region = ['R1', 'R2', 'R3', 'R4', 'R5']
#    for r in region:
#    path = rede_pt.IN_CSV_LIST+'yearly/'
#    for file in os.listdir(path):
#        if os.path.isfile(path+file):
#            print('elemento da pasta: {}'.format(file))
#            df = rede_pp.LoadCSV(path, file)
#            df_rain, df_norain, size_norain, size_rain = rede_pp.ThresholdRainNoRain(df)
#            #month = file[9:16]
#            region = file[12:14]
#            filename = 'boxplot_yearly_'+region+'.hmtl'
#            #filename = 'boxplot_'+month+'_'+region+'.hmtl'
#            gb = GraphicsBuilder()
#            gb.box_plotter(df_rain=df_rain, df_norain=df_norain, yvalue1=df_norain['PCT89'], ytitle1='Pixels - no rain',
#                           yvalue2=df_rain['PCT89'], ytitle2='Pixels - rain', size_norain=size_norain, size_rain=size_rain,
#                           chart_title='Box Plot PCT89 - Yearly - '+region,
#                           output_file_name=os.path.join(OUT_CSV_LIST+'yearly', filename))
     # ------------------------------------------------------------------------------
    # regioes = ['R1', 'R2', 'R3', 'R4', 'R5']
    # for r in regioes:
    #     frames = []
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
