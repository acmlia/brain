from __future__ import absolute_import, division, print_function

import os
import time
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


print('TF version '+tf.__version__)

class Validation:
    """
    This module is intended to automate the TensorFlow Neural Network training.
    """
    ymlv = ''
    ymlp = ''
    ymlf = ''
    path_csv = ''
    file_csv = ''
    path_fig = ''
    vrn = ''
    fig_title = ''

    def __init__(self, yaml_version='',
                 yaml_path='',
                 yaml_file='',
                 path_csv='',
                 file_csv='',
                 path_fig='',
                 version_tag='',
                 figure_title=''):

        self.ymlv = yaml_version
        self.ymlp = yaml_path
        self.ymlf = yaml_file
        self.path_csv = path_csv
        self.file_csv = file_csv
        self.path_fig = path_fig
        self.vrn = yaml_version + version_tag
        self.fig_title = figure_title

    def autoVld(self):
        ## load YAML and create model
        yaml_file = open(yaml_path + 'tf_regression_' + yamlversion + '.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        loaded_model.load_weights(yaml_path + 'tf_regression_' + yamlversion + '.h5')
        print("Loaded model from disk")
        # ------------------------------------------------------------------------------
        df_pred = pd.read_csv(os.path.join(path_csv, file_csv), sep=',', decimal='.')

        vrn = yamlversion + '_195_SCR'
        fig_title = 'Validation_' + vrn + '_'

        # ------------------------------------------------------------------------------
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
        plt.plot(np.cumsum(pca1.explained_variance_ratio_))
        plt.xlabel('Number of components for TB1')
        plt.ylabel('Cumulative explained variance');
        fig_name = 'PCA_TB1_' + vrn + '.png'
        plt.savefig(path_fig + fig_name)
        # ---
        pca_trans1 = PCA(n_components=2)
        pca1 = pca_trans1.fit(TB1)
        TB1_transformed = pca_trans1.transform(TB1)
        print("original shape:   ", TB1.shape)
        print("transformed shape:", TB1_transformed.shape)
        # ------------------------------------------------------------------------------
        pca = PCA()
        pca2 = pca.fit(TB2)
        plt.plot(np.cumsum(pca2.explained_variance_ratio_))
        plt.xlabel('Number of components for TB2')
        plt.ylabel('Cumulative explained variance');
        plt.savefig("PCA_TB2.png")
        # ---
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

        # threshold_rain = 0.1
        # rain_pixels = np.where((dataset['sfcprcp_s1'] >= threshold_rain))
        # dataset=dataset.iloc[rain_pixels]
        SCR_pixels = np.where((dataset['SCR'] == 1))
        dataset = dataset.iloc[SCR_pixels]

        SCR = dataset.pop('SCR')
        y_true = dataset.pop('sfcprcp_s1')
        y_true[y_true == -9999.0] = np.NaN

        x_normed = dataset.values
        y_pred = loaded_model.predict(x_normed).flatten()

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        plt.figure()
        plt.scatter(y_true, y_pred)
        plt.xlabel('True Values [sfcprcp]')
        plt.ylabel('Predictions [sfcprcp]')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0, plt.xlim()[1]])
        plt.ylim([0, plt.ylim()[1]])
        plt.plot([-100, 100], [-100, 100])
        fig_name = fig_title + "_plot_scatter_y_test_vs_y_pred.png"
        plt.savefig(path_fig + fig_name)
        plt.clf()

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(y_true, y_pred, 'o', c='blue', alpha=0.07, markeredgecolor='none')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('True Values [sfcprcp]')
        ax.set_ylabel('Predictions [sfcprcp]')
        plt.plot([-100, 100], [-100, 100])
        fig_name = fig_title + "_plot_scatter_LOG_y_test_vs_y_pred.png"
        plt.savefig(path_fig + fig_name)
        plt.clf()

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # It looks like our model predicts reasonably well.
        # Let's take a look at the error distribution.

        error = y_pred - y_true
        plt.hist(error, bins=25)
        plt.xlabel("Prediction Error [sfcprcp]")
        plt.ylabel("Count")
        fig_name = fig_title + "_prediction_error.png"
        plt.savefig(path_fig + fig_name)
        plt.clf()