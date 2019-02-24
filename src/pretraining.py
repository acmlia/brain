import numpy as np
import pandas as pd
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os, sys
import logging



class PreTraining:
    """
    This module treat the data for to be ready for the training.
    It can call the functions:
        
    """
    
    def __init__(self, RAIN_CSV=None,
                 NORAIN_CSV=None,
                 COLUMN_TYPES=None):
        self.RAIN_CSV = RAIN_CSV
        self.NORAIN_CSV = NORAIN_CSV
        self.COLUMN_TYPES = COLUMN_TYPES
        
        
        
    def AdditionalInputVariables(self, path, file):
        '''
        Create new input variables from the dataset, as PCT, SSI, MPDI, etc...
        '''

        if file.startswith(".", 0, len(file)):
            print("File name starts with point: {} - Skipping...".format(file))
        elif file.endswith(".csv"):
            try:
                df = pd.DataFrame()
                df = pd.read_csv(os.path.join(path, file), sep=',', decimal='.',
                                        dtype=self.COLUMN_TYPES)
                print('Dataframe {} was loaded'.format(file))

                df['10VH'] = df['10V'] - df['10H']
                df['18VH'] = df['18V'] - df['18H']
                df['36VH'] = df['36V'] - df['36H']
                df['89VH'] = df['89V'] - df['89H']
                df['166VH'] = df['166V'] - df['166H']
                df['183VH'] = df['186V'] - df['190H']
                df['SSI'] = df['18V'] - df['36V']
                df['delta_neg'] = df['18V'] - df['18H']
                df['delta_pos'] = df['18V'] + df['18H']
                df['MPDI'] = np.divide(df['delta_neg'], df['delta_pos'])
                df['MPDI_scaled'] = df['MPDI']*600


    def AnalysisPCA(self, ):
        '''
        Analyse the PCA decomposition to help with the input variables decision.
        Need to change the INPUT VARIABLES.

        :param path: sets the csv files path (string)
        :param file: file name or file list (string)
        :return:  dataframe (DataFrame)

        '''
        
         if file.startswith(".", 0, len(file)):
            print("File name starts with point: {} - Skipping...".format(file))
        elif file.endswith(".csv"):
            try:
                dataframe = pd.DataFrame()
                dataframe = pd.read_csv(os.path.join(path, file), sep=',', skipinitialspace=True, decimal='.',
                                        dtype=self.COLUMN_TYPES)
                print('Dataframe {} was loaded'.format(file))
                
                
                # Extracting the INPUT variables from the dataframe
                cols = dataframe.columns
                input_var=cols[8:21]
                
                # Creating a copy of the dataframe to be used in PCA decomposition
                dataframe_input = dataframe[input_var]
                
                #--------------------------------------------------------------
                # PCA DECOMPOSITION --> choosing the number of components:

                pca_verification = PCA().fit(dataframe_input)
                plt.plot(np.cumsum(pca_verification.explained_variance_ratio_))
                plt.xlabel('Number of components')
                plt.ylabel('Cumulative explained variance');
              #  figure_name = 'Verification_PCA_components.png'
                plt.savefig(os.path.join(path, 'Verification_PCA_components.png'))
                
                #--------------------------------------------------------------
                # Comparing the dimensions before and after the transformation in components:

                pca = PCA(n_components=2)
                pca.fit(dataframe_input)
                dataframe_input_transformed = pca.transform(dataframe_input)
                print("original shape:   ", dataframe_input.shape)
                print("transformed shape:", dataframe_input_transformed.shape)
                
                #--------------------------------------------------------------
                # Saving the PCA attributes in a dictionary:
                
                pca_attributes = {'components': pca.explained_variance_ratio_,
                                  'explained_variance': pca.explained_variance_,
                                  'explained_variance_ratio': pca.explained_variance_ratio_,
                                  'singular_values': pca.singular_values_,
                                  'mean': pca.mean_,
                                  'n_components': pca.n_components_,
                                  'noise_variance': pca.noise_variance_}
                #--------------------------------------------------------------
                #  Plot the first two principal components of each point to learn about the data:
                plt.figure(figsize=(12,8))
                plt.scatter(TB_transformed[:, 0], TB_transformed[:, 1],
                            c=df[rr], edgecolor='none', alpha=0.5,
                            cmap=plt.cm.get_cmap('YlGn', 10))
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.colorbar();
                #--------------------------------------------------------------

