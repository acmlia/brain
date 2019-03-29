from __future__ import absolute_import, division, print_function

import os
import time
import glob
import h5py
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
    path_hdf5 = ''

    def __init__(self, yaml_version='',
                 yaml_path='',
                 path_csv='',
                 file_csv='',
                 path_fig='',
                 path_hdf5=''):

        self.ymlv = yaml_version
        self.ymlp = yaml_path
        self.path_csv = path_csv
        self.file_csv = file_csv
        self.path_fig = path_fig
        self.vrn = yaml_version
        self.path_hdf5 = path_hdf5

    def autoVld(self):
        ## load YAML and create model
        yaml_file = open(self.ymlp + 'tf_reg_' + self.ymlv + '.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        loaded_model.load_weights(self.ymlp + 'tf_reg_' + self.ymlv + '.h5')
        print("Loaded model from disk")
        # ------------------------------------------------------------------------------
        df_pred = pd.read_csv(os.path.join(self.path_csv, self.file_csv), sep=',', decimal='.')

        vrn = self.ymlv + '_195_SCR'
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
        plt.savefig(self.path_fig + fig_name)
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
        fig_name = 'PCA_TB2_' + vrn + '.png'
        plt.savefig(self.path_fig + fig_name)
        
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
        plt.savefig(self.path_fig + fig_name)
        plt.clf()

        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        ax = plt.gca()
        ax.plot(y_true, y_pred, 'o', c='blue', alpha=0.07, markeredgecolor='none')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('True Values [sfcprcp]')
        ax.set_ylabel('Predictions [sfcprcp]')
        plt.plot([-100, 100], [-100, 100])
        fig_name = fig_title + "_plot_scatter_LOG_y_test_vs_y_pred.png"
        plt.savefig(self.path_fig + fig_name)
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
        plt.savefig(self.path_fig + fig_name)
        plt.clf()

    def read_hdf5_1CGMI(self):
        #start of the program
        print("starting the conversion from  1C-GMI (HDF5) to DataFrame (CSV):")
        print("")
        
        #list of methods 
        #this method will print all of the names of the hdf internal files
        print("defining methods")
        def printname(name):
            print(name)
        print("method definitions complete")
        print("")
        
        #assign current working directory
        dir = self.path_hdf5
        print("the HDF5 directory is: "+dir)
        print("")
        
        #make directory folder (if it does not already exist) and directory variable for output text files
        print("Testing directory for output text files")
        if not os.path.exists(dir+"/1CGMI_csv/"):
            os.makedirs(dir+"/1CGMI_csv/")
            print("CSV file directory created")
        csvdir = dir + "1CGMI_csv/"
        print("CSV DIR NAME", csvdir)
        
        print("")
        
        #list of hdf files to be converted
        print(">>> list of hdf files")
        hdflist = glob.glob(os.path.join(self.path_hdf5,'*1C*.HDF5'))
#        hdflist=glob.glob(self.path_hdf5+'*2B*.HDF5')
#        hdflist = [os.path.basename(s) for s in hdflist if "1C" in s]
        print("")
        #available datasets in hdf files
        print("available datasets in HDF5 files: ")
        singlehdflist=hdflist[0]
        insidehdffile=h5py.File(singlehdflist,"r+")
        insidehdffile.visit(printname)
        insidehdffile.close()
        print("")
        
        #datatype conversion 
        #this loop outputs the indvidual lat long and precip datasets available within the hdf file as indivdual text files 
        for hdffile in hdflist:
            #read and write hdf file
            print("reading the hdf file: " + hdffile)
            currenthdffile=h5py.File(hdffile,"r+")
            print("reading hdf file complete")
            print("")
            
            #data retrieval 
            #This is where you extract the datasets you want to output as text files
            #you can add more variables if you would like
            #this is done in the format varible=hdffilename['dataset']
            print("Creating DF with lat, lon, sfccode, TBs!")
            
            lat=currenthdffile['S1/Latitude/']
            lon=currenthdffile['S1/Longitude']
            TB_S1=currenthdffile['S1/Tc']
            TB_S2=currenthdffile['S2/Tc']
            
            lat=np.ravel(lat)
            lon=np.ravel(lon)
            TB_S1=np.array(TB_S1)
            TB_S1_t=TB_S1.transpose(2,0,1).reshape(9,-1)
            TB_S1=TB_S1_t.transpose()
        
            TB_S2=np.array(TB_S2)
            TB_S2_t=TB_S2.transpose(2,0,1).reshape(4,-1)
            TB_S2=TB_S2_t.transpose()
        
            df=pd.DataFrame()
            df['lat']=lat
            df['lon']=lon
            df['10V']=TB_S1[:,0]
            df['10H']=TB_S1[:,1]
            df['18V']=TB_S1[:,2]
            df['18H']=TB_S1[:,3]
            df['23V']=TB_S1[:,4]
            df['36V']=TB_S1[:,5]
            df['36H']=TB_S1[:,6]
            df['89V']=TB_S1[:,7]
            df['89H']=TB_S1[:,8]
            df['166V']=TB_S2[:,0]
            df['166H']=TB_S2[:,1]
            df['186V']=TB_S2[:,2]
            df['190V']=TB_S2[:,3]
        
            print("Dataframe created!")
            print("")
        
            file_name = os.path.basename(hdffile.replace(".HDF5", ".csv"))
            print("NAME:\n", file_name, '\n')
            print('CSVDIR: ', csvdir)
            print("TESTE:\n", csvdir + file_name, '\n')
            df.to_csv((csvdir + file_name), index=False, sep=",", decimal='.')
            print("Dataframe saved!")
            
    def read_hdf5_2AGPROF(self):
        #start of the program
        print("starting the conversion from  2A GPROF (HDF5)to DataFrame (CSV):")
        print("")
        #list of methods 
        #this method will print all of the names of the hdf internal files
        print("defining methods")
        def printname(name):
            print(name)
        print("method definitions complete")
        print("")
        
        #assign current working directory
        dir = self.path_hdf5
        print("the HDF5 directory is: "+dir)
        print("")
        
        #make directory folder (if it does not already exist) and directory variable for output text files
        print("creating a directory for output text files")
        if not os.path.exists(dir+"/"+"2AGPROF_csv/"):
            os.makedirs(dir+"/"+"2AGPROF_csv/")
        csvdir=dir+"2AGPROF_csv/"
        print("text file directory created")
        print("")
        
        #list of hdf files to be converted
        print("list of hdf files")
        hdflist = glob.glob(os.path.join(self.path_hdf5,'*2A*.HDF5'))
        print(hdflist)
        print("")
        
        #available datasets in hdf files
        print("available datasets in HDF5 files: ")
        singlehdflist=hdflist[0]
        insidehdffile=h5py.File(singlehdflist,"r+")
        insidehdffile.visit(printname)
        insidehdffile.close()
        print("")
    
        #datatype conversion 
        #this loop outputs the indvidual lat long and precip datasets available within the hdf file as indivdual text files 
        for hdffile in hdflist:
            #read and write hdf file
            print("reading the hdf file: "+hdffile)
            currenthdffile=h5py.File(hdffile,"r+")
            print("reading hdf file complete")
            print("")
            
            #data retrieval 
            #This is where you extract the datasets you want to output as text files
            #you can add more variables if you would like
            #this is done in the format varible=hdffilename['dataset']
            print("Creating DF with lat, lon, sfccode, TBs(simulated GMI)!")
            
            lat=currenthdffile['S1/Latitude/']
            lon=currenthdffile['S1/Longitude']
            sfccode=currenthdffile['S1/surfaceTypeIndex']
            sfcprcp=currenthdffile['S1/surfacePrecipitation']
            T2m=currenthdffile['S1/temp2mIndex']
            tcwv=currenthdffile['S1/totalColumnWaterVaporIndex']
            
            lat=np.ravel(lat)
            lon=np.ravel(lon)
            sfccode=np.ravel(sfccode)
            sfcprcp=np.ravel(sfcprcp)
            T2m=np.ravel(T2m)
            tcwv=np.ravel(tcwv)
            
            df=pd.DataFrame()
            df['lat']=lat
            df['lon']=lon
            df['sfccode']=sfccode
            df['sfcprcp']=sfcprcp
            df['T2m']=T2m
            df['tcwv']=tcwv

            print("Dataframe created!")
            print("")

            file_name = os.path.basename(hdffile.replace(".HDF5", ".csv"))
            print("NAME:\n", file_name, '\n')
            print('CSVDIR: ', csvdir)
            print("TESTE:\n", csvdir + file_name, '\n')
            df.to_csv((csvdir + file_name), index=False, sep=",", decimal='.')
            print("Dataframe saved!")


    def read_hdf5_2BCMB(self):
        #start of the program
        print("starting the conversion from 2BCMB (HDF5) to DatFrame (CSV):")
        print("")
        
        #list of methods 
        #this method will print all of the names of the hdf internal files
        print("defining methods")
        def printname(name):
            print(name)
        print("method definitions complete")
        print("")
        
        #assign current working directory
        dir = self.path_hdf5
        print("the HDF5 directory is: "+dir)
        print("")
    
        #make directory folder (if it does not already exist) and directory variable for output text files
        print("creating a directory for output CSV files")
        if not os.path.exists(dir+"/"+"2BCMB_csv/"):
            os.makedirs(dir+"/"+"2BCMB_csv/")
        csvdir=dir+"2BCMB_csv/"
        print("text file directory created")
        print("")
        
        #list of hdf files to be converted
        print("list of hdf files")
        hdflist = glob.glob(os.path.join(self.path_hdf5,'*2B*.HDF5'))
        print(hdflist)
        print("")
        
        #available datasets in hdf files
        print("available datasets in HDF5 files: ")
        singlehdflist=hdflist[0]
        insidehdffile=h5py.File(singlehdflist,"r+")
        insidehdffile.visit(printname)
        insidehdffile.close()
        print("")
        
        #datatype conversion 
        #this loop outputs the indvidual lat long and precip datasets available within the hdf file as indivdual text files 
        for hdffile in hdflist:
            #read and write hdf file
            print("reading the hdf file: "+hdffile)
            currenthdffile=h5py.File(hdffile,"r+")
            print("reading hdf file complete")
            print("")
            
            #data retrieval 
            #This is where you extract the datasets you want to output as text files
            #you can add more variables if you would like
            #this is done in the format varible=hdffilename['dataset']
            print("Creating DF with lat, lon, sfccode, TBs(simulated GMI)!")
            
            lat=currenthdffile['NS/Latitude/']
            lon=currenthdffile['NS/Longitude']
            sfccode=currenthdffile['NS/Input/surfaceType']
            sfcprcp=currenthdffile['NS/surfPrecipTotRate']
            sim_TB=currenthdffile['NS/simulatedBrightTemp']
            
            lat=np.ravel(lat)
            lon=np.ravel(lon)
            sfccode=np.ravel(sfccode)
            sfcprcp=np.ravel(sfcprcp)
            
            TBS=np.array(sim_TB)
            TBS_2D=TBS.transpose(2,0,1).reshape(13,-1)
            TBS=TBS_2D.transpose()
            
            df=pd.DataFrame()
            df['lat']=lat
            df['lon']=lon
            df['sfccode']=sfccode
            df['sfcprcp']=sfcprcp
            df['10V']=TBS[:,0]
            df['10H']=TBS[:,1]
            df['18V']=TBS[:,2]
            df['18H']=TBS[:,3]
            df['23V']=TBS[:,4]
            df['36V']=TBS[:,5]
            df['36H']=TBS[:,6]
            df['89V']=TBS[:,7]
            df['89H']=TBS[:,8]
            df['166V']=TBS[:,9]
            df['166H']=TBS[:,10]
            df['186V']=TBS[:,11]
            df['190V']=TBS[:,11:12]
        
            print("Dataframe created!")
            print("")
    
            file_name = os.path.basename(hdffile.replace(".HDF5", ".csv"))
            print("NAME:\n", file_name, '\n')
            print('CSVDIR: ', csvdir)
            print("TESTE:\n", csvdir + file_name, '\n')
            df.to_csv((csvdir + file_name), index=False, sep=",", decimal='.')
            print("Dataframe saved!")

