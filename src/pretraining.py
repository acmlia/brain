import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly 
plotly.tools.set_credentials_file(username='lia_amaral', api_key='yymuh9KYQM2uJphpnkSc')
import plotly.plotly as py
import plotly.graph_objs as go
import cufflinks as cf


#import seaborn as sns
#import matplotlib.pyplot as plt
##from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler


class PreTraining:
    """
    This module treat the data for to be ready for the training.
    It can call the functions:
        
    """
    
    def __init__(self, IN_CSV_LIST=None,
                 OUT_CSV_LIST=None,
                 RAIN_CSV=None,
                 NORAIN_CSV=None,
                 COLUMN_TYPES=None):
        self.IN_CSV_LIST = IN_CSV_LIST
        self.RAIN_CSV = RAIN_CSV
        self.NORAIN_CSV = NORAIN_CSV
        self.COLUMN_TYPES = COLUMN_TYPES
        
        
        
    def AdditionalInputVariables(self, path, file):
        '''
        Create new input variables from the dataset, as PCT, SSI, MPDI, etc...
        '''
        df = pd.DataFrame()
        if file.startswith(".", 0, len(file)):
            print("File name starts with point: {} - Skipping...".format(file))
            return None
        elif file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, file), sep=',', decimal='.',
                                    dtype=self.COLUMN_TYPES)
            print('Dataframe {} was loaded!'.format(file))

            df['10VH'] = df['10V'] - df['10H']
            df['18VH'] = df['18V'] - df['18H']
            df['36VH'] = df['36V'] - df['36H']
            df['89VH'] = df['89V'] - df['89H']
            df['166VH'] = df['166V'] - df['166H']
            df['183VH'] = df['186V'] - df['190V']
            df['SSI'] = df['18V'] - df['36V']
            df['delta_neg'] = df['18V'] - df['18H']
            df['delta_pos'] = df['18V'] + df['18H']
            df['MPDI'] = np.divide(df['delta_neg'], df['delta_pos'])
            df['MPDI_scaled'] = df['MPDI']*600
            
            # Inclugin the PCT formulae: PCTf= (1+alfa)*TBfv - alfa*TBfh
            alfa10 = 1.5
            alfa18 = 1.4
            alfa36 = 1.15
            alfa89 = 0.7
            
            df['PCT10'] = (1+ alfa10)*df['10V'] - alfa10*df['10H']
            df['PCT18'] = (1+ alfa18)*df['18V'] - alfa18*df['18H']
            df['PCT36'] = (1+ alfa36)*df['36V'] - alfa36*df['36H']
            df['PCT89'] = (1+ alfa89)*df['89V'] - alfa89*df['89H']
            
            file_name = os.path.splitext(file)[0] + "_OK.csv"
            df.to_csv(os.path.join(path, file_name), index=False, sep=",", decimal='.')
            print("The file ", file_name, " was genetared!")
        if not df.empty:
            return df
        else:
            print("Unexpected file format: {} - Skipping...".format(file))
            return None

#    def ExploratoryAnalysis(self, path, file):
#        '''
#        Create the plots and graphics to help the exploratory analysis.
#        '''
#        if file.startswith(".", 0, len(file)):
#            print("File name starts with point: {} - Skipping...".format(file))
#            return None
#        elif file.endswith(".csv"):
#            df = pd.DataFrame()
#            df = pd.read_csv(os.path.join(path, file), sep=',', decimal='.',
#                                    dtype=self.COLUMN_TYPES)
#            print('Dataframe {} was loaded!'.format(file))
#        
#        gb = graphics_builder(df)

            
            