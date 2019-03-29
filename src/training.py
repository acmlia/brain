from __future__ import absolute_import, division, print_function

import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
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
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import model_from_yaml

print('TF version '+tf.__version__)

# ------------------------------------------------------------------------------

def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))
# ------------------------------------------------------------------------------


class Training:
    """
    This module is intended to automate the TensorFlow Neural Network training.
    """
    PCA = PCA()
    seed = 0
    run_prefix = ''
    tver = ''
    vernick = ''
    file = ''
    path = ''
    fig_title = ''
    path_fig = ''
    mod_out_pth = ''
    mod_out_name = ''

    def __init__(self, random_seed=0,
                 run_prefix='',
                 version='',
                 version_nickname='',
                 csv_entry='',
                 csv_path='',
                 figure_path='',
                 model_out_path='',
                 model_out_name=''):

        self.run_prefix = run_prefix
        self.seed = random_seed
        self.tver = version
        self.vernick = version_nickname
        self.file = csv_entry
        self.path = csv_path
        self.path_fig = figure_path
        self.fig_title = run_prefix + version + version_nickname
        self.mod_out_pth = model_out_path
        self.mod_out_name = model_out_name
    # -------------------------------------------------------------------------
    # DROP DATA OUTSIDE INTERVAL
    # -------------------------------------------------------------------------
       
    @staticmethod
    def keep_interval(keepfrom: 0.0, keepto: 1.0, dataframe, target_col: str):
        keepinterval = np.where((dataframe[target_col] >= keepfrom) &
                                (dataframe[target_col] <= keepto))
        result = dataframe.iloc[keepinterval]
        return result

    # -------------------------------------------------------------------------
    # BUILD MODELS DEFINITIONS : CLAS = CLASSIFICATION and REG = REGRESSION
    # -------------------------------------------------------------------------

    @staticmethod
    def build_class_model():
        '''
         Fucntion to create the instance and configuration of the keras
         model(Sequential and Dense).
        '''
        # Create the Keras model:
        model = Sequential()
        model.add(Dense(8, input_dim=4, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(2, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'],)
        return model

    @staticmethod
    def build_reg_model(input_size):
        '''
         Fucntion to create the instance and configuration of the keras
         model(Sequential and Dense).
        '''
        model = Sequential()
        model.add(GaussianNoise(0.01, input_shape=(input_size,)))
        model.add(Dense(24, activation='linear'))
        model.add(Dense(10, activation='linear'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        return model

    # -------------------------------------------------------------------------
    # EXECUTION OF READING INPUT ATTRIBUTES, SCALING, PCA, SPLIT AND RUN MODEL!
    # -------------------------------------------------------------------------

    def autoExecClass(self):

        # Fix random seed for reproducibility:
        np.random.seed(self.seed)

        # Load dataset:
        df = pd.read_csv(os.path.join(self.path, self.file), sep=',', decimal='.')
        x, y= df.loc[:,['36V', '89V', '166V', '190V']], df.loc[:,['TagRain']]
        
        x_arr = np.asanyarray(x)
        y_arr = np.asanyarray(y)
        y_arr = np.ravel(y_arr)

        # Scaling the input paramaters:
#       scaler_min_max = MinMaxScaler()
        norm_sc = Normalizer()
        x_normalized= norm_sc.fit_transform(x_arr)

        # Split the dataset in test and train samples:
        x_train, x_test, y_train, y_test = train_test_split(x_normalized,
                                                            y_arr, test_size=0.10,
                                                            random_state=101)

        # Create the instance for KerasRegressor:
        model = KerasClassifier(build_fn=self.build_class_model, batch_size=10,
                                epochs=100, verbose=0)
        tic()
        screening_trained = model.fit(x_train,y_train)
        tac()

# ------------------------------------------------------------------------------
        # Saving model to YAML:

        model_yaml = screening_trained.to_yaml()
        with open(self.mod_out_pth + self.mod_out_name + '.yaml', 'w') as yaml_file:
            yaml_file.write(model_yaml)

        # serialize weights to HDF5
        model.save_weights(self.mod_out_pth + self.mod_out_name + '.h5')
        print("Saved model to disk")

    # ------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------

    def autoExecReg(self):

        # Fix random seed for reproducibility:
        np.random.seed(self.seed)
# ------------------------------------------------------------------------------

        df_orig = pd.read_csv(os.path.join(self.path, self.file), sep=',', decimal='.')

        df_input = df_orig.loc[:, ['10V', '10H', '18V', '18H', '36V', '36H', '89V', '89H',
                                   '166V', '166H', '183VH', 'sfccode', 'T2m', 'tcwv', 'PCT36', 'PCT89', '89VH',
                                   'lat']]

        colunas = ['10V', '10H', '18V', '18H', '36V', '36H', '89V', '89H',
                   '166V', '166H', '183VH', 'sfccode', 'T2m', 'tcwv', 'PCT36', 'PCT89', '89VH',
                   'lat']

        scaler = StandardScaler()

        normed_input = scaler.fit_transform(df_input)
        df_normed_input = pd.DataFrame(normed_input[:],
                                       columns=colunas)
        ancillary = df_normed_input.loc[:, ['183VH', 'sfccode', 'T2m', 'tcwv', 'PCT36', 'PCT89', '89VH',
                                            'lat']]
        # regions=df_orig.loc[:,['R1','R2','R3','R4','R5']]
        # ------------------------------------------------------------------------------
        # Choosing the number of components:

        TB1 = df_normed_input.loc[:, ['10V', '10H', '18V', '18H']]
        TB2 = df_normed_input.loc[:, ['36V', '36H', '89V', '89H', '166V', '166H']]

        # ------------------------------------------------------------------------------
        # Verifying the number of components that most contribute:
        pca = self.PCA
        pca1 = pca.fit(TB1)
        plt.plot(np.cumsum(pca1.explained_variance_ratio_))
        plt.xlabel('Number of components for TB1')
        plt.ylabel('Cumulative explained variance');
        plt.savefig(self.path_fig + self.tver + '_PCA_TB1.png')
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
        plt.savefig(self.path_fig + self.tver + 'PCA_TB2.png')
        # ---
        pca_trans2 = PCA(n_components=2)
        pca2 = pca_trans2.fit(TB2)
        TB2_transformed = pca_trans2.transform(TB2)
        print("original shape:   ", TB2.shape)
        print("transformed shape:", TB2_transformed.shape)
        # ------------------------------------------------------------------------------
        # JOIN THE TREATED VARIABLES IN ONE SINGLE DATASET AGAIN:

        PCA1 = pd.DataFrame(TB1_transformed[:],
                            columns=['pca1_1', 'pca_2'])
        PCA2 = pd.DataFrame(TB2_transformed[:],
                            columns=['pca2_1', 'pca2_2'])

        dataset = PCA1.join(PCA2, how='right')
        dataset = dataset.join(ancillary, how='right')
        dataset = dataset.join(df_orig.loc[:, ['sfcprcp']], how='right')
        # ------------------------------------------------------------------------------

        dataset = self.keep_interval(0.5, 100.0, dataset, 'sfcprcp')

        # ----------------------------------------
        # SUBSET BY SPECIFIC CLASS (UNDERSAMPLING)
        n = 0.90
        to_remove = np.random.choice(
            dataset.index,
            size=int(dataset.shape[0] * n),
            replace=False)
        dataset = dataset.drop(to_remove)

        # ------------------------------------------------------------------------------
        # Split the data into train and test
        # Now split the dataset into a training set and a test set.
        # We will use the test set in the final evaluation of our model.

        train_dataset = dataset.sample(frac=0.8, random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

        # ------------------------------------------------------------------------------
        # Inspect the data:
        # Have a quick look at the joint distribution of a few pairs of columns from the training set.

        colunas = list(dataset.columns.values)

        # ------------------------------------------------------------------------------
        # Also look at the overall statistics:
        train_stats = train_dataset.describe()
        train_stats.pop("sfcprcp")
        train_stats = train_stats.transpose()

        # ------------------------------------------------------------------------------
        # Split features from labels:
        # Separate the target value, or "label", from the features.
        # This label is the value that you will train the model to predict.

        y_train = train_dataset.pop('sfcprcp')
        y_test = test_dataset.pop('sfcprcp')

        # ------------------------------------------------------------------------------
        # Normalize the data:

        scaler = StandardScaler()
        normed_train_data = scaler.fit_transform(train_dataset)
        normed_test_data = scaler.fit_transform(test_dataset)

        # ------------------------------------------------------------------------------
        # Build the model:

        model = self.build_reg_model(len(train_dataset.keys()))
        # ------------------------------------------------------------------------------
        # Inspect the model:
        # Use the .summary method to print a simple description of the model

        model.summary()

        # ------------------------------------------------------------------------------
        # It seems to be working, and it produces a result
        # of the expected shape and type.

        # Train the model:
        # Train the model for 1000 epochs, and record the training
        # and validation accuracy in the history object.

        # ------------------------------------------------------------------------------
        # Display training progress by printing a single dot for each completed epoch

        class PrintDot(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                if epoch % 100 == 0: print('')
                print('.', end='')

        EPOCHS = 1000

        history = model.fit(
            normed_train_data, y_train,
            epochs=EPOCHS, validation_split=0.2, verbose=0,
            callbacks=[PrintDot()])
        print(history.history.keys())

        # ------------------------------------------------------------------------------
        # Visualize the model's training progress using the stats
        # stored in the history object.

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        hist.tail()

        self.plot_history(history)
        # ------------------------------------------------------------------------------

        model = self.build_reg_model(len(train_dataset.keys()))

        # The patience parameter is the amount of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        history = model.fit(normed_train_data, y_train, epochs=EPOCHS,
                            validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

        # ------------------------------------------------------------------------------
        # Ploting again, but with the EarlyStopping apllied:

        self.plot_history_EarlyStopping(history)

        # The graph shows that on the validation set, the average error
        # is usually around +/- 2 MPG. Is this good?
        # We'll leave that decision up to you.
        # ------------------------------------------------------------------------------
        # Let's see how well the model generalizes by using
        # the test set, which we did not use when training the model.
        # This tells us how well we can expect the model to predict
        # when we use it in the real world.

        loss, mae, mse = model.evaluate(normed_test_data, y_test, verbose=0)

        print("Testing set Mean Abs Error: {:5.2f} sfcprcp".format(mae))

        # Make predictions
        # Finally, predict SFCPRCP values using data in the testing set:

        test_predictions = model.predict(normed_test_data).flatten()

        plt.figure()
        plt.scatter(y_test, test_predictions)
        plt.xlabel('True Values [sfcprcp]')
        plt.ylabel('Predictions [sfcprcp]')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0, plt.xlim()[1]])
        plt.ylim([0, plt.ylim()[1]])
        plt.plot([-100, 100], [-100, 100])
        fig_name = self.fig_title + "_plot_scatter_y_test_vs_y_pred.png"
        plt.savefig(self.path_fig + fig_name)
        plt.clf()

        #------------------------------------------------------------------------------
        ax = plt.gca()
        ax.plot(y_test,test_predictions, 'o', c='blue', alpha=0.07, markeredgecolor='none')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('True Values [sfcprcp]')
        ax.set_ylabel('Predictions [sfcprcp]')
        plt.plot([-100, 100], [-100, 100])
        fig_name = self.fig_title + "_plot_scatter_LOG_y_test_vs_y_pred.png"
        plt.savefig(self.path_fig+fig_name)
        plt.clf()
#------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # It looks like our model predicts reasonably well.
        # Let's take a look at the error distribution.

        error = test_predictions - y_test
        plt.hist(error, bins=25)
        plt.xlabel("Prediction Error [sfcprcp]")
        plt.ylabel("Count")
        fig_name = self.fig_title + "_prediction_error.png"
        plt.savefig(self.path_fig + fig_name)
        plt.clf()

        # ------------------------------------------------------------------------------
        # Saving model to YAML:

        model_yaml = model.to_yaml()
        with open(self.mod_out_pth + self.mod_out_name + '.yaml', 'w') as yaml_file:
            yaml_file.write(model_yaml)

        # serialize weights to HDF5
        model.save_weights(self.mod_out_pth + self.mod_out_name + '.h5')
        print("Saved model to disk")

    # -------------------------------------------------------------------------
    # FUNCTIONS TO MAKE PLOTS ABOUT TRAINING:
    # -------------------------------------------------------------------------
    def plot_history(self, history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [sfcprcp]')
        plt.plot(hist['epoch'], hist['mean_absolute_error'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
                 label='Val Error')
        ylim_max = hist.val_mean_absolute_error.max() + 10
        plt.ylim([0, ylim_max])
        plt.legend()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$scfprcp^2$]')
        plt.plot(hist['epoch'], hist['mean_squared_error'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_squared_error'],
                 label='Val Error')
        ylim_max = hist.val_mean_squared_error.max() + 10
        plt.ylim([0, ylim_max])
        plt.legend()
        # plt.show()
        fig_name = self.fig_title + "_error_per_epochs_history.png"
        plt.savefig(self.path_fig + fig_name)

    def plot_history_EarlyStopping(self, history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [sfcprcp]')
        plt.plot(hist['epoch'], hist['mean_absolute_error'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
                 label='Val Error')
        ylim_max = hist.val_mean_absolute_error.max() + 10
        plt.ylim([0, ylim_max])

        plt.legend()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$sfcprcp^2$]')
        plt.plot(hist['epoch'], hist['mean_squared_error'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_squared_error'],
                 label='Val Error')
        ylim_max = hist.val_mean_squared_error.max() + 10
        plt.ylim([0, ylim_max])

        plt.legend()

        fig_name = self.fig_title + "_error_per_epochs_EarlyStopping.png"
        plt.savefig(self.path_fig + fig_name)
