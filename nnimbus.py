#! /usr/bin/env python3

# ,----------------,
# | PYTHON IMPORTS |----------------------------------------------------------------------------------------------------
# '----------------'

import logging
import sys
import os

from decouple import config
from core.pre_process import PreProcess
from core.training import Training
from core.prediction import Prediction
from core.validation import Validation
from core import utils

# ,----------------------,
# | ENVIRONMENT SETTINGS |----------------------------------------------------------------------------------------------
# '----------------------'

# Setting up information logs
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.DEBUG)

# Path for the CSV's (input and output):
IN_CSV_LIST = config('IN_CSV_LIST', default='')
OUT_CSV_LIST = config('OUT_CSV_LIST', default='')
RAIN_CSV = config('RAIN_CSV', default='')
NORAIN_CSV = config('NORAIN_CSV', default='')

# Geographical coordinates for regional subset:
# The parameters follow the structure: LAT_LIMIT = [min, max] and LON_LIMIT = [min, max]
LAT_LIMIT = [-34.0, 6.0]
LON_LIMIT = [-75.0, -35.0]

# Minimal threshold of rain rate:
THRESHOLD_RAIN = 0.1

# ,--------------------------,
# | NNIMBUS WORKFLOW OPTIONS |------------------------------------------------------------------------------------------
# '--------------------------'

workflow = {
    'read_raw_csv': False,
    'read_alternative_csv': False,
    'extract_region': False,
    'concatenate_csv_list_to_df': True,
    'compute_additional_variables': True,
    'training': False,
    'pre_process_HDF5': False,
    'prediction': False,
    'validation': False,
    'save_data': True
}

# ,-----------,
# | RUN MODEL |---------------------------------------------------------------------------------------------------------
# '-----------'


def main():
    logging.info(f'Starting NNIMBUS\n')

    # Initializing core classes
    preprocess = PreProcess()
    preprocess.status()

    training = Training()
    training.status()

    prediction = Prediction()
    prediction.status()

    validation = Validation()
    validation.status()

    # ,----------------------,
    # | Reading raw CSV Data |------------------------------------------------------------------------------------------
    # '----------------------'
    if workflow['read_raw_csv']:
        logging.info(f'Reading RAW Randel CSV data')
        training_data = preprocess.load_raw_csv(IN_CSV_LIST, 'CSU.LSWG.201409.bin.csv')
    else:
        logging.info(f'Process skipped by the user: Reading RAW Randel CSV data')
    # ,------------------------------,
    # | Reading alternative CSV Data |----------------------------------------------------------------------------------
    # '------------------------------'
    if workflow['read_alternative_csv']:
        logging.info(f'Reading alternative CSV data')
        training_data = preprocess.load_csv(IN_CSV_LIST, 'subset_CSU.LSWG.201409.csv')
    else:
        logging.info(f'Process skipped by the user: Reading alternative CSV Data')
    # ,------------------------------------------,
    # | Extracting region of interest by LAT LON |----------------------------------------------------------------------
    # '------------------------------------------'
    if workflow['extract_region']:
        logging.info(f'Extracting region of interest')
        training_data = preprocess.extract_region(dataframe=training_data,
                                                   lat_min=LAT_LIMIT[0],
                                                   lat_max=LAT_LIMIT[1],
                                                   lon_min=LAT_LIMIT[0],
                                                   lon_max=LAT_LIMIT[1])
    else:
        logging.info(f'Process skipped by the user: Extracting region of interest by LAT LON')
    # ,------------------------------,
    # | Concatenating CSV dataframes |----------------------------------------------------------------------
    # '------------------------------'
    if workflow['concatenate_csv_list_to_df']:
        logging.info(f'Reading CSV to generate a list of dataframes.')
        training_data = preprocess.load_csv_list(IN_CSV_LIST)
        logging.info(f'Concatenating list of CSV into a single dataframe')
        training_data = preprocess.concatenate_df_list(training_data)
    else:
        logging.info(f'Process skipped by the user: Concatenating list of CSV into a single dataframe')
    # ,------------------------------,
    # | Compute additional variables |----------------------------------------------------------------------------------
    # '------------------------------'
    if workflow['compute_additional_variables']:
        logging.info(f'Computing additional variables')
        logging.info(f'Input dataset columns: {list(training_data.columns.values)}')
        training_data = preprocess.compute_additional_input_vars(training_data)
        logging.info(f'Output dataset columns: {list(training_data.columns.values)}')
    else:
        logging.info(f'Process skipped by the user: Compute additional variables')
    # ,----------,
    # | Training |------------------------------------------------------------------------------------------------------
    # '----------'
    if workflow['training']:
        logging.info(f'Training')
    else:
        logging.info(f'Process skipped by the user: Training')

    # ,-----------,
    # | Read HDF5 |-----------------------------------------------------------------------------------------------------
    # '-----------'
    if workflow['pre_process_HDF5']:
        logging.info(f'Reading HDF5')
    else:
        logging.info(f'Process skipped by the user: Read HDF5')

    # ,------------,
    # | Prediction |----------------------------------------------------------------------------------------------------
    # '------------'
    if workflow['prediction']:
        logging.info(f'Predicting stuff')
    else:
        logging.info(f'Process skipped by the user: Prediction')

    # ,------------,
    # | Validation |----------------------------------------------------------------------------------------------------
    # '------------'
    if workflow['validation']:
        logging.info(f'Validating stuff')
    else:
        logging.info(f'Process skipped by the user: Validation')
    # ,-----------,
    # | Save data |----------------------------------------------------------------------------------------------------
    # '-----------'
    if workflow['save_data']:
        logging.info(f'Saving stuff')
        file_name = 'concatenated_data.csv'
        utils.tic()
        training_data.to_csv(os.path.join(OUT_CSV_LIST, file_name), index=False, sep=",", decimal='.')
        t_hour, t_min, t_sec = utils.tac_api()
        logging.info(f'Dataframe successfully saved as CSV in {t_hour}h:{t_min}m:{t_sec}s')
    else:
        logging.info(f'Process skipped by the user: Save data')


if __name__ == '__main__':
    utils.tic()
    main()
    utils.tac()
