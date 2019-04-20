#! /usr/bin/env python3

# ,----------------,
# | PYTHON IMPORTS |----------------------------------------------------------------------------------------------------
# '----------------'

import logging
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

# Path for the CSV's (input and output):
IN_CSV_LIST = config('IN_CSV_LIST', default='')
OUTPUT_DIR = config('OUTPUT_DIR', default='')

# Final training data saving name
FTRD_SVN = config('FTRD_SVN', default='')

# Geographical coordinates for regional subset
# The parameters follow the structure:
# LAT_LIMIT = [min, max] and
# LON_LIMIT = [min, max]
LAT_LIMIT = [-34.0, 6.0]
LON_LIMIT = [-75.0, -35.0]

# Minimal threshold of rain rate:
THRESHOLD_RAIN = 0.1

# NNIMBUS logs
LOGFILE = OUTPUT_DIR+'nnimbus.log'
VERFILE = OUTPUT_DIR+'nnimbus.ver'

# Setting up information logs for every NNIMBUS execution in an external file
logging.basicConfig(filename=LOGFILE, format='%(asctime)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.DEBUG)

# check for existing version file to retrieve the number of the last NNIMBUS execution
if os.path.isfile(VERFILE):
    with open(VERFILE, 'r+') as f:
        str_current_version = f.readline().rstrip().split('@')[1]
        NN_RUN = int(str_current_version) + 1
        f.seek(0)
        f.write('ver@'+str(NN_RUN))
        log_found_message = f'Version file found. Saving logs at {os.path.abspath(LOGFILE)}'
else:
    NN_RUN = 0
    version_warning = f'No version file found in "{os.path.abspath(VERFILE)}" - Tagging this execution as version:{NN_RUN}'
    print(version_warning)
    logging.info(version_warning)
    with open(VERFILE, 'w+') as f:
        f.seek(0)
        f.write(f'ver@{NN_RUN}\n')



# ,--------------------------,
# | NNIMBUS WORKFLOW OPTIONS |------------------------------------------------------------------------------------------
# '--------------------------'

workflow = {
    'read_raw_csv': False,
    'read_alternative_csv': False,
    'extract_region': False,
    'concatenate_csv_list_to_df': False,
    'compute_additional_variables': False,
    'training': True,
    'pre_process_HDF5': False,
    'prediction': False,
    'validation': False,
    'save_data': False
}

# ,-----------,
# | RUN MODEL |---------------------------------------------------------------------------------------------------------
# '-----------'


def main():
    greatings_header = f'| Starting NNIMBUS # NN_RUN:{NN_RUN} |'
    separator = utils.repeat_to_length('-', len(greatings_header) - 2)
    logging.info(f',{separator},')
    logging.info(greatings_header)
    logging.info(f'\'{separator}\'')
    print(f',{separator},\n' +
          greatings_header + '\n' +
          f'\'{separator}\'')

    # Initializing core classes
    preprocess = PreProcess()
    training = Training()
    prediction = Prediction()
    validation = Validation()

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
        training.status()
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
        file_name = FTRD_SVN
        utils.tic()
        training_data.to_csv(os.path.join(OUTPUT_DIR, file_name), index=False, sep=",", decimal='.')
        t_hour, t_min, t_sec = utils.tac_api()
        logging.info(f'Dataframe successfully saved as CSV in {t_hour}h:{t_min}m:{t_sec}s')
    else:
        logging.info(f'Process skipped by the user: Save data')


if __name__ == '__main__':
    utils.tic()
    main()
    t_hour, t_min, t_sec = utils.tac_api()
    final_message = f'Elapsed execution time: {t_hour}h : {t_min}m : {t_sec}s\n'
    logging.info(utils.repeat_to_length('-', len(final_message)))
    logging.info(final_message)

