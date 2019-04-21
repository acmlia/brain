#! /usr/bin/env python3

# ,----------------,
# | PYTHON IMPORTS |----------------------------------------------------------------------------------------------------
# '----------------'

import logging
import os

from pprint import pformat
from decouple import config
from decouple import Csv
from core.pre_process import PreProcess
from core.prediction import Prediction
from core.validation import Validation
from core import utils

# ,----------------------,
# | ENVIRONMENT SETTINGS |----------------------------------------------------------------------------------------------
# '----------------------'
"""
Environment settings comes from the external .env file through the use of the python-decouple module.
 
"""
# Tag with the PC name (please avoid using special characteres)
PCTAG = config('PCTAG', default='default')

# Path for the CSV's (input and output):
IN_CSV_PATH = config('IN_CSV_PATH', default='')
IN_CSV_NAME = config('IN_CSV_NAME', default='')
OUTPUT_DIR = config('OUTPUT_DIR', default='')

# Final training data saving name
TRNGCSV_TO_SAVE = config('TRNGCSV_TO_SAVE', default='')

# Random seed for reproducible training
RANDOM_SEED = config('RANDOM_SEED', default=0, cast=int)

# Geographical coordinates for regional subset
LAT_LIMIT = config('LAT_LIMIT', default=0, cast=Csv(float))
LON_LIMIT = config('LON_LIMIT', default=0, cast=Csv(float))

# Minimal threshold of rain rate:
THRESHOLD_RAIN = 0.1

# NNIMBUS logs
LOGFILE = OUTPUT_DIR+f'nnimbus_{PCTAG}.log'
VERFILE = OUTPUT_DIR+f'nnimbus_{PCTAG}.ver'

# Setting up information logs for every NNIMBUS execution in an external file
logging.basicConfig(filename=LOGFILE, format='%(asctime)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.DEBUG)

# check for existing version file to retrieve the number of the last NNIMBUS execution
if os.path.isfile(VERFILE):
    with open(VERFILE, 'r+') as f:
        str_current_version = f.readline().rstrip().split('#')[1]
        NN_RUN = int(str_current_version) + 1
        f.seek(0)
        f.write('NN_Run #'+str(NN_RUN))
        log_found_message = f'Version file found. Saving logs at {os.path.abspath(LOGFILE)}'
else:
    NN_RUN = 0
    version_warning = f'> No version file found in "{os.path.abspath(VERFILE)}"\n' \
        f'> Tagging this execution as version: #{NN_RUN}'
    print(version_warning)
    logging.info(version_warning)
    with open(VERFILE, 'w+') as f:
        f.seek(0)
        f.write(f'NN_Run #{NN_RUN}\n')

# ,--------------------------,
# | NNIMBUS WORKFLOW OPTIONS |------------------------------------------------------------------------------------------
# '--------------------------'

workflow = {
    'read_raw_csv': False,
    'read_alternative_csv': False,
    'extract_region': False,
    'concatenate_csv_list_to_df': False,
    'compute_additional_variables': False,
    'training': False,
    'pre_process_HDF5': False,
    'prediction': False,
    'validation': False,
    'save_data': False
}

# ,-----------,
# | RUN MODEL |---------------------------------------------------------------------------------------------------------
# '-----------'


def main():
    greatings_header = f'| Starting NNIMBUS @ {PCTAG} # NN_RUN:{NN_RUN} |'
    separator = utils.repeat_to_length('-', len(greatings_header) - 2)
    logging.info(f',{separator},')
    logging.info(greatings_header)
    logging.info(f'\'{separator}\'')
    print(f',{separator},\n' +
          greatings_header + '\n' +
          f'\'{separator}\'')

    # Initializing core classes
    preprocess = PreProcess()
    prediction = Prediction()
    validation = Validation()
    # ,----------------------------,
    # | SAVING MODEL CONFIG TO LOG |------------------------------------------------------------------------------------
    # '----------------------------'
    logging.info(f'User-defined workflow:\n\n{pformat(workflow)}\n')
    logging.info(f'Environment variables:\n\n'
                 f'PCTAG = {PCTAG}\n'
                 f'IN_CSV_PATH = {IN_CSV_PATH}\n'
                 f'IN_CSV_NAME = {IN_CSV_NAME}\n'
                 f'OUTPUT_DIR = {OUTPUT_DIR}\n'
                 f'TRNGCSV_TO_SAVE = {TRNGCSV_TO_SAVE}\n'
                 f'LAT_LIMIT = {LAT_LIMIT}\n'
                 f'LON_LIMIT = {LON_LIMIT}\n'
                 f'THRESHOLD_RAIN = {THRESHOLD_RAIN}\n'
                 f'VERSION_FILE = {os.path.abspath(VERFILE)}\n'
                 f'LOG_FILE = {os.path.abspath(LOGFILE)}\n'
                 f'RANDOM_SEED = {RANDOM_SEED}\n')

    # ,----------------------,
    # | Reading raw CSV Data |------------------------------------------------------------------------------------------
    # '----------------------'
    if workflow['read_raw_csv']:
        logging.info(f'Reading RAW Randel CSV data')
        training_data = preprocess.load_raw_csv(IN_CSV_PATH, 'CSU.LSWG.201409.bin.csv')

    # ,------------------------------,
    # | Reading alternative CSV Data |----------------------------------------------------------------------------------
    # '------------------------------'
    if workflow['read_alternative_csv']:
        logging.info(f'Reading alternative CSV data')
        training_data = preprocess.load_csv(IN_CSV_PATH, 'subset_CSU.LSWG.201409.csv')

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

    # ,------------------------------,
    # | Concatenating CSV dataframes |----------------------------------------------------------------------
    # '------------------------------'
    if workflow['concatenate_csv_list_to_df']:
        logging.info(f'Reading CSV to generate a list of dataframes.')
        training_data = preprocess.load_csv_list(IN_CSV_PATH)
        logging.info(f'Concatenating list of CSV into a single dataframe')
        training_data = preprocess.concatenate_df_list(training_data)

    # ,------------------------------,
    # | Compute additional variables |----------------------------------------------------------------------------------
    # '------------------------------'
    if workflow['compute_additional_variables']:
        logging.info(f'Computing additional variables')
        logging.info(f'Input dataset columns: {list(training_data.columns.values)}')
        training_data = preprocess.compute_additional_input_vars(training_data)
        logging.info(f'Output dataset columns: {list(training_data.columns.values)}')

    # ,----------,
    # | Training |------------------------------------------------------------------------------------------------------
    # '----------'
    if workflow['training']:
        training_import_warning = f'> Importing training modules, this may take a while...'
        print(training_import_warning)
        logging.info(training_import_warning)

        from core.training import Training

        retrieval = Training(random_seed=7,
                             version='NN'+str(NN_RUN),
                             csv_entry=IN_CSV_NAME,
                             csv_path=IN_CSV_PATH,
                             figure_path=OUTPUT_DIR + 'ann_training/',
                             model_out_path=OUTPUT_DIR + 'ann_training/',
                             model_out_name=f'final_ann_{NN_RUN}')
        retrieval.status()
        retrieval.autoExecReg()
    # ,-----------,
    # | Read HDF5 |-----------------------------------------------------------------------------------------------------
    # '-----------'
    if workflow['pre_process_HDF5']:
        logging.info(f'Reading HDF5')

    # ,------------,
    # | Prediction |----------------------------------------------------------------------------------------------------
    # '------------'
    if workflow['prediction']:
        logging.info(f'Predicting stuff')

    # ,------------,
    # | Validation |----------------------------------------------------------------------------------------------------
    # '------------'
    if workflow['validation']:
        logging.info(f'Validating stuff')

    # ,-----------,
    # | Save data |----------------------------------------------------------------------------------------------------
    # '-----------'
    if workflow['save_data']:
        logging.info(f'Saving stuff')
        file_name = TRNGCSV_TO_SAVE
        utils.tic()
        training_data.to_csv(os.path.join(OUTPUT_DIR, file_name), index=False, sep=",", decimal='.')
        t_hour, t_min, t_sec = utils.tac()
        logging.info(f'Dataframe successfully saved as CSV in {t_hour}h:{t_min}m:{t_sec}s')


if __name__ == '__main__':
    utils.tic()
    main()
    t_hour, t_min, t_sec = utils.tac()
    final_message = f'| Elapsed execution time: {t_hour}h : {t_min}m : {t_sec}s |'
    separator = utils.repeat_to_length('-', len(final_message) - 2)
    logging.info(f',{separator},')
    logging.info(final_message)
    logging.info(f'\'{separator}\'')
