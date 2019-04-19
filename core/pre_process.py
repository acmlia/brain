import logging
import os
import sys

import pandas as pd
import numpy as np
from core import utils

class PreProcess:
    """
    Nada.
    """
    COLUMN_TYPES = {'numpixs': 'int64', 'lat': 'float64', 'lon': 'float64', 'sfccode': 'float64', 'T2m': 'float64',
                    'tcwv': 'float64', 'skint': 'float64', 'sfcprcp': 'float64', 'cnvprcp': 'float64',
                    '10V': 'float64', '10H': 'float64', '18V': 'float64', '18H': 'float64',
                    '23V': 'float64', '36V': 'float64', '36H': 'float64', '89V': 'float64',
                    '89H': 'float64', '166V': 'float64', '166H': 'float64', '186V': 'float64',
                    '190V': 'float64', 'emis10V': 'float64', 'emis10H': 'float64', 'emis18V': 'float64',
                    'emis18H': 'float64', 'emis23V': 'float64', 'emis36V': 'float64', 'emis36H': 'float64',
                    'emis89V': 'float64', 'emis89H': 'float64', 'emis166V': 'float64', 'emis166H': 'float64',
                    'emis186V': 'float64', 'emis190V': 'float64'}

    @staticmethod
    def status():
        """
        Shows the settings of the main parameters necessary to process the algorithm.
        """
        logging.info(f'{__name__} OK')
        pass

    def load_raw_csv(self, path, file):
        """
        WARNING: This function is intended only to read raw randel csv data:
            /path/to/dir/CSU.LSWG.201409.bin.csv
            /path/to/dir/CSU.LSWG.201410.bin.csv
            /path/to/dir/CSU.LSWG.201411.bin.csv
        """
        df = pd.DataFrame()
        if file.startswith(".", 0, len(file)):
            logging.info(f"File name starts with point: {file} - Skipping...")
        elif file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, file), sep=',', decimal='.',
                             dtype=self.COLUMN_TYPES, header=5, skipinitialspace=True)
            logging.info(f'Dataframe {file} was loaded')
        if not df.empty:
            return df
        else:
            logging.info(f"Unexpected file format: {file} - Skipping...")
            return None

    def load_csv(self, path, file):
        """
        Nada.
        """
        df = pd.DataFrame()
        if file.startswith(".", 0, len(file)):
            logging.info(f"File name starts with point: {file} - Skipping...")
        elif file.endswith(".csv"):
            utils.tic()
            df = pd.read_csv(os.path.join(path, file), sep=',', decimal='.')
            t_hour, t_min, t_sec = utils.tac_api()
            logging.info(f'{file} successfully loaded into a dataframe in {t_hour}h:{t_min}m:{t_sec}s')
        if not df.empty:
            return df
        else:
            logging.info(f"Unexpected file format: {file}")
            sys.exit('Empty CSV is invalid - System halt by unmet conditions.')

    def load_csv_list(self, path):
        """
        This function will read all csv files inside the given path:
        """
        df_list = []
        logging.info(f'Reading files:\n{os.listdir(path)}\n')
        for file in os.listdir(path):
            df = self.load_csv(path, file)
            df_list.append(df)
        logging.info(f'Final list size after loading: {len(df_list)} files.')
        return df_list

    @staticmethod
    def extract_region(dataframe, lat_max, lat_min, lon_max, lon_min):
        """
        Extract regional areas from the global dataset (original)
        :param dataframe: original global dataframe (DataFrame)
        :return:  dataframe_regional (DataFrame)
        """
        logging.info(f'Extracting user-specified regions from dataframe using defined limits:'
                     f'LAT: "{[lat_min, lat_max]}"'
                     f'LON: "{[lon_min, lon_max]}"')

        subset = dataframe.where(
            (dataframe['lat'] <= lat_max) &
            (dataframe['lat'] >= lat_min) &
            (dataframe['lon'] <= lon_max) &
            (dataframe['lon'] >= lon_min))

        regional_dataframe = dataframe.iloc[subset]
        return regional_dataframe

    def concatenate_df_list(self, df_list):
        """
        Concatenate a given list of dataframes into one single dataframe.
        """
        # Diagnose the size of the input list
        checksum = 0
        logging.info(f'Total number of dataframes in list: {len(df_list)}')
        for idx, df in enumerate(df_list):
            logging.info(f'Dataframe #{idx} size: {len(df)}')
            checksum += len(df)
        # Concatenation of the dataframes inside the list into a single dataframe:
        try:
            logging.info(f'Concatenating files...')
            concatenated_df = pd.concat(df_list, sort=False, ignore_index=True, verify_integrity=True)
            logging.info(f'Dataframe concatenation completed!\n'
                         f'Sum of dataframes in the input list: {checksum}\n'
                         f'Final dataframe size --------------: {len(concatenated_df)}')
        except ValueError as e:
            logging.error("ValueError:", e)
            sys.exit(1)

        return concatenated_df

    @staticmethod
    def compute_additional_input_vars(df):
        '''
        Create new input variables from the dataset, as PCT, SSI, MPDI, etc...
        '''
        expected_columns = ['10V', '10H', '18V', '18H', '36V', '36H',
                            '89V', '89H', '166V', '166H', '186V', '190V']
        if set(expected_columns).issubset(set(list(df.columns.values))):
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
            df['MPDI_scaled'] = df['MPDI'] * 600
            df['SI'] = df['23V'] - df['89V']

            # Inclugin the PCT formulae: PCTf= (1+alfa)*TBfv - alfa*TBfh
            alfa10 = 1.5
            alfa18 = 1.4
            alfa36 = 1.15
            alfa89 = 0.7

            df['PCT10'] = (1 + alfa10) * df['10V'] - alfa10 * df['10H']
            df['PCT18'] = (1 + alfa18) * df['18V'] - alfa18 * df['18H']
            df['PCT36'] = (1 + alfa36) * df['36V'] - alfa36 * df['36H']
            df['PCT89'] = (1 + alfa89) * df['89V'] - alfa89 * df['89H']
            return df
        else:
            logging.info(f'Some of the expected columns where not present in the input dataframe.')
            logging.info('Expected columns:'
                         '\n{}\n'
                         'Found columns:'
                         '\n{}\n'
                         'System halt by unmet conditions.'.format(expected_columns, list(df.columns.values)))
            sys.exit(1)
