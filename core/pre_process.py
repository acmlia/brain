import numpy as np
import pandas as pd
import os, sys
import logging


class PreProcess:
    """
    Core module for cleaning the input data and assemble of the inputs for training and post-processing.
    It can call the functions:
    LoadCSV, ExtractRegion, ThresholdRainNoRain,
    ConcatenationMonthlyDF and FitConcatenationDF.

    :param infile: sets the input file path (string)
    :param file: file name of the input CSV data (string)
    :param outfile: sets the output file of the network (string)
    """

    def print_settings(self):
        """
        Shows the settings of the main parameters necessary to process the algorithm.
        """
        logging.info(f'whois: core.pre-process')
        pass


