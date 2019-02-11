#! /usr/bin/env python3

import logging
import settings
from src.preprocess import Preprocess


def main():
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S', level=logging.DEBUG)

    IN_CSV_LIST = settings.IN_CSV_LIST
    OUT_CSV_LIST = settings.OUT_CSV_LIST
    RAIN_CSV = settings.RAIN_CSV
    NORAIN_CSV = settings.NORAIN_CSV
    LAT_LIMIT = settings.LAT_LIMIT
    LON_LIMIT = settings.LON_LIMIT
    THRESHOLD_RAIN = settings.THRESHOLD_RAIN
    COLUMN_TYPES = settings.COLUMN_TYPES

    # ,---------------------,
    # | Code starts here :) |
    # '---------------------'

    minharede = Preprocess(
        IN_CSV_LIST, OUT_CSV_LIST, RAIN_CSV,
        NORAIN_CSV, LAT_LIMIT, LON_LIMIT,
        THRESHOLD_RAIN, COLUMN_TYPES)

    # Print the network initial settings
    minharede.print_settings()


if __name__ == '__main__':
    main()
