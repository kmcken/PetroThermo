import datetime
import logging
import os

# LOGGING
root_path = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(filename=root_path + '/Logs/run.log', level=logging.DEBUG)
# Logging Levels:
# CRITICAL
# ERROR
# WARNING
# INFO
# DEBUG


def read_file(file):
    t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    logging.info('{0} INFO: Read from {1} file.'.format(t, file))

    txt_file = list()
    f = open(file, 'r', encoding='utf-8-sig')
    for line in f:
        txt_file.append(line.strip().split(','))
    f.close()

    return txt_file


def heat_capacity_constants(name=None, symbol=None, file=None):
    t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    logging.info('{0} INFO: Getting heat capacities.'.format(t))
    """Heat Capacity Constants for Cp = a + b/1e2 T + c/1e5 T^2 + d/1e9 T^3"""

    if name is None and symbol is None:
        print('No name or symbol input')
        t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        logging.warning('{0} WARNING: No name or symbol input.'.format(t))
        raise ValueError

    if file is None:
        file = root_path + '/MolarHeatCapacities.txt'

    a, b, c, d, temp_range = None, None, None, None, [None, None]
    with open(file, 'r', encoding='utf-8-sig') as f:
        for line in f:
            parts = line.split(',')
            if parts[0] == name:
                a, b, c, d, temp_range = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]), \
                                         [float(parts[6]), float(parts[7])]
            if parts[1] == symbol:
                a, b, c, d, temp_range = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]), \
                                         [float(parts[6]), float(parts[7])]

    return a, b, c, d, temp_range
