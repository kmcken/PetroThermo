import datetime
import logging
import os
import sys

import ReadFromFile as read
import UnitConverter as unit

# LOGGING
root_path = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(filename=root_path + '/Logs/run.log', level=logging.DEBUG)
# Logging Levels:
# CRITICAL
# ERROR
# WARNING
# INFO
# DEBUG

t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
logging.info('{0} START: Starting data analysis.'.format(t))

length = 1
print(str(length) + ' m = ' + str(unit.from_si(length, 'ft')) + ' ft.')

cp_file = root_path + '/MolarHeatCapacities.txt'
print(read.heat_capacity_constants(name='Air'))

t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
logging.info('{0} END: Target Destroyed.'.format(t))
