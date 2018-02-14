import datetime
import logging
import os
import sys

import numpy as np
import PengRobinson as pr
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

print(pr.pressure(400, 1.3685e-4, 369.83, 4.248e6, 0.1523)/1e6)
print(pr.volume(400, 10e6, 369.83, 4.248e6, 0.1523))
print(pr.volume(310, 1e6, 369.83, 4.248e6, 0.1523))

t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
logging.info('{0} END: Target Destroyed.'.format(t))
