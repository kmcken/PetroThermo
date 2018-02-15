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

print(273.15-213.4)
length = 1
print(str(length) + ' m = ' + str(unit.from_si(length, 'ft')) + ' ft.')
farenheit = 100
print(str(farenheit) + ' F = ' + str(unit.to_si(farenheit, 'degF')) + ' K.')
psi = 100
print('14.7 psia = ' + str(unit.to_si(14.7, 'psi')) + ' Pa.')
print(str(100) + ' psia = ' + str(unit.to_si(psi, 'psi')) + ' Pa.')
#
# print("{:.2e}".format(pr.pressure(400, pr.volume(400, 10e6, 369.83, 4.248e6, 0.1523)[0], 369.83, 4.248e6, 0.1523)))
# print(pr.volume(310, 1e6, 369.83, 4.248e6, 0.1523))
# print(pr.volume(400, 10e6, 369.83, 4.248e6, 0.1523))
# print(pr.dPdV(310, 1e6, 369.83, 4.248e6, 0.1523))
# print(pr.dPdV(400, 10e6, 369.83, 4.248e6, 0.1523))
# print(pr.dPdT(310, 1e6, 369.83, 4.248e6, 0.1523))
# print(pr.dPdT(400, 10e6, 369.83, 4.248e6, 0.1523))
# print(read.heat_capacity_constants('Methane'))

print(read.get_heat_capacity_constants(formula='Air'))
print(read.get_phase_change_data(name='Air'))

t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
logging.info('{0} END: Target Destroyed.'.format(t))
