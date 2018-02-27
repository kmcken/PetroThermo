import datetime
import logging
import os
import sys

import numpy as np
import PengRobinson as pr
import PVT
import ReadFromFile as read
import UnitConverter as unit


# LOGGING
def setup_logger(name, log_file, level=logging.INFO):
    # Logging Levels:
    # CRITICAL
    # ERROR
    # WARNING
    # INFO
    # DEBUG

    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


# Setup Log Files
root_path = os.path.dirname(os.path.realpath(__file__))
runlog = setup_logger('runlog', root_path + '/Logs/run.log', level=logging.DEBUG)
alglog = setup_logger('alglog', root_path + '/Logs/alg.log')

runlog.info('START Thermodynamic Analysis of Multi-Phase Petroleum Fluids.')

#
# print("{:.2e}".format(pr.pressure(400, pr.volume(400, 10e6, 369.83, 4.248e6, 0.1523)[0], 369.83, 4.248e6, 0.1523)))

temperature = unit.to_si(25, 'degC')
pressure = unit.to_si(14.7, 'psi')
substance = 'Methane'
MW, Tc, Pc, Ttrip, Ptrip, Acentric = read.get_phase_change_data(name=substance)


print(pr.volume(temperature, pressure, Tc, Pc, Acentric))
print(pr.dPdV(temperature, pressure, Tc, Pc, Acentric))
print('Cp of ' + substance + ': ' + str(np.round(PVT.cp(temperature, substance) / MW, 4)) + ' kJ/(kg-K) at ' + str(np.round(unit.from_si(temperature, 'degF'), 2)) + ' F and ' + str(np.round(unit.from_si(pressure, 'psi'), 2)) + ' psi')
print('Cv of ' + substance + ': ' + str(np.round(PVT.cv(temperature, pressure, substance) / MW, 4)) + ' kJ/(kg-K) at ' + str(np.round(unit.from_si(temperature, 'degF'), 2)) + ' F and ' + str(np.round(unit.from_si(pressure, 'psi'), 2)) + ' psi')
print('Speed of Sound in ' + substance + ': ' + str(np.round(PVT.speed_of_sound(temperature,  pressure, substance), 2)) + ' m/s')


t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
logging.info('{0} END: Target Destroyed.'.format(t))
