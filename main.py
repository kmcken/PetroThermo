import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import PengRobinson as pr
import ReadFromFile as read

import UnitConverter as units


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

properties = read.get_phase_change_data('Water')
Tc = properties[1]
Pc = properties[2]
MW = properties[0]
omega = properties[5]

Pr = list()
Hv = list()
Pr_comp = list()
Hv_comp = list()

for comp in range(0, 3):
    component = 'Methane'
    Tr = np.linspace(0.48, 0.93, 100)
    if comp == 1:
        component = 'Ethane'
        Tr = np.linspace(0.3, 0.93, 100)
    if comp == 2:
        component = 'Propane'
        Tr = np.linspace(0.29, 0.93, 100)

    properties = read.get_phase_change_data(component)
    Tc = properties[1]
    Pc = properties[2]
    omega = properties[5]

    for i in range(0, len(Tr)):
        saturation = pr.saturation(Tr[i] * Tc, Tc, Pc, omega)
        Pr_comp.append(saturation[0] / Pc)
        Hv_comp.append(saturation[1] / 1000)

    Pr.append(list(Pr_comp))
    Hv.append(list(Hv_comp))
    Pr_comp.clear()
    Hv_comp.clear()

fig = plt.figure()
plt.plot(np.linspace(0.48, 0.93, 100), Pr[0], 'b', label='Methane', linestyle='-')
plt.plot(np.linspace(0.3, 0.93, 100), Pr[1], 'b', label='Ethane', linestyle='--')
plt.plot(np.linspace(0.29, 0.93, 100), Pr[2], 'b', label='Propane', linestyle=':')
plt.legend()
plt.xlabel('Reduced Temperature, Tr')
plt.ylabel('Reduced Pressure Saturation, Pr')
plt.grid(True)

fig = plt.figure()
plt.plot(np.linspace(0.48, 0.93, 100), Hv[0], 'b', label='Methane', linestyle='-')
plt.plot(np.linspace(0.3, 0.93, 100), Hv[1], 'b', label='Ethane', linestyle='--')
plt.plot(np.linspace(0.29, 0.93, 100), Hv[2], 'b', label='Propane', linestyle=':')
plt.legend()
plt.xlabel('Reduced Temperature, Tr')
plt.ylabel('Enthalpy of Vaporization, Hv (kJ/mol)')
plt.grid(True)

plt.show()

runlog.info('END Target Destroyed')
