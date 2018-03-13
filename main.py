import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import SingleComponent as Single
import MultiComponent as Multi
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

Tc = (425.1, 469.7)
Pc = (units.to_si(37.96, 'bar'), units.to_si(33.7, 'bar'))
w = (0.2, 0.252)
z = (0.3563, 0.6437)
delta = ([0, 0], [0, 0])

print(Multi.a_factor(100, Tc, Pc, z, w, delta))
print(Multi.b_factor(Tc, Pc, z))

poles = np.array([-1, 1.2, 1.5, 2, 3])
print((np.extract(poles < 1, poles) > 0).any())
print((poles < 0).any())
if not (poles < 0).any() or not (poles > 1).any():
    print('No solution')

print(np.min(np.extract(poles > 1, poles)))
print(np.max(np.extract(poles < 0, poles)))
print([np.max(np.extract(poles < 0, poles)), np.min(np.extract(poles > 1, poles))])
print(np.where(poles > 1)[0])
print(np.extract(poles > 1, poles))

sys.exit()

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
        saturation = Single.saturation(Tr[i] * Tc, Tc, Pc, omega)
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
