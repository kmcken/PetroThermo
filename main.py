import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import PengRobinson as pr
import ReadFromFile as read


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

methane_prop = read.get_phase_change_data('Methane')
Tc = methane_prop[1]
Pc = methane_prop[2]
omega = methane_prop[5]

Pr = np.linspace(0.1, 30, 1000)
Tr = np.linspace(0.5, 4, 36)

T = Tr * Tc
P = Pr * Pc

plt.figure()
DeltaH = list()
for t in range(0, len(Tr)):
    print('Tr = ' + str(Tr[t]))
    for p in range(0, len(Pr)):
        DeltaH.append(pr.departure_H(T[t], P[p], Tc, Pc, omega)/Tc * -1 /4.184)

    plt.plot(Pr, DeltaH, 'b')
    DeltaH.clear()

plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.5)
plt.xscale('log')
plt.xlabel('Reduced Pressure, Pr')
plt.ylabel('(H^IG - H)/Tc, cal/mol-K')
plt.show()

plt.figure()
DeltaS = list()
for t in range(0, len(Tr)):
    print('Tr = ' + str(Tr[t]))
    for p in range(0, len(Pr)):
        DeltaS.append(pr.departure_S(T[t], P[p], Tc, Pc, omega) * -1 /4.184)

    plt.plot(Pr, DeltaS, 'b')
    DeltaS.clear()

plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.5)
plt.xscale('log')
plt.xlabel('Reduced Pressure, Pr')
plt.ylabel('(S^IG - S), cal/mol-K')
plt.show()

runlog.info('END Target Destroyed')
