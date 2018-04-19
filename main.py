import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import SCNComposition as scn
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

names, number, fractions = read.scn_composition(root_path + '/Data/composition.txt')

zc7p = scn.z_c7plus(number, fractions)

exp_const = scn.exp_regression(number[7:], fractions[7:], zc7p)
lbc_const = scn.lbc_regression(number[5:-2], fractions[5:-2], fractions[5])

n = np.linspace(1, 40, 400)
z_exp = scn.exp_compostion(n, *exp_const)
z_lbc = scn.lbc_compostion(n, *lbc_const)

fig = plt.figure()
ax = plt.axes()
plt.bar(number, fractions)
plt.plot(n, z_exp, 'k--', label='Exponential')
plt.plot(n, z_lbc, 'b--', label='Lorenz-Bray-Clark')
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
plt.xlabel('Single Carbon Number (SCN) Group')
plt.ylabel('Mole Fraction')
plt.legend()
plt.show()

print('Target Destroyed')
runlog.info('END Target Destroyed')
