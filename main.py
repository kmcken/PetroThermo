import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import MultiComponent as multi
import SCNComposition as scn
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

names, number, fractions = read.scn_composition(root_path + '/Data/composition.txt')
delta = read.get_binary_interations(scn_list=number)

mw_raw = list()
for n in range(0, len(number)):
    mw_raw.append(read.get_phase_change_data(scn=number[n])[0])

mw30 = list()
for n in range(0, 30):
    mw30.append(read.get_phase_change_data(scn=n + 1)[0])

# sys.exit()
zc7p = scn.z_c7plus(number, fractions)

z = 6
# pdf_const = scn.pdf_regression(mw_raw[z:-2], fractions[z:-2], mw_raw[z-1])
# exp_const = scn.exp_regression(number[7:], fractions[7:], zc7p)
# lbc_const = scn.lbc_regression(number[z:-2], fractions[z:-2], fractions[z])

### Bubble Point Calculation
T = units.to_si(220, 'degF')
P = units.to_si(1.31998e7, 'Pa')
Tc, Pc, w = list(), list(), list()
for n in range(0, len(number)):
    Tc.append(read.get_phase_change_data(scn=number[n])[1])
    Pc.append(units.to_si(read.get_phase_change_data(scn=number[n])[2], 'MPa'))
    w.append(read.get_phase_change_data(scn=number[n])[3])

# temp, press, z, temp_crit, press_crit, w, delta=None, tolerance=1e-10
print(str(multi.flash(T, P, fractions, Tc, Pc, w, delta)[2]))

### PLOTTING
# n = np.linspace(7, 30, 300)
# z_pdf = list()
# for i in range(0, 30):
#     z_pdf.append(scn.pdf_composition(mw30[i], pdf_const[0], pdf_const[1], pdf_const[2]))
#
# print(np.sum(np.array(z_pdf[z:])))
# print(zc7p)
#
# z_exp = scn.exp_compostion(n, *exp_const)
# z_lbc = scn.lbc_compostion(n, *lbc_const)
# z_katz = scn.katz_composition(n, zc7p)
#
# fig = plt.figure()
# ax = plt.axes()
# plt.bar(number, fractions)
# plt.plot(np.linspace(1, 30, 30), z_pdf, 'r', label='PDF')
# plt.plot(n, z_exp, 'k--', label='Exponential')
# plt.plot(n, z_lbc, 'b-.', label='Lorenz-Bray-Clark')
# plt.plot(n, z_katz, 'g:', label='Katz')
# ax.xaxis.set_major_locator(plt.MultipleLocator(5))
# ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
# ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
# plt.xlabel('Single Carbon Number (SCN) Group')
# plt.ylabel('Mole Fraction')
# plt.legend()
# plt.show()

print('Target Destroyed')
runlog.info('END Target Destroyed')
