import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import MultiComponent as multi
import SingleComponent as single
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
# delta = read.get_binary_interations(scn_list=number)

mw_raw = list()
for n in range(0, len(number)):
    mw_raw.append(read.get_phase_change_data(scn=number[n])[0])

mw30 = list()
for n in range(0, 30):
    mw30.append(read.get_phase_change_data(scn=n + 1)[0])

zc7p = scn.z_c7plus(number, fractions)
Mc7p = scn.MW_c7plus(number, fractions)
gc7p = scn.specific_gravity_c7plus(number, fractions)

fraction7 = list()
for i in range(0, 6):
    fraction7.append(fractions[i])
fraction7.append(zc7p)


pdf_const = scn.pdf_regression(mw_raw[:5], fractions[:5])
exp_const = scn.exp_regression(number[0:5], fractions[0:5], zc7p)
# lbc_const = scn.lbc_regression(number[6:], fractions[6:], fractions[5])
# ped_const = scn.pedersen_regression(number[:5], fractions[:5])


n30 = list(np.linspace(1, 30, 30))
delta30 = read.get_binary_interations(scn_list=n30)

frac = list()
for i in range(0, 6):
    frac.append(fractions[i])
for i in range(6, 30):
    # frac.append(scn.exp_compostion(i, *exp_const))
    # frac.append(scn.katz_composition(i, zc7p))
    frac.append(scn.lbc_compostion(i, fractions[5], -0.00873587, -0.00987387))


def error(x):
    return (x[0] - x[1]) ** 2 / (x[0] * x[1])

print(frac)
print(len(frac))

### Bubble Point Calculation
T = units.to_si(220, 'degF')
P = units.to_si(12.952, 'MPa')    # LBC
# P = units.to_si(12.2501, 'MPa') # Katz
# P = units.to_si(1.2763e7, 'Pa') # Exponential
# P = units.to_si(1.31998e7, 'Pa')# Exact
Tc, Pc, w, g = list(), list(), list(), list()
for n in range(0, len(n30)):
    Tc.append(read.get_phase_change_data(scn=n30[n])[1])
    Pc.append(units.to_si(read.get_phase_change_data(scn=n30[n])[2], 'MPa'))
    w.append(read.get_phase_change_data(scn=n30[n])[3])
    g.append(read.get_phase_change_data(scn=n30[n])[4])

# temp, press, z, temp_crit, press_crit, w, delta=None, tolerance=1e-10
# print(str(multi.flash(T, P, frac, Tc, Pc, w, delta30)[2]))
print(error([12.952, 13.1998]))
print(error([12.2501, 13.1998]))
print(error([12.763, 13.1998]))
# sys.exit()

kw = scn.watson_factor(Mc7p, gc7p)
Tb = scn.watson_boiling_pt(kw, gc7p)
Tcs = scn.sancet_Tc(Mc7p)
Pcs = scn.sancet_Pc(Mc7p)
Pck = scn.kessler_Pc(kw, Tb/Tcs)
# sys.exit()

### PLOTTING
# fig = plt.figure()
# ax = plt.axes()
# width = 0.35
# plt.bar(number - width/2, fractions, width, color='b', label='Actual Composition')
# plt.bar([1 + width/2, 2 + width/2, 3 + width/2, 4 + width/2, 5 + width/2, 6 + width/2, 7 + width/2], fraction7, width, color='k', label='C1 to C7+ Composition')
# ax.xaxis.set_major_locator(plt.MultipleLocator(5))
# ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
# ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
# plt.xlabel('Single Carbon Number (SCN) Group')
# plt.ylabel('Mole Fraction')
# plt.legend()
# plt.show()

o = np.linspace(17, 300, 300)
m = np.linspace(2, 30, 29)
z_frac = list()
for i in range(1, 30):
    z_frac.append(scn.pdf_composition(16.0, *pdf_const, [mw30[i - 1], mw30[i]]))

z_pdf = list()
for i in range(0, 300):
    z_pdf.append((scn.pdf_whitson(o[i], mw_raw[0], *pdf_const)))
#
# plt.figure()
# plt.plot(o, z_pdf, 'k', label=r'$\alpha = {0}$, $\beta ={1}$'.format(np.round(pdf_const[0], 4), np.round(pdf_const[1], 4)))
# plt.yscale('log')
# plt.xlabel('Molecular Weight')
# plt.ylabel(r'Probability Density Function, $\rho(M)$')
# plt.legend()
# plt.show()

xi3, w3 = scn.gauss_laguerre_quadrature_consts(3)
xi4, w4 = scn.gauss_laguerre_quadrature_consts(4)


def mw_gauss(chi, eta, beta):
    return chi * beta + eta


for i in range(0, 3):
    print(mw_gauss(xi3[i], mw30[6], pdf_const[1]), w3[i])

for i in range(0, 4):
    print(mw_gauss(xi4[i], mw30[6], pdf_const[1]), w4[i])
sys.exit()
# print(np.sum(np.array(z_pdf[z:])))

n = np.linspace(1, 30, 300)
z_exp = scn.exp_compostion(n, *exp_const)
z_lbc = scn.lbc_compostion(n, fractions[5], -0.00873587, -0.00987387)
z_katz = scn.katz_composition(n, zc7p)
# # z_ped = scn.pedersen_composition(n, *ped_const)

plt.figure()
ax = plt.axes()
plt.bar(number, fractions)
plt.plot(n, z_exp, ':', label='Exponential')
plt.plot(n, z_lbc, '-.', label='Lorenz-Bray-Clark')
plt.plot(n, z_katz, '--', label='Katz')
plt.plot(m, z_frac, 'k', label='Whitson')
# plt.plot(n, z_ped, label='Pedersen')
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
plt.xlabel('Single Carbon Number (SCN) Group')
plt.ylabel('Mole Fraction')
plt.legend()
plt.show()

print('Target Destroyed')
runlog.info('END Target Destroyed')
