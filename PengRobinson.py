import datetime
import logging
import os

import numpy as np
import UnitConverter as unit
import QRDecomposition as qr

# LOGGING
root_path = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(filename=root_path + '/Logs/run.log', level=logging.DEBUG)
# Logging Levels:
# CRITICAL
# ERROR
# WARNING
# INFO
# DEBUG


def pressure(temp, vol, temp_crit, press_crit, acentric_factor):
    """Peng-Robinson Equation of State
    Inputs:
        temp: Temperature, T (K)
        volume: Volume, V (m^3)
        temp_crit: Critical Temperature, Tc (K)
        press_crit: Critical Pressure, Pc (Pa)
        accntric_factor: Acentric factor, omega (unitless)
    Outputs:
        press: Pressure, P (Pa)
    """
    t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    logging.info('{0} INFO: Calculating pressure from the Peng-Robinson Equation of State.'.format(t))

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    a = a_factor(temp, temp_crit, press_crit, acentric_factor)
    b = b_factor(temp_crit, press_crit)

    press = R * temp / (vol - b) - a / (vol**2 + 2 * b * vol - b**2)
    return press


def volume(temp, press, temp_crit, press_crit, acentric_factor):
    """Peng-Robinson Equation of State
    Inputs:
        temp: Temperature, T (K)
        press: Pressure, P (Pa)
        temp_crit: Critical Temperature, Tc (K)
        press_crit: Critical Pressure, Pc (Pa)
        accntric_factor: Acentric factor, omega (unitless)
    Outputs:
        volume: Volume, V (m^3)
    """
    t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    logging.info('{0} INFO: Calculating volume from the Peng-Robinson Equation of State.'.format(t))

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    a = a_factor(temp, temp_crit, press_crit, acentric_factor)
    b = b_factor(temp_crit, press_crit)

    a1 = b - R * temp / press
    a2 = a / press - 3 * b**2 - (2 * b * R * temp) / press
    a3 = b**3 + b**2 * R * temp / press - a * b / press

    vol = qr.cubic_root(a1, a2, a3)

    volume = np.array(np.zeros(3))
    for i in range(0, 3):
        if np.iscomplex(vol[i]):
            volume[i] = np.nan
        else:
            volume[i] = vol[i].real

    volume = np.sort(volume)
    return volume


def a_factor(temp, temp_crit, press_crit, acentric_factor):
    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    ac = 0.45723553 * R**2 * temp_crit**2 / press_crit
    k = 0.37464 + 1.54226 * acentric_factor - 0.26992 * acentric_factor**2
    a = ac * (1 + k * (1 - np.sqrt(temp / temp_crit)))**2
    return a


def b_factor(temp_crit, press_crit):
    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    b = 0.07779607 * R * temp_crit / press_crit
    return b
