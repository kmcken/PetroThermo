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
    """Pressure from Peng-Robinson Equation of State
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
    """Volume Peng-Robinson Equation of State
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
        if np.iscomplex(vol[i]) or vol[i].real < b:
            volume[i] = np.nan
        else:
            volume[i] = vol[i].real

    volume = np.sort(volume)
    return volume


def dPdV(temp, press, temp_crit, press_crit, acentric_factor):
    """dP/dV_T from Peng-Robinson Equation of State
    Inputs:
        temp: Temperature, T (K)
        press: Pressure, P (Pa)
        temp_crit: Critical Temperature, Tc (K)
        press_crit: Critical Pressure, Pc (Pa)
        accntric_factor: Acentric factor, omega (unitless)
    Outputs:
        dpdv: Pressure Derivative with Respect to Volume at Constant Temperature, dP/dV_T (Pa/(mol-m^3))
    """
    t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    logging.info('{0} INFO: Calculating volume from the Peng-Robinson Equation of State.'.format(t))

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    a = a_factor(temp, temp_crit, press_crit, acentric_factor)
    b = b_factor(temp_crit, press_crit)

    vol = volume(temp, press, temp_crit, press_crit, acentric_factor)

    dpdv = np.array([np.nan, np.nan, np.nan])

    for i in range(0, 3):
        dpdv[i] = (- R * temp / (vol[i] - b)**2 + 2 * a * (vol[i] + b) / (vol[i]**2 + 2 * b * vol[i] - b**2)**2)

    return dpdv


def dPdT(temp, press, temp_crit, press_crit, acentric_factor):
    """dP/dT_V from Peng-Robinson Equation of State
    Inputs:
        temp: Temperature, T (K)
        press: Pressure, P (Pa)
        temp_crit: Critical Temperature, Tc (K)
        press_crit: Critical Pressure, Pc (Pa)
        accntric_factor: Acentric factor, omega (unitless)
    Outputs:
        dpdv: Pressure Derivative with respect to Temperature at constant Volume, dP/dT_V (Pa/K)
    """
    t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    logging.info('{0} INFO: Calculating volume from the Peng-Robinson Equation of State.'.format(t))

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    b = b_factor(temp_crit, press_crit)
    dadt = dadT(temp, temp_crit, press_crit, acentric_factor)
    vol = volume(temp, press, temp_crit, press_crit, acentric_factor)

    dpdt = np.array([np.nan, np.nan, np.nan])
    for i in range(0, 3):
        dpdt[i] = R / (vol[i] - b) + dadt / (vol[i]**2 + 2 * b * vol[i] - b**2)
    return dpdt


def dadT(temp, temp_crit, press_crit, acentric_factor):
    a = a_factor(temp, temp_crit, press_crit, acentric_factor)
    k = kappa(acentric_factor)
    return - k * a / ((1 + k * (1 - np.sqrt(temp/temp_crit)))*np.sqrt(temp * temp_crit))


def ddadT2(temp, temp_crit, press_crit, acentric_factor):
    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    ac = 0.45723553 * R ** 2 * temp_crit ** 2 / press_crit
    k = kappa(acentric_factor)
    return ac * k * np.sqrt(temp_crit / temp) * (1 + k) / (2 * temp * temp_crit)


def a_factor(temp, temp_crit, press_crit, acentric_factor):
    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    ac = 0.45723553 * R**2 * temp_crit**2 / press_crit
    k = kappa(acentric_factor)
    a = ac * (1 + k * (1 - np.sqrt(temp / temp_crit)))**2
    return a


def b_factor(temp_crit, press_crit):
    """
    Calculates the b value for Peng-Robinson Equation of State

    :param temp_crit: The substance critical temperature (K)
    :type temp_crit: float
    :param press_crit: The substance critical pressure (Pa)
    :type press_crit: float
    :return: Value of b (m**3)
    :rtype: float
    """
    R = 8.314459848  # Gas Constant: m**3 Pa / (mol K)

    if press_crit == 0:
        t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        logging.error('{0} b_factor: divide by zero error.'.format(t))
        raise ZeroDivisionError

    return 0.077796074 * R * temp_crit / press_crit



def kappa(acentric_factor):
    """
    Calculates the kappa value for Peng-Robinson Equation of State

    :param acentric_factor: The substance acentric factor (unitless)
    :type acentric_factor: float
    :return: Value of kappa
    :rtype: float
    """
    return 0.37464 + 1.54226 * acentric_factor - 0.26992 * acentric_factor ** 2
