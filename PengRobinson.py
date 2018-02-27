import datetime
import logging
import os

import numpy as np
import UnitConverter as unit
import QRDecomposition as qr

root_path = os.path.dirname(os.path.realpath(__file__))
runlog = logging.getLogger('runlog')
alglog = logging.getLogger('alglog')


def departure_H(temp, press, temp_crit, press_crit, acentric_factor):
    """
    Enthalpy Departure Function using the Peng-Robingson Equation of State

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param press: Current Pressure, P (Pa)
    :type press: float
    :param temp_crit: Substance Critical Temperature, Tc (K)
    :type temp_crit: float
    :param press_crit: Substance Critical Pressure, Pc (Pa)
    :type press_crit: float
    :param acentric_factor: Acentric Factor, omega (unitless)
    :type acentric_factor: float
    :return: Enthalpy Departure Function, Delta H
    :rtype: float
    """

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    dadt = dadT(temp, temp_crit, press_crit, acentric_factor)
    a = a_factor(temp, temp_crit, press_crit, acentric_factor)
    b = b_factor(temp_crit, press_crit)

    vol = volume(temp, press, temp_crit, press_crit, acentric_factor)
    if np.isnan(vol[2]):
        vol = vol[0]
    else:
        vol = vol[2]

    Z = press * vol / (R * temp)
    B = (b * press) / (R * temp)

    dH = R * temp * (Z - 1) - (temp * dadt - a)/(2 * np.sqrt(2) * b) * np.log((Z + (1 - np.sqrt(2)) * B) /
                                                                              (Z + (1 + np.sqrt(2)) * B))
    return dH


def departure_S(temp, press, temp_crit, press_crit, acentric_factor):
    """
    Entropy Departure Function using the Peng-Robingson Equation of State

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param press: Current Pressure, P (Pa)
    :type press: float
    :param temp_crit: Substance Critical Temperature, Tc (K)
    :type temp_crit: float
    :param press_crit: Substance Critical Pressure, Pc (Pa)
    :type press_crit: float
    :param acentric_factor: Acentric Factor, omega (unitless)
    :type acentric_factor: float
    :return: Entropy Departure Function, Delta S
    :rtype: float
    """

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    dadt = dadT(temp, temp_crit, press_crit, acentric_factor)
    b = b_factor(temp_crit, press_crit)

    vol = volume(temp, press, temp_crit, press_crit, acentric_factor)

    if np.isnan(vol[2]):
        vol = vol[0]
    else:
        vol = vol[2]

    Z = press * vol / (R * temp)
    B = (b * press) / (R * temp)

    dS = R * np.log(Z - B) - dadt / (2 * np.sqrt(2) * b) * np.log((Z + (1 - np.sqrt(2)) * B) /
                                                                  (Z + (1 + np.sqrt(2)) * B))
    return dS


def departure_G(temp, press, temp_crit, press_crit, acentric_factor):
    """
    Gibbs Free-Energy Departure Function using the Peng-Robingson Equation of State

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param press: Current Pressure, P (Pa)
    :type press: float
    :param temp_crit: Substance Critical Temperature, Tc (K)
    :type temp_crit: float
    :param press_crit: Substance Critical Pressure, Pc (Pa)
    :type press_crit: float
    :param acentric_factor: Acentric Factor, omega (unitless)
    :type acentric_factor: float
    :return: Gibbs Free-Energy Departure Function, Delta G
    :rtype: float
    """

    dS = departure_S(temp, press, temp_crit, press_crit, acentric_factor)
    dH = departure_H(temp, press, temp_crit, press_crit, acentric_factor)

    return dH - temp * dS


def departure_U(temp, press, temp_crit, press_crit, acentric_factor):
    """
    Internal Energy Departure Function using the Peng-Robingson Equation of State

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param press: Current Pressure, P (Pa)
    :type press: float
    :param temp_crit: Substance Critical Temperature, Tc (K)
    :type temp_crit: float
    :param press_crit: Substance Critical Pressure, Pc (Pa)
    :type press_crit: float
    :param acentric_factor: Acentric Factor, omega (unitless)
    :type acentric_factor: float
    :return: Internal Energy Departure Function, Delta G
    :rtype: float
    """

    vol = volume(temp, press, temp_crit, press_crit, acentric_factor)
    if np.isnan(vol[2]):
        vol = vol[0]
    else:
        vol = vol[2]
    dH = departure_H(temp, press, temp_crit, press_crit, acentric_factor)

    return dH - press * vol


def departure_A(temp, press, temp_crit, press_crit, acentric_factor):
    """
    Helmholtz Free-Energy Departure Function using the Peng-Robingson Equation of State

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param press: Current Pressure, P (Pa)
    :type press: float
    :param temp_crit: Substance Critical Temperature, Tc (K)
    :type temp_crit: float
    :param press_crit: Substance Critical Pressure, Pc (Pa)
    :type press_crit: float
    :param acentric_factor: Acentric Factor, omega (unitless)
    :type acentric_factor: float
    :return: Internal Energy Departure Function, Delta G
    :rtype: float
    """

    dU = departure_U(temp, press, temp_crit, press_crit, acentric_factor)
    dS = departure_S(temp, press, temp_crit, press_crit, acentric_factor)

    return dU - temp * dS


def pressure(temp, vol, temp_crit, press_crit, acentric_factor):
    """
    Pressure using the Peng-Robinson Equation of State

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param vol: Current Volume, V (m**3)
    :type vol: float
    :param temp_crit: Substance Critical Temperature, Tc (K)
    :type temp_crit: float
    :param press_crit: Substance Critical Pressure, Pc (Pa)
    :type press_crit: float
    :param acentric_factor: Acentric Factor, omega (unitless)
    :type acentric_factor: float
    :return: Pressure, P (Pa)
    :rtype: float
    """

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    a = a_factor(temp, temp_crit, press_crit, acentric_factor)
    b = b_factor(temp_crit, press_crit)

    press = R * temp / (vol - b) - a / (vol**2 + 2 * b * vol - b**2)
    return press


def volume(temp, press, temp_crit, press_crit, acentric_factor):
    """
    Calculates fluid from from Peng-Robinson Equation of State.
    If multiphase conditions exist, returns all three volumes.

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param pressure: Current Pressure, P (Pa)
    :type pressure: float
    :param temp_crit: Substance Critical Temperature, Tc (K)
    :type temp_crit: float
    :param press_crit: Substance Critical Pressure, Pc (Pa)
    :type press_crit: float
    :param acentric_factor: Acentric Factor, omega (unitless)
    :type acentric_factor: float
    :return: Volume, V (m**3)
    :rtype np.array
    """

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
