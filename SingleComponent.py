import logging
import os
import sys

import copy
import numpy as np
import scipy.optimize as optimize
import UnitConverter as unit
import QRDecomposition as qr

root_path = os.path.dirname(os.path.realpath(__file__))
runlog = logging.getLogger('runlog')
alglog = logging.getLogger('alglog')


def min_gibbs_volume(temp, press, temp_crit, press_crit, acentric_factor):
    """
    Volume and Z-factor for the minimum Gibb's Free Energy solution.

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
    :return: Volume and Z-factor, [V, Z]
    :rtype: [float, float]
    """

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    a = a_factor(temp, temp_crit, press_crit, acentric_factor)
    b = b_factor(temp_crit, press_crit)
    vol = volume(temp, press, temp_crit, press_crit, acentric_factor)
    Z = list()

    if np.isnan(vol[2]):
        return vol[0], press * vol[0] / (R * temp), None

    Z.append(press * vol[0] / (R * temp))
    Z.append(press * vol[2] / (R * temp))

    G = list()
    G.append(press * vol[0] - R * temp * np.log(vol[0] - b) + a / (2 * b * np.sqrt(2)) *
             np.log((vol[0] + b * (1 - np.sqrt(2))) /
                    (vol[0] + b * (1 + np.sqrt(2)))))

    G.append(press * vol[2] - R * temp * np.log(vol[2] - b) + a / (2 * b * np.sqrt(2)) *
             np.log((vol[2] + b * (1 - np.sqrt(2))) /
                    (vol[2] + b * (1 + np.sqrt(2)))))

    if G[1] - G[0] > 0:  # Liquid
        return vol[0], Z[0], 1

    if G[1] - G[0] < 0:  # Gas
        return vol[2], Z[1], 3

    # Liquid and Gas
    return [vol[0], vol[2]], Z, np.nan


def spinodal_pts(temp, temp_crit, press_crit, acentric_factor):
    """
    Finds the spinodal points from the Peng-Robinson Equation of State when T <= T_c

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param temp_crit: Substance Critical Temperature, Tc (K)
    :type temp_crit: float
    :param press_crit: Substance Critical Pressure, Pc (Pa)
    :type press_crit: float
    :param acentric_factor: Acentric Factor, omega (unitless)
    :type acentric_factor: float
    :return: Two Spinodal Points [[V1, P1], [V2, P2]]
    """

    if temp >= temp_crit:
        return np.nan, np.nan

    a = a_factor(temp, temp_crit, press_crit, acentric_factor)
    b = b_factor(temp_crit, press_crit)

    def dpdv_fun(vol):
        R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
        return - R * temp / (vol - b) ** 2 + 2 * a * (vol + b) / (vol ** 2 + 2 * b * vol - b ** 2) ** 2

    def ddpdv2_fun(vol):
        R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
        return 2 * R * temp * (vol ** 2 + 2 * b * vol - b ** 2) ** 3 - 4 * a * (2 * b + 1) * (vol - b) ** 3

    roots = list()
    roots.append(optimize.root(dpdv_fun, b + 1e-5).x[0])
    roots.append(optimize.root(dpdv_fun, roots[0] * 10).x[0])

    return roots


def saturation(temp, temp_crit, press_crit, acentric_factor, tolerance=1e-8):
    """
    Saturation Vapor Pressure and Enthalpy of Vaporization of a pure substance at given a temperature.

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param temp_crit: Substance Critical Temperature, Tc (K)
    :type temp_crit: float
    :param press_crit: Substance Critical Pressure, Pc (Pa)
    :type press_crit: float
    :param acentric_factor: Acentric Factor, omega (unitless)
    :type acentric_factor: float
    :param tolerance: Saturation Pressure iteration tolerance (Pa)
    :type tolerance: float
    :return: Saturation Vapor Pressure, Ps, and Enthalpy of Vaporization, Hv, [P_sat, H_v]
    :rtype: [float, float]
    """

    if temp >= temp_crit:
        return np.nan, np.nan

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    a = a_factor(temp, temp_crit, press_crit, acentric_factor)
    b = b_factor(temp_crit, press_crit)
    dadt = dadT(temp, temp_crit, press_crit, acentric_factor)

    # Find Saturation Pressure, P_sat
    i = 0
    P_sat_prev = 0
    spinodal_vol = spinodal_pts(temp, temp_crit, press_crit, acentric_factor)
    spinodal_press = [pressure(temp, spinodal_vol[0], temp_crit, press_crit, acentric_factor),
                      pressure(temp, spinodal_vol[1], temp_crit, press_crit, acentric_factor)]
    P_sat = press_crit
    for i in range(0, 2):
        if spinodal_press[i] < 0:
            spinodal_press[i] = 0

    p_limits = spinodal_press
    stable = min_gibbs_volume(temp, P_sat, temp_crit, press_crit, acentric_factor)[2]

    while stable is not np.nan:
        delta_P_sat = np.abs(P_sat - P_sat_prev)
        if delta_P_sat < tolerance:
            break

        P_sat_prev = copy.copy(P_sat)
        if stable is None:
            p_limits[1] = copy.copy(P_sat)
            P_sat = P_sat - 0.5 * (p_limits[1] - p_limits[0])
        else:
            if stable == 1:
                p_limits[1] = copy.copy(P_sat)
                P_sat = P_sat - 0.5 * (p_limits[1] - p_limits[0])
            else:
                p_limits[0] = copy.copy(P_sat)
                P_sat = P_sat + 0.5 * (p_limits[1] - p_limits[0])

        stable = min_gibbs_volume(temp, P_sat, temp_crit, press_crit, acentric_factor)[2]
        i += 1

    vol = volume(temp, P_sat, temp_crit, press_crit, acentric_factor)
    H_v = list()
    H_v.append(temp * R * np.log(vol[0] - b) + temp * dadt / (2 * b * np.sqrt(2)) *
               np.log((vol[0] + b * (1 - np.sqrt(2))) /
                      (vol[0] + b * (1 + np.sqrt(2)))))
    H_v.append(temp * R * np.log(vol[2] - b) + temp * dadt / (2 * b * np.sqrt(2)) *
               np.log((vol[2] + b * (1 - np.sqrt(2))) /
                      (vol[2] + b * (1 + np.sqrt(2)))))
    return P_sat, H_v[1] - H_v[0]


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

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1

    vol = volume(temp, press, temp_crit, press_crit, acentric_factor)
    if np.isnan(vol[2]):
        vol = vol[0]
    else:
        vol = vol[2]
    dH = departure_H(temp, press, temp_crit, press_crit, acentric_factor)

    return dH - press * vol + R * temp


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

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    dH = departure_H(temp, press, temp_crit, press_crit, acentric_factor)
    dS = departure_S(temp, press, temp_crit, press_crit, acentric_factor)

    vol = volume(temp, press, temp_crit, press_crit, acentric_factor)
    if np.isnan(vol[2]):
        vol = vol[0]
    else:
        vol = vol[2]

    return dH - temp * dS - press * vol + R * temp


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
    :param press: Current Pressure, P (Pa)
    :type press: float
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
    :return: Pressure Derivative with Respect to Volume at Constant Temperature, (dP/dV)_T (Pa/(mol-m^3))
    :rtype np.array
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
    """
    dP/dT_V from Peng-Robinson Equation of State

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
    :return: Pressure Derivative with respect to Temperature at constant Volume, (dP/dT)_V (Pa/K)
    :rtype np.array
    """

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    b = b_factor(temp_crit, press_crit)
    dadt = dadT(temp, temp_crit, press_crit, acentric_factor)
    vol = volume(temp, press, temp_crit, press_crit, acentric_factor)

    dpdt = np.array([np.nan, np.nan, np.nan])
    for i in range(0, 3):
        dpdt[i] = R / (vol[i] - b) + dadt / (vol[i]**2 + 2 * b * vol[i] - b**2)
    return dpdt


def dadT(temp, temp_crit, press_crit, omega):
    """
    Partial derivative of Van der Waals a-value with respect to temperature at constant pressure (da/dT)_P

    :param temp: Absolute Temperature, T (K)
    :type temp: float
    :param temp_crit: Critical Temperature, Tc (K)
    :type temp_crit: float
    :param press_crit: Critical Pressure, Pc (Pa)
    :type press_crit: float
    :param omega: Acentric Factor (unitless)
    :type omega: float
    :return: Van der Waals a-value
    :rtype: float
    """
    a = a_factor(temp, temp_crit, press_crit, omega)
    k = kappa(omega)
    return - k * a / ((1 + k * (1 - np.sqrt(temp/temp_crit)))*np.sqrt(temp * temp_crit))


def ddadT2(temp, temp_crit, press_crit, omega):
    """
    Second derivative of Van der Waals a-value with respect to temperature at constant pressure (d**2 a/dT**2)_P

    :param temp: Absolute Temperature, T (K)
    :type temp: float
    :param temp_crit: Critical Temperature, Tc (K)
    :type temp_crit: float
    :param press_crit: Critical Pressure, Pc (Pa)
    :type press_crit: float
    :param omega: Acentric Factor (unitless)
    :type omega: float
    :return: Van der Waals a-value
    :rtype: float
    """
    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    ac = 0.45723553 * R ** 2 * temp_crit ** 2 / press_crit
    k = kappa(omega)
    return ac * k * np.sqrt(temp_crit / temp) * (1 + k) / (2 * temp * temp_crit)


def a_factor(temp, temp_crit, press_crit, omega):
    """
    Calculates the Van der Waals a-value for the Peng-Robinson Equation of State.

    :param temp: Absolute Temperature, T (K)
    :type temp: float
    :param temp_crit: Critical Temperature, Tc (K)
    :type temp_crit: float
    :param press_crit: Critical Pressure, Pc (Pa)
    :type press_crit: float
    :param omega: Acentric Factor (unitless)
    :type omega: float
    :return: Van der Waals a-value
    :rtype: float
    """
    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    ac = 0.45723553 * R**2 * temp_crit**2 / press_crit
    k = kappa(omega)
    a = ac * (1 + k * (1 - np.sqrt(temp / temp_crit)))**2
    return a


def b_factor(temp_crit, press_crit):
    """
    Calculates the Van der Waals b-value for the Peng-Robinson Equation of State.

    :param temp_crit: The substance critical temperature (K)
    :type temp_crit: float
    :param press_crit: The substance critical pressure (Pa)
    :type press_crit: float
    :return: Value of b (m**3)
    :rtype: float
    """
    R = 8.314459848  # Gas Constant: m**3 Pa / (mol K)

    if press_crit == 0:
        alglog.error('b_factor: divide by zero error.')
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
