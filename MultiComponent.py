import datetime
import logging
import os

import copy
import numpy as np
import SingleComponent as Single
import UnitConverter as Units
import QRDecomposition as qr

root_path = os.path.dirname(os.path.realpath(__file__))
runlog = logging.getLogger('runlog')
alglog = logging.getLogger('alglog')


def volume_min_G(temp, press, temp_crit, press_crit, acentric_factor):
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


def saturation(temp, temp_crit, press_crit, acentric_factor, tolerance=0.001):
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
    P_sat = 0.5 * press_crit
    P_sat_prev = 0
    p_limits = [0, press_crit]
    stable = volume_min_G(temp, P_sat, temp_crit, press_crit, acentric_factor)[2]

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

        stable = volume_min_G(temp, P_sat, temp_crit, press_crit, acentric_factor)[2]
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



def pressure(temp, vol, z, temp_crit, press_crit, w, delta):
    """
    Pressure using the Peng-Robinson Equation of State

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param vol: Current Volume, P (Pa)
    :type vol: ()
    :param z: Vector that contains the composition of the mixture (unitless)
    :type z: ()
    :param temp_crit: Vector of Critical Temperature, Tc (K)
    :type temp_crit: ()
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :type press_crit: ()
    :param w: Vector of Acentric Factor, omega (unitless)
    :type w: ()
    :param delta: Square matrix containing the binary interaction coefficients
    :return: Pressure, P (Pa)
    :rtype: float
    """

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    a = a_factor(temp, temp_crit, press_crit, z, w, delta)
    b = b_factor(temp_crit, press_crit, z)

    press = R * temp / (vol[0] - b) - a / (vol[0]**2 + 2 * b * vol[0] - b**2)
    return press


def volume(temp, press, z, temp_crit, press_crit, w, delta):
    """
    Multicomponent volume from the Peng-Robinson Equation of State.
    If multiphase conditions exist, returns all three volumes.

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param press: Current Pressure, P (Pa)
    :type press: float
    :param z: Vector that contains the composition of the mixture (unitless)
    :type z: ()
    :param temp_crit: Vector of Critical Temperature, Tc (K)
    :type temp_crit: ()
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :type press_crit: ()
    :param w: Vector of Acentric Factor, omega (unitless)
    :type w: ()
    :param delta: Square matrix containing the binary interaction coefficients
    :return: Volume, vol (m**3)
    :rtype np.array
    """

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    a = a_factor(temp, temp_crit, press_crit, z, w, delta)
    b = b_factor(temp_crit, press_crit, z)

    a1 = b - R * temp / press
    a2 = a / press - 3 * b**2 - (2 * b * R * temp) / press
    a3 = b**3 + b**2 * R * temp / press - a * b / press

    vol = qr.cubic_root(a1, a2, a3)

    vol = np.array(np.zeros(3))
    for i in range(0, 3):
        if np.iscomplex(vol[i]) or vol[i].real < b:
            vol[i] = np.nan
        else:
            vol[i] = vol[i].real

    vol = np.sort(vol)
    return vol


def dPdV(temp, press, z, temp_crit, press_crit, w, delta):
    """
    Multicomponent partial derivative of pressure with respect to volume at constant temperature, (dP/dV)_T,
    from Peng-Robinson Equation of State

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param press: Current Pressure, P (Pa)
    :type press: float
    :param z: Vector that contains the composition of the mixture (unitless)
    :type z: ()
    :param temp_crit: Vector of Critical Temperature, Tc (K)
    :type temp_crit: ()
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :type press_crit: ()
    :param w: Vector of Acentric Factor, omega (unitless)
    :type w: ()
    :param delta: Square matrix containing the binary interaction coefficients
    :return: Pressure Derivative with Respect to Volume at Constant Temperature, (dP/dV)_T (Pa/(mol-m^3))
    :rtype np.array
    """

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    a = a_factor(temp, temp_crit, press_crit, z, w, delta)
    b = b_factor(temp_crit, press_crit, z)

    vol = volume(temp, press, z, temp_crit, press_crit, w, delta)

    dpdv = np.array([np.nan, np.nan, np.nan])

    for i in range(0, 3):
        dpdv[i] = (- R * temp / (vol[i] - b)**2 + 2 * a * (vol[i] + b) / (vol[i]**2 + 2 * b * vol[i] - b**2)**2)

    return dpdv


def dPdT(temp, press, temp_crit, press_crit, z, w, delta):
    """
    Multicomponent partial derivative of pressure with respect to temperature at constant volume, (dP/dT)_V,
    from Peng-Robinson Equation of State

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param press: Current Pressure, P (Pa)
    :type press: float
    :param temp_crit: Vector of Critical Temperature, Tc (K)
    :type temp_crit: ()
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :type press_crit: ()
    :param z: Vector that contains the composition of the mixture (unitless)
    :type z: ()
    :param w: Vector of Acentric Factor, omega (unitless)
    :type w: ()
    :param delta: Square matrix containing the binary interaction coefficients
    :return: Pressure Derivative with respect to Temperature at constant Volume, (dP/dT)_V (Pa/K)
    :rtype np.array
    """

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1
    b = b_factor(temp_crit, press_crit, z)
    dadt = dadT(temp, temp_crit, press_crit, z, w, delta)
    vol = volume(temp, press, z, temp_crit, press_crit, w, delta)

    dpdt = np.array([np.nan, np.nan, np.nan])
    for i in range(0, 3):
        dpdt[i] = R / (vol[i] - b) + dadt / (vol[i]**2 + 2 * b * vol[i] - b**2)
    return dpdt


def dadT(temp, temp_crit, press_crit, z, w, delta):
    """
    Partial derivative of Van der Waals a-value with respect to temperature at constant pressure (da/dT)_P

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param temp_crit: Vector of Critical Temperature, Tc (K)
    :type temp_crit: ()
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :type press_crit: ()
    :param z: Vector that contains the composition of the mixture (unitless)
    :type z: ()
    :param w: Vector of Acentric Factor, omega (unitless)
    :type w: ()
    :param delta: Square matrix containing the binary interaction coefficients
    :return: dadT, (da/dT)_P
    :rtype float
    """

    N = len(temp_crit)
    if len(press_crit) and len(z) and len(w) and len(delta) is not N:
        raise ValueError('Size of inputs do not agree.')

    a_vector = list()
    for i in range(0, N):
        a_vector.append(Single.a_factor(temp, temp_crit[i], press_crit[i], w[i]))

    dadt_vector = list()
    for i in range(0, N):
        dadt_vector.append(Single.dadT(temp, temp_crit[i], press_crit[i], w[i]))

    dadt = 0
    for i in range(0, N):
        for j in range(0, N):
            dadt += 0.5 * z[i] * z[j] * (1 - delta[j][i]) * (np.sqrt(a_vector[j] / a_vector[i]) * dadt_vector[i] +
                                                             np.sqrt(a_vector[i] / a_vector[j]) * dadt_vector[j])

    return dadt


def ddadT2(temp, temp_crit, press_crit, z, w, delta):
    """
    Second derivative of Van der Waals a-value with respect to temperature at constant pressure (d**2 a/dT**2)_P

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param temp_crit: Vector of Critical Temperature, Tc (K)
    :type temp_crit: ()
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :type press_crit: ()
    :param z: Vector that contains the composition of the mixture (unitless)
    :type z: ()
    :param w: Vector of Acentric Factor, omega (unitless)
    :type w: ()
    :param delta: Square matrix containing the binary interaction coefficients
    :return: ddadT2, (d**2 a/dT**2)_P
    :rtype float
    """

    N = len(temp_crit)
    if len(press_crit) and len(z) and len(w) and len(delta) is not N:
        raise ValueError('Size of inputs do not agree.')

    a_vector = list()
    for i in range(0, N):
        a_vector.append(Single.a_factor(temp, temp_crit[i], press_crit[i], w[i]))

    dadt_vector = list()
    for i in range(0, N):
        dadt_vector.append(Single.dadT(temp, temp_crit[i], press_crit[i], w[i]))

    ddadt2_vector = list()
    for i in range(0, N):
        ddadt2_vector.append(Single.ddadT2(temp, temp_crit[i], press_crit[i],w[i]))

    ddat2 = 0
    for i in range(0, N):
        for j in range(0, N):
            ddat2 += 0.5 * 0.5 * z[i] * z[j] * (1 - delta[j][i]) * \
                    (dadt_vector[i] * dadt_vector[j] / np.sqrt(a_vector[i] * a_vector[j]) +
                     ddadt2_vector[i] * np.sqrt(a_vector[j]) / np.sqrt(a_vector[i]) +
                     ddadt2_vector[j] * np.sqrt(a_vector[i]) / np.sqrt(a_vector[j]) -
                     0.5 * (dadt_vector[i] ** 2 * np.sqrt(a_vector[j]) / np.sqrt(a_vector[i] ** 3) +
                            dadt_vector[j] ** 2 * np.sqrt(a_vector[i]) / np.sqrt(a_vector[j] ** 3)))


def a_factor(temp, temp_crit, press_crit, z, w, delta):
    """
    Calculates the Van der Waals a-value of a multicomponent mixture for the Peng-Robinson Equation of State.

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param temp_crit: Vector of Critical Temperature, Tc (K)
    :type temp_crit: ()
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :type press_crit: ()
    :param z: Vector that contains the composition of the mixture (unitless)
    :type z: ()
    :param w: Vector of Acentric Factor, omega (unitless)
    :type w: ()
    :param delta: Square matrix containing the binary interaction coefficients
    :return: a-value
    :rtype float
    """

    N = len(temp_crit)
    if len(press_crit) and len(z) and len(w) and len(delta) is not N:
        raise ValueError('Size of inputs do not agree.')

    a_vector = list()
    for i in range(0, N):
        a_vector.append(Single.a_factor(temp, temp_crit[i], press_crit[i], w[i]))

    a = 0
    for i in range(0, N):
        for j in range(0, N):
            a += w[i] * w[j] * np.sqrt(a_vector[i] * a_vector[j]) * (1 - delta[j][i])

    return a


def b_factor(temp_crit, press_crit, z):
    """
    Calculates the Van der Waals b-value of a multicomponent mixture for the Peng-Robinson Equation of State.

    :param temp_crit: Vector of Critical Temperature, Tc (K)
    :type temp_crit: ()
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :type press_crit: ()
    :param z: Vector that contains the composition of the mixture (unitless)
    :type z: ()
    :return: Value of b (m**3)
    :rtype: float
    """

    N = len(temp_crit)
    if len(press_crit) and len(z) is not N:
        raise ValueError('Size of inputs do not agree.')

    b = 0
    for i in range(0, N):
        b += z[i] * Single.b_factor(temp_crit[i], press_crit[i])

    return b
