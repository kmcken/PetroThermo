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


def RachfordRice(z, K, tolerance=1e-10):
    """
    Rachford-Rice equation
    :param z: Vector of the composition of the mixture (unitless)
    :param K: Vector that contains the composition of the mixture (unitless)
    :param tolerance: Tolerance control (fraction)
    :type tolerance: float
    :return: Vector of the liquid composition, x (unitless)
    :return: Vector of the gas compositiion, y (unitless)
    :return: Scalar of the liquid fraction (unitless)
    :return: x, y, nL
    """
    N = len(z)
    if len(K) is not N:
        raise ValueError('Size of inputs do not agree.')

    poles = np.array(np.zeros(N))
    for i in range(0, len(K)):
        poles[i] = (-K[i]/(1-K[i]))

    # Check if solution exists
    if (np.extract(poles < 1, poles) > 0).any():
        print('Pole between 0 and 1')
    if not (poles < 0).any() or not (poles > 1).any():
        raise ValueError('No solution along the nL[0, 1] interval.')

    def rr_f(z, k, n_L, n):
        f = 0
        for i in range(0, n):
            f += z[i] * (1 - k[i]) / (n_L + k[i] * (1 - n_L))
        return f

    def rr_dfdn(z, k, n_L, n):
        f = 0
        for i in range(0, n):
            f -= z[i] * (1 - k[i]) ** 2 / (n_L * (1 - k[i]) + k[i]) ** 2
        return f

    nL = 0
    nL_next = 1
    error = np.abs(nL - nL_next)

    while error > tolerance:
        nL_next = nL - rr_f(z, K, nL, N) / rr_dfdn(z, K, nL, N)
        error = np.abs(nL - nL_next)
        nL = nL_next

    x, y = list(), list()
    for i in range(0, N):
        x.append(z[i] / (nL + K[i] * (1 - nL)))
        y.append(z[i] * K[i] / (nL + K[i] * (1 - nL)))

    return x, y, nL


def flash(temp, press, z, temp_crit, press_crit, w, delta):
    """
    Multicomponent flash calculation for the Peng-Robinson Equation of State.

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param press: Current Pressure, P (Pa)
    :type press: float
    :param z: Vector that contains the composition of the mixture (unitless)
    :param temp_crit: Vector of Critical Temperature, Tc (K)
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :param w: Vector of Acentric Factor, omega (unitless)
    :param delta: Square matrix containing the binary interaction coefficients
    :return: Vector containing the liquid composition by component, x
    :return: Vector containing the gas composition by component, y
    :return: Molar specific volume of the liquid, VL
    :return: Molar specific volume of the gas, VG
    :return: Liquid Fraction, nL
    """

    N = len(temp_crit)
    if len(press_crit) and len(z) and len(w) and len(delta) is not N:
        raise ValueError('Size of inputs do not agree.')

    K = np.array(np.ones(N) * 0.5)[0]

    try:
        x, y, nL = RachfordRice(z, K)
    except ValueError:
        alglog.info('No solution along the nL[0, 1] interval.')
        raise ValueError('No solution along the nL[0, 1] interval.')

    x, y, VL, VG, nL = None, None, None, None, None
    return x, y, VL, VG, nL


def pressure(temp, vol, z, temp_crit, press_crit, w, delta):
    """
    Pressure using the Peng-Robinson Equation of State

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param vol: Current Volume, P (Pa)
    :param z: Vector that contains the composition of the mixture (unitless)
    :param temp_crit: Vector of Critical Temperature, Tc (K)
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :param w: Vector of Acentric Factor, omega (unitless)
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
    :param temp_crit: Vector of Critical Temperature, Tc (K)
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :param w: Vector of Acentric Factor, omega (unitless)
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
    :param temp_crit: Vector of Critical Temperature, Tc (K)
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :param w: Vector of Acentric Factor, omega (unitless)
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
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :param z: Vector that contains the composition of the mixture (unitless)
    :param w: Vector of Acentric Factor, omega (unitless)
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
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :param z: Vector that contains the composition of the mixture (unitless)
    :param w: Vector of Acentric Factor, omega (unitless)
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
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :param z: Vector that contains the composition of the mixture (unitless)
    :param w: Vector of Acentric Factor, omega (unitless)
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
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :param z: Vector that contains the composition of the mixture (unitless)
    :param w: Vector of Acentric Factor, omega (unitless)
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
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :param z: Vector that contains the composition of the mixture (unitless)
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
