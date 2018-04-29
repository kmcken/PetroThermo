import logging
import os

from copy import copy
import numpy as np
import scipy.optimize as optimze
import SingleComponent as Single
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
        poles[i] = (- K[i] / (1 - K[i]))

    def rr_f(n_L, z, k, n):
        f = 0
        for i in range(0, n):
            f += z[i] * (1 - k[i]) / (n_L + k[i] * (1 - n_L))
        return f

    # Check if solution exists
    # nL = optimze.newton(rr_f, 0.5, args=(z, K, N))

    poles = np.sort(poles)
    if not (poles < 0).any():
        nL = 0
    else:
        if not (poles > 1).any():
            nL = 1
        else:
            rr0 = rr_f(0, z, K, N)
            rr1 = rr_f(1, z, K, N)
            print(rr0, rr1)

            if rr1 >= 1:
                nL = 1
            else:
                if rr0 <= 0:
                    nL = 0
                else:
                    nL = optimze.newton(rr_f, 0, args=(z, K, N), tol=tolerance)
                    if nL < 0:
                        nL = 0.5
                    if nL > 1:
                        nL = 0.5

    x, y = list(), list()
    for i in range(0, N):
        x.append(z[i] / (nL + K[i] * (1 - nL)))
        y.append(z[i] * K[i] / (nL + K[i] * (1 - nL)))

    return x, y, nL


def flash(temp, press, z, temp_crit, press_crit, w, delta=None, tolerance=1e-10):
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
    :param tolerance: Tolerance control (fraction)
    :type tolerance: float
    :return: Vector containing the liquid composition by component, x
    :return: Vector containing the gas composition by component, y
    :return: Molar specific volume of the liquid, VL
    :return: Molar specific volume of the gas, VG
    :return: Liquid Fraction, nL
    """

    N = len(temp_crit)

    if delta is None:
        delta = list()
        for i in range(0, N):
            delta.append(list(np.zeros(N)))

    if len(press_crit) and len(z) and len(w) and len(delta) is not N:
        raise ValueError('Size of inputs do not agree.')

    def error(x1, x2):
        err = 0
        for i in range(0, N):
            err += (x1[i] - x2[i]) ** 2 / (x1[i] * x2[i])
        return err

    K_prev = np.array(np.ones(N) * 0.5)
    K = np.array(np.ones(N) * 0.1)
    err = error(K, K_prev)
    x, y, nL = 0, 0, 0

    while np.abs(err) > tolerance:
        K_prev = copy(K)
        x, y, nL = RachfordRice(z, K, tolerance)
        phi_L, phi_G = fugacity(temp, press, temp_crit, press_crit, x, y, z, w, delta)
        K = np.exp(phi_L - phi_G)
        err = error(K, K_prev)

    return x, y, nL


def fugacity(temp, press, temp_crit, press_crit, x, y, z, w, delta):
    """
    Calculates fugacity of the gas components in a mixture.

    :param temp: Current Temperature, T (K)
    :type temp: float
    :param press: Current Pressure, P (Pa)
    :type press: float
    :param temp_crit: Vector of Critical Temperature, Tc (K)
    :type temp_crit: nd.array
    :param press_crit: Vector of Critical Pressure, Pc (Pa)
    :type press_crit: nd.array
    :param x: Vector containing the liquid composition by component, x
    :type x: np.array
    :param y: Vector containing the gas composition by component, y
    :type y: np.array
    :param z: Vector that contains the composition of the mixture (unitless)
    :type z: np.array
    :param w: Vector of Acentric Factor, omega (unitless)
    :type w: nd.array
    :param delta: Square matrix containing the binary interaction coefficients
    :type delta: list
    :return: Two vector of the fugacity by component, (phi_liquid, phi_gas)
    """

    N = len(temp_crit)
    if len(press_crit) and len(z) and len(w) and len(delta) is not N:
        raise ValueError('Size of inputs do not agree.')

    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1

    a, b = np.array(np.zeros(N)), np.array(np.zeros(N))
    alpha, aT = np.array(np.zeros(N)), np.array(np.zeros(N))
    for j in range(0, N):
        Tr = temp / temp_crit[j]
        alpha[j] = (1 + (0.37464 + 1.54226 * w[j] - 0.26992 * w[j] ** 2) * (1 - np.sqrt(Tr))) ** 2
        a[j] = 0.45723553 * R**2 * temp_crit[j]**2 / press_crit[j]
        b[j] = Single.b_factor(temp_crit[j], press_crit[j])
        aT[j] = alpha[j] * a[j]

    def aTb_fluid(aTj, bj, x, delta):
        a, b = 0, 0
        for i in range(0, len(x)):
            b += x[i] * bj[i]
            for j in range(0, len(x)):
                a += x[i] * x[j] * np.sqrt(aTj[i] * aTj[j]) * (1 - delta[i][j])
        return a, b

    def ABprime(aTj, aT, b, b_mix, x, delta):
        A, B = np.array(np.zeros(len(x))), np.array(np.zeros(len(x)))
        for j in range(0, len(x)):
            a = 0
            for i in range(0, len(x)):
                a += x[i] * np.sqrt(aTj[i]) * (1 - delta[j][i])
            A[j] = 1 / aT * (2 * np.sqrt(aTj[j]) * a)
            B[j] = b[j] / b_mix
        return A, B

    ab_liq = aTb_fluid(aT, b, x, delta)
    AB_liq_p = ABprime(aT, ab_liq[0], b, ab_liq[1], x, delta)
    ab_gas = aTb_fluid(aT, b, y, delta)
    AB_gas_p = ABprime(aT, ab_gas[0], b, ab_gas[1], y, delta)

    AB_liq = ab_liq[0] * press / (R ** 2 * temp ** 2), ab_liq[1] * press / (R * temp)
    AB_gas = ab_gas[0] * press / (R ** 2 * temp ** 2), ab_gas[1] * press / (R * temp)

    z_factor = np.array(np.zeros(2))
    z_factor[0] = np.real(qr.cubic_root(AB_liq[1] - 1, AB_liq[0] - 2 * AB_liq[1] - 3 * AB_liq[1] ** 2,
                                        - AB_liq[0] * AB_liq[1] + AB_liq[1] ** 2 + AB_liq[1] ** 3)[0])
    z_factor[1] = np.real(qr.cubic_root(AB_gas[1] - 1, AB_gas[0] - 2 * AB_gas[1] - 3 * AB_gas[1] ** 2,
                                        - AB_gas[0] * AB_gas[1] + AB_gas[1] ** 2 + AB_gas[1] ** 3)[0])

    phi_gas, phi_liq = np.array(np.zeros(N), dtype=np.float64), np.array(np.zeros(N), dtype=np.float64)
    for j in range(0, N):
        phi_gas[j] = - np.log(z_factor[1] - AB_gas[1]) + (z_factor[1] - 1) * AB_gas_p[1][j] - AB_gas[0] / \
                     (np.power(2, 1.5) * AB_gas[1]) * (AB_gas_p[0][j] - AB_gas_p[1][j]) * \
                     np.log((z_factor[1] + (1 + np.sqrt(2)) * AB_gas[1]) / (z_factor[1] + (1 - np.sqrt(2)) * AB_gas[1]))
        phi_liq[j] = - np.log(z_factor[0] - AB_liq[1]) + (z_factor[0] - 1) * AB_liq_p[1][j] - AB_liq[0] / \
                     (np.power(2, 1.5) * AB_liq[1]) * (AB_liq_p[0][j] - AB_liq_p[1][j]) * \
                     np.log((z_factor[0] + (1 + np.sqrt(2)) * AB_liq[1]) / (z_factor[0] + (1 - np.sqrt(2)) * AB_liq[1]))

    return phi_liq, phi_gas


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

    volume = np.array(np.zeros(3))
    for i in range(0, 3):
        if np.iscomplex(vol[i]) or vol[i].real < b:
            volume[i] = np.nan
        else:
            volume[i] = vol[i].real

    volume = np.sort(volume)
    return volume


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
