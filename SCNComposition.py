import logging
import os

import numpy as np
import ReadFromFile as read
import scipy.optimize as optimize
import scipy.special as special

root_path = os.path.dirname(os.path.realpath(__file__))
runlog = logging.getLogger('runlog')
alglog = logging.getLogger('alglog')


def MW_c7plus(n, fractions):
    z_c7p = z_c7plus(n, fractions)
    index7 = 0
    while n[index7] < 7:
        index7 += 1
    index7 -= 1

    MW_c7p = 0
    for i in range(index7, len(n)):
        MW_c7p += read.get_phase_change_data(scn=n[i])[0] * fractions[i] / z_c7p
    return MW_c7p


def z_c7plus(n, fractions):
    index7 = 0
    while n[index7] < 7:
        index7 += 1
    index7 -= 1

    z_c7p = 0
    for i in range(index7, len(n)):
        z_c7p += fractions[i]
    return z_c7p


def katz_composition(n, z_c7plus):
    return 1.38205 * z_c7plus * np.exp(-0.25903 * n)


def lnexp_compostion(n, zc7plus, beta):
    return zc7plus + np.exp((7 - n) * beta) + beta


def exp_compostion(n, alpha, beta):
    return np.exp(alpha - n * beta)


def exp_regression(n, z, zc7plus):
    """
    Mix-Integer Nonlinear Program to regression fit the exponential approximation of mole fraction.

    :param n: n-number of known data
    :type n: np.array
    :param z: Mole fraction of known data
    :type z: np.array
    :return: alpha, beta
    :rtype np.array
    """

    popt, pcov = optimize.curve_fit(lnexp_compostion, n, z, bounds=((zc7plus - 1e-10, 0), (zc7plus, np.inf)))

    def alpha(zc7p, beta):
        return np.log(zc7p) + 7 * beta + np.log(beta)

    a = alpha(zc7plus, popt[1])
    return a, popt[1]


def lbc_compostion(n, z6, alpha, beta):
    return z6 * np.exp(alpha * (n - 6) - beta * (n - 6) ** 2)


def lbc_regression(n, z, z6):
    """
    Mix-Integer Nonlinear Program to regression fit the Lorenz-Bray-Clark approximation of mole fraction.

    :param n: n-number of known data
    :type n: np.array
    :param z: Mole fraction of known data
    :type z: np.array
    :return: alpha, beta
    :rtype np.array
    """

    popt, pcov = optimize.curve_fit(lbc_compostion, n, z, bounds=((z6 - 1e-8, 0, 0), (z6, np.inf, np.inf)))

    return popt


def watson_factor(MW, SG):
    """
    Watson Characterization Factor for C7+

    :param MW: molecular weight
    :type MW: float
    :param SG: specific gravity
    :type SG: float
    :return: kw, gamma
    :return: Watson characterization factor
    :rtype: float
    """

    return 4.5579 * np.power(MW, 0.15178) * np.power(SG, -0.84573)


def watson_gamma(MW, kw):
    """Specific gravity from Watson characterization factor.

    :param MW: molecular weight
    :type MW: float
    :param kw: Watson Characterization Factor
    :type kw: float
    :return: SG specific gravity
    :rtype: float
    """

    return np.power(kw / (4.5579 * np.power(MW, 0.15178)), 0.84573)
    

def watson_boiling_pt(kw, SG):
    return np.power(kw * SG, 3)


def pdf_composition(MW, tau, gamma, beta):
    euler = special.gamma(gamma)
    chi = MW - tau
    alpha = 1 / (gamma * beta * euler)
    return np.power(chi, gamma - 1) * np.exp(-1 * chi / beta) * alpha


def pdf_regression(MW, z, tau=None):
    """
    Mix-Integer Nonlinear Program to regression fit the probability density fraction of molecular weights.

    :param MW: molecular weight
    :type MW np.array
    :param z: Mole fraction of known data
    :type z: np.array
    :param tau: minimum molecular weight
    :type tau: float
    :return: tau, gamma, beta
    :rtype np.array
    """

    if tau is None:
        tau = 0

    popt, pcov = optimize.curve_fit(pdf_composition, MW, z, bounds=((tau, 0, -np.inf), (tau + 1e-10, np.inf, np.inf)))
    return popt


def gauss_lumping(tau, gamma, beta, order, z_c7p):
    """
    Hydrocarbon lumping using Gauss-Laguerre Quadrature

    :param tau: Minimum molecular weight
    :type tau: float
    :param gamma: Fitting parameter
    :type gamma: float
    :param beta: Fitting parameter
    :type beta: float
    :param order: Gauss-Laguerre approximation order
    :type order: int
    :param z_c7p: C7+ Fraction
    :type z_c7p: float
    :return: MW, Zi, and Z_c7+
    """
    def molecular_weight(chi, tau, beta):
        return chi * beta + tau

    def gl_fractions(chi, gamma):
        return np.power(chi, gamma - 1) / (special.gamma(gamma))

    chi, w = gauss_laguerre_quadrature_consts(order)
    mw, fraction= list(), list()

    for i in range(0, order):
        mw.append(molecular_weight(chi[i], tau, beta))
        fraction.append(w[i] * gl_fractions(chi[i], gamma) * z_c7p)

    return np.array(mw), np.array(fraction), np.sum(np.array(fraction))


def gauss_laguerre_quadrature_consts(order):
    """
    Returns the Guass-Laguerre Quadrature constants.

    :param order: Order of the quadrature
    :type order: int
    :return: xi, w
    """

    if order != 2 and order != 3 and order != 4:
        raise ValueError('Gauss-Laguerre order is not 2, 3, or 4.')

    if order == 2:
        xi = (0.5858, 3.4142)
        w = (0.8536, 0.1464)
    else:
        if order == 3:
            xi = (0.4158, 2.2943, 6.2960)
            w = (0.7111, 0.2785, 0.0104)
        else:
            if order == 4:
                xi = (0.3226, 1.7458, 4.5366, 9.3951)
                w = (0.6032, 0.3574, 0.0384, 0.0005)

    return xi, w


def acentric(M):
    return -0.3 + np.exp(-6.252 + 3.64457 * np.power(M, 0.1))
