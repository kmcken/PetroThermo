import logging
import os

import numpy as np
import ReadFromFile as read
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.special as special

root_path = os.path.dirname(os.path.realpath(__file__))
runlog = logging.getLogger('runlog')
alglog = logging.getLogger('alglog')


def MW_c7plus(n, fractions):
    z_c7p = z_c7plus(n, fractions)
    index7 = 0
    while n[index7] < 8:
        index7 += 1
    index7 -= 1

    MW_c7p = 0
    for i in range(index7, len(n)):
        MW_c7p += read.get_phase_change_data(scn=n[i])[0] * fractions[i] / z_c7p
    return MW_c7p


def z_c7plus(n, fractions):
    index7 = 0
    while n[index7] < 8:
        index7 += 1
    index7 -= 1

    z_c7p = 0
    for i in range(index7, len(n)):
        z_c7p += fractions[i]
    return z_c7p


def katz_composition(n, z_c7plus):
    return 1.38205 * z_c7plus * np.exp(-0.25903 * n)


def pedersen_composition(n, alpha, beta):
    return np.exp((n - alpha) / beta)


def pedersen_regression(n, z):
    popt, pcov = optimize.curve_fit(pedersen_composition, n, z)
    return popt


def lnexp_compostion(n, zc7plus, beta):
    return np.log(zc7plus) + (7 - n) * beta + np.log(beta)


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

    popt, pcov = optimize.curve_fit(lnexp_compostion, n, np.log(z), bounds=((zc7plus - 1e-10, 0), (zc7plus, np.inf)))

    def alpha(zc7p, beta):
        return np.log(zc7p) + 7 * beta + np.log(beta)

    a = alpha(zc7plus, popt[1])
    return a, popt[1]


def lbc_compostion(n, z_6, alpha, beta):
    return z_6 * np.exp(alpha * (n - 6) ** 2 + beta * (n - 6))


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


def pdf_composition(n, alpha, beta, limits):
    return integrate.quad(pdf_whitson, limits[0], limits[1], args=(n, alpha, beta))[0]


def pdf_whitson(MW, n, alpha, beta):
    euler = special.gamma(alpha)
    return np.power(MW - n, alpha - 1) * np.exp(-1 * (MW - n) / beta) / (np.power(beta, alpha) * euler)


def pdf_regression(MW, fraction, n=None):
    if n is None:
        n = 16.0425

    def min_pdf(x, m, z, n):
        pdf = np.abs(pdf_composition(n, x[0], x[1], [n, m[0]]) - z[0])
        for i in range(1, len(m)):
            pdf += np.abs(pdf_composition(n, x[0], x[1], [m[i - 1], m[i]]) - z[i])
        return pdf

    opt = optimize.minimize(min_pdf, x0=np.array([1, 10]), method='Nelder-Mead', bounds=((0.5, -np.inf), (3, np.inf)), args=(MW, fraction, n))
    return opt.x


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
