import logging
import os

import numpy as np
import ReadFromFile as read
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.special as special
import UnitConverter as units

root_path = os.path.dirname(os.path.realpath(__file__))
runlog = logging.getLogger('runlog')
alglog = logging.getLogger('alglog')


def specific_gravity_c7plus(n, fractions):
    M, gamma = list(), list()
    for i in range(0, len(n)):
        M.append(read.get_phase_change_data(scn=n[i])[0])
        gamma.append(read.get_phase_change_data(scn=n[i])[4])

    sumM = 0
    for i in range(0, len(n)):
        sumM += fractions[i] * M[i]

    fraction7 = 0
    for i in range(6, len(n)):
        fraction7 += fractions[i] * M[i]

    density = 0
    for i in range(6, len(n)):
        density += fractions[i] * M[i] / (fraction7 * gamma[i])

    return 1 / density


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
    return units.to_si((kw * SG) ** 3, 'degR')


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


def gauss_lumping(eta, gamma, beta, order, z_c7p):
    """
    Hydrocarbon lumping using Gauss-Laguerre Quadrature

    :param eta: Minimum molecular weight
    :type eta: float
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
    def molecular_weight(chi, eta, beta):
        return chi * beta + eta

    def gl_fractions(chi, gamma):
        return np.power(chi, gamma - 1) / (special.gamma(gamma))

    chi, w = gauss_laguerre_quadrature_consts(order)
    mw, fraction, Tc, Tb, Pc, acentric = list(), list(), list(), list(), list(), list()
    Pb = units.to_si(1, 'atm')
    for i in range(0, order):
        mw.append(molecular_weight(chi[i], eta, beta))
        fraction.append(w[i] * gl_fractions(chi[i], gamma) * z_c7p)
        Tc.append(sancet_Tc(mw[i]))
        Tb.append(sancet_Tb(mw[i]))
        Pc.append(sancet_Pc(mw[i]))
        acentric.append(kessler_acentric(Tb[i] / Tc[i], Pbr=Pb / Pc[i]))

    return np.array(mw), np.array(fraction), np.sum(np.array(fraction)), np.array(Tc), np.array(Pc), np.array(acentric)


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


def sancet_Pc(M):
    return units.to_si(82.82 + 653 * np.exp(-0.007427 * M), 'psi')


def sancet_Tc(M):
    return units.to_si(-778.5 + 383.5 * np.log(M - 4.075), 'degR')


def sancet_Tb(M):
    Tc = -778.5 + 383.5 * np.log(M - 4.075)
    return units.to_si(194 + 0.001241 * np.power(Tc, 1.869), 'degR')


def kessler_acentric(Tbr, Pbr=None, kw=None):
    if kw is not None:
        return -7.904 + 0.1352 * kw - 0.007465 * kw ** 2 + 8.359 * Tbr + (1.408 - 0.01063 * kw) / Tbr
    if Pbr is not None:
        f0 = np.log(Pbr) - 5.92714 / Tbr + 1.28862 * np.log(Tbr) - 0.169347 * Tbr ** 6
        f1 = 15.2518 - 15.6875 / Tbr - 13.4721 * np.log(Tbr) + 0.43577 * Tbr ** 6
        return f0 / f1
    raise ValueError('Need Pbr or kw for Lee-Kesler Eccentric Factor')


def kessler_Pc(kw, Tbr):
    w = kessler_acentric(Tbr, kw=kw)
    f1 = 5.92714 - 6.09648 / Tbr - 1.28862 * np.log(Tbr) + 0.169347 * Tbr ** 6
    f2 = 15.2518 - 15.6875 / Tbr - 13.4721 * np.log(Tbr) + 0.43577 * Tbr ** 6
    return units.to_si(1, 'atm') / np.exp(f1 + w * f2)
