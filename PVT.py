import datetime
import logging
import os

import numpy as np
import SingleComponent as pr
import ReadFromFile as read

# LOGGING
root_path = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(filename=root_path + '/Logs/run.log', level=logging.DEBUG)
# Logging Levels:
# CRITICAL
# ERROR
# WARNING
# INFO
# DEBUG


def cp(temp, substance=None, formula=None):
    """
    Calculates the constant pressure heat capacity of a substance.
    Cp = A + B/1e2 T + C/1e5 T^2 + D/1e9 T^3

    :param substance: The substance
    :type substance: str
    :param formula: The substance formula
    :type formula: str
    :param temp: The substance temperature (K)
    :type temp: float
    :return: cp (J/mol-K)
    :rtype: float
    """
    if substance is None and formula is None:
        t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        logging.error('{0} Heat Capacity: No name or formula input.'.format(t))
        raise ValueError

    const = read.get_heat_capacity_constants(name=substance, formula=formula)
    R = 8.314459848  # Gas Constant: m^3 Pa mol^-1 K^-1 3.64*R -1.101e-3*R*temp +2.466e-6*R*temp**2 -0.942e-9*R*temp**3
    return const[0] + const[1]*1e-2 * temp + const[2]*1e-5 * temp**2 + const[3]*1e-9 * temp**3


def cv(temp, press, substance=None, formula=None):
    """
    Calculates the volumetric heat capacity of a substance.
    Cv = Cp + T * dVdP * dPdT**2

    :param substance: The substance
    :type substance: str
    :param formula: The substance formula
    :type formula: str
    :param temp: The substance temperature (K)
    :type temp: float
    :param press: The substance pressure (Pa)
    :type press: float
    :return: cv (J/mol-K)
    :rtype: float
    """
    if substance is None and formula is None:
        t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        logging.error('{0} Volumetric Heat Capacity: No name or formula input.'.format(t))
        raise ValueError

    try:
        MW, Tc, Pc, Ttrip, Ptrip, Acentric = read.get_phase_change_data(name=substance, formula=formula)
    except:
        t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        logging.error('{0} Volumetric Heat Capacity: Error loading phase change data.'.format(t))
        raise ValueError

    Cp = cp(temp, substance=substance, formula=formula)
    dPdT = pr.dPdT(temp, press, Tc, Pc, Acentric)[0]
    dVdP = 1/pr.dPdV(temp, press, Tc, Pc, Acentric)[0]

    return Cp + temp * dVdP * dPdT**2


def speed_of_sound(temp, press, substance=None, formula=None):
    """
    Calculates the speed of sound for a pure substance

    :param substance: The substance
    :type substance: str
    :param formula: The substance formula
    :type formula: str
    :param temp: The substance temperature (K)
    :type temp: float
    :param press: The substance pressure (Pa)
    :type press: float
    :return: Speed of Sound, vs (m/s)
    :rtype: float
    """

    Cp = cp(temp, substance=substance, formula=formula)
    Cv = cv(temp, press, substance=substance, formula=formula)
    try:
        MW, Tc, Pc, Ttrip, Ptrip, Acentric = read.get_phase_change_data(name=substance, formula=formula)
    except:
        t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        logging.error('{0} Volumetric Heat Capacity: Error loading phase change data.'.format(t))
        raise ValueError

    dPdV = pr.dPdV(temp, press, Tc, Pc, Acentric)
    vol = pr.volume(temp, press, Tc, Pc, Acentric)
    return np.sqrt(-1/(MW*0.001) * Cp/Cv * dPdV) * vol
