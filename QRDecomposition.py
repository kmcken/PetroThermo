import datetime
import logging
import os
import sys

import numpy as np

# LOGGING
root_path = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(filename=root_path + '/Logs/run.log', level=logging.DEBUG)
# Logging Levels:
# CRITICAL
# ERROR
# WARNING
# INFO
# DEBUG


def cubic_root(a1, a2, a3):
    """Finds the cubic root for the generalized equation:
        x**3 + a1 x**2 + a2 x + a3 = 0
    Inputs:
        a1, a2, a3: constants
    Outputs:
        X: [x1, x2, x3] unsorted."""
    t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    logging.info('{0} INFO: Finding the cubic root using QR Decomposition.'.format(t))

    Q = (3 * a2 - a1**2) / 9
    R = (9 * a1 * a2 - 27 * a3 - 2 * a1**3) / 54
    D = Q**3 + R**2

    S = np.cbrt(R + np.sqrt(D))
    T = np.cbrt(R - np.sqrt(D))

    x = np.array(np.zeros(3), dtype=complex)

    if D >= 0:
        x[0] = S + T - a1 / 3
        x[1] = -(S + T)/2 - a1/3 + 1/2 * 1j * np.sqrt(3) * (S - T)
        x[2] = -(S + T)/2 - a1/3 - 1/2 * 1j * np.sqrt(3) * (S - T)
        return x

    theta = np.arccos(R / np.sqrt(-Q**3))
    x[0] = 2 * np.sqrt(-Q) * np.cos(theta/3) - a1/3
    x[1] = 2 * np.sqrt(-Q) * np.cos(theta/3 + 2/3 * np.pi) - a1/3
    x[2] = 2 * np.sqrt(-Q) * np.cos(theta/3 + 4/3 * np.pi) - a1/3
    return x
