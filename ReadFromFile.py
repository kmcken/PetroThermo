import datetime
import logging
import os
import sqlite3

# LOGGING
root_path = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(filename=root_path + '/Logs/run.log', level=logging.DEBUG)
# Logging Levels:
# CRITICAL
# ERROR
# WARNING
# INFO
# DEBUG


def read_file(file):
    t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    logging.info('{0} Read from {1} file.'.format(t, file))

    txt_file = list()
    f = open(file, 'r', encoding='utf-8-sig')
    for line in f:
        txt_file.append(line.strip().split(','))
    f.close()

    return txt_file


def get_phase_change_data(name=None, formula=None, database=None):
    """
    Gets the phase change data from the .db file for a specific substance.
    Input name OR formula.

    :param name: Substance name
    :type name: str
    :param formula: Substance formula
    :type formula: str
    :param database: Database file path
    :type database: str
    :return: [MW, Tc, Pc, Ttrip, Ptrip, Acentric]: Phase Change
    :rtype: [MW, Tc, Pc, Ttrip, Ptrip, Acentric]: float
    """
    t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    logging.info('{0} Getting heat capacity constants.'.format(t))

    if name is None and formula is None:
        t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        logging.error('{0} Chemistry Database Query: No name or formula input.'.format(t))
        raise ValueError

    if database is None:
        database = root_path + '/ChemistryData.db'

    try:
        (cursor, conn) = open_database(database)
    except FileNotFoundError:
        raise FileNotFoundError
    except sqlite3.InterfaceError:
        raise FileNotFoundError

    if name is not None:
        cursor.execute('SELECT MW, Tc, Pc, Ttrip, Ptrip, Acentric FROM PhaseChange WHERE Name=?', [name])
    else:
        cursor.execute('SELECT MW, Tc, Pc, Ttrip, Ptrip, Acentric FROM PhaseChange WHERE Formula=?', [formula])
    constants = cursor.fetchall()
    close_database(cursor, conn)

    if not constants:
        t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        logging.error('{0} Chemistry Database Query: Missing substance input.'.format(t))
        print('Chemistry Database Query: Missing substance input')
        raise ValueError

    return constants[0]


def get_heat_capacity_constants(name=None, formula=None, database=None):
    """
    Gets the Heat Capacity Constants from the .db file
    Cp = A + B/1e2 T + C/1e5 T^2 + D/1e9 T^3
    Input name OR formula.

    :param name: Substance name
    :type name: str
    :param formula: Substance formula
    :type formula: str
    :param database: Database file path
    :type database: str
    :return: [A, B, C, D]: Heat Capacity Constants
    :rtype: [A, B, C, D]: float
    """
    t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    logging.info('{0} Getting heat capacity constants.'.format(t))

    if name is None and formula is None:
        t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        logging.error('{0} Heat Capacity Database Query: No name or formula input.'.format(t))
        raise ValueError

    if database is None:
        database = root_path + '/ChemistryData.db'

    try:
        (cursor, conn) = open_database(database)
    except FileNotFoundError:
        raise FileNotFoundError
    except sqlite3.InterfaceError:
        raise FileNotFoundError

    if name is not None:
        cursor.execute('SELECT A, B, C, D, LowTemperature, HighTemperature FROM MolarHeatCapacities WHERE Name=?',
                       [name])
    else:
        cursor.execute('SELECT A, B, C, D, LowTemperature, HighTemperature FROM MolarHeatCapacities WHERE Formula=?',
                       [formula])
    constants = cursor.fetchall()
    close_database(cursor, conn)

    if not constants:
        t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        logging.error('{0} Heat Capacity Database Query: Missing substance input.'.format(t))
        print('Heat Capacity Database Query: Missing substance input')
        raise ValueError

    return constants[0]


def open_database(file=None):
    t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    logging.info('{0} DATABASE: Opening {1} database.'.format(t, file))

    if file is None:
        logging.error('{0} Missing database file input.'.format(t))
        raise FileNotFoundError

    try:
        conn = sqlite3.connect(file)
    except sqlite3.InterfaceError:
        t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        logging.error('{0} Interface error with file {1}.'.format(t, file))
        raise sqlite3.InterfaceError
    else:
        cursor = conn.cursor()
        t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        logging.info('{0} Database opened'.format(t))
        return cursor, conn


def close_database(cursor, conn):
    t = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    cursor.close()
    conn.close()
    logging.info('{0} Closed database.'.format(t))
