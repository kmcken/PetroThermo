import logging
import os
import sqlite3

root_path = os.path.dirname(os.path.realpath(__file__))
runlog = logging.getLogger('runlog')
alglog = logging.getLogger('alglog')


def read_file(file):
    runlog.info('Read from {0} file.'.format(file))

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

    if name is None and formula is None:
        runlog.error('PHASE CHANGE: No chemical name or formula input.')
        raise ValueError('PHASE CHANGE: No chemical name or formula input.')

    if database is None:
        database = root_path + '/ChemistryData.db'

    runlog.info('PHASE CHANGE: Read data from {0} file.'.format(database))

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
        runlog.error('PHASE CHANGE: Missing data for the requested chemical.')
        print('PHASE CHANGE: Missing data for the requested chemical.')
        raise ValueError('PHASE CHANGE: Missing data for the requested chemical.')

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

    if name is None and formula is None:
        runlog.error('HEAT CAPACITY: No chemical name or formula input.')
        raise ValueError('HEAT CAPACITY: No chemical name or formula input.')

    if database is None:
        database = root_path + '/ChemistryData.db'

    runlog.info('HEAT CAPACITY: Read data from {0} file.'.format(database))

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
        runlog.error('HEAT CAPACITY: Missing data for the requested chemical.')
        print('HEAT CAPACITY: Missing data for the requested chemical.')
        raise ValueError('HEAT CAPACITY: Missing data for the requested chemical.')

    return constants[0]


def open_database(file=None):
    if file is None:
        runlog.error('DATABASE: Missing file input.')
        raise FileNotFoundError('DATABASE: Missing file input.')

    runlog.info('DATABASE: Opening {0} database.'.format(file))
    try:
        conn = sqlite3.connect(file)
    except sqlite3.InterfaceError:
        runlog.error('DATABASE: Database interface error.')
        raise sqlite3.InterfaceError
    else:
        cursor = conn.cursor()
        runlog.info('DATABASE: Database {0} opened.'.format(file))
        return cursor, conn


def close_database(cursor, conn):
    cursor.close()
    conn.close()
    runlog.info('DATABASE: Database closed.')
