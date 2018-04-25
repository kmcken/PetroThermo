import logging
import numpy as np
import os
import sqlite3

root_path = os.path.dirname(os.path.realpath(__file__))
runlog = logging.getLogger('runlog')
alglog = logging.getLogger('alglog')


def scn_composition(file):
    """
    Reads SCN composition model.

    :param file: File location
    :type file: str
    :return:
    """

    runlog.info('Read from {0} file.'.format(file))

    string = read_file(file)
    name = list()
    number = list()
    fraction = list()
    for i in range(1, len(string)):
        name.append(string[i][0]), number.append(float(string[i][1])), fraction.append(float(string[i][2]))
    return tuple(name), np.array(number), np.array(fraction)


def read_file(file):
    """
    Reads generic .csv file.

    :param file: File location
    :type file: str
    :return: Comma delimited list by line
    """

    runlog.info('Read from {0} file.'.format(file))

    txt_file = list()
    f = open(file, 'r', encoding='utf-8-sig')
    for line in f:
        txt_file.append(line.strip().split(','))
    f.close()

    return txt_file


def get_phase_change_data(name=None, formula=None, scn=None, database=None):
    """
    Gets the phase change data from the .db file for a specific substance.
    Input name OR formula.

    :param name: Substance name
    :type name: str
    :param formula: Substance formula
    :type formula: str
    :param scn: Single Carbon Number
    :type scn: int
    :param database: Database file path
    :type database: str
    :return: [MW, Tc, Pc, Acentric]: Phase Change
    :rtype: list
    """

    if name is None and formula is None and scn is None:
        runlog.error('PHASE CHANGE: No chemical name or formula input.')
        raise ValueError('PHASE CHANGE: No chemical name or formula input.')

    if database is None:
        database = root_path + '/Data/ChemistryData.db'

    runlog.info('PHASE CHANGE: Read data from {0} file.'.format(database))

    try:
        (cursor, conn) = open_database(database)
    except FileNotFoundError:
        raise FileNotFoundError
    except sqlite3.InterfaceError:
        raise FileNotFoundError

    if name is not None:
        cursor.execute('SELECT MW, Tc, Pc, Acentric FROM PhaseChange WHERE Name=?', [name])
    else:
        if formula is not None:
            cursor.execute('SELECT MW, Tc, Pc, Acentric FROM PhaseChange WHERE Formula=?', [formula])
        else:
            cursor.execute('SELECT MW, Tc, Pc, Acentric FROM PhaseChange WHERE SCN=?', [scn])
    constants = cursor.fetchall()
    close_database(cursor, conn)

    if not constants:
        runlog.error('PHASE CHANGE: Missing data for the requested chemical.')
        print('PHASE CHANGE: Missing data for the requested chemical.')
        raise ValueError('PHASE CHANGE: Missing data for the requested chemical.')

    return constants[0]


def get_binary_interations(database=None):
    """
    Gets the binary interaction coefficient
    :param database: Database file path
    :type database: str
    :return: delta: Binary Interaction Matrix
    :rtype: np.array
    """

    if database is None:
        database = root_path + '/Data/ChemistryData.db'

    runlog.info('BINARY INTERACTIONS: Read data from {0} file.'.format(database))

    try:
        (cursor, conn) = open_database(database)
    except FileNotFoundError:
        raise FileNotFoundError
    except sqlite3.InterfaceError:
        raise FileNotFoundError

    delta = np.array(np.zeros((30, 30)))
    for i in range(0, 30):
        cursor.execute('SELECT * FROM BinaryInteractions WHERE SCN=?', [i + 1])
        binary = cursor.fetchall()[0]
        for j in range(0, 9):
            if binary[j + 1] is None:
                delta[i][j] = 0.
            else:
                delta[i][j] = binary[j + 1]

    close_database(cursor, conn)
    return delta


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
