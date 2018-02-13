"""
This module will provide unit conversion capabilities based on the Energistics Unit of Measure Standard v1.0.

For more information regarding the Energistics standard, please see
http://www.energistics.org/asset-data-management/unit-of-measure-standard

Author: Espen Solbu
August 17 - 2016
"""

from __future__ import division

import warnings
import functools

from lxml import etree
try:
    import numpy as np
    __numpyEnabled = True
except:
    __numpyEnabled = False
    #log.info("Numpy not installed. performance will be degraded")

import pkg_resources
import math # needed for PI operations to work properly

resource_package = __name__



# load Energistics symbols and factors
xmlFile = pkg_resources.resource_string(resource_package, "/units.xml")
root = etree.fromstring(xmlFile)

tag = etree.QName('http://www.energistics.org/energyml/data/uomv1', 'unit')
__units = {}
for unitXml in root.iter(tag.text):
    unit = {}
    isBase = False
    for field in unitXml:
        t = etree.QName(field.tag)
        # print(t.localname)
        # print field.text

        try:
            unit[t.localname] = float(eval(field.text.replace("PI", "math.pi")))
        except:
            unit[t.localname] = field.text

        if t.localname=="isBase":
            unit["A"] = 0.0
            unit["B"] = 1.0
            unit["C"] = 1.0

    __units[unit["symbol"]] = unit


#CustomUnits
#TODO, move these to API
#if "tf" not in Units.keys():
#    Units["tf"]={'symbol':'tf','name':"Metric Ton Force","A":0.0,"B":9.80665*1000.0,"C":1.0,"D":0.0}


def add_custom_unit(symbol, name, a, b, c, d=0, force=False):
    """
    Adds a custom unit defined as:\n
    y=(a+b*value)/(c+d*value)\n
    where\n
    offset = a/c\n
    scale = b/c\n

    All current Units have d=0, so this can safely be ignored

    Set the force flag to True to force an override of existing symbol
    """
    #global Units
    if symbol not in __units.keys():
        __units[symbol] = {'symbol': symbol, 'name': name, "A": a, "B": b, "C": c, "D": d}


def from_si(value, targetUnit):
    """
    Takes value(s) in SI, and converts it to a value in the desired TARGETUNIT

    :param value: The value to convert (can be a list)
    :type value: float
    :param targetUnit: The relevant unitsymbol as a string
    :type targetUnit: str
    :return: The value converted to TARGETUNIT
    :rtype: float
    """
    global __numpyEnabled
    offset = __units[targetUnit]["A"] * 1.0 / __units[targetUnit]["C"]
    scale = __units[targetUnit]["B"] * 1.0 / __units[targetUnit]["C"]
    if __numpyEnabled:
        y = np.divide(value, scale) - np.divide(offset, scale)
    else:
        scaledOffset = offset / scale
        if hasattr(value, "__iter__"):

            y = [(v / scale)-scaledOffset for v in value]
        else:
            y = value/scale - scaledOffset

    return y


def to_si(value, sourceUnit):
    """
    Takes value(s) in SOURCEUNIT and converts it to a value in the SI unit for the relevant quantity

    :param value: The value to convert (can be a list)
    :type value: float
    :param sourceUnit: The relevant unitsymbol as a string
    :type sourceUnit: str
    :return: The value converted to SI
    :rtype: float
    """
    global __numpyEnabled
    offset = __units[sourceUnit]["A"] * 1.0 / __units[sourceUnit]["C"]
    scale = __units[sourceUnit]["B"] * 1.0 / __units[sourceUnit]["C"]
    if __numpyEnabled:
        y = np.multiply(value,scale) + offset
    else:
        if hasattr(value,"__iter__"):
            y = [(v * scale) + offset for v in value]
        else:
            y = value*scale + offset
    return y


def set_numpy_enabled(enabled):
    global __numpyEnabled
    __numpyEnabled = enabled


