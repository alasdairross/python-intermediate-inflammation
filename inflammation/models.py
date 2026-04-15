"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):  
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array."""
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array."""
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array."""
    return np.min(data, axis=0)

def patient_normalise(data):
    """ Normalise patient data by the max for that patient (max of a given row)"""
    # check data validity
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be numpy array")

    if len(data.shape) != 2:
            raise ValueError('inflammation array should be 2-dimensional')
    
    # dont allow negative data.
    if np.any(data < 0):
        raise ValueError('Inflammation values should not be negative')

    max = np.nanmax(data, axis = 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data/ max[:,np.newaxis]

    normalised[np.isnan(normalised)] = 0. #set nan values to zero
    normalised[normalised < 0] = 0 # set negative normalised values to 0
    return normalised