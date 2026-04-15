"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt

import pytest 
from inflammation.models import daily_mean

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


from inflammation.models import patient_normalise

@pytest.mark.parametrize(
        "test, expected, expected_raise",
        [ 
            ([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 0]], None ),
            ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], 
             [[1, 1, 1], [1, 1, 1], [1, 1, 1]], None ),
            ( [[1,2,3], [4,5,6]], # max = [3,6]
               [[1/3,2/3,1],[4/6,5/6,1]], None),  
            ( [[-1,2,3], [4,5,6]], # max = [3,6]
               [[1/3,2/3,1],[4/6,5/6,1]], ValueError),  
            ( "not an array", # max = [3,6]
               None, TypeError),  
            ]
)
def test_normalise(test,expected,expected_raise):
    """ 
    Test for normalisation for positive integers
    Test with relative tolerance for float division of 0.01, and also absolute tolerance of 0.01
    """
    if expected_raise is not None:
            with pytest.raises(expected_raise):
                patient_normalise(np.array(test))
    else:
        result = patient_normalise(np.array(test))
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)