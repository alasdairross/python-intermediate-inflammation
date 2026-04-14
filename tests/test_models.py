"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt

from inflammation.models import daily_mean, daily_max, daily_min
import pytest

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


def test_daily_max():
    """Test daily max function with array of positive integers"""
    test_input = np.array([[4,3,1],[3,9,5]])

    test_output = np.array([4,9,5])

    npt.assert_array_equal(daily_max(test_input), test_output)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ]
)
def test_daily_mean(test,expected):
    npt.assert_array_equal(daily_mean(np.array(test)),np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [5, 6]),
    ]
)
def test_daily_max(test,expected):
    npt.assert_array_equal(daily_max(np.array(test)),np.array(expected))



@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [1, 2]),
    ]
)
def test_daily_min(test,expected):
    npt.assert_array_equal(daily_min(np.array(test)),np.array(expected))


