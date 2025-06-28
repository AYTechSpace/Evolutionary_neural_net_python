import pytest
import numpy as np
from project import generate_test_data, bin_format, sigmoid, rounded_sigmoid, mutate

def test_generate_test_data():
    assert len(generate_test_data(4, 2, False)) == 2

def test_bin_format():
    #print(bin_format(0, 2, False))
    #print(bin_format(7, 4, False))
    #print(bin_format(-6, 6, True))
    assert (bin_format(0, 2, False) == np.array([0, 0])).all()
    assert (bin_format(7, 4, False) == np.array([0, 1, 1, 1])).all()
    assert (bin_format(-6, 6, True) == np.array([1, 0, 0, 1, 1, 0])).all()

    #print(bin_format(11, 4, False))

    assert (bin_format(11, 4, False) == np.array([1, 0, 1, 1])).all()
    assert (bin_format(2, 4, False) == np.array([0, 0, 1, 0])).all()
    assert (bin_format(13, 5, False) == np.array([0, 1, 1, 0, 1])).all()

def test_sigmoid():
    assert 0.119 < sigmoid(-2) < 0.120
    assert sigmoid(0) == 0.5
    assert 0.880 < sigmoid(2) < 0.881

def test_rounded_sigmoid():
    assert rounded_sigmoid(-2) == 0
    assert rounded_sigmoid(0) == 0
    assert rounded_sigmoid(2) == 1

def test_mutate():
    assert (2 > mutate(1, 1, 1) > 0)
    assert (mutate(1, 1, 1) != 1)