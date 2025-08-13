from numpy import asarray
from numpy.testing import assert_array_equal

from hankel.data import id_coder_from, int_tensorise, int_untensorise, one_hot_coder_from, one_hot_tensorise, one_hot_untensorise

TOKEN_TO_ID = dict(zip(['a', 'bc', 'd'], range(3)))
ID_TO_TOKEN = {id: token for token, id in TOKEN_TO_ID.items()}


def test_one_hot_tensorise():
    assert_array_equal(one_hot_tensorise(['a', 'bc', 'd', 'a'], TOKEN_TO_ID),
                       asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]))


def test_one_hot_untensorise():
    assert_array_equal(one_hot_untensorise(asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]), ID_TO_TOKEN),
                       ['a', 'bc', 'd', 'a'])


def test_int_tensorise():
    assert_array_equal(int_tensorise(['a', 'bc', 'd', 'a'], TOKEN_TO_ID), asarray([0, 1, 2, 0]))


def test_int_untensorise():
    assert_array_equal(int_untensorise(asarray([0, 1, 2, 0]), ID_TO_TOKEN), ('a', 'bc', 'd', 'a'))


def test_id_coder():
    coder = id_coder_from(TOKEN_TO_ID)

    assert_array_equal(coder.tensorise(['a', 'bc', 'd', 'a']), asarray([0, 1, 2, 0]))
    assert_array_equal(coder.untensorise(asarray([0, 1, 2, 0])), ('a', 'bc', 'd', 'a'))


def test_one_hot_coder():
    coder = one_hot_coder_from(TOKEN_TO_ID)

    assert_array_equal(coder.tensorise(['a', 'bc', 'd', 'a']), asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    assert_array_equal(coder.untensorise(asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])), ('a', 'bc', 'd', 'a'))
