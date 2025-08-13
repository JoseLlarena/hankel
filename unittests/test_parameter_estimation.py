from numpy import asarray, eye, stack

from hankel.spectral import estimate_parameters
from unittests import assert_close


def test_estimates_wfsa_parameters_from_hankel_matrix():
    hankel = asarray([[[0, 0, 1],
                       [0, 0, 1],
                       [1, 1, 0]],

                      [[0, 0, 1],
                       [0, 0, 1],
                       [1, 1, 0]],

                      [[1, 1, 0],
                       [1, 1, 0],
                      [0, 0, 1.]]])

    initial, transitions, final = estimate_parameters(hankel)

    assert_close(initial, asarray([.841, 0]), rtol=0, atol=1e-3)
    assert_close(final, asarray([0, -.841]), rtol=0, atol=1e-3)
    assert_close(transitions, stack([eye(2),  asarray([[0, -.707], [-.707*2, 0]]).T], axis=-3), rtol=0, atol=1e-3)
