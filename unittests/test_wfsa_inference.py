from typing import Final

from numpy import eye, stack, asarray
from numpy.testing import assert_allclose, assert_equal
from numpy.typing import NDArray

from hankel.models import WFSA, VectorWFSA

EXAMPLES: Final[NDArray] = asarray([[[1., 0.],  # 000
                                     [1., 0.],
                                     [1., 0.]],
                                    [[0., 1.],  # 101
                                     [1., 0.],
                                     [0., 1.]]])


def test_WFSA_with_scalar_output():
    print()
    xs = stack([asarray([[1, 0], [1, 0], [1, 0]]),
                asarray([[1, 0], [0, 1], [1, 0]])])
    wfsa = VectorWFSA(eye(2)[0],
                      stack([eye(2), asarray([[0., 0.], [1., 1.]]).T], axis=-3),
                      eye(2)[-1],
                      binary=True)

    assert_allclose(wfsa(xs), stack([asarray([1, 0]), asarray([0, 1])]))


def test_WFSA_with_vector_output():
    print()
    xs = stack([asarray([[1, 0], [1, 0], [1, 0]]),
                asarray([[1, 0], [0, 1], [1, 0]])])
    wfsa = WFSA(eye(2)[0],
                stack([eye(2), asarray([[0., 0.], [1., 1.]]).T], axis=-3),
                eye(2)[-1])

    assert_allclose(wfsa(xs), stack([asarray(0), asarray(1)]))
