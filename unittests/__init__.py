from math import radians, sin, cos

from numpy import ndarray, array, float32
from numpy.testing import assert_allclose


def rot3d(deg: float = 90., axis: str = 'z') -> ndarray:
    """
    Returns rotation matrix by an angle of `deg` degrees about the `axis` vector

    :param deg:
    :param axis:
    :return:
    """

    theta = radians(deg)

    if axis == 'x':
        return array([[1, 0, 0],
                      [0, cos(theta), -sin(theta)],
                      [0, sin(theta), cos(theta)]], dtype=float32)  # anti/clockwise deg rotations about x axis

    if axis == 'y':
        return array([[cos(theta), 0, sin(theta)],
                      [0, 1, 0],
                      [-sin(theta), 0, cos(theta)]], dtype=float32)  # anti/clockwise deg rotations about y axis

    return array([[cos(theta), -sin(theta), 0],  # anti/clockwise deg rotations about z axis
                  [sin(theta), cos(theta), 0],
                  [0, 0, 1]], dtype=float32)


def assert_close(actual: ndarray, expected: ndarray, rtol: float = 0, atol: float = 0):
    print('\nACTUAL  :', actual, 'EXPECTED:', expected, sep='\n')
    assert_allclose(actual, expected, rtol=rtol, atol=atol)
