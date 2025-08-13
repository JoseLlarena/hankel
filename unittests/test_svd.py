from hankel.spectral import nmf_of, svd_of
from unittests import assert_close
from numpy import diag, fliplr


def test_computes_full_svd():
    mat = diag([1., 2., 3.]) @ fliplr(diag([4., 5., 6.]))

    U, S, Vt = svd_of(mat, dim=3)

    assert_close(U @ S@Vt, mat, rtol=1e-6)


def test_computes_dimensionality_truncated_svd():
    mat = diag([1., 2., 0.]) @ diag([4., 5., 6.])

    U, S, Vt = svd_of(mat, dim=2)

    assert_close(U @ S@Vt, mat, rtol=1e-6)


def test_computes_singular_value_truncated_svd():
    mat = diag([1., 2., 0.]) @ diag([4., 5., 6.])

    U, S, Vt = svd_of(mat, dim=-1, sv_ratio=1e-9)

    assert_close(U @ S@Vt, mat, rtol=1e-6)
    assert tuple(U.shape) == (3, 2)
    assert tuple(S.shape) == (2, 2)
    assert tuple(Vt.shape) == (2, 3)
