from numpy import eye, stack, asarray

from hankel.model_io import WFSA, load_wfsa, save_wfsa


def test_WFSA_with_scalar_output():
    wfsa = WFSA(eye(2)[0],
                stack([eye(2), asarray([[0., 0.], [1., 1.]]).T], axis=-3),
                eye(2)[-1])

    save_wfsa(wfsa, 'wfsa.npz', dict(kind='binary'))

    assert load_wfsa('wfsa.npz') == (wfsa, dict(kind='binary'))
