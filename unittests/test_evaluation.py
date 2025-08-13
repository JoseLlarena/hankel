from numpy import eye, stack, asarray

from hankel import px
from hankel.evaluation import _evaluate_batch, zero_one_loss, evaluate
from hankel.models import VectorWFSA
"""
OR parameters
[1 0]
   F T
F [1 0]
T [0 1]
   F T
F [0 0]
T [1 1]
[0 1]
"""
OR_WFSA = VectorWFSA(eye(2)[0],
                     stack([eye(2), asarray([[0., 0.],
                                             [1., 1.]]).T], axis=-3),
                     eye(2)[-1],
                     binary=True)


def test_batch_evaluation():
    print()
    xs = stack([asarray([[1, 0], [1, 0], [1, 0]]),
                asarray([[1, 0], [0, 1], [1, 0]])])
    ys = stack([asarray([1, 0]),
                asarray([0, 1])])
    
    assert _evaluate_batch(OR_WFSA, xs, ys, px(zero_one_loss, tol=1e-2)) == 0


def test_unbatched_evaluation():
    print()
    xs = [asarray([[1, 0], [1, 0]]), asarray([[1, 0], [0, 1], [1, 0]]), asarray([[1, 0], [0, 1], [0, 1]])]
    ys = [asarray([1, 0]), asarray([0, 1]),  asarray([0., 1])]

    assert evaluate(OR_WFSA, xs, ys, zero_one_loss) == 0
