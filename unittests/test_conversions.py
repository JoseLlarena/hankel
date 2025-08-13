from collections.abc import Sequence
from typing import List

from hypothesis import given
from hypothesis.strategies import integers
from numpy import asarray, eye, int32, ones, stack, vstack
from numpy.random import Generator, default_rng, randint
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from hankel.conversions import as_unweighted, hmm_to_wfsa, with_single_start
from hankel.model_io import WFSA

XOR_WFSA = WFSA(initial=asarray([0, .841]),
                transitions=stack([eye(2), asarray([[0, 1.565], [.639, 0]]).T], axis=-3),
                final=asarray([.760, 0]))

OR_WFSA = WFSA(initial=asarray([.479, .666]),
               transitions=stack([eye(2), asarray([[1.366, .983], [-.509, -.366]]).T], axis=-3),
               final=asarray([.602, -.433]))

BOOL_AND_WFSA = WFSA(initial=asarray([1]),
                    transitions=stack([asarray([[0]]), asarray([[1]])], axis=-3),
                    final=asarray([1]))

BOOL_OR_WFSA = WFSA(initial=eye(2)[0],
                   transitions=stack([eye(2), asarray([[0, 0], [1., 1.]]).T], axis=-3),
                   final=eye(2)[-1])


@given(integers(min_value=2, max_value=4), integers(min_value=2, max_value=4), integers(min_value=1, max_value=5))
def test_hmm_to_wfsa(num_states: int, vocab_size: int, seq_len: int):
    gen: Generator = default_rng()

    init: NDArray = gen.dirichlet(ones(num_states))
    trans: NDArray = vstack([gen.dirichlet(ones(num_states)) for _ in range(num_states)])
    obs: NDArray = vstack([gen.dirichlet(ones(num_states)) for _ in range(vocab_size)])

    seq: List[int] = randint(0, vocab_size, size=seq_len).tolist()
    one_hot_seq: NDArray = eye(vocab_size)[seq].astype(int)[None, ...]  # Converts to one-hot for WFSA (B=1, T, V)

    assert_allclose(WFSA(*hmm_to_wfsa(init, trans, obs))(one_hot_seq)
                    [0], prob(seq, init, trans, obs), rtol=0, atol=1e-6)


def test_as_unweighted_returns_the_same_input_if_unweighted():
    xs = stack([asarray([[1, 0], [1, 0], [1, 0]]),
                asarray([[1, 0], [0, 1], [1, 0]])])
    initial, transitions, final = with_single_start(OR_WFSA.initial, OR_WFSA.transitions, OR_WFSA.final)
    unweighted = WFSA(*as_unweighted(initial, transitions, final, quant=2))

    assert_allclose(unweighted(xs), XOR_WFSA(xs), rtol=0, atol=1e-3)


def test_with_single_start_returns_the_same_input_as_original():

    xs = stack([asarray([[1, 0], [1, 0], [1, 0]]),
                asarray([[1, 0], [0, 1], [1, 0]])])

    wfsa = WFSA(*with_single_start(OR_WFSA.initial, OR_WFSA.transitions, OR_WFSA.final))

    assert_allclose(wfsa(xs), OR_WFSA(xs), rtol=0, atol=1e-6)


def prob(seq: Sequence[int] | NDArray[int32], init: NDArray, transitions: NDArray, observations: NDArray) -> float:
    """
    Computes the probability of the sequence `seq` given the HMM parameters:
    - init: initial state probabilities, shape (N,)
    - transitions: state transition matrix, shape (N, N)
        (rows are the "from" states, columns are the "to" states; i.e.,
        transitions[i, j] is the probability of transitioning from state i to state j)
    - observations: observation probability matrix, shape (M, N)
    - seq: sequence of observed symbols, length T, each in 0..M-1

    Returns:
        Probability of observing the sequence under the HMM.
    """

    seq_: NDArray = asarray(seq, dtype=int)
    T: int = len(seq)

    # Forward algorithm
    alpha: NDArray = init * observations[seq_[0]]  # shape (N,)
    for t in range(1, T):
        alpha = (alpha @ transitions) * observations[seq_[t]]

    return float(alpha.sum())
