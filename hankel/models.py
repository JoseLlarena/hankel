"""
Classes and function to create WFSAs.
"""
from typing import Tuple

import deal
from numpy import array_equal, broadcast_to, einsum, expand_dims, squeeze, transpose, zeros
from numpy.typing import NDArray
from pydantic import validate_call

from hankel import PYDANTIC_CONFIG 


class WFSA:
    """
    Linear representation of a Weighted Finite State Automaton (WFSA) with scalar output.
    """
    @deal.pre(lambda self, initial, transitions, final:
              initial.ndim == 1 and initial.size and
              transitions.ndim == 3 and transitions.size and
              transitions.shape[-2] == transitions.shape[-1] and transitions.shape[-2] == initial.shape[0] and
              final.size and final.shape == initial.shape)
    @validate_call(config=PYDANTIC_CONFIG, validate_return=True)
    def __init__(self, initial: NDArray, transitions: NDArray, final: NDArray):
        """
        Constructs a WFSA.

        Args:
            initial (NDArray): An initial weight vector of size d.
            transitions (NDArray): a rank-3 tensor of shape v x d x d containing per-symbol transition matrices of 
                shape d x d, where rows index start states and columns index end states.
            final (NDArray): A final weight vector of size d.
        """
        self._initial: NDArray = expand_dims(expand_dims(initial, 0), -1)  # D -> 1xD -> 1xDx1
        self._transitions: NDArray = transitions.copy()  # VxDxD
        self._final: NDArray = final

    @property
    def initial(self) -> NDArray:
        return squeeze(squeeze(self._initial, -1), 0)  # 1xDx1 -> 1xD -> D

    @property
    def transitions(self) -> NDArray:
        return self._transitions  # VxDxD

    @property
    def final(self) -> NDArray:
        return self._final

    @property
    def trans_mats(self) -> Tuple[NDArray, ...]:

        return tuple(mat.T for mat in self.transitions)

    @property
    def parameters(self) -> Tuple[NDArray, ...]:
        return (self.initial, ) + self.trans_mats + (self.final, )

    def __str__(self) -> str:
        return (f'WFSA(initial={self.initial},\ntransitions=\n{self.trans_mats},'
                f'\nfinal={self.final})')

    def __eq__(self, other) -> bool:
        if not isinstance(other, WFSA):
            return NotImplemented

        return (array_equal(self.initial, other.initial) and array_equal(self.transitions, other.transitions)
                and array_equal(self.final, other.final))

    def __hash__(self) -> int:
        return hash((self.initial.tobytes(), self.transitions.tobytes(), self.final.tobytes()))

    @deal.pre(lambda self, xs: xs.ndim in (2, 3))
    @validate_call(config=PYDANTIC_CONFIG)
    def __call__(self, xs: NDArray) -> NDArray:
        # Expands init vector to match batch size: 1xDx1 -> BxDx1
        state: NDArray = broadcast_to(self._initial, (xs.shape[0], self._initial.shape[1], 1)).copy()

        if xs.ndim == 3:  # this is to support empty strings
            # Pre-computes transition tensor using einsum: 'BTV,VDd -> BTdD'
            trans: NDArray = einsum('BTV,VDd->BTdD', xs, self.transitions)

            for t in range(xs.shape[-2]):
                state = trans[:, t] @ state  # [BxTxDxD -> BxDxD] @ BxDx1 -> BxDx1

        return state.squeeze(-1) @ self.final  # [BxDx1 -> BxD] @ D(xK) -> B(xK)


class VectorWFSA(WFSA):
    """
    Linear representation of WFSA with 2-dimensional vector output.
    """

    @validate_call(config=PYDANTIC_CONFIG, validate_return=True)
    def __init__(self, initial: NDArray, transitions: NDArray, final: NDArray, binary: bool):
        """
        Constructs a WFSA with 2-dimensional vector-valued output. The spectral learning algorithm returns a scalar-
        valued WFSA, but the evaluation code expects a vector-valued one for acceptors (binary classifiers).

        Args:
            initial (NDArray): An initial weight vector of size d.
            transitions (NDArray): a rank-3 tensor of shape v x d x d containing per-symbol transition matrices of 
                shape d x d, where rows index start states and columns index end states.
            final (NDArray): A final weight vector of size d.
            binary (bool): True if the output should be binary, False if it should be polar.
        """
        super().__init__(initial, transitions, final)
        self.binary: bool = binary

    def __call__(self, *args, **kwargs) -> NDArray:
        scalar_out: NDArray = super().__call__(*args, **kwargs)
        vector_out: NDArray = zeros((scalar_out.shape[0], 2))
        vector_out[:, 1] = scalar_out
        vector_out[:, 0] = (1 - scalar_out) if self.binary else - scalar_out
        return vector_out


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def with_vector_output(wfsa: WFSA, binary: bool) -> WFSA:

    return VectorWFSA(initial=wfsa.initial, transitions=wfsa.transitions, final=wfsa.final, binary=binary)
