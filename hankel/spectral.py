"""
Functions to learn Weighted Finite State Automata using the Spectral method
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product
from logging import DEBUG, Logger, getLogger
from operator import concat
from typing import Any, Dict, Final, Iterable, Literal, Mapping, Sequence, Tuple, TypeAlias, TypeVar
from warnings import filterwarnings

import deal
from more_itertools import flatten
from numpy import diag, float32, full, stack
from numpy.linalg import matrix_rank
from numpy.typing import NDArray
from pydantic import validate_call
from scipy.linalg import pinv, svd
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

from hankel import PYDANTIC_CONFIG, Fn, hankel_out, px
from hankel._bases import by_all_affixes, by_freq, by_length, by_split_pmi

T = TypeVar('T')
Kind: TypeAlias = Literal['binary', 'lm', 'polar']

LOG: Final[Logger] = getLogger(__package__)
MIN_PROB: Final[float] = 1e-6
DEFAULT_BASIS_ARGS: Final[Mapping[str, Any]] = dict(base_vocab=(), topk=.05, topk_pref=.05, topk_suff=.05)
DEFAULT_FACTOR_ARGS: Final[Mapping[str, Any]] = dict(dim=-1, tol=1e-1, algo='svd')
KINDS: Final[Tuple[Kind, Kind, Kind]] = ('binary', 'lm', 'polar')

filterwarnings('ignore', category=ConvergenceWarning, module='sklearn.decomposition._nmf')


@deal.pre(lambda kind, xs, ys=(), basis_kwargs=None, factor_kwargs=None: kind in KINDS)
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def learn_wfsa(kind: Kind,
               xs: Sequence[Sequence[T]],
               ys: Sequence[float] = (),
               basis_kwargs: Dict[str, Any] | None = None,
               factor_kwargs: Dict[str, Any] | None = None) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Learns Weighted Finite State Automaton (WFSA) using the Spectral method.

    Args:
        kind (Kind): The type of WFSA to learn. One of:
            'binary'   - binary two-class classification {0, 1}
            'polar'    - polar two-class classification {-1, 1}
            'lm'       - language modeling (probabilities in [0, 1])
        xs (Sequence[Sequence[T]]): Input training data, as a sequence of sequences.
        ys (Iterable[float], optional): Target training data for acceptors. Empty for language models. Defaults to ().
        basis_kwargs (Dict[str, Any] | None, optional): Dictionary of arguments for basis selection 
            (e.g., algorithm, parameters). See `select_basis` for supported options. Defaults to None.
        factor_kwargs (Dict[str, Any] | None, optional): Dictionary of arguments for Hankel matrix 
            factorization (e.g., rank, tolerance, algorithm). See `estimate_parameters` for supported options. 
            Defaults to None.

    Returns:
        Tuple[NDArray, NDArray, NDArray]: A 3-tuple with the WFSA parameters: initial weight vector with shape (d,),
            transition tensor with shape (v, d, d) and final weight vector with shape (d,).
    """
    LOG.debug('Selecting basis...')
    prefs, suffs = select_basis(xs, **(DEFAULT_BASIS_ARGS | (basis_kwargs or {})))
    LOG.debug('Computing targets...')
    targets: Mapping[Sequence[T], float] = dict(zip(xs, ys)) if ys else estimate_targets(xs, kind=kind)
    LOG.debug('Filling hankel matrix...')
    hankel: NDArray = fill_hankel(targets, prefs, suffs, default=-1. if kind == 'polar' else 0.)
    LOG.debug('Estimating parameters...')
    if LOG.level == DEBUG:
        LOG.debug(hankel_out(hankel[:1], prefs, suffs))
    params: Tuple = estimate_parameters(hankel, **(DEFAULT_FACTOR_ARGS | (factor_kwargs or {})))

    return params


def select_basis(sequences: Iterable[Sequence[T]], algo: str = 'pmi', **kwargs) -> Tuple[Tuple[T, ...], Tuple[T, ...]]:
    """
    Builds a prefix-closed Hankel basis

    Args:
        sequences (Iterable[Sequence[T]]): The input data to build the basis froml.
        algo (str, optional): The heuristic to use to build the basis. One of `all`, `freq`, `length` or `pmi`. 
            Defaults to 'pmi'.
        kwargs: A dictionary with extra parameters. Supported are `base_vocab`, `topk` for `pmi` and  `topk_pref` and
            `topk_suff` for `freq` and `length`

    Raises:
        ValueError: If an unsupported algorithm is passed in.

    Returns:
        Tuple[Tuple[T, ...], Tuple[T, ...]]: A 2-tuple with the basis' prefixes and suffixes
    """

    base_vocab: Tuple[T, ...] = kwargs.get('base_vocab', ())
    match algo:
        case 'all':
            return by_all_affixes(sequences, base_vocab=base_vocab)
        case 'freq':
            return by_freq(sequences,
                           base_vocab=base_vocab,
                           topk_pref=kwargs.get('topk_pref', .05),
                           topk_suff=kwargs.get('topk_suff', .05))
        case 'length':
            return by_length(sequences,
                             base_vocab=base_vocab,
                             topk_pref=kwargs.get('topk_pref', .05),
                             topk_suff=kwargs.get('topk_suff', .05))
        case 'pmi':
            return by_split_pmi(sequences, base_vocab=base_vocab, topk=kwargs.get('topk', .05))

        case _:
            raise ValueError(f'unknown basis algorithm [{algo}]')


def estimate_targets(data: Iterable[Sequence[T]], kind: Kind) -> Mapping[Sequence[T], float]:
    """
    Estimates the cells of the Hankel matrix, containing the target values for the task to be learnt by the WFSA.
    When the kind is `lm`, the cells will be filled with smoothed relative frequencies, using add-delta smoothing with 
    delta = `MIN_PROB`.

    Args:
        data (Iterable[Sequence[T]]): A collection of sequences/
        kind (Kind): : Type of targets. One of `binary`, `lm` or `polar`.

    Returns:
        Mapping[Sequence[T], float]: A mapping from each sequence in `data` to a target value.
    """

    default: Fn = px(float, -1 if kind == 'polar' else 0 if kind == 'binary' else MIN_PROB)
    counts: Counter[Sequence[T]] = Counter(data)
    seq_to_value: Mapping[Sequence[T], float]

    match kind:
        case 'lm':  # implements delta smoothing
            den: float = counts.total() + len(counts)*MIN_PROB
            seq_to_value = (defaultdict(default, {seq: (c - MIN_PROB) / den for seq, c in counts.items()}))
        case _:
            seq_to_value = defaultdict(default, {seq: min(count, 1) for seq, count in counts.items()})

    return seq_to_value


def fill_hankel(seq_to_value: Mapping[Sequence[T], float],
                prefs: Sequence[Sequence[T]],
                suffs: Sequence[Sequence[T]],
                default: float = 0.) -> NDArray:
    """
    Fills a Hankel matrix with basis given by `prefs` and `suffs` and with cell values given by `seq_to_value`. If the
    string corresponding to a cell is in `seq_to_value` else they are set to `default`. The data is assumed to represent
    the empty string with an empty sequence.

    The returned tensor is rank 3, where the first dimension (tube) is the subblock, of size |V|+1, the second
    dimension (row) stands for prefixes, of size |P'|, and the third dimension (column) stands for suffixes, of size |S|
    |V| is the vocabulary size, |P'| is the number of root prefixes and |S| the number of suffixes.

    Args:
        seq_to_value (Mapping[Sequence[T], float]): F mapping from sequence to target value, to go in the hankel cell.
        prefs (Sequence[Sequence[T]]): The list of closed prefixes, assumed to be sorted by the last token.
        suffs (Sequence[Sequence[T]]): The list of suffixes.
        default (float, optional): The value to give a cell if prefix-suffix is not in `seq_to_value`. Defaults to 0.

    Returns:
        NDArray: A rank-3 tensor of size |V|+1x|P'|x|S|
    """

    hankel: NDArray = full((len(prefs), len(suffs)), default)  # PxS
    for (i, prefix), (j, suffix) in product(enumerate(prefs), enumerate(suffs)):
        hankel[i, j] = seq_to_value.get(tuple(concat(prefix, suffix)), default)

    vocab_n: int = len(set(flatten(prefs))) + 1  # V, assumes prefixes contain full vocab, then adds the empty string
    return hankel.reshape((vocab_n, -1, len(suffs)))  # PxS -> V+1xP'xS


def estimate_parameters(hankel: NDArray, dim: int = -1, sv_ratio: float = 1e-1, algo: str = 'svd', **kwargs) \
        -> Tuple[NDArray, NDArray, NDArray]:
    """
    Estimates initial weights, final weights and transition matrices of a non-deterministic WFSA. The first row in 
    `hankel` is assumed to correspond to the empty-string (epsilon) subblock. Extra parameters are passed to 
    `hankel.spectral.nmf_of`. When `dim` = `-1`, the dimensionality of the WFSA, ie, the number of minimal states, is 
    found by keeping singular values that are at least `sv_ratio` times the largest singular value of the complete 
    subblock. When `dim` is not `-1`, the dimensionality is set to that value, in which case `sv_ratio` is ignored.

    Args:
        hankel (NDArray): Hankel block as a |V| x |P'| x |S| rank-3 tensor, where V is vocab, P' root prefixes and 
            S suffixes
        dim (int, optional): Dimension of WFSA, if `dim` =-1, the dimension found by SVD, in conjunction with 
            `sv_ratio`, is used. Defaults to -1.
        sv_ratio (float, optional):  Minimum fraction of the largest singular value singular values need to have to be 
            kept; ignored when `dim` != -1. Defaults to 1e-1.
        algo (str, optional): Specifies which matrix factorisation algorithm to use, one of `nmf` or `svd`. 
            Defaults to 'svd'.

    Returns:
        Tuple[NDArray, NDArray, NDArray]: a 3-tuple with the initial, transition and final weights of the learned WFSA,
            of sizes D, VxDxD and D.
    """

    complete_block: NDArray = hankel[0]  # VxP'xS -> P'xS
    first_row: NDArray = complete_block[0, :]  # P'xS -> S
    first_col: NDArray = complete_block[:, 0]  # P'xS -> P'

    if dim != -1:
        if dim > (max_rank := min(complete_block.shape)):
            LOG.warning(f'Truncating dim = [{dim}] as it is larger than maximum rank [{max_rank}]...')
        dim = min(dim, max_rank)

    P, S = _low_rank_factorise(complete_block, dim, sv_ratio, algo, **kwargs)  # D = dim; P'S -> P'xD, DxS
    # formulae have matrices transposed wrt literature, ie, Lemma 5.2.1 in Balle & al. (2013).
    # "Learning finite-state machines: Statistical and algorithmic aspects", as I use pre-multiplication.
    pinv_P: NDArray = pinv(P)  # P'xD -> DxP'
    pinv_S: NDArray = pinv(S)  # DxS -> SxD

    init: NDArray = pinv_S.T @ first_row  # [[SxD -> DxS] @ S] -> D
    final: NDArray = pinv_P @ first_col  # DxP' @  P' -> D
    trans: NDArray = stack([(pinv_P @ (block @ pinv_S)) for block in hankel[1:]], axis=-3)  # transpose twice = original
    # [[DxP' @ P'xS @ SxD] -> DxD]*V -> VxDxD

    return init, trans, final  # D, VxDxD, D


# --------------------------------------------- DELEGATE FUNCTIONS -----------------------------------------------------


def _low_rank_factorise(complete_block: NDArray, dim: int, sv_ratio: float, algo: str, **kwargs) \
        -> Tuple[NDArray, NDArray]:
    if algo == 'nmf':
        return nmf_of(complete_block, dim=dim, sv_ratio=sv_ratio, **kwargs)  # P'xS -> P'xD, DxS

    U, Z, Vt = svd_of(complete_block, dim=dim, sv_ratio=sv_ratio)  # P'xS -> P'xD, DxD, DxS

    Z = Z**.5
    return U @ Z, Z @ Vt  # [P'xD @ DxD] -> P'xD, [DxD @ DxS] -> DxS (nicer weights than U @ Z, Vt)


@deal.pre(lambda m, dim=-1, sv_ratio=0:
          m.ndim == 2 and m.size > 0 and
          ((1 <= dim <= max(m.shape)) or dim == -1) and
          0 <= sv_ratio <= 1)
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def svd_of(matrix: NDArray, dim: int = -1, sv_ratio: float = 0.) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Computes a possibly truncated Singular Value Decomposition of the given PxS `data` array. Supports two scenarios:

    a) if `dim` >= 1, a truncated SVD will be performed such that only the top `dim` singular values are preserved, ie,
        matrix[PxS] ~ U[PxD] S[DxD] Vt[DxS]. In this case, the `sv_ratio` parameter will be ignored. Use when
        you know exactly how many dimensions you want.

    b) if `dim` = -1, a truncated SVD will be computed such that matrix[PxS] ~ U[PxD] S[DxD] Vt[DxS], where D is the
        number of singular values that are greater or equal than `sv_ratio` times the largest singular value. Use when
        you want only the dimensions that explain the most variance. When `sv_ratio`=0, the number of
        dimensions will be the rank of the matrix.

    The current implementation is a thin wrapper around `scipy.linalg.svd`

    Args:
        matrix (NDArray): The matrix to compute the SVD on, a rank-2 PxS array.
        dim (int, optional): The number of dimensions/singular values to keep. Defaults to -1.
        sv_ratio (float, optional): The minimum fraction of the largest singular value that other singular values must 
            have to be kept; ignored if dim!= -1. Defaults to 0.

    Returns:
        Tuple[NDArray, NDArray, NDArray]: a 3-tuple containing the U, S and Vt factors of the SVD decomposition, with
            shapes PxD, DxD, DxS.
    """

    U, s, Vt = svd(matrix, full_matrices=False)  # PxS -> PxR, R, RxS
    if dim == -1:
        dim = sum(s >= s[0] * sv_ratio)

    return U[:, :dim], diag(s[:dim]), Vt[:dim, :]  # PxR, R, RxS -> PxD, DxD, DxS


@deal.pre(lambda m, dim=-1, sv_ratio=0, init='nndsvd', shuffle=False, seed=42:
          m.ndim == 2 and m.size > 0 and
          ((1 <= dim <= max(m.shape)) or dim == -1) and
          0 <= sv_ratio <= 1 and
          init in ('svd', 'random', 'nndsvd', 'nndsvda', 'nndsvdar'))
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def nmf_of(matrix: NDArray,
           dim: int = -1,
           sv_ratio: float = 0.,
           init: str = 'nndsvd',
           shuffle: bool = False,
           seed: int = 42) -> Tuple[NDArray, NDArray]:
    """
    Finds two matrices, Encoder and Decoder, that when multiplied together approximate the given non-negative matrix,
    using Non-negative Matrix Factorisation (NMF). Supports 2 scenarios:

    a) if `dim` is not `-1`, an NMF is computed on `matrix` truncated to the number of dimensions specified in `dim`, in
        which case, the parameter `sv_ratio` is ignored

    b) if `dim=-1`, a NMF is computed with the number of dimensions decided by how many singular values are
        `tol` times the largest singular value.

    The current implementation is a thin wrapper around `sklearn.decomposition.NMF`. 
    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

    Args:
        matrix (NDArray): The matrix to be factorised into encoder and decoder. Must contain only non-negative values.
        dim (int, optional): Inner dimension of factor matrices. Defaults to -1.
        sv_ratio (float, optional): The minimum fraction of the largest singular value that a singular value must have 
            to be kept; ignored if dim!= -1. Defaults to 0..
        init (str, optional): initialisation strategy for NMF, one of `svd, random, nndsvd, nndsvda, nndsvdar`. 
            Defaults to 'nndsvd'.
        shuffle (bool, optional): Whether the coordinates should be shuffled. Defaults to False.
        seed (int, optional): Random seed, for reproducibility. Defaults to 42.

    Returns:
        Tuple[NDArray, NDArray]: Encoder and decoder matrices such that matrix ~ encoder @ decoder.
    """

    if init == 'svd':
        U, S, Vt = svd_of(matrix, dim=dim, sv_ratio=sv_ratio)  # PxS -> PxD, DxD, DxS
        S = S**.5
        E, D = abs(U @ S), abs(S @ Vt)  # [P'xD @ DxD] -> P'xD, [DxD @ DxS] -> DxS
        nmf: NMF = NMF(n_components=E.shape[-1], random_state=seed, max_iter=1000, init='custom', shuffle=shuffle)
        dec: NDArray = nmf.fit_transform(matrix, W=E, H=D).astype(float32)  # PxS, PxD, DxS -> DxS

    else:
        dim = matrix_rank(matrix, tol=sv_ratio) if dim == -1 else dim
        nmf = NMF(n_components=dim, random_state=seed, max_iter=1000, init=init, shuffle=shuffle)
        dec = nmf.fit_transform(matrix).astype(float32)  # PxS -> DxS

    enc: NDArray = nmf.components_.astype(float32)  # DxS
    return dec, enc  # PxD, DxS
