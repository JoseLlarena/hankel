"""
Functions to evaluate WFSAs.
"""
from collections.abc import Iterable
from typing import Final, Sequence

import deal
from more_itertools import bucket
from numpy import abs, all, asarray, float32, floating, log2, log10, stack
from numpy.typing import NDArray
from pydantic import validate_call

from hankel import PYDANTIC_CONFIG, Fn

MIN_PROB: Final[float] = 1e-16  # smallest non-zero probability to avoid infinity when applying logs


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def evaluate(model: Fn[[NDArray], NDArray], xs: Sequence[NDArray], ys: Sequence[NDArray],
             loss_fn: Fn[[NDArray, NDArray, str], NDArray]) -> float:
    """
    Evaluates the given model on the given data with the given loss function.

    Args:
        model (Fn[[NDArray], NDArray]): The model to evaluate.
        xs (Sequence[NDArray]): Input data.
        ys (Sequence[NDArray]): Target data.
        loss_fn (Fn[[NDArray, NDArray, str], NDArray]): Loss function.

    Returns:
        float: The aggregate loss over the input+target data
    """
    grouped_by_x_len: bucket = bucket(zip(xs, ys), key=lambda x_y: len(x_y[0]))

    loss: float = 0
    n: int = 0
    for x_len in grouped_by_x_len:
        x_batch, y_batch = map(stack, zip(*grouped_by_x_len[x_len]))
        loss += _evaluate_batch(model, x_batch, y_batch, loss_fn, red='sum')  # BxTxD -> BxTxK
        n += len(x_batch)

    return loss / n


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def score(model: Fn[[NDArray], NDArray], xs: Iterable[NDArray], ppl: bool = True, normed: bool = False) -> float:
    """
    Scores the given language model.

    Args:
        model (Fn[[NDArray], NDArray]): The model to score.
        xs (Iterable[NDArray]): The input data.
        ppl (bool, optional): True if the loss should be perplexity, False for cross-entropy. Defaults to True.
        normed (bool, optional): True if the loss should be normalised by the number of tokens, False otherwise. 
            Defaults to False.

    Returns:
        float: The aggregate perplexity or cross-entropy.
    """
    loss: float = 0
    num_tokens: int = 0
    for batch in _make_batches(xs):
        length: int = 1 if batch.ndim == 2 or not normed else batch.shape[-2]
        loss += -log2(model(batch).clip(MIN_PROB, 1)).sum()  # BxTxK -> BxK
        num_tokens += len(batch)*length

    return 2 ** (loss/num_tokens) if ppl else (loss / num_tokens)


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def lm_predict(model: Fn[[NDArray], NDArray], xs: Iterable[NDArray], normed: bool = False) -> Iterable[float]:
    """
    Predicts the base-10 negative log likelihoods for each input sequence using the model. If the model output is
    non-positive, a minimum probability of 1e-16 is used.

    Args:
        model (Fn[[NDArray], NDArray]): A language model
        xs (Iterable[NDArray]): A collection of input sequences
        normed (bool, optional): If the probabilities should be normalised by the sequence length.

    Returns:
        Iterable[float]: A collection of possibly normalised base-10 negative log likelihoods for each input sequence
    """
    for batch in _make_batches(xs):
        length: int = 1 if batch.ndim == 2 or not normed else batch.shape[-2]
        yield from -log10(model(batch).clip(MIN_PROB, 1))/length  # clipping as no guarantee output in (0,1]


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def class_predict(model: Fn[[NDArray], NDArray], xs: Iterable[NDArray]) -> Iterable[float]:
    """
    Predicts the class of each input sequence using the model.

    Args:
        model (Fn[[NDArray], NDArray]): Classification model
        xs (Iterable[NDArray]): Collection of input sequences

    Returns:
        Iterable[float]: A collection of class predictions for each input sequence
    """

    yield from (pred for batch in _make_batches(xs) for pred in model(batch))


@deal.pre(lambda y_hats, ys, tol= 1e-5, reduction='mean':
          y_hats.ndim in (2, 3) and y_hats.size and
          ys.shape == y_hats.shape and y_hats.size and
          tol >= 0 and
          reduction in ('none', 'mean', 'sum'))
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def zero_one_loss(y_hats: NDArray, ys: NDArray, tol: float | floating = 1e-5, reduction: str = 'mean') -> NDArray:
    """
    Computes the 0-1 loss between targets and outputs, defined by per-row equality comparison.

    :param y_hats: an Nxk 2nd-order array with N k-dimensional one-hot-encoded targets, one per row
    :param ys: an Nxk 2nd-order array with N k-dimensional one-hot-encoded outputs, one per row
    :param tol: tolerance for equality comparison
    :param reduction: type of loss aggregation, one of 'sum', 'mean', 'none'
    :return: a N-sized boolean array or a scalar with the losses
    """

    equal: NDArray = all(abs(y_hats - ys) <= tol, axis=1)

    losses: NDArray = (~equal).astype(float32)  # Loss is 1 if prediction and target are not "equal enough"

    match reduction:
        case 'none':
            return losses
        case 'sum':
            return asarray(losses.sum())
        case 'mean':
            return asarray(losses.mean())
        case _:
            raise ValueError(f'Invalid reduction [{reduction}]')


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def avg_ternariness_of(params: Iterable[NDArray], tol: float) -> float:
    """
    Computes the average ternariness of the given set of parameters. Ternariness is defined as the closeness of
    a parameter to -1, 0 or 1, and it's an extension of binarity (or booleanness).

    Args:
        params (Iterable[NDArray]): Parameters to compute the average ternariness for.
        tol (float): how close the parameter has to be to count as -1, 0 or 1.

    Returns:
        float: The computed average ternariness.
    """
    params = tuple(params)
    return float(sum(_ternariness_of(p, tol, normed=False) for p in params) / sum(p.size for p in params))


def _ternariness_of(weights: NDArray, tol: float = 1e-2, normed: bool = True) -> float:
    """ computes the (possibly normalised) number of parameters that are close-to-binary in the given array"""
    s: NDArray = (abs(weights) <= 0 + tol).sum() + ((abs(weights) > 1 - tol) & (abs(weights) < 1 + tol)).sum()

    return s.item() / (weights.size if normed else 1)


def _evaluate_batch(model: Fn[[NDArray], NDArray],
                    xs: NDArray,
                    ys: NDArray,
                    loss_fn: Fn[[NDArray, NDArray, str], NDArray],
                    red: str = 'mean') -> float:
    return loss_fn(model(xs), ys, reduction=red).item()  # BxTxD -> BxTxK


def _make_batches(xs: Iterable[NDArray]) -> Iterable[NDArray]:
    grouped_by_x_len: bucket = bucket(xs, key=len)

    yield from (stack(tuple(grouped_by_x_len[x_len])) for x_len in grouped_by_x_len)
