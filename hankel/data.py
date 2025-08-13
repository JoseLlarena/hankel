"""
Functions and classes to transform data
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Callable, Generic, Iterable, List, Mapping, Tuple, TypeVar
from os.path import exists
import deal
from more_itertools import unzip
from numpy import arange, asarray, concatenate, empty, float32, int32, ones, zeros
from numpy.typing import NDArray
from pydantic import validate_call

from hankel import PYDANTIC_CONFIG, X, Y, T, Fn
from hankel.spectral import KINDS, Kind

T_co = TypeVar('T_co', covariant=True)


class Coder(Generic[T_co]):
    """
    Encapsulates the encoding and decoding of sequences into and from arrays.
    """
    @validate_call(config=PYDANTIC_CONFIG, validate_return=True)
    def __init__(self,
                 vocab: Iterable[T_co],
                 tens: Fn[[Sequence[T_co], ...], NDArray] = lambda seq, _: asarray(seq),
                 untens: Fn[[NDArray, ...], Tuple[T_co, ...]] = lambda a, b: (a, ),
                 *,
                 bias: bool = False):
        """
        :param vocab: vocabulary
        :param tens: sequence-to-array function
        :param untens: array-to-sequence function
        :param bias: whether to add an extra dimension to account for a linear bias
        """
        self.idx_to_token: Mapping[int, T_co] = dict(enumerate(vocab))
        self.token_to_idx: Mapping[T_co, int] = {token: idx for idx, token in self.idx_to_token.items()}
        self.tens: Fn[[Sequence[T_co], Any], NDArray] = tens
        self.untens: Fn[[NDArray, Any], Tuple[T_co, ...]] = untens
        self.bias: bool = bias

    @property
    def n(self) -> int:
        """
        Returns the number of tokens in the vocabulary.

        :return: the size of the vocabulary
        """
        return len(self.token_to_idx)

    @validate_call(config=PYDANTIC_CONFIG, validate_return=True)
    def tensorise(self, seq: Sequence[T_co], flat: bool = False) -> NDArray:
        """
        Converts a sequence to an array

        :param seq: sequence to be converted
        :param flat: True for a V(+1)-shaped array , False for a NxV(+1)-shaped array
        :return: a tensorised sequence
        """
        as_array: NDArray = self.tens(seq, self.token_to_idx)
        if self.bias:
            as_array = with_bias(as_array)
        return as_array.reshape(1, -1).squeeze(0) if flat else as_array

    def untensorise(self, seq: NDArray) -> Tuple[T_co, ...]:
        """
        Converts an array into a sequence

        :param seq: tensorised sequence to be untensorised
        :return: sequence
        """
        if self.bias:
            seq = without_bias(seq)

        return self.untens(seq, self.idx_to_token)

    def has_bias(self) -> bool:
        return self.bias


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def to_labelled_dataset(inputs: Iterable[Sequence[X]],
                        targets: Callable[[Sequence[X]], Y] | Iterable[Y],
                        x_coder: Coder[X],
                        y_coder: Coder[Y]) -> Tuple[Sequence[NDArray], Sequence[NDArray]]:
    """_summary_

    Args:
        inputs (Iterable[Sequence[X]]): _description_
        targets (Callable[[Sequence[X]], Y] | Iterable[Y]): _description_
        x_coder (Coder[X]): _description_
        y_coder (Coder[Y]): _description_

    Returns:
        Tuple[Sequence[NDArray], Sequence[NDArray]]: _description_
    """
    XY: Iterable[Tuple[Sequence[X], Y]] = (zip(inputs, targets) if isinstance(targets, Iterable) else
                                           ((seq, targets(seq)) for seq in inputs))
    xs, ys = unzip((x_coder.tensorise(x), y_coder.tensorise([y], flat=True)) for x, y in XY)

    return tuple(xs), tuple(ys)


# @validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def to_unlabelled_dataset(inputs: Iterable[Sequence[X]], x_coder: Coder[X]) -> Tuple[NDArray,...]:
    """
    Encodes a collection of unabelled sequences into one-hot encoded sequences

    Args:
        inputs (Iterable[Sequence[X]]): A collection of sequences
        x_coder (Coder[X]): A coder for translating sequences into numpy arrays

    Returns:
        Iterable[NDArray]: A collection of one-hot encoded sequences
    """
    return tuple(x_coder.tensorise(x) for x in inputs)


@deal.pre(lambda value, kind: kind in KINDS)
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def scalar_encode(value: str, kind: Kind) -> float:
    """
    :param value: scalar value to be encoded
    :return: encoded scalar value
    """
    return 2*float(value) - 1 if kind == 'polar' else float(value)


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def one_hot_coder_from(vocab: Iterable[T]) -> Coder[T]:
    """
    Creates a one-hot coder
    :param vocab: vocabulary of sequences to encode/decode
    :return: coder
    """
    vocab = tuple(vocab)
    if not vocab:
        raise ValueError("Vocabulary cannot be empty")

    return Coder(vocab, one_hot_tensorise, one_hot_untensorise)


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def polar_coder_from(vocab: Iterable[T]) -> Coder[T]:
    """
    Creates a polar {-1, 1} coder
    :param vocab: vocabulary of sequences to encode/decode
    :return: coder
    """
    vocab = tuple(vocab)

    def polar_tensorise(seq: Sequence[T], token_to_id: Mapping[T, int]) -> NDArray:
        tens = one_hot_tensorise(seq, token_to_id)
        tens[tens == 0] = -1
        return tens

    def polar_untensorise(seq_array: NDArray, id_to_token: Mapping[int, T]) -> Tuple[T, ...]:
        seq_array[seq_array == -1] = 0
        return one_hot_untensorise(seq_array, id_to_token)

    return Coder(vocab, polar_tensorise, polar_untensorise)


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def id_coder_from(vocab: Iterable[T]) -> Coder[T]:
    """
    Creates an integer-id coder, meant for embedding layers

    :param vocab: vocabulary of tokens to encode/decode
    :return: coder
    """
    vocab = tuple(vocab)
    if len(vocab) == 0:
        raise ValueError("Vocabulary cannot be empty")

    return Coder(vocab, int_tensorise, int_untensorise)



@deal.pre(lambda seq, token_to_id: seq.ndim == 1 and len(token_to_id))
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def int_untensorise(seq_array: NDArray[int32], id_to_token: Mapping[int, T]) -> Tuple[T | str, ...]:
    """
    Converts array into sequence of tokens. Assumes `seq_array` is a 1st-order array, ie, it's a sequence of
    ids and looks them up

    If a token is not found, the index of the `UNK` special string is used instead. This is meant for handling OOVs.

    :param seq_array: an array of shape N
    :param id_to_token: an id-to-token mapping, should contain the `UNK` token
    :return: a tuple of tokens
    """

    return tuple(map(id_to_token.__getitem__, seq_array.tolist()))


@deal.pre(lambda seq, token_to_id: len(token_to_id))
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def one_hot_tensorise(sequence: Sequence[T], token_to_id: Mapping[T, int]) -> NDArray:
    """
    Converts a sequence of N tokens into an array of size NxK, where N is the length of the
    sequence and K is the size of `token_to_id`, ie, the vocabulary. In other words, a matrix whose rows correspond to
    the one-hot encoding of each token in the sequence, with a 1 in the dimension corresponding to the token index;
    meant for input to dense layers. This is a localist encoding.

    If a token is not found, the index of the `UNK` special token is used instead. This is meant for handling OOVs.

    :param sequence: the sequence of K tokens to encode
    :param token_to_id: the mapping from token to identifier
    :return: an array of size NxK
    :raises: KeyError if any token, including `UNK`, is missing from `token_to_id`
    """

    if not sequence:
        return empty(0)

    indices: NDArray[int32] = int_tensorise(sequence, token_to_id)
    vocab_size: int = len(token_to_id)

    result: NDArray = zeros((len(indices), vocab_size), dtype=float32)
    result[arange(len(indices)), indices] = 1

    return result


@deal.pre(lambda seq, token_to_id: not seq.ndim or
          (seq.ndim == 2 and seq.shape[-1] == len(token_to_id)) and len(token_to_id))
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def one_hot_untensorise(seq_array: NDArray, id_to_token: Mapping[int, T]) -> Tuple[T, ...]:
    """
    Converts NxK array into sequence of N tokens. Assumes `seq_array` is rank-2, ie, it's a sequence of
    one-hot-encoded vectors and looks up the index with the highest value in each row

    :param seq_array: an NxK array representing a sequence of tokens
    :param id_to_token: an index-to-token mapping
    :return: an untensorised tuple of N strings
    """
    if not seq_array.ndim:
        return ()

    indices: NDArray[int32] = seq_array.argmax(axis=-1).astype(int32)

    return tuple(map(id_to_token.__getitem__, indices.tolist()))


@deal.pre(lambda seq, token_to_id: len(seq) and len(token_to_id))
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def int_tensorise(sequence: Sequence[T], token_to_id: Mapping[T, int]) -> NDArray[int32]:
    """
    Converts a sequence of tokens into an array of size K where K is the length of the sequence.
    This is meant as input to embedding layers. This is a localist encoding.

    If a token is not found, the index of the `UNK` special token is used instead. This is meant for handling OOVs.

    :param sequence: the sequence of tokens to encode
    :param token_to_id: the mapping from token to identifier
    :return: an array of size K
    :raises: RuntimeError if any token is missing from `token_to_id`
    """

    return asarray([token_to_id[token] for token in sequence])


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def with_bias(seq: NDArray) -> NDArray:
    """
    Add a bias dimension to the tensorised sequence

    :param seq: unbiased tensorised sequence
    :return: biased tensorised sequence
    """
    if seq.ndim != 2 or seq.shape[-1] == 0:
        raise ValueError(f'Invalid array shape [{seq.shape}]')
    
    return concatenate([seq, ones((seq.shape[0], 1))], axis=-1)  # NxD -> NxD+1


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def without_bias(seq: NDArray) -> NDArray:
    """
    Removes bias dimension from the tensorised sequence

    :param seq: biased tensorised sequence
    :return: unbiased tensorised sequence
    """
    if seq.ndim != 2 or seq.shape[-1] == 0:
        raise ValueError(f'Invalid array shape [{seq.shape}]')

    return seq[:, :-1]  # NxD -> NxD-1

@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def load_labelled_data(path: str) -> Iterable[Tuple[Tuple[str, ...], str]]:
    """
    Loads labelled data from a file. The file must contain one row per data point, made up of white-separated tokens,
    followed by a tab followed by a single character: "1" for a positive example and "0" for a negative one.

    Args:
        path (str): Path to data file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file doesn't have a valid format

    Yields:
        Iterator[Iterable[Tuple[Tuple[str, ...], str]]]: a collection of data rows with the input and target strings.
    """
    
    if not exists(path):
        raise FileNotFoundError(f"Data file [{path}] does not exist")

    with open(path, 'r', encoding='utf8') as f:

        for line in f:  # FIXME USE csv MODULE
            seq_target: List[str] = line.rstrip().split('\t')
            if len(seq_target) != 2:
                raise ValueError(f"Row [{line}] should be two tab-separated columns")

            seq, target = seq_target
            if target not in tuple('01'):
                raise ValueError(f"Target can only be '0' or '1', found [{target}]")

            yield (tuple(seq.split()), target)


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def load_unlabelled_data(path: str) -> Iterable[Tuple[str, ...]]:
    """
    Loads unlabelled data from a file. The file should contain one sequence per line, with tokens separated by 
    whitespace.

    Args:
        path (str): path to unlabelled data file
        bounded (bool): True if each string should be bounded by the beggining and end of sequence markers.

    Raises:
        FileNotFoundError: If the file does not exist

    Yields:
        Iterable[Tuple[str, ...]]: an iterable of tuples of strings
    """
    if not exists(path):
        raise FileNotFoundError(f'Data file [{path}] does not exist')

    with open(path, 'r', encoding='utf8') as f:
        yield from (tuple(line.split())for line in f)
