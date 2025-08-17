import os
import sys
from functools import partial, update_wrapper
from itertools import zip_longest
from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from numbers import Number
from typing import Callable, Final, Sequence, Tuple, TypeAlias, TypeVar

import deal
import numpy as np
from click import BadParameter, Context, Option, Parameter, ParamType
from numpy import asarray, floating, integer, ndarray, random
from numpy.typing import NDArray
from pydantic import ConfigDict, validate_call

__all__ = ['cli', 'conversions', 'data', 'evaluation', 'hp_search', 'model_io', 'models', 'spectral']


V = TypeVar('V')
X = TypeVar('X')
Y = TypeVar('Y')
T = TypeVar('T')
Z = TypeVar('Z')
Fn: TypeAlias = Callable
Num: TypeAlias = int | float | floating | integer


EPS: str = '𝝺'
PAD: str = '★'
UNK: str = '⊡'

PYDANTIC_CONFIG: Final[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def config_logging(logger: Logger, path: str = '', sparse: bool | None = True, level: int = INFO) -> Logger:
    """
    Configures given logger 

    Args:
        logger (Logger): The logger to configure.
        path (str, optional): If not empty, the file to log to, in addition to the console. Defaults to ''.
        sparse (bool | None, optional): wether the logging format should be short (True), long (False) or minimal 
            (None). Defaults to True.
        level (int, optional): Logger's logging level. Defaults to INFO.

    Returns:
        Logger: the configured logger
    """

    formatter: Formatter
    if sparse is None:
        formatter = Formatter('[%(asctime)s] %(message)s')
    elif sparse:
        formatter = Formatter('[%(asctime)s][%(levelname)1s] %(message)s')
    else:
        formatter = Formatter('[%(asctime)s][%(levelname)s][%(name)s.%(funcName)s:%(lineno)3d] %(message)s')

    formatter.datefmt = '%Y%m%d %H:%M:%S'

    logger.handlers.clear()
    logger.setLevel(level)

    handler = StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if path:
        handler_ = FileHandler(path)
        handler_.setLevel(level)
        handler_.setFormatter(formatter)
        logger.addHandler(handler_)

    return logger


LOG: Final[Logger] = config_logging(getLogger(__package__), sparse=True)


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def hankel_out(hankel: NDArray, 
               prefs: Sequence[Tuple[str, ...]], 
               suffs: Sequence[Tuple[str, ...]], 
               ints:int = 2,
               fracs: int = 1) -> str:
    """
    Returns a string representation of the Hankel matrix

    Args:
        hankel (NDArray): Hankel matrix, a rank-3 tensor of size (v, d, d).
        prefs (Sequence[Tuple[str, ...]]): Prefixes indexing the rows of the Hankel matrix.
        suffs (Sequence[Tuple[str, ...]]): Suffixes indexing the rows of the Hankel matrix.
        fracs (int, optional): Number of fractions to show. Defaults to 6.

    Returns:
        str: A string represetation of the given Hankel matrix with prefixes and suffixes.
    """
    return "Hankel Matrix's ϵ-block\n\n"+nout(hankel,  # V x R x S
                                    row_hs=[' '.join(p) if p else 'ϵ' for p in prefs[:hankel.shape[1]]],
                                    col_hs=[' '.join(s) if s else 'ϵ' for s in suffs],
                                    tube_hs=[' '.join(p) if p else 'ϵ' for p in prefs[:: hankel.shape[1]]],
                                    indent=-1,
                                    ints=ints,
                                    fracs=fracs,
                                    console=False)


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def px(fn: Fn[[...], T], *args, **kwargs) -> Fn[[...], T]:
    """
    Replacement for `functools.partial` that's concise and inherits the properties of the partialled out function

    Args:
        fn (Fn[[...], T]): Function to partial out.
        args: Position parameters.
        kwargs: Keyword parameters.
    Returns:
        Fn[[...], T]: Partialled out function.
    """    
    return update_wrapper(partial(fn, *args, **kwargs), fn)


@deal.pre(lambda item, ints=2, fracs=3, tol=1e-9, indent=-1, col_hs=(), row_hs=(), tube_hs=(), sep='', console=True:
          ints >= 0 and fracs >= 0 and tol >= 0 and indent >= -1)  # TODO HARDEN THIS
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def nout(item: Num |
         NDArray[floating | integer] |
         Sequence[Num] |
         Sequence[Sequence[Num]] |
         Sequence[Sequence[Sequence[Num]]],
         ints: int = 2,
         fracs: int = 3,
         tol: float = 1e-9,
         indent: int = -1,
         col_hs: Sequence[str] = (),
         row_hs: Sequence[str] = (),
         tube_hs: Sequence[str] = (),
         *,
         sep:str='',
         console: bool = True) -> str:
    """
    Returns and optionally prints formatted output for numerical data.

    Args:
        item (Num | NDArray[floating | integer] | Sequence[Num] | Sequence[Sequence[Num]] | 
            Sequence[Sequence[Sequence[Num]]]): numerical item to format
        ints (int, optional): number of integer digits. Defaults to 2.
        fracs (int, optional): number of fractional digits. Defaults to 3.
        tol (float, optional): minum number to format as special symbols, for -1, 0 and 1. Defaults to 1e-9.
        indent (int, optional): row/tube indentation. Defaults to -1.
        col_hs (Sequence[str], optional): column headers. Defaults to ().
        row_hs (Sequence[str], optional): row headers. Defaults to ().
        tube_hs (Sequence[str], optional): tube headers. Defaults to ().
        sep (str, optional): separator string to print after the number. Defaults to ''.
        console (bool, optional): Whether the formatted output should also be printed to console. Defaults to True.

    Raises:
        ValueError: if `item` is not a number of if nested 4 levels or more, or if the array is 4th order or higher.

    Returns:
        str: a formatted string representation of the numerical data
    """

    s: str = ''

    if isinstance(item, (ndarray, Sequence)):
        # +1 adds a gap between a row and the left-most cell
        indent = max(map(len, row_hs), default=0)+1 if indent == -1 else indent
        matrix: Final[NDArray[floating | integer]] = to_ndarray(item)
        m = matrix.shape[0]
        col_width: Final[int] = fracs + ints + 2

        header: Final[str] = (f'{"":{indent}.{indent}}' +
                              f'{"".join(f"{col_h:^{col_width}.{col_width}}" for col_h in col_hs)}\n\n') if col_hs else ''

        s += header
        match matrix.ndim:
            case 1 | 2:
                if matrix.ndim == 1:
                    m = 1 # stops the wrong m, as it would be the length of the vector
                for row_h, r in zip_longest(row_hs or ('',)*m, [matrix] if matrix.ndim == 1 else matrix, fillvalue=''):
                    s += f'{row_h:<{indent}.{indent}}' + format_row(r, ints=ints, fracs=fracs, tol=tol)+'\n'
                s = s.rstrip()  # remove trailing newline in the last row
            case 3:
                s = ''
                tube_indent: int = max(map(len, tube_hs), default=0)+1
                for tube_h, t in zip(tube_hs or map(str, range(m)), matrix):
                    s += f'{tube_h:{tube_indent}.{tube_indent}}\n'
                    s += nout(t, ints=ints, fracs=fracs, tol=tol, indent=indent,
                              col_hs=col_hs, row_hs=row_hs, console=False)+'\n\n'
            case _:
                raise ValueError(f'Tensors of 4th and higher order are not supported. Found [{matrix.ndim}D array]')

    elif isinstance(item, Number):
        s = format_scalar(item, ints=ints, fracs=fracs, tol=tol)
    else:
        raise ValueError(f'Unsupported data type [{type(item).__name__}]. Must be numeric')
    
    if sep:
        s+=f'\n{sep}'
        
    if console:
        print(s)

    return s


@deal.pre(lambda row, ints=2, fracs=3, tol=1e-9: ints >= 0 and fracs >= 0 and tol >= 0)
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def format_row(row: NDArray[floating | integer] | Sequence[Num],
               ints: int = 2,
               fracs: int = 3,
               tol: float = 1e-9) -> str:
    """
    Formats row for easier visualisation of matrix rows and row vectors.

    Args:
        row (NDArray[floating  |  integer] | Sequence[Num]): The vector to format.
        ints (int, optional): The number of fractional digits to print. Defaults to 2.
        fracs (int, optional): The number of integer digits to print. Defaults to 3.
        tol (float, optional): The minimum number that switches the display of -1's, 0's and 1's to a symbol. 
            Defaults to 1e-9.

    Returns:
        str: The formatted vector.
    """   
    return f'{"".join(format_scalar(num.item(), ints=ints, fracs=fracs, tol=tol) for num in to_ndarray(row))}'


@deal.pre(lambda num, ints=2, fracs=3, tol=1e-9: ints >= 0 and fracs >= 0 and tol >= 0)
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def format_scalar(num: Num, fracs: int = 3, ints: int = 1, tol: float = 1e-9) -> str:
    """ 
    Formats number for easier visualisation of matrices and vectors.

    Args:
        num (Num): The number to format.
        fracs (int, optional): How many characters wide the decimal part should be. Defaults to 3.
        ints (int, optional): How many characters wide integer part should be. Defaults to 1.
        tol (float, optional): How close a number has to be to -1, 0 or 1 to be displayed with special symbols. 
            Defaults to 1e-9.

    Returns:
        str: The formatted number.
    """    
    width: Final[int] = ints + fracs + 2  # 2 = 1 for decimal point + 1 for negative sign

    symbol: str = ''
    if abs(num) < tol:
        symbol = '∘'
    elif abs(num - 1) < tol:
        symbol = '■'
    elif abs(num + 1) < tol:
        symbol = '□'

    if symbol:  # centres symbol with 3 leading spaces
        return f'{symbol:^{width}.{width}s}'
    # the replace operation implements no leading zero format for fractional numbers
    return f'{num:{width}.{fracs}f}'  # .replace('0.', ' .' if abs(num) < 1 else '0.')


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def to_ndarray(matrix: NDArray[floating | integer] |
               Sequence[Num] |
               Sequence[Sequence[Num]] |
               Sequence[Sequence[Sequence[Num]]]) -> NDArray[floating | integer]:
    """
    Converts input matrix to ndarray if it's not already.

    Args:
        matrix (NDArray[floating | integer] | Sequence[Num] | Sequence[Sequence[Num]] | 
            Sequence[Sequence[Sequence[Num]]]): a possibly nested collection of numbers

    Returns:
        NDArray[floating | integer]: a numpy array
    """

    return matrix if isinstance(matrix, ndarray) else asarray(matrix)


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def make_deterministic(seed: int = 42):
    """
    Makes execution deterministic by fixing seeds and preventing multithreading.

    Args:
        seed (int, optional): The seed supplied to random number generators. Defaults to 42.
    
    Returns:
        nothing but mutates the application state
    """   

    random.seed(seed)
    np.random.seed(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

class RangeOfSteppedPercentages(ParamType):
    """
    Parameter containing a range of floating-point percentages plus a step, separated by colons.
    """
    name: str = 'range of percentages with step'

    def convert(self, value: str, param: Parameter | None, ctx: Context | None) -> Tuple[float, float, float]:
        try:
        
            start, end, step = map(float, value.split(':'))
            if start < 0 or start >= 100:
                self.fail(f'Start of range must be at least 0 and at most less than 100, you entered "{value}"', param, ctx)
            if start > end:
                self.fail(f'Start of range must be <= end, you entered "{value}"', param, ctx)
            if end <= 0 or end > 100:
                self.fail(f'End of range must be greater than 0 and at most 100, you entered "{value}"', param, ctx)    
            if step <= 0 or step > end - start:
                self.fail(f'Step must be positive and less than the range, you entered "{value}"', param, ctx)
            return (start, end, step)
        
        except ValueError:
            self.fail(f'"{value}" is not a valid float range with step (e.g. 0:.01:.001)', param, ctx)


class TripleSplit(ParamType):
    """
    Parameter containing 3 colon-separated integer defining a three-way data split..
    """
    name: str = 'triple split'

    def convert(self, value: str, param: Parameter | None, ctx: Context | None) -> Tuple[int, int, int]:
        try:
            train, valid, test = map(int, value.split(':'))
            if train < 1:
                self.fail(f'training split must be at least 1%, you entered "{value}"', param, ctx)
            if valid < 0 or test < 0:
                self.fail(f'Validation and test splits must be at least 0%, you entered "{value}"', param, ctx)
            if train + valid + test > 100:
                self.fail(f'Splits must add up to 100, you entered "{value}"', param, ctx)
            return (train, valid, test)
        except ValueError:
            self.fail(f'"{value}" is not a valid triple split (e.g. 80:10:10)', param, ctx)


class CSVList(ParamType):
    """
    Parameter containing a comma-separated list of values
    """
    name: str = "csv"

    def __init__(self, max_len: int) -> None:
        super().__init__()
        self.max_len: int = max_len

    def convert(self, value: str, param: Parameter | None, ctx: Context | None) -> Tuple[str, ...]:
        try:
            values: Tuple[str, ...] = tuple(v.strip() for v in value.split(","))
            if not values or len(values) > self.max_len or len(values) != len(set(values)):
                raise ValueError()
            return values
        except ValueError:
            self.fail(f'Must be a non-empty comma-separated list without repetitions with a maximum of '
                      f'{self.max_len} elements. You entered "{value}"', param, ctx)


def validate_special_int(ctx: Context, param: Option, value: int):
    if value < -1 or value == 0:
        raise BadParameter(f'Must be -1 or a positive integer, you entered "{value}"')
    return value
