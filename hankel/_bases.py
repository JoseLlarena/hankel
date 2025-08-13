"""
Functions implementing different basis selection heuristics.
"""
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence, Set
from functools import lru_cache
from heapq import nsmallest
from itertools import product, starmap
from logging import Logger, getLogger
from math import log2, prod
from operator import concat
from typing import DefaultDict, Final, FrozenSet, List, Mapping, Protocol, Tuple, TypeAlias, TypeVar, runtime_checkable

from more_itertools import flatten


@runtime_checkable
class SupportsLessThan(Protocol):

    def __lt__(self, other: object) -> bool:
        ...


T = TypeVar('T', bound=SupportsLessThan)
Word: TypeAlias = Sequence[str]

LOG: Final[Logger] = getLogger(__package__)


def by_all_affixes(sequences: Iterable[Word], base_vocab: Iterable[T] = ()) -> Tuple[Tuple[T, ...], Tuple[T, ...]]:
    """
    Constructs a prefix-closed basis with all prefixes and suffixes in the data.

    Args:
        sequences (Iterable[Word]): The data to build a basis from.
        base_vocab (Iterable[T], optional): Extra tokens to add to the prefixes. Defaults to ().

    Returns:
        Tuple[Tuple[T, ...], Tuple[T, ...]]: A basis with prefixes and suffixes.
    """
    sequences = set(sequences)

    root_prefixes: FrozenSet[Word] = _affixes_of(sequences, prefix=True)
    suffixes: FrozenSet[Word] = _affixes_of(sequences, prefix=False)
    infixes: FrozenSet[Word] = frozenset({(sym, )
                                          for sym in set(base_vocab).union(flatten(sequences))}
                                         | {()})

    prefixes: Iterable[Word] = _pclose(root_prefixes, infixes)

    return tuple(prefixes), tuple(sorted(suffixes))


def by_length(sequences: Iterable[Word], topk_pref: float, topk_suff: float, base_vocab: Iterable[T] = ()) \
        -> Tuple[Tuple[T, ...], Tuple[T, ...]]:
    """
    Constructs a prefix-closed basis by combining the shortest prefixes and suffixes in the data. All vocab items
    are also added to both prefixes and suffixes. 

    Args:
        sequences (Iterable[Word]): The data to build a basis from.
        topk_pref (float): fraction of all prefixes sorted by increasing length to include in the basis.
        topk_suff (float): fraction of all suffixes sorted by increasing length to include in the basis.
        base_vocab (Iterable[T], optional): Extra tokens to add to the prefixes and suffixes. Defaults to ().
    Returns:
        Tuple[Tuple[T, ...], Tuple[T, ...]]: A basis with prefixes and suffixes.
    """

    sequences = tuple(map(tuple, sequences))
    vocab: FrozenSet[Tuple[T, ...]] = _vocab_of(sequences, base_vocab)

    root_suffs = tuple(_affixes_of(sequences, prefix=False))
    root_suffs = nsmallest(int(topk_suff*len(root_suffs)), root_suffs, key=lambda suffix: (len(suffix), suffix))

    root_prefs = tuple(_affixes_of(sequences))
    root_prefs = nsmallest(int(topk_pref*len(root_prefs)), root_prefs, key=lambda prefix: (len(prefix), prefix))

    vocab: Set[Word] = set(_vocab_of(sequences, base_vocab))
    
    suffixes: List[Word] = sorted(set(root_suffs) | vocab, key=_epsilon_first)
    prefixes: Iterable[Word] = _pclose(root_prefs, vocab)

    return tuple(prefixes), tuple(suffixes)


def by_freq(sequences: Iterable[Word], topk_pref: float, topk_suff: float, base_vocab: Iterable[T] = ()) \
        -> Tuple[Tuple[T, ...], Tuple[T, ...]]:
    """
    Constructs a prefix-closed basis by combining the most frequent prefixes and suffixes in the data. All vocab items
    are also added to both prefixes and suffixes. 

    Args:
        sequences (Iterable[Word]): The data to build a basis from.
        topk_pref (float): fraction of all prefixes sorted by decreasing frequency to include in the basis.
        topk_suff (float): fraction of all suffixes sorted by decreasing frequency to include in the basis.
        base_vocab (Iterable[T], optional): Extra tokens to add to the prefixes and suffixes. Defaults to ().
    Returns:
        Tuple[Tuple[T, ...], Tuple[T, ...]]: A basis with prefixes and suffixes.
    """    

    seqs: Tuple[Tuple[T, ...], ...] = tuple(map(tuple, sequences))
    root_suffs = set()
    if topk_suff:
        root_suffs = Counter(_affixes_of(seqs, prefix=False, unique=False))
        root_suffs, _ = zip(*root_suffs.most_common(int(max(1, round(topk_suff*len(root_suffs))))))
        root_suffs = set(root_suffs)
    root_prefs = set()
    if topk_pref:
        root_prefs = Counter(_affixes_of(seqs, unique=False))
        root_prefs, _ = zip(*root_prefs.most_common(int(max(1, round(topk_pref*len(root_prefs))))))
        root_prefs = set(root_prefs)

    vocab: Set[Word] = set(_vocab_of(sequences, base_vocab))
   
    suffixes: List[Word] = sorted(set(root_suffs) | vocab, key=_epsilon_first)
    prefixes: Iterable[Word] = _pclose(root_prefs, vocab)

    return tuple(prefixes), tuple(suffixes)


def by_split_pmi(sequences: Iterable[Word], topk: float, base_vocab: Iterable[T] = ()) \
        -> Tuple[Tuple[T, ...], Tuple[T, ...]]:
    """
    Constructs a prefix-closed basis by combining pairs of least correlated prefixes and suffixes in the data. The 
    correlation is measured with the normalised Pointwise Mutual Information (PMI). All vocab items
    are also added to both prefixes and suffixes.

    Args:
        sequences (Iterable[Word]): The data to build a basis from.
        topkf (float): fraction of all prefix-suffix pairs sorted by increasing PMI to include in the basis.
        base_vocab (Iterable[T], optional): Extra tokens to add to the prefixes and suffixes. Defaults to ().
    Returns:
        Tuple[Tuple[T, ...], Tuple[T, ...]]: A basis with prefixes and suffixes.
    """
    affix_to_pmi: Mapping[Tuple[Word, Word], float] = _compute_affix_pmis(sequences)

    root_prefs, root_suffs = set(), set()
    if topk:
        top_k_npmis: Iterable[Tuple[Word, Word]] = nsmallest(int(max(1, round(topk*len(affix_to_pmi)))),
                                                             affix_to_pmi.keys(),
                                                             key=lambda hankel:
                                                             (affix_to_pmi[hankel], len(hankel[0]) + len(hankel[-1]), hankel))

        root_prefs, root_suffs = map(set, zip(*top_k_npmis))

    vocab: FrozenSet[Word] = set(_vocab_of(sequences, base_vocab))
   

    suffixes: List[Word] = sorted(set(root_suffs) | vocab, key=_epsilon_first)
    prefixes: Iterable[Word] = _pclose(root_prefs, vocab)

    return tuple(prefixes), tuple(suffixes)


# ---------------------------------------------------- DELEGATE FUNCTIONS ----------------------------------------------


def _vocab_of(sequences: Iterable[Word], base_vocab: Iterable[T]) -> FrozenSet[Tuple[T, ...]]:
    return frozenset({(sym, ) for sym in set(base_vocab).union(flatten(sequences))} | {()})


def _pclose(root_prefixes: Iterable[Word], infixes: Iterable[Word]) -> Iterable[Word]:
    return starmap(concat,
                   sorted(product(sorted(set(root_prefixes) | {()}), sorted(infixes, key=_epsilon_first)),
                          key=lambda seq: seq[::-1]))


def _compute_affix_pmis(sequences: Iterable[Word]) -> Mapping[Tuple[Word, Word], float]:
    affix_to_count: Counter[Tuple[Word, Word]] = Counter(_splits_of(sequences))

    pref_to_count: DefaultDict[Word, int] = defaultdict(int)
    suff_to_count: DefaultDict[Word, int] = defaultdict(int)
    N: int = 0
    for (prefix, suffix), count in affix_to_count.items():
        pref_to_count[prefix] += count
        suff_to_count[suffix] += count
        N += count

    affix_to_pmi: Mapping[Tuple[Word, Word], float] = {
        (p, s): (_norm_pmi_of(N, joint, pref_to_count[p], suff_to_count[s]))
        for (p, s), joint in affix_to_count.items()
    }

    return affix_to_pmi


def _splits_of(sequences: Iterable[Word]) -> Iterable[Tuple[Word, Word]]:
    return flatten([(seq[:i], seq[i:]) for i in range(len(seq) + 1)] for seq in sequences)


def _affixes_of(sequences: Iterable[Word],
                prefix: bool = True,
                unique: bool = True) -> FrozenSet[Word] | Iterable[Word]:
    affixes: Iterable[Word] = flatten(
        map(lambda i: seq[:i] if prefix else seq[i:], range(len(seq) + 1)) for seq in sequences)
    return frozenset(affixes) if unique else affixes


def _epsilon_first(item: Word) -> Tuple[int, Word]:
    return len(item), item


@lru_cache
def _norm_pmi_of(N: int, joint: int, *marginals: int) -> float:
    log_joint: float = log2(joint / N)
    return (log_joint - log2(prod([m / N for m in marginals]))) / -log_joint
