"""
Module for conversions from, to and between different types of WFSAs and other structures.
"""
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence, Set
from itertools import product
from logging import Logger, getLogger
from math import atan2, degrees, log2
from typing import Deque, Dict, Final, FrozenSet, List, Literal, Mapping, Sequence, Set, Tuple, TypeAlias

import deal
from graphviz import Digraph
from numpy import (argmax, around, array_equal, broadcast_to, diag, einsum, expand_dims, eye, float32, ones, outer,
                   squeeze, stack, zeros, zeros_like)
from numpy.typing import NDArray
from pydantic import validate_call
from scipy.linalg import inv, norm, svd

from hankel import PYDANTIC_CONFIG, Fn, px
from hankel.models import WFSA

Port: TypeAlias = Literal['n', 's', 'e', 'w', 'nw', 'sw', 'se', 'ne']

ALL_PORTS: Final[Tuple[Port, ...]] = ('n', 's', 'e', 'w', 'nw', 'sw', 'se', 'ne')
SYMBOL_TO_SUBSCRIPT: Final[Mapping[str, str]] = dict(zip('0123456789-', '₀₁₂₃₄₅₆₇₈₉₋')) 
SYMBOL_TO_SUPERSCRIPT: Final[Mapping[str, str]] = {"0": "⁰",
                                                   "1": "¹",
                                                   "2": "²",
                                                   "3": "³",
                                                   "4": "⁴",
                                                   "5": "⁵",
                                                   "6": "⁶",
                                                   "7": "⁷",
                                                   "8": "⁸",
                                                   "9": "⁹",
                                                   "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ", "f": "ᶠ", "g": "ᵍ",
                                                   "h": "ʰ", "i": "ⁱ", "j": "ʲ", "k": "ᵏ", "l": "ˡ", "m": "ᵐ", "n": "ⁿ",
                                                   "o": "ᵒ", "p": "ᵖ", "q": "q", "r": "ʳ", "s": "ˢ", "t": "ᵗ", "u": "ᵘ",
                                                   "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ",
                                                   "A": "ᴬ", "B": "ᴮ", "C": "ᶜ", "D": "ᴰ", "E": "ᴱ", "F": "ᶠ", "G": "ᵍ",
                                                   "H": "ᴴ", "I": "ᴵ", "J": "ʲ", "K": "ᴷ", "L": "ᴸ", "M": "ᴹ", "N": "ᴺ",
                                                   "O": "ᴼ", "P": "ᴾ", "Q": "q", "R": "ᴿ", "S": "ˢ", "T": "ᵀ", "U": "ᵁ",
                                                   "V": "ⱽ", "W": "ᵂ", "X": "ˣ", "Y": "ʸ", "Z": "ᶻ"}

LOG: Final[Logger] = getLogger(__package__)


@deal.pre(lambda initial, transitions, final:
          initial.ndim == 1 and initial.size and
          transitions.ndim == 3 and transitions.size and
          transitions.shape[-2] == transitions.shape[-1] and transitions.shape[-1] == initial.shape[0] and
          final.size and final.shape == initial.shape)
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def with_single_start(initial: NDArray, transitions: NDArray, final: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Transforms WFSA parameters to functional equivalent ones but where there's a single initial state, ie, the initial
    weight vector is a one hot vector.

    Args:
        initial (NDArray): Initial weight vector.
        transitions (NDArray): State transition tensor.
        final (NDArray): Final Weight Vector.

    Returns:
        Tuple[NDArray, NDArray, NDArray]: Transformed parameters
    """
    if array_equal(initial, zeros_like(initial)):
        LOG.warning("WFSA's initial state is the origin so it can't be converted to a single start state.")
        return initial, transitions, final

    target: NDArray = zeros_like(initial)
    target[initial.argmax()] = 1
    m: NDArray = _find_v2v_matrix(initial, target)
    return target, inv(m) @ transitions @ m, inv(m) @ final


@deal.pre(lambda initial, transitions, final, length=-1, quant=7:
          initial.ndim == 1 and initial.size and
          transitions.ndim == 3 and transitions.size and
          transitions.shape[-2] == transitions.shape[-1] and transitions.shape[-2] == initial.shape[0] and
          final.size and final.shape == initial.shape and
          (length == -1 or length > 0) and
          quant >= 0)
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def as_unweighted(initial: NDArray, transitions: NDArray, final: NDArray, length: int = -1, quant: int = 7) \
        -> Tuple[NDArray, NDArray, NDArray]:
    """
    Returns the unweighted version of the given WFSA parameters. Experimentatl. When it works, the resulting parameters 
    are functionally equivalent but only have boolean weights. This only works for acceptors not for language models.
    The current implementation is a hack and uses a path through the WFSA to sample the embeddings/states of the 
    corresponding FSA.

    Args:
        initial (NDArray): Initial weight vector.
        transitions (NDArray): State transition tensor.
        final (NDArray): Final weight Vector.
        length (int, optional): The length of the path to use to sample the embeddings. Defaults to -1.
        quant (int, optional): The quantisation level. Defaults to 7.

    Returns:
        Tuple[NDArray, NDArray, NDArray]: Parameters of the FSA corresponding to the given WFSA.
    """
    symbols: Sequence[int] = tuple(range(transitions.shape[0]))
    length = int(6/log2(len(symbols))) if length == -1 else length  # hack fixing number of probing points to <= 2**6
    combo: Iterable[Tuple[int, ...]] = tuple(product(symbols, repeat=length))
    LOG.debug(f'Converting WFSA to unweighted with length {length} and precision {quant} = {len(combo)} combinations')
    batch: NDArray = stack(tuple(tuple(eye(len(symbols), dtype=int)[idx] for idx in seq) for seq in combo))

    recs: FrozenSet[Tuple[Tuple[float, ...], int, Tuple[float, ...], int]] = \
        _sample_transitions(initial, transitions, final, batch, quant)
    state_to_id: Mapping[Tuple[float, ...], int] = dict(
        zip(states := {q for (q_prev, x, q, final) in recs}, range(len(states))))
    dim: int = len(state_to_id)

    initial_: NDArray = zeros((dim, ))
    final_: NDArray = zeros((dim, ))
    transitions_: List[NDArray] = [zeros((dim, dim)) for _ in range(len(symbols))]
    for (q_prev, x, q, is_final) in recs:
        if not q_prev:
            initial_[state_to_id[q]] = 1
        else:
            transitions_[x][state_to_id[q]][state_to_id[q_prev]] = 1
        final_[state_to_id[q]] = is_final

    return initial_, stack([m.T for m in transitions_], axis=-3), final_


@deal.pre(lambda wfsa, symbol_names=(), state_names=(), unweighted=False, title="WFSA State Transition Diagram":
          not symbol_names or len(symbol_names) == len(wfsa.transitions) and
          not state_names or len(state_names) == len(wfsa.initial))
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def wfsa_to_graphviz(wfsa: WFSA,
                     symbol_names: Sequence[str] = (),
                     state_names: Sequence[str] = (),
                     unweighted: bool = False,
                     title: str = 'WFSA') -> Digraph:
    """
    Generates a Graphviz Digraph with the state-transition diagram of the given WFSA.

    Args:
        wfsa (WFSA): The (W)WFSA
        symbol_names (Sequence[str], optional): Names for the input symbols. If not provided, they 
            will be named as "σ" followed by a subscipt integer indicating their order. Defaults to ().
        state_names (Sequence[str], optional): Names for the states. If not provided, they will named as "q" followed
            by an integer subscript indicated their order starting with the initial state and following a path
            from their to the final state(s).Defaults to ().
        unweighted (bool, optional): True if the WFSA should be displayed as a FSA. Defaults to False.
        title (str, optional): Title of Diagraph when rendered. Defaults to "WFSA".

    Raises:
        ImportError: If graphviz is not installed.

    Returns:
        Digraph: The Digraph with the state-transition diagram.
    """
    if title == 'WFSA' and unweighted:
        title = title[1:]
    try:
        import graphviz
    except ImportError:
        raise ImportError("graphviz package is required. Install with: pip install graphviz")

    initials: NDArray = wfsa.initial
    finals: NDArray = wfsa.final
    transits: Tuple[NDArray, ...] = wfsa.trans_mats
    vocab_size: int = len(wfsa.trans_mats)

    symbol_names = tuple(symbol_names or [f"σ{num_to_subs(i)}" for i in range(vocab_size)])
    state_names = tuple(state_names or _compute_state_names(initials, transits))

    graph: Digraph = Digraph(engine='circo')
    graph.attr(margin='.1', pad='.2', dpi='110', overlap='false', label=title+'\n\n\n',  labelloc='t',
               splines='true', mindist=('1' if unweighted else '2.5'),
               outputorder='edgesfirst', root=state_names[argmax(initials)])

    graph = _draw_states(state_names, unweighted, initials, finals, graph, fontsize='12')
    graph = _draw_transitions(graph, wfsa, state_names, symbol_names, unweighted, fontsize='12')

    return graph


@deal.pre(lambda ps, Ts, Os:
          Ts.ndim == 2 and Ts.size and Ts.shape[0] == Ts.shape[1] and
          Os.ndim == 2 and Os.size and Os.shape[1] == Ts.shape[1] and
          ps.ndim == 1 and ps.size and ps.shape[0] == Ts.shape[0])
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def hmm_to_wfsa(init: NDArray, transitions: NDArray, observations: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Converts a Hidden Markov Model's parameters to a probabilistic WFSA's parameters.

    Args:
        init (NDArray): Initial probabilities.
        transitions (NDArray): Transition probabilities.
        observations (NDArray): Observation probabilities.

    Returns:
        Tuple[NDArray, NDArray, NDArray]: The initial weight, transition tensor and final weight of an equivalent WFSA.
    """
    trans: NDArray = stack([diag(symbol_obs) @ transitions for symbol_obs in observations])
    return init, trans, ones(init.shape[0])

# --------------------------------------------- DELEGATE FUNCTIONS -----------------------------------------------------


def _find_v2v_matrix(a: NDArray, b: NDArray) -> NDArray:
    """
    Calculates the linear transformation needed to convert `a` into `b` using a rotation/reflection `R` followed by
    a scalar scaling S, such that `b = SRa`.

    Args:
        a (NDArray): Start vector.
        b (NDArray): End vector.

    Raises:
        ValueError: If the dimensions of `a` and `b` differ, or if they are not vectors or if  `a` is the null vector 
        and `b` is not.

    Returns:
        NDArray: A matrix that when pre-multiplied by `a` results in `b`.
    """

    if a.shape != b.shape:
        raise ValueError(f'The dimensions of vector a [{a.shape}] are different from those of b [{b.shape}]')
    if a.ndim != 1:
        raise ValueError(f'The rank of the vectors must be 1 but the arguments have rank [{a.ndim}]')

    dim: int = a.shape[0]

    a_norm: float32 = norm(a)
    b_norm: float32 = norm(b)

    if b_norm == 0.:  # simple case that doesn't require SVD
        return zeros((dim, dim))

    if a_norm == 0.:  # impossible case, must be rejected
        raise ValueError('Vector a is null and vector b is not, and that is not allowed')

    # normal case where we calculate the singular value decomposition of the outer product of the vectors
    U, _, Vh = svd(outer(b / b_norm, a / a_norm))

    # Combine rotation and scaling in one step
    return (b_norm / a_norm) * (U @ Vh).T


def _sample_transitions(initial: NDArray, transitions: NDArray, final: NDArray, xs: NDArray, precision: int = 7)\
        -> FrozenSet[Tuple[Tuple[float, ...], int, Tuple[float, ...], int]]:
    """
    Samples transitions from the given WFSA to construct a corresponding FSA.

    Args:
        initial (NDArray): Initial weight vector.
        transitions (NDArray): State transition tensor.
        final (NDArray): Final weight Vector.
        xs (NDArray): The input to sample transitions with.
        precision (int, optional): The level of qunatisation to avoid spurious states due to
            numerical imprecisions. Defaults to 7.

    Returns:
        FrozenSet[Tuple[Tuple[float, ...], int, Tuple[float, ...], int]]: A collection of tuples representing 
            transitions as [start state index, symbol index, end state index, whether end state is a final state]
    """
    quant: Fn = px(around, decimals=precision)

    init: NDArray = expand_dims(expand_dims(initial, 0), -1)  # D -> 1xD -> 1xDx1
    state: NDArray = broadcast_to(init, (xs.shape[0], init.shape[1], 1)).copy()
    prev_state: NDArray = state
    unique_transitions: Set[Tuple[Tuple[float, ...], int, Tuple[float, ...], int]] = {
        ((), -1, tuple(map(float, initial)), int(quant(initial @ final)))}
    if xs.ndim == 3:  # this is to support empty strings

        trans: NDArray = einsum('BTV,VDd->BTdD', xs, transitions)  # Pre-computes transitions: 'BTV,VDd -> BTdD'

        for t in range(xs.shape[-2]):
            state = trans[:, t] @ state  # [BxTxDxD -> BxDxD] @ BxDx1 -> BxDx1
            transitions_ = ((tuple(map(float, q_prev)),
                             int(argmax(x)),
                             tuple(map(float, q)),
                             int(quant(q @ final)))
                            for q_prev, x, q in zip(quant(squeeze(prev_state, -1)),
                                                    xs[:, t, :],
                                                    quant(squeeze(state, -1))))
            unique_transitions.update(transitions_)
            prev_state = state

    return frozenset(unique_transitions)


def _draw_states(state_names: Tuple[str, ...],
                 unweighted: bool,
                 initial: NDArray,
                 finals: NDArray,
                 graph: Digraph,
                 fontsize: str) -> Digraph:
    """
    Draws the states of the WFSA as as vertices in a graph.

    Args:
        state_names (Sequence[str,]): Nemes of each state.
        unweighted (bool): True if this is part of an unweighted FSA.
        initial (NDArray): Vector of initial weights.
        finals (NDArray): Vector of final weights.
        graph (Digraph): Graph to draw state in.
        fontsize (int): Size of the font for state names and final weightes.

    Returns:
        Digraph: Graph with states drawn.
    """
    graph.node('', shape='point', width='0', height='0', style='invis', label='')  # invisible root node for layout

    for i, state_name in enumerate(state_names):
        initial_w: float = initial[i]
        final_w: float = finals[i]
        node_label: str = state_name if unweighted else _in_node_box(state_name, _simplify_weight(final_w))
        size: str = '.5' if unweighted else '.8'
        shape: str = 'circle'
        color: str = 'black'
        style: str = 'filled'
        margin: str = '0'

        if _non_zero(initial_w):
            color = 'crimson'
            style = 'filled, bold'

        if _non_zero(final_w):
            shape = 'doublecircle'
            size = '.45' if unweighted else '.7'

        graph.node(state_name,
                   node_label,
                   fillcolor='white',
                   shape=shape,
                   color=color,
                   margin=margin,
                   width=size,
                   height=size,
                   fontname="Dejavu Sans Mono",
                   style=style,
                   fixedsize='true',
                   fontsize=fontsize)
    return graph


def _draw_transitions(graph: Digraph,
                      wfsa: WFSA,
                      state_names: Tuple[str, ...],
                      symbol_names: Tuple[str, ...],
                      unweighted: bool,
                      fontsize: str) -> Digraph:
    """
    Draws WFSA transitions as labelled edges in a graph.

    Args:
        graph (Digraph): The grap to draw the transitions in.
        wfsa (WFSA): The WFSA.
        state_names (Tuple[str, ...]): The WFSA state names.
        symbol_names (Tuple[str, ...]): The WFSA input symbol names.
        unweighted (bool): True if this is part of an unweighted FSA.
        fontsize (str): Size of the font for edge labels.

    Returns:
        Digraph: The graph with transitions drawn in it.
    """
    # Collects all transitions from the same state
    num_states: int = len(wfsa.initial)
    vocab_size: int = len(wfsa.trans_mats)
    trans_to_symbolweights = defaultdict(list)

    for symbol_idx in range(vocab_size):
        symbol: str = symbol_names[symbol_idx]
        transitions: NDArray = wfsa.trans_mats[symbol_idx]

        for from_state in range(num_states):
            for to_state in range(num_states):
                weight: float = transitions[to_state, from_state]
                if _non_zero(weight) or not unweighted:
                    trans_to_symbolweights[from_state, to_state].append((symbol, weight))

    # Draws the standard transitions
    for (from_state, to_state), symbolweights in trans_to_symbolweights.items():
        if from_state == to_state:
            continue
        symbols, weights = zip(*[(symbol, _simplify_weight(weight) if not unweighted else None)
                                 for symbol, weight in symbolweights])
        graph.edge(state_names[from_state],
                   state_names[to_state],
                   label=_in_edge_box(symbols, weights),
                   fontsize=fontsize,
                   color='#00000080',
                   fontname="Dejavu Sans Mono",
                   penwidth='0.5',
                   arrowsize='.75')

    # Draws the self-loops
    state_to_pos: Mapping[str, Tuple[float, float]] = _find_node_positions(graph)
    state_to_ports: Mapping[str, FrozenSet[str]] = _find_occupied_ports(wfsa, state_names, state_to_pos)

    for (from_state, to_state), symbolweights in trans_to_symbolweights.items():
        if from_state != to_state:
            continue
        symbols, weights = zip(*[(symbol, _simplify_weight(weight) if not unweighted else None)
                                 for symbol, weight in symbolweights])

        head, tail = _find_available_ports(state_to_ports, state_names[from_state])
        graph.edge(state_names[from_state],
                   state_names[to_state],
                   label=_in_edge_box(symbols, weights),
                   fontsize=fontsize,
                   color='#00000080',
                   fontname="Dejavu Sans Mono",
                   penwidth='0.5',
                   headport=head,
                   tailport=tail,
                   arrowsize='.75')
    return graph


def _simplify_weight(weight: float) -> float:
    rounded: float = round(float(weight), 2)

    return 0 if abs(rounded) < 1e-6 else rounded


def _compute_state_names(initial: NDArray, transits: Tuple[NDArray, ...]) -> Tuple[str, ...]:
    """
    Compute state names for a WFSA using a BFS-based labelling approach.

    This function performs a breadth-first search (BFS) starting from the initial state
    and assigns new state names based on the BFS order. Unreachable states are labeled last.

    Args:
        initial (NDArray): Initial state weights.
        transits (Tuple[NDArray, ...]): A tuple of transition matrices, one for each symbol in the vocabulary.

    Returns:
        Tuple[str, ...]: A tuple of new state names, where each name is a string in the format "q₀", "q₁", etc.
            where the index indicates the step in the path from the initial state to the final state(s).
    """

    initial_state: int = int(argmax(initial))
    num_states: int = len(initial)
    vocab_size: int = len(transits)

    if num_states == 1:
        return ('q₀',)

    # Step 1: Builds adjacency list
    state_to_adjacents: List[List[int]] = [[] for _ in range(num_states)]
    for symbol_idx in range(vocab_size):
        transitions: NDArray = transits[symbol_idx]
        for from_state in range(num_states):
            for to_state in range(num_states):
                if abs(transitions[to_state, from_state]) > 1e-6:
                    state_to_adjacents[from_state].append(to_state)

    # Step 2: BFS to determine naming order
    visited: List[bool] = [False] * num_states
    visited[initial_state] = True
    queue: Deque = deque([initial_state])
    bfs_order: List[int] = []

    while queue:
        state: int = queue.popleft()
        bfs_order.append(state)
        for neighbor in state_to_adjacents[state]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    # Step 3: Assigns state names based on BFS order
    state_id_to_label: Dict[int, str] = {}
    for new_index, old_index in enumerate(bfs_order):
        state_id_to_label[old_index] = f"q{num_to_subs(new_index)}"

    # Fills in unreachable states with labels after BFS
    counter: int = len(bfs_order)
    for state_id in range(num_states):
        if state_id not in state_id_to_label:
            state_id_to_label[state_id] = f"q{num_to_subs(counter)}"
            counter += 1

    return tuple(state_id_to_label[i] for i in range(num_states))


def num_to_subs(num: float | int | str) -> str:
    """
    Converts a number into a subscript string representation

    Args:
        num (float | int | str): number to convert.

    Returns:
        str: subscript format of the number or the number itself if there's no subscript equivalent.
    """
    return ''.join(SYMBOL_TO_SUBSCRIPT.get(digit, digit) for digit in str(num))

def num_to_super(num: float | int | str) -> str:
    """
    Converts a number into a superscript string representation

    Args:
        num (float | int | str): number to convert.

    Returns:
        str: superscript format of the number or the number itself if there's no superscript equivalent.
    """
    return ''.join(SYMBOL_TO_SUPERSCRIPT.get(digit, SYMBOL_TO_SUPERSCRIPT.get(digit.lower(), digit)) for digit in str(num))



def _non_zero(weight: float) -> bool:
    """
    Checks if a weight is non-zero within a small tolerance.
    """
    return abs(weight) > 1e-6


def _in_node_box(state: str, weight: float) -> str:
    return f"""<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="1" BGCOLOR="white" >
                    <TR><TD>{state}</TD></TR>
                    <TR><TD><sub>{weight}</sub></TD></TR>
                </TABLE>>"""


def _in_edge_box(symbols: Tuple[str, ...], weights: Tuple[str, ...]) -> str:

    symbols, weights = zip(* [(symbol, weight) for symbol, weight in zip(symbols, weights) if weight])

    label: str = ''.join(f'<TD BORDER="1" BGCOLOR="white" COLOR="#f0f0f0" VALIGN="middle">{symbol}'
                         f'<sub>&thinsp;{weight}</sub></TD>' if weight is not None else
                         f'<TD BORDER="1" BGCOLOR="white" COLOR="#f0f0f0" VALIGN="middle">{symbol}</TD>'
                         for symbol, weight in zip(symbols, weights))

    return (f"""<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" BGCOLOR="white" COLOR="white" """
            f"""ALIGN="center" VALIGN="middle">"""
            f"""<TR>{label}</TR></TABLE>>""")


def _find_node_positions(graph: Digraph) -> Mapping[str, Tuple[float, float]]:
    """
    Finds the x-y coordinates of all nodes/states.

    Args:
        graph (Digraph): The graph.

    Returns:
        Mapping[str, Tuple[float, float]]: A map from state names to theyr x-y coordinates.
    """

    plain_output: str = graph.pipe(format='plain').decode('utf-8')
    node_positions: Dict[str, Tuple[float, float]] = {}

    for line in plain_output.splitlines():
        parts: List[str] = line.split()
        if parts[0] == 'node':
            name: str = parts[1]
            x: float = float(parts[2])
            y: float = float(parts[3])
            node_positions[name] = (x, y)

    return node_positions


def _find_occupied_ports(wfsa: WFSA, state_names: Tuple[str, ...], node_positions: Mapping[str, Tuple[float, float]])\
        -> Mapping[str, FrozenSet[Port]]:
    """

    Finds ports (N, NE, E, SE, S, SW, W, NW) that are being used for incoming or outgoing edges for each node.

    Args:
        wfsa (WFSA): The WFSA.
        state_names (Tuple[str, ...]): The state names.
        node_positions (Mapping[str, Tuple[float, float]]): A map with the x-y coordinates for each node.

    Returns:
        Mapping[str, FrozenSet[Port]]: A map with all the occupied ports for each node.
    """
    occupied_ports: Dict[str, Set[Port]] = {state: set() for state in state_names}

    # Gets transitions
    transits: Tuple[NDArray, ...] = wfsa.trans_mats
    num_states: int = len(state_names)

    for symbol_idx in range(len(transits)):
        transitions: NDArray = transits[symbol_idx]

        for from_state in range(num_states):
            for to_state in range(num_states):
                weight: float = transitions[to_state, from_state]

                if _non_zero(weight) and from_state != to_state:
                    from_name: str = state_names[from_state]
                    to_name: str = state_names[to_state]

                    # Calculates which ports are used based on relative positions
                    from_pos: Tuple[float, float] = node_positions[from_name]
                    to_pos: Tuple[float, float] = node_positions[to_name]

                    # Determines outgoing port from from_node
                    outgoing_port: Port = _port_from_positions(from_pos, to_pos)
                    occupied_ports[from_name].add(outgoing_port)

                    # Determines incoming port to to_node
                    incoming_port: Port = _port_from_positions(to_pos, from_pos)
                    occupied_ports[to_name].add(incoming_port)

    return {state: frozenset(ports) for state, ports in occupied_ports.items()}


def _find_available_ports(occupied_ports: Mapping[str, FrozenSet[Port]], state_name: str) -> Tuple[Port, Port]:
    """
    Finds available port pairs for self-loops that don't conflict with occupied ports by incoming/outgoing edges.

    Args:
        occupied_ports (Mapping[str, FrozenSet[Port]]): A map with occupied ports for each node.
        state_name (str): Name of the state to find free ports for.

    Returns:
        Tuple[Port, Port]: Start and end ports for the given state's self-loop. If none found, the south port is 
            returned as a last resort.
    """
    available: Set[Port] = set(ALL_PORTS) - set(occupied_ports.get(state_name, set()))

    for port in ALL_PORTS:
        if port in available:
            return port, port

    return 's', 's'


def _port_from_positions(from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> Port:
    """
    Finds port (n, ne, e, se, s, sw, w, nw) at the the given coordinates.

    Args:
        from_pos (Tuple[float, float]): The star position.
        to_pos (Tuple[float, float]): The end position.

    Returns:
        Port: The port.
    """

    dx: float = to_pos[0] - from_pos[0]
    dy: float = to_pos[1] - from_pos[1]

    angle: float = degrees(atan2(dy, dx))

    # Normalises to 0-360 degrees
    if angle < 0:
        angle += 360

    # Maps angle to compass directions
    if angle < 22.5 or angle >= 337.5:
        return 'e'
    elif angle < 67.5:
        return 'ne'
    elif angle < 112.5:
        return 'n'
    elif angle < 157.5:
        return 'nw'
    elif angle < 202.5:
        return 'w'
    elif angle < 247.5:
        return 'sw'
    elif angle < 292.5:
        return 's'
    else:
        return 'se'
