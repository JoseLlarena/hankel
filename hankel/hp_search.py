"""
Functions to search the hyper-parameter space.
"""
from itertools import product
from logging import Logger, getLogger
from string import Template
from typing import Any, Dict, Final, Iterable, Mapping, Sequence, Tuple
from pydantic import validate_call
import deal
from more_itertools import interleave_longest
from numpy import floating, ndarray
from numpy.typing import NDArray

from hankel import Fn, V, X, Y, px, PYDANTIC_CONFIG
from hankel.conversions import as_unweighted, with_single_start
from hankel.data import (Coder, one_hot_coder_from, polar_coder_from, scalar_encode, to_labelled_dataset,
                         to_unlabelled_dataset)
from hankel.evaluation import avg_ternariness_of, evaluate, score, zero_one_loss
from hankel.models import WFSA, with_vector_output
from hankel.spectral import KINDS, Kind, learn_wfsa

LOG: Final[Logger] = getLogger(__package__)
MAX_LOSS: Final[float] = 1e22
RUN_TEMPLATE: Final[Template] = Template('train. loss [${t_loss}] valid. loss [${v_loss}] tern. [${tern}] '
                                         'dim. [${dim}]')
LOOP_TEMPLATE: Final[Template] = Template("\tbasis selection algos\t= ${basis_algos}\n"
                                          "\tfactorisation algos\t= ${factor_algos}\n"
                                          "\ttop-k affixes\t\t= ${topks}\n"
                                          "\ttop-k prefixes\t\t= ${topk_prefs}\n"
                                          "\ttop-k suffixes\t\t= ${topk_suffs}\n"
                                          "\ttrain. loss functions\t= ${t_loss_fns}\n"
                                          "\tvalid. loss functions\t= ${v_loss_fns}\n"
                                          "\tdimensions\t\t= ${dims}\n"
                                          "\tsingular value ratios\t= ${sv_ratios}\n\n"
                                          "\ttrain. size\t\t= ${t_data_size}\n"
                                          "\tvalid. size\t\t= ${v_data_size}\n"
                                          "\ttest   size\t\t= ${e_data_size}\n"
                                          "\tnum. runs\t\t= ${runs}\n"
                                          "\tlogs every\t\t= ${period}\n"
                                          "\tstop loss\t\t= ${stop_loss}\n")


def ppl(dummy: NDArray) -> NDArray:
    return dummy


NAME_TO_LOSS_FN: Final[Dict[str, Fn[[NDArray, NDArray], NDArray]]] = dict(xent=lambda dummy: dummy,
                                                                          ppl=ppl,
                                                                          zero_one_5=px(zero_one_loss, tol=1e-5),
                                                                          zero_one_4=px(zero_one_loss, tol=1e-4),
                                                                          zero_one_3=px(zero_one_loss, tol=1e-3),
                                                                          zero_one_2=px(zero_one_loss, tol=1e-2),
                                                                          zero_one_1=px(zero_one_loss, tol=1e-1),
                                                                          zero_one=px(zero_one_loss, tol=.49),
                                                                          zo5=px(zero_one_loss, tol=1e-5),
                                                                          zo4=px(zero_one_loss, tol=1e-4),
                                                                          zo3=px(zero_one_loss, tol=1e-3),
                                                                          zo2=px(zero_one_loss, tol=1e-2),
                                                                          zo1=px(zero_one_loss, tol=1e-1),
                                                                          zo=px(zero_one_loss, tol=.49))


@deal.pre(lambda kind,
          x_vocab,
          y_vocab,
          hyper_params,
          t_data,
          v_data,
          e_data,
          stop_loss=1e-6,
          period=.1,
          unweighted=False,
          quant=7,
          **meatadata: kind in KINDS and
          x_vocab and
          hyper_params and
          t_data and
          v_data and
          stop_loss >= 0 and
          0 <= period <= 1 and
          quant >= 0)
@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def grid_search(kind: Kind,
                x_vocab: Sequence[X],
                y_vocab: Sequence[Y],
                hyper_params: Mapping[str, Sequence[V]],
                t_data: Sequence[Tuple[Sequence[X], Y] | Sequence[X]],
                v_data: Sequence[Tuple[Sequence[X], Y] | Sequence[X]],
                e_data: Sequence[Tuple[Sequence[X], Y] | Sequence[X]] = (),
                *,
                stop_loss: float = 1e-6,
                period: float = .1,
                unweighted: bool = False,
                quant: int = 7,
                **metadata) -> Tuple[WFSA, float, float, float | None, Dict[str, Any]]:
    """
    Exhaustively searches hyper-parameter space for Spectral learning. The best model, as per the validation loss
    is returned, along with its training and validation losses and optionally the test loss, plus its hyper-parameter
    specification.

    Supported hyper-parameters:

        `basis_algos`   : basis selection algorithms, one of `all`, `auto`, `freq`, `length`, `pmi`, 
            passed to `hankel.spectral.learn_wfsa`.
        `factor_algos`  : factorisation algorithms, one of `svd`, `nmf`, passed to `hankel.spectral.learn_wfsa`.
        `topks`         : sequence of percentages of pref-suffix splits to try in `pmi` or `auto`, 
            passed to `hankel.spectral.learn_wfsa`.
        `topk_prefs`    : sequence of percentages of prefixes to try in `freq`, `length` or `auto`, 
            passed to `hankel.spectral.learn_wfsa`.
        `topk_suffs`    : sequence of percentages of suffixes to try in `freq`, `length` or `auto`, 
            passed to `hankel.spectral.learn_wfsa`.
        `inits`         : initialisation strategies for NMF, one of `svd, random, nndsvd, nndsvda, nndsvdar`, 
            passed to `hankel.spectral.learn_wfsa`. 
        `shuffles`      : shuffling flags for MF, one of `True`, `False`, passed to `hankel.spectral.learn_wfsa`.
        `tern_tols`     : the minimum distances from +-1 and 0 to consider a parameter ternary.

        `t_loss_fns`    : training loss functions to try.  FIXME SHOULDN'T BE A HYPERPARAMETER
        `v_loss_fns`    : validation loss functions to try, also used for evaluating on the test dataset. SAME AS ABOVE
        `dims`          : number of dimensions for the WFSA, passed to `hankel.spectral.learn_wfsa`. SAME AS ABOVE
        `base_vocabs`   : extra tokens to include in the input vocabulary, passed to `hankel.spectral.learn_wfsa`. SAME
        `sv_ratios`     : singular value ratios, passed to `hankel.spectral.learn_wfsa`. SAME

    Args:
        kind (Kind): The type of WFSA to learn, one of `binary`, `lm`, `polar`.
        x_vocab (Sequence[X]): Input vocabulary.
        y_vocab (Sequence[Y]): Output vocabulary.
        hyper_params (Mapping[str, Sequence[V]]): Gridsearch hyper-parameters.
        t_data (Sequence[Tuple[Sequence[X], Y] | Sequence[X]]): Training data, a collection of 2-tuples containing
            the input and target, for acceptors, or just input for an LM.
        v_data (Sequence[Tuple[Sequence[X], Y] | Sequence[X]]): Validation data, a collection of 2-tuples containing
            the input and target, for acceptors, or just input for an LM.
        e_data (Sequence[Tuple[Sequence[X], Y] | Sequence[X]], optional): Test data, a collection of 2-tuples containing
            the input and target, for acceptors, or just input for an LM. Defaults to ().
        stop_loss (float, optional): The loss value that stops the gridsearch early. Defaults to 1e-6.
        period (float, optional): The logging period as a fraction of the total search runs. Defaults to .1.
        unweighted (bool, optional): `True` if an FSA should be returned, `False` for a WFSA. Defaults to `False`.
        quant (int, optional): The level of quantisation to pass to the function that extracts an FSA from the 
            learned WFSA. Ignored if `unweighted` is `False`. Defaults to 7.
        metadata: Extra data to print at the beggining of the gridsearch.

    Raises:
        RuntimeError: If an error prevented the gridsearch to run at all.

    Returns:
        Tuple[WFSA, float, float, float | None, Dict[str, Any]]: A 5-tuple contianing the best WFSA found, its training
            loss, validation loss, test loss, and its hyperparameters.
    """

    specs: tuple[Mapping[str, Any], ...] = tuple(_combine(**hyper_params))
    period = max(1, int(period * len(specs)))

    # (tensor) datasets are needed for evaluation only, not for training
    x_coder: Coder[X] = one_hot_coder_from(x_vocab)
    y_coder: Coder[Y] | None = None if kind == 'lm' else polar_coder_from(
        y_vocab) if kind == 'polar' else one_hot_coder_from(y_vocab)

    t_dataset, train_xs, train_ys = _prepare_data(t_data, x_coder, y_coder, kind)
    v_dataset, *_ = _prepare_data(v_data, x_coder, y_coder, kind)

    info: str = _loop_info(dict(hyper_params) |
                           dict(t_data_size=len(t_data),
                                v_data_size=len(v_data),
                                e_data_size=len(e_data),
                                runs=len(specs),
                                period=period,
                                stop_loss=stop_loss) |
                           metadata)
    LOG.info(f'running tuning loop with specs:\n\n{info}')

    best_wfsa, best_spec, best_v_loss, best_t_loss, best_ternariness = None, {}, MAX_LOSS, MAX_LOSS, -1

    for run,  spec in enumerate(specs, start=1):
        wfsa: WFSA = _make_wfsa(*learn_wfsa(kind, train_xs, train_ys, **_learn_args(spec)), kind)

        t_loss_fn, v_loss_fn, tern_tol = _evaluation_args(spec)
        t_loss, v_loss, ternariness = _compute_metrics(wfsa, t_dataset, v_dataset, t_loss_fn, v_loss_fn, kind, tern_tol)

        if not run % period or run == 1:
            info = _run_info(dict(t_loss=t_loss, v_loss=v_loss, tern=ternariness, dim=len(wfsa.initial)),
                             len(specs),
                             run)
            LOG.info(info)

        if v_loss < best_v_loss or (v_loss == best_v_loss and ternariness > best_ternariness):
            best_ternariness = max(best_ternariness, ternariness)

            best_wfsa, best_spec, best_v_loss, best_t_loss = wfsa, spec, v_loss, t_loss
            best_spec['ternariness'] = float(ternariness)
            best_spec['run'] = run

            if best_v_loss <= stop_loss:
                if run % period and run != 1:
                    info = _run_info(dict(t_loss=t_loss, v_loss=v_loss, tern=ternariness, dim=len(wfsa.initial)),
                                     len(specs),
                                     run)
                    LOG.info(info)
                LOG.info(f'stop loss [{stop_loss:.0e}] achieved, stopping early...')
                break

    if not best_wfsa:
        raise RuntimeError('Could not run tuning loop.')

    if unweighted:
        best_wfsa = with_vector_output(
            WFSA(*as_unweighted(best_wfsa.initial, best_wfsa.transitions, best_wfsa.final, quant)), kind == 'binary')
        t_loss_fn, v_loss_fn, tern_tol = _evaluation_args(best_spec)
       
        best_t_loss, best_v_loss, best_ternariness = _compute_metrics(
            best_wfsa, t_dataset, v_dataset, t_loss_fn, v_loss_fn, kind, tern_tol)
        best_spec['ternariness'] = float(best_ternariness)
       
        info = _run_info(dict(t_loss=best_t_loss,
                              v_loss=best_v_loss,
                              tern=best_ternariness,
                              dim=len(best_wfsa.initial)))
        LOG.info('after conversion to FSA: '+info[24:])

    e_loss: float | None = None
    if e_data:
        e_dataset, *_ = _prepare_data(e_data, x_coder, y_coder, kind)
        e_loss = score(best_wfsa, e_dataset) if kind == 'lm' else evaluate(best_wfsa, *e_dataset, v_loss_fn)

    return best_wfsa, best_t_loss, best_v_loss, e_loss, best_spec


def _prepare_data(data: Sequence[Tuple[Sequence[X], Y] | Sequence[X]],
                  x_coder: Coder[X],
                  y_coder: Coder[Y] | None,
                  kind: Kind) \
    -> Tuple[Tuple[Sequence[NDArray], Sequence[NDArray]] | Iterable[NDArray],
             Sequence[Sequence[X]],
             Tuple[float, ...],]:
    xs, ys = zip(*data) if kind != 'lm' else (data, ())

    dataset: Tuple[Sequence[NDArray], Sequence[NDArray]] | Iterable[NDArray] = \
        (to_unlabelled_dataset(xs, x_coder) if kind == 'lm' else
            to_labelled_dataset(xs, ys, x_coder, y_coder))

    train_ys: Tuple[float, ...] = tuple(scalar_encode(y, kind=kind) for y in ys) if kind != 'lm' else ()

    return dataset, xs, train_ys


def _make_wfsa(initial: NDArray, transitions: NDArray, final: NDArray, kind: Kind) -> WFSA:
    params: Tuple[NDArray, NDArray, NDArray] = with_single_start(initial, transitions, final)
    wfsa: WFSA = WFSA(*params)
    return wfsa if kind == 'lm' else with_vector_output(wfsa, kind == 'binary')


def _compute_metrics(wfsa: WFSA, t_dataset, v_dataset, t_loss_fn, v_loss_fn, kind: Kind, tern_tol: float) \
        -> Tuple[float, float, float]:

    t_loss: float = score(wfsa, t_dataset, ppl=t_loss_fn == ppl) if kind == 'lm' else \
        evaluate(wfsa, *t_dataset, t_loss_fn)

    v_loss: float = score(wfsa, v_dataset, ppl=v_loss_fn == ppl) if kind == 'lm' else \
        evaluate(wfsa, *v_dataset, v_loss_fn)

    return t_loss, v_loss, avg_ternariness_of(wfsa.parameters, tern_tol)


def _learn_args(spec: Dict[str, Any]) -> Dict[str, Any]:
    # TODO ADD SANITY CHECKS
    factor_kwargs: Dict[str, Any] = dict(algo=spec['factor_algo'],
                                         tol=spec.get('sv_ratio', 1e-1),
                                         dim=spec.get('dim', -1))

    basis_kwargs: Dict[str, Any] = dict(algo=spec['basis_algo'], base_vocab=spec.get('base_vocab', ()))
    if basis_kwargs['algo'] in {'auto', 'pmi'}:
        basis_kwargs['topk'] = spec['topk']

    elif basis_kwargs['algo'] in ('freq', 'length'):
        basis_kwargs['topk_pref'] = spec['topk_pref']
        basis_kwargs['topk_suff'] = spec['topk_suff']

    return dict(basis_kwargs=basis_kwargs, factor_kwargs=factor_kwargs)


def _evaluation_args(spec: Dict[str, Any]) -> Tuple[Fn, Fn, float]:
    tern_tol: float = spec.get('tern_tol', 1e-2)
    t_loss_fn: str = spec['t_loss_fn'].replace('-', '_')
    v_loss_fn: str = spec['v_loss_fn'].replace('-', '_')

    return NAME_TO_LOSS_FN[t_loss_fn], NAME_TO_LOSS_FN[v_loss_fn], tern_tol


def _combine(**specs) -> Iterable[Dict[str, Any]]:

    # special case for 'auto' basis selection algo
    # interleaving helps it find a solution faster if one of the algos does not work at all
    if set(specs['basis_algos']) == {'pmi', 'freq', 'length'}:

        pmi_specs = dict(specs)
        pmi_specs['basis_algos'] = ('pmi',)
        del pmi_specs['topk_prefs']
        del pmi_specs['topk_suffs']

        freq_specs = dict(specs)
        freq_specs['basis_algos'] = ('freq',)
        del freq_specs['topks']

        len_specs = dict(specs)
        len_specs['basis_algos'] = ('length',)
        del len_specs['topks']

        return interleave_longest(_make_combos(pmi_specs), _make_combos(freq_specs), _make_combos(len_specs))

    return _make_combos(specs)


def _make_combos(specs: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for k, v in dict(specs).items():
        if isinstance(v, Sequence) and not v:
            del specs[k]

    keys: Tuple[str, ...] = tuple(key[:-1] for key in specs)  # assumes keys end in 's'
    return map(lambda vals: dict(zip(keys, vals)), product(*(specs.values())))


def _run_info(spec: Dict[str, V], num_runs: int = -1, run: int = -1) -> str:
    return (f'[{run:5,d}] of [{num_runs:4,d}] runs: ' +
            RUN_TEMPLATE.substitute({k: f'{_format_value(k,v)}' for k, v in spec.items()}))


def _loop_info(specs: Dict[str, V]) -> str:
    return LOOP_TEMPLATE.substitute({k: f'{_format_value(k,v)}' for k, v in specs.items()})


def _format_value(key: str, value: Any) -> str:
    match (key, value):

        case ('dim', value):
            return f'{value:2,d}'

        case ('tern', value):
            return f'{value:.1f}'

        case ('stop_loss', value):
            return f'{value:.0e}'

        case (key, value) if key.endswith('size') or key.endswith('runs') or key.endswith('period'):
            return f'{value:5,d}'

        case (key, value) if isinstance(value, Sequence) and not isinstance(value, str):
            if len(value) > 2:
                value = (value[0], '...', value[-1])
            return '('+', '.join(map(str, value))+')'

        case (key, value) if isinstance(value, (float, floating, ndarray)):
            return f'{value:.3e}'

        case _:
            return f'{value}'
