#!/usr/bin/env python3
"""
CLI for learning Weighted Finite State Automata using the Spectral method.
"""

import webbrowser
from collections.abc import Iterable
from functools import reduce
from io import BufferedReader, BufferedWriter, BytesIO
from logging import DEBUG, Logger, getLogger
from os.path import abspath
from string import Template
from typing import Any, Dict, Final, Mapping, Tuple

import sys
from click import BadParameter, Choice, File, FloatRange, IntRange, Path, argument, get_binary_stream, group, option
from numpy import allclose, arange, eye, stack, zeros_like
from numpy.linalg import matrix_power
from numpy.typing import NDArray

from hankel import CSVList, Fn, RangeOfSteppedPercentages, TripleSplit, config_logging, nout, validate_special_int
from hankel.conversions import num_to_super, wfsa_to_graphviz
from hankel.data import load_labelled_data, load_unlabelled_data, one_hot_coder_from, to_unlabelled_dataset
from hankel.evaluation import class_predict, lm_predict, make_batches
from hankel.hp_search import grid_search
from hankel.model_io import load_wfsa, save_wfsa
from hankel.spectral import Kind

LOG: Final[Logger] = getLogger(__package__)

LEARN_HELP: Final[
    str
] = """Learn a non-deterministic (Weighted) Finite State Automaton (W/FSA) using Spectral Learning.\n
                        It runs a grid search over the specified hyperparameters, using the validation
                        split to choose the lowest validation loss.\n
                        positional arguments:\n
                          INFILE\tPath to the input data file. It must contain examples as sequences of
                        whitespace-separated tokens. For acceptors, it must also be followed
                        by a tab, and then the target value: '1' for a positive example, and '0' for a negative
                        example.\n
                          e.g.: 0 1 1 0    1\n
                          MODEL: <FILE>.npz or stdout\tPath to output data file. It contains weights with metadata
                        in a numpy compressed archive with keys 'initial', 'transitions', 'final' and 'metadata'.
                        'initial' will contain the d-dimensional initial weight vector, 'transitions' will contain a 3rd order tensor of
                        shape (v, d, d) the per-symbol transition weights, and 'final' will contain the d-dimensional
                        vector of final weights.
                        'metadata' is a dictionary with the following keys:\n
                            'vocab': the input vocabulary, of size v\n
                            'kind' : the kind of WFSA, with value in {'binary', 'lm', 'polar'}\n
                            'unweighted': with value in {False, True}\n
                          \n
                        """

KIND_HELP: Final[str] = """The type of WFSA to learn:\n
                                binary: acceptor (binary classifier) with outputs {0, 1}\n
                                polar:  acceptor (binary classifier) with outputs {-1, 1}\n
                                lm: a language model with outputs in [0, 1]\n
                                \t\n
                                """
DIM_HELP: Final[str] = """Number of dimensions/states of the WFSA, either a positive integer or -1. If the positive
                            integer is greater than the number of non-zero singular values i will be set to the
                            rank of the hankel matrix min(|rows|, |cols|).
                            When set to -1, the dimensionality is determined by the singular value ratio option
                            (--sv_ratio).\n
                            \t\n"""

SV_RATIO_HELP: Final[str] = """The minimum fraction of the Hankel matrix's SVD's largest singular value other singular
                                values must have in order to be kept. It controls the number states in the WFSA. A value
                                of 0 will keep all singular values. A value of 1 will keep only the largest singular
                                value(s). Ignored
                                if --dim is not set to -1.\n
                                \t\n"""

SPLITS_HELP: Final[str] = """A triple of integers representing the percentages of the data to be used for each
                                training/validation/test data split. They must add up to 100. If the validation split
                                is 0, the training split will be used for validation instead.\n
                                \t\n"""
LOSS_FN_HELP: Final[str] = """The type of loss function to evaluate the model on the training/validation/test sets:\n
                                ppl: perplexity\n
                                zo:  one-zero loss with a .5 threshold\n
                                zo1: one-zero loss with a 1e-1 threshold\n
                                zo2: one-zero loss with a 1e-2 threshold\n
                                zo3: one-zero loss with a 1e-3 threshold\n
                                zo4: one-zero loss with a 1e-4 threshold\n
                                zo5: one-zero loss with a 1e-5 threshold\n
                                \t\n
                                """
LOG_HELP: Final[str] = """Fraction of the total number grid search runs to log progress at
                            """

STOP_LOSS_HELP: Final[str] = """Value of the loss the grid search will stop at.
                                """
UNWEIGHTED_HELP: Final[str] = """Wether to return an unweighted FSA.
                                Only valid when --kind is 'binary' or 'polar'"""
FAIL_STATES_HELP: Final[str] = """Wether to keep fail states in the unweighted FSA. Some automata, e.g. Tomita 2, are
                                conventionally represented with a fail state, whereas others, e.g. Reber, aren't. This
                                flags helps choose which one is more appropriate. Ignored if --unweighted is not used.
                                """
QUANT_HELP: Final[str] = """The level of rounding precision to use when estimating an unweighted FSA. This estimation is
                            done by sampling and quantising the WFSA's embedding space. Lower values can merge distinct
                            states. Higher values can lead to spurious states. Ignored if --kind is 'lm' or --unweighted
                            is not used.
                            """

BASIS_HELP: Final[str] = """The type of basis selection algorithm to use:\n
                                auto: will try 'pmi' first, then 'freq' then 'length', in a roundrobin fashion. Its
                                slower than individual algorithms but more likely to find a complete basis.\n
                                all: use all prefixes and suffixes in the training data\n
                                freq: use the k-most frequent prefixes and suffixes in the training data.
                                The associated hyperparameters are --topk_pref and --topk_suff\n
                                length: use the k shortest prefixes and suffixes in the training data.
                                The associated hyperparameters are --topk_pref and --topk_suff\n
                                pmi: use the k-most uncorrelated prefix-suffix pairs in the training data, using their
                                pointwise mutual information (PMI). The associated hyperparameter is --topk\n
                                The actual number of prefixes and suffixes used will be their cartesian product,
                                augmented with all vocabulary items.\n
                                \t\n
                                """

TOP_K_HELP: Final[str] = """The range of percentages of all prefix-suffix pairs to try during the
                            grid search, with a step. Only valid
                            when --basis is 'pmi' or 'auto'. Must be positive.
                            e.g.: -t 0:5:.2\n
                            """
TOP_K_PREF_HELP: Final[str] = """The range of percentages of all prefixes to try during the grid search, with a step.
                                Only valid when --basis is 'freq', 'length' or 'auto'. Must be positive.
                                e.g.: -t 0:5:.2\n
                        """
TOP_K_SUFF_HELP: Final[str] = """The range of percentages of all suffixes to try during the grid search, with a step.
                                Only valid when --basis is 'freq', 'length' or 'auto'. Must be positive.
                                e.g.: -t 0:5:.2\n
                                """

SORT_HELP: Final[str] = """Whether to sort the predictions by value, then by example length, then by example\n
                                """
EXTRA_VOCAB_HELP: Final[str] = """Extra tokens to add to the automatically computed vocabulary, taken from the input
                                    dataset. Necessary for languages like 1* where
                                  the token '0' is missing in unlabelled datasets (for language modelling)."""

LOSS_FUNCTIONS: Final[Tuple[str, ...]] = 'ppl', 'zo', 'zo1', 'zo2', 'zo3', 'zo4', 'zo5'
BASES: Final[Tuple[str, ...]] = 'auto', 'all', 'freq', 'length', 'pmi'

SPEC_TEMPLATE: Final[Template] = Template(
    '\n\tBest run:\n\n'
    '\trun\t\t= ${run}\n'
    '\ttrain. loss \t= ${t_loss}\n'
    '\tvalid. loss\t= ${v_loss}\n'
    '\ttest   loss\t= ${e_loss}\n'
    '\tternariness\t= ${ternariness}\n'
    '\tbasis selection\t= ${basis_algo}\n'
    '\ttop-k affix\t= ${topk}\n'
    '\ttop-k prefix\t= ${topk_pref}\n'
    '\ttop-k suffix\t= ${topk_suff}\n'
    '\tdim.\t\t= ${dim}\n'
)


@group()
def cli():
    pass


@cli.command(help=LEARN_HELP)
@argument('infile', type=Path(exists=True, dir_okay=False))
@argument('model', type=File(mode='wb', lazy=False), default=None, required=False)
@option('--kind', '-k', default='binary', show_default=True, type=Choice(['binary', 'polar', 'lm']), help=KIND_HELP)
@option('--dim', default=-1, show_default=True, type=int, callback=validate_special_int, help=DIM_HELP)
@option('--sv_ratio', '-sr', default=1e-1, show_default=True, type=FloatRange(0, 1), help=SV_RATIO_HELP)
@option('--splits', '-s', default='80:10:10', show_default=True, type=TripleSplit(), help=SPLITS_HELP)
@option('--loss-fn', '-lf', default='zo2', show_default=True, type=Choice(LOSS_FUNCTIONS), help=LOSS_FN_HELP)
@option('--log', default=0.1, type=FloatRange(0, 1), show_default=True, help=LOG_HELP)
@option('--stop-loss', '-sl', default=1e-6, show_default=True, type=FloatRange(0), help=STOP_LOSS_HELP)
@option('--unweighted', '-u', is_flag=True, help=UNWEIGHTED_HELP)
@option('--fail-states', '-fs', is_flag=True, help=FAIL_STATES_HELP)
@option('--quant', '-q', default=7, show_default=True, type=IntRange(0), help=QUANT_HELP)
@option('--basis', '-b', default='auto', show_default=True, type=Choice(BASES), help=BASIS_HELP)
@option('--topk', '-t', type=RangeOfSteppedPercentages(), help=TOP_K_HELP)
@option('--topk-pref', '-tp', type=RangeOfSteppedPercentages(), help=TOP_K_PREF_HELP)
@option('--topk-suff', '-ts', type=RangeOfSteppedPercentages(), help=TOP_K_SUFF_HELP)
@option('--extra-tokens', '-e', type=CSVList(2 ** 16), required=False, help=EXTRA_VOCAB_HELP)
@option('--verbose', '-v', is_flag=True, help='Verbose output')
def learn(
        infile: str,
        model: BufferedWriter | None,
        kind: Kind,
        dim: int,
        sv_ratio: float,
        splits: Tuple[int, int, int],
        loss_fn: str,
        stop_loss: float,
        unweighted: bool,
        fail_states: bool,
        quant: int,
        log: float,
        basis: str,
        topk: Tuple[float, float, float] | None,
        topk_pref: Tuple[float, float, float] | None,
        topk_suff: Tuple[float, float, float] | None,
        extra_tokens: Tuple[str, ...],
        verbose: bool,
):
    if verbose:
        config_logging(LOG, sparse=False, level=DEBUG)

    try:
        if kind == 'lm':
            if loss_fn != 'ppl':
                LOG.warning('Setting loss function to perplexity...')
                loss_fn = 'ppl'
            if kind == 'lm' and unweighted:
                raise BadParameter('Unweighted mode can only be used with the [lm] kind')

        if kind in ('binary', 'polar'):
            if loss_fn == 'ppl':
                raise BadParameter('[ppl] loss function can only be used with the [lm] kind')

        if basis == 'pmi':
            if not topk:
                raise BadParameter('--topk must be used with the [pmi] basis')

            if topk_pref or topk_suff:
                raise BadParameter('--topk-pref and --topk-suff can only be used with [freq] or [length] bases')

        if basis in ('freq', 'length'):
            if not topk_pref or not topk_suff:
                raise BadParameter(f'--topk-pref and --topk-suff must be used with [{basis}]')
            if topk:
                raise BadParameter('--topk can only be used with the [pmi] basis')

        if basis == 'all':
            if topk or topk_pref or topk_suff:
                raise BadParameter('[all] does not take any parameters')

        bases: Tuple[str, ...] = ('pmi', 'freq', 'length') if basis == 'auto' else (basis,)
        if basis == 'auto':
            topk = topk or (0, 5, 0.5) if kind == 'lm' else (0, 25, 0.5)
            topk_pref = topk_pref or (0, 5, 0.5) if kind == 'lm' else (0, 25, 0.5)
            topk_suff = topk_suff or (0, 5, 0.5) if kind == 'lm' else (0, 25, 0.5)
        topks = () if not topk else tuple(arange(topk[0], topk[1] + topk[2], topk[2]) / 100)
        topk_prefs = (
            () if not topk_pref else tuple(arange(topk_pref[0], topk_pref[1] + topk_pref[2], topk_pref[2]) / 100)
        )
        topk_suffs = (
            () if not topk_suff else tuple(arange(topk_suff[0], topk_suff[1] + topk_suff[2], topk_suff[2]) / 100)
        )

        extra_tokens = extra_tokens or ()

        if not model and sys.stdout.isatty():
            raise BadParameter('Output file must be specified when not piping to stdout')

        if isinstance(model, str) and not model.endswith('.npz'):
            raise BadParameter(f'Output file [{model}] must end with .npz')

        args: Final[Mapping[str, Any]] = dict(
            infile=infile,
            model=model,
            kind=kind,
            dim=dim,
            sv_ratio=sv_ratio,
            splits=splits,
            loss_fn=loss_fn,
            stop_loss=stop_loss,
            unweighted=unweighted,
            fail_states=fail_states,
            quant=quant,
            log=log,
            basis=basis,
            topk=topk,
            topk_prefix=topk_pref,
            topk_suffix=topk_suff,
            extra_tokens=extra_tokens,
            verbose=verbose,
        )
        info: str = Template(reduce(lambda t, item: t + f'\t{item[0]:15s} {item[-1]}\n', args.items(), '')).substitute(
            **args
        )
        LOG.info(f'Running tuning loop with args:\n\n{info}')

        t_data, v_data, e_data, x_vocab, y_vocab = _extract_data(infile, kind, splits, extra_tokens)

        wfsa, t_loss, v_loss, e_loss, spec = grid_search(
            kind=kind,
            x_vocab=x_vocab,
            y_vocab=y_vocab,
            hyper_params=dict(
                basis_algos=bases,
                factor_algos=('svd',),
                topks=topks,
                topk_prefs=topk_prefs,
                topk_suffs=topk_suffs,
                t_loss_fns=(loss_fn,),
                v_loss_fns=(loss_fn,),
                dims=(dim,),
                base_vocabs=(extra_tokens,),
                sv_ratios=(sv_ratio,),
            ),
            t_data=t_data,
            v_data=v_data,
            e_data=e_data,
            stop_loss=stop_loss,
            period=log,
            unweighted=unweighted,
            fail_states=fail_states,
            quant=quant,
        )
        LOG.info(
            SPEC_TEMPLATE.substitute(
                t_loss=f'{t_loss:.2e}',
                v_loss=f'{v_loss:.2e}',
                e_loss=f'{e_loss:.2e}' if e_loss is not None else '---',
                basis_algo=spec['basis_algo'],
                run=spec['run'],
                topk=spec.get('topk', 'N/A'),
                topk_pref=spec.get('topk_pref', 'N/A'),
                topk_suff=spec.get('topk_suff', 'N/A'),
                ternariness=f'{spec["ternariness"]:.1f}',
                dim=wfsa.initial.shape[0],
            )
        )

        metadata: Dict[str, Any] = dict(
            basis=spec['basis_algo'],
            t_loss=t_loss,
            v_loss=v_loss,
            e_loss=e_loss,
            run=spec['run'],
            kind=kind,
            vocab=x_vocab,
            unweighted=unweighted,
        )
        if unweighted:
            metadata['quant'] = quant

        if spec['basis_algo'] == 'all':
            pass
        elif spec['basis_algo'] == 'pmi':
            metadata['topk'] = float(spec['topk'])
        else:
            metadata['topk_pref'] = float(spec['topk_pref'])
            metadata['topk_suff'] = float(spec['topk_suff'])

        out = get_binary_stream('stdout') if not model else model
        LOG.info(f'Saving WFSA to [{out.name}]...')
        save_wfsa(wfsa, out, metadata)

    except Exception as e:
        LOG.exception('Error during hyper-parameter search', exc_info=e)
        sys.exit(1)


# ----------------------------------------------------------------------------------------------------------------------


PREDICT_HELP: Final[str] = """Predict values for unlabelled data with a (Weighted) Finite State Automaton (W/FSA)\n
                            For kind 'lm' the predicted values are base-10 negative log likelihoods. For kind 'binary'
                            and 'polar', they are  0/1 and -1/1 respectively.\n
                            positional arguments:\n
                              INFILE\tpath to the input data file. It must contain examples as sequences of
                            whitespace-separated tokens.\n
                              MODEL\tpath to the model file. It must be a numpy compressed archive with keys
                            (d = number of states, v = vocabulary size): \n
                                initial: a 1D numpy array with initial weights, of shape (d,). It must be a one-hot vector\n
                                transitions: a 3D numpy array with transition weights of shape (v, d, d).\n
                                final: a 1D numpy array with final weights of shape (d,).\n
                                meta: a dict containing a key named 'kind' with value in {'binary', 'lm', 'polar'}\n
                            """
PRED_OUT_HELP: Final[str] = """A comma-separated list of output formats for the prediction:\n
                                  cons: outputs predictions to console with format <EXAMPLE><TAB><PREDICTION>\n
                                  <FILE>.<EXT>: writes prediction to the given file, with format
                                <EXAMPLE><TAB><PREDICTION>\n
                                e.g.: -o cons,wfsa.txt\n
                                \t\n
                                """


@cli.command(help=PREDICT_HELP)
@argument('infile', type=Path(exists=True, dir_okay=False))
@argument('model', type=Path(exists=True, dir_okay=False))
@option('--output', '-o', default='cons', required=False, type=CSVList(2), help=PRED_OUT_HELP)
@option('--sort', '-s', default=False, is_flag=True, help=SORT_HELP)
@option('--verbose', '-v', is_flag=True, help='Verbose output')
def predict(infile: str, model: str, output: Tuple[str, ...], sort: bool, verbose: bool):
    if verbose:
        config_logging(LOG, sparse=False, level=DEBUG)

    wfsa, metadata = load_wfsa(model)

    data: Iterable[Tuple[str, ...]] = tuple(load_unlabelled_data(infile))
    x_vocab: Tuple[str, ...] = tuple(sorted(metadata['vocab']))
    dataset: Iterable[NDArray] = tuple(to_unlabelled_dataset(data, one_hot_coder_from(x_vocab)))

    format_: str
    predict_fn: Fn
    match metadata['kind']:
        case 'lm':
            predict_fn = lm_predict
            format_ = '.3'

        case 'binary' | 'polar':
            predict_fn = class_predict
            format_ = '4.2f'
        case _:
            raise ValueError(f'Unknown WFSA kind [{metadata["kind"]}]')

    preds: Iterable[Tuple[Tuple[str, ...], float]] = zip(
        (tuple(datum) for batch in make_batches(data) for datum in batch), predict_fn(wfsa, dataset)
    )

    if sort:
        preds = sorted(preds, key=lambda data_pred: (-data_pred[-1], len(data_pred[0]), data_pred[0]))

    out_str: str = reduce(lambda out, pred: out + f'{" ".join(pred[0]):24s}\t{pred[-1]:{format_}}\n', preds, '')

    if 'cons' in output:
        print(out_str)

    if '.' in ''.join(output):
        with open(output[0] if '.' in output[0] else output[-1], 'w', encoding='utf8') as f:
            f.write(out_str)


# ----------------------------------------------------------------------------------------------------------------------


SHOW_HELP: Final[str] = """Show WFSA's details:\n
                             MODEL\tpath to the model file. It must be a numpy compressed archive either from stdin or
                             ending in ".npz"
                             with keys (d = number of states, v = vocabulary size): \n
                                initial: a 1D numpy array with initial weights, of shape (d,). It must be a one-hot vector\n
                                transitions: a 3D numpy array with transition weights of shape (v, d, d).\n
                                final: a 1D numpy array with final weights of shape (d,).\n
                                meta: a dict containing:\n
                                   a key named 'kind' with value in {'binary', 'lm', 'polar'}\n
                                   a key named 'unweighted' with value in {False, True}\n
                                   a key named 'vocab' with a collection of strings as values. It must be of length v.\n
                            """

SHOW_OUT_HELP: Final[str] = """A comma-separated list of output formats for the WFSA:\n
                                  cons: outputs weights to console in tabular form\n
                                  <FILE>.png: saves state-transition diagram to a png file (opens in browser)\n
                                  <FILE>.pdf: saves state-transition diagram to a pdf file (opens in browser)\n
                                e.g.: -o cons,wfsa.png,wfsa.pdf\n
                                The state-transition diagrams show a transition's input symbols and weight near the
                                edges and a state's name and final weight inside the vertices. The initial weight
                                is not shown because it's always 1 for the initial state and 0 for the rest as the
                                (W)FSAs are guaranteed have a single start state. For unweighted FSAs, only the state 
                                name and the transition input symbol is shown. The initial state is displayed with a
                                red bold circle. The final states are shown with a double circle.\n
                                \t\n
                                """


@cli.command(help=SHOW_HELP)
@argument('model', type=File(mode='rb'), default=None, required=False)
@option('--output', '-o', default='cons', type=CSVList(3), help=SHOW_OUT_HELP)
@option('--verbose', '-v', is_flag=True, help='Verbose output')
def show(model: str | BufferedReader | None, output: Tuple[str, ...], verbose: bool):
    if verbose:
        config_logging(LOG, sparse=False, level=DEBUG)

    if not model and sys.stdin.isatty():
        raise BadParameter('Model file must be specified when not piping to stdin')

    if isinstance(model, str) and not model.endswith('.npz'):
        raise BadParameter(f'Model file [{model}] must end with .npz')

    wfsa, metadata = load_wfsa(BytesIO(get_binary_stream('stdin').read()) if not model else model)

    if 'kind' not in metadata or 'unweighted' not in metadata or 'vocab' not in metadata:
        raise ValueError('Invalid model file. Must contain keys "kind", "unweighted", and "vocab"')

    kind: str = metadata['kind']
    unweighted: bool = metadata['unweighted']
    vocab: Tuple[str, ...] = tuple(sorted(metadata['vocab']))
    fracs: int = 4 if kind == 'lm' else 2

    if len(vocab) != len(wfsa.trans_mats):
        raise ValueError('Invalid model file. Vocabulary size does not match the number of transition matrices')

    for path in output:
        ext: str = path if path == 'cons' else '.' + path.split('.')[-1]

        match ext:
            case 'cons':
                print(
                    f'\n{kind.upper()} FSA Weights\t(∘ = 0   ■ = 1 )\n\n'
                    if unweighted
                    else f'\n{kind.upper()} WFSA Weights\t( □ = -1  ∘ = 0   ■ = 1 )\n\n'
                )

                nout(wfsa.initial, fracs=fracs, row_hs=['α'], indent=2)
                print()

                nout(wfsa.final, fracs=fracs, row_hs=['ω'], indent=2)
                print('\n')

                nout(stack(wfsa.trans_mats), fracs=fracs, tube_hs=tuple(f'A{num_to_super(s)}' for s in vocab), indent=2)

                print('PERIODICITY OF TRANSITION MATRICES\n')
                zeroes: NDArray = zeros_like(wfsa.trans_mats[0])
                identity: NDArray = eye(wfsa.trans_mats[0].shape[0])
                for s, m in zip(vocab, wfsa.trans_mats):
                    kind: str = 'Non-periodic'

                    m_power: NDArray = identity
                    for power in range(1, 11):
                        m_power = m @ m_power

                        if allclose(m_power, zeroes, rtol=0, atol=1e-5):
                            kind = (
                                f'Null: M = 0'
                                if power == 1
                                else f'Nilpotent index {power}: M{num_to_super(power)} = 0, '
                                     f'M{num_to_super(power - 1)} ≠ 0'
                            )
                            break

                        if allclose(m_power, identity, rtol=0, atol=1e-5):
                            match power:
                                case 1:
                                    kind = 'Identity:    M = I'
                                case 2:
                                    kind = f'Involutory: M{num_to_super(2)} = I, M ≠ I'
                                case _:
                                    kind = f'{power}{"rd" if power == 3 else "th"} power of Identity\t'
                                    f': M{num_to_super(power)} = I, M{num_to_super(power - 1)} ≠ I'
                            break

                        if power >= 2 and allclose(m_power, m, rtol=0, atol=1e-5):
                            match power:
                                case 2:
                                    kind = f'Idempotent: M{num_to_super(2)} = M'
                                case 3:
                                    kind = f'Tripotent: M{num_to_super(3)} = M, M{num_to_super(2)} ≠ M'
                                case _:
                                    kind = f'{power}-potent: M{num_to_super(power)} = M, M{num_to_super(power - 1)} ≠ M'

                            break

                        if allclose(m_power, matrix_power(m, power - 1), rtol=0, atol=1e-5):
                            kind = (
                                f'Idempotent index {power}: M{num_to_super(power)} = M{num_to_super(power - 1)}, '
                                f'M{num_to_super(power - 1)} ≠ M{num_to_super(power - 2)}'
                            )
                            break

                    print(f'A{num_to_super(s)}  {kind}')

                print('\n')

            case '.png' | '.pdf':
                digraph = wfsa_to_graphviz(wfsa, symbol_names=vocab, unweighted=unweighted)

                digraph.render(path.removesuffix(ext), format=ext[1:], cleanup=True)
                webbrowser.open(f'file://{abspath(path)}')

            case _:
                raise BadParameter(f'Unsupported output format [{path}]')


# --------------------------------------------------- DELEGATE FUNCTIONS -----------------------------------------------

def _extract_data(infile: str, kind: str, splits: Tuple[int, int, int], extra_tokens: Tuple[str, ...]):
    data: Tuple[Tuple[Tuple[str, ...], str] | Tuple[str, ...], ...]
    x_vocab: set[str] = set(extra_tokens)
    y_vocab: set[str] = set()

    if kind == 'lm':
        data = tuple(load_unlabelled_data(infile))

        for x in data:
            x_vocab.update(x)

    else:
        data = tuple(load_labelled_data(infile))
        for x, y in data:
            x_vocab.update(x)
            y_vocab.add(y)

    if not splits[1]:
        LOG.warning('No validation split specified, using training data instead...')

    t, v, e = (int(len(data) * split / 100) for split in splits)
    # if no validation split is provided, it uses training data instead
    # acceptors' training set is added to the validation set to improve performance, especially with small datasets
    t_data = data[:t] if (v or e) else data  # all data is used for training when no validation or test data
    v_data = data[(t if kind == 'lm' and v else 0): t + v]  # if validation split is 0, it uses training data instead
    e_data = data[t + v:] if e else ()  # ensures 0 split is honoured

    return t_data, v_data, e_data, tuple(sorted(x_vocab)), tuple(sorted(y_vocab))


if __name__ == '__main__':
    cli()
