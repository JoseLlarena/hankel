"""
Functions to save and load WFSAs
"""

from io import BufferedReader, BufferedWriter, BytesIO
from logging import Logger, getLogger
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, Final, Tuple, Dict
from numpy.lib.npyio import NpzFile
from numpy import (asarray, float32, savez_compressed, load)
from pydantic import validate_call
from hankel.models import WFSA
from hankel import PYDANTIC_CONFIG

LOG: Final[Logger] = getLogger(__package__)


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def save_wfsa(wfsa: WFSA, path: str | PathLike | BinaryIO |BufferedWriter, metadata: Dict[str, Any] | None = None):
    """
    Saves given WFSA parameters with the given metadata. The saved file is a numpy compressed archive.

    Args:
        wfsa (WFSA): The WFSA whose parameters must be saved.
        path (str | PathLike | BinaryIO | BufferedWriter): The path to the target to save the WFSA to. If path-like, it
            should end in .npz 
        metadata (Dict[str, Any] | None, optional): A set of key-value pairs to save along the parameters. 
            Defaults to None.
    """
    if isinstance(path, (str, Path)) and not str(path).endswith('.npz'):
        LOG.warning(f'The file [{path}] does not end in .npz. It will be saved with an .npz extension.')

    savez_compressed(path,
                     initial=wfsa.initial.astype(float32),
                     transitions=wfsa.transitions.astype(float32),
                     final=wfsa.final.astype(float32),
                     metadata=asarray(metadata or {}))


@validate_call(config=PYDANTIC_CONFIG, validate_return=True)
def load_wfsa(path: str | PathLike | BufferedReader | BytesIO) -> Tuple[WFSA, Dict[str, Any]]:
    """
    Loads a WFSA and its metadata from the given file path or buffer.

    Note: This function assumes the file contains keys named 'initial', 'transitions', 'final', and 'metadata'.

    Args:
        path (str | PathLike | BufferedReader | BinaryIO): file path or buffer to load from, if it is a file path
            it should end with ".npz"

    Returns:
        Tuple[WFSA, Dict[str, Any]]: A WFSA and its metadata
    """
    if isinstance(path, (str, Path)) and not str(path).endswith('.npz'):
        LOG.warning(f'The file [{path}] does not end in .npz.')

    data: NpzFile = load(path, allow_pickle=True)
    return WFSA(data['initial'], data['transitions'], data['final']), data['metadata'].item()
