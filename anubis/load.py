import numpy as np
import warnings
import dill
from anubis.exceptions import ANUBISException
from pathlib import Path

def save_density(draws, folder = '.', name = 'density'):
    """
    Exports a list of figaro.mixture instances to file

    Arguments:
        :list draws:         list of mixtures to be saved
        :str or Path folder: folder in which the output file will be saved
        :str name:           name to be given to output file
    """
    with open(Path(folder, name+'.pkl'), 'wb') as f:
        dill.dump(draws, f)
        
def load_density(path):
    """
    Loads a list of anubis.het_mixture instances from path.

    Arguments:
        :str or Path path: path with draws (file or folder)

    Returns
        :list: anubis.het_mixture object instances
    """
    path = Path(path)
    if path.is_file():
        return _load_density_file(path)
    else:
        return [_load_density_file(file) for file in path.glob('*.[jp][sk][ol]*')]

def _load_density_file(file):
    """
    Loads a list of anubis.het_mixture instances from file.

    Arguments:
        :str or Path file: file with draws

    Returns
        :list: anubis.het_mixture object instances
    """
    file = Path(file)
    try:
        with open(file, 'rb') as f:
            draws = dill.load(f)
        return draws
    except FileNotFoundError:
            raise ANUBISException("{0} not found. Please provide it or re-run the inference.".format(file.name))
