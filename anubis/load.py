import numpy as np
import dill
from figaro.load import load_data as load_data_figaro, load_density as load_density_figaro
from anubis.exceptions import ANUBISException
from pathlib import Path

def save_density(draws, folder = '.', name = 'density'):
    """
    Exports a list of anubis.het_mixture instances to file

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
    path = Path(path).resolve()
    if path.is_file():
        if not path.parts[-1].split('.')[-1] == 'pkl':
            path = Path(path, '.pkl')
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

def load_data(path_samples, path_mixtures, *args, **kwargs):
    """
    Loads the data from .txt files (for simulations) or .h5/.hdf5/.dat files (posteriors from GWTC-x) along with their DPGMM reconstruction (must be available in advance).
    Default cosmological parameters from Planck Collaboration (2021) in a flat Universe (https://www.aanda.org/articles/aa/pdf/2020/09/aa33910-18.pdf)
    Not all GW parameters are implemented: run figaro.load.available_gw_pars() for a list of available parameters.
    
    Arguments:
        str or Path path_samples:  folder with samples files
        str or Path path_mixtures: folder with mixtures files
        bool seed:                 fixes the seed to a default value (1) for reproducibility
        list-of-str par:           list with parameter(s) to extract from GW posteriors
        int n_samples:             number of samples for (random) downsampling. Default -1: all samples
        double h:                  Hubble constant H0/100 [km/(s*Mpc)]
        double om:                 matter density parameter
        double ol:                 cosmological constant density parameter
        str waveform:              waveform family to be used ('combined', 'seob', 'imr')
        double snr_threhsold:      SNR threshold for event filtering. For injection analysis only.
        double far_threshold:      FAR threshold for event filtering. For injection analysis only.
        bool verbose:              show progress bar

    Returns:
        iterable:   iterable storing samples and reconstructions
        np.ndarray: names
    """
    samples, names = load_data_figaro(path_samples, *args, **kwargs)
    mixtures       = [load_density_figaro(Path(path_mixtures, 'draws_'+ev+'.pkl')) for ev in names]
    return [[ss, mm] for ss, mm in zip(samples, mixtures)], names
