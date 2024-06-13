import numpy as np
from anubis.exceptions import ANUBISException

def get_samples(draws):
    """
    Extract parameter samples from a list of draws.
    
    Arguments:
        :iterable draws: instances of het_mixture class
    
    Returns:
        :np.ndarray: samples
    """
    n_shared_pars = [-d.n_shared_pars if d.n_shared_pars > 0 else len(d.models[d.augment].pars) for d in draws]
    ll = [[list(d.models[i+d.augment].pars[:n]) for i in range(len(d.models)-d.augment)] + [list(d.models[d.augment].pars[n:])]  for d, n in zip(draws, n_shared_pars)]
    # Funny way of doing the right thing. I may rewrite it in the future, but it works for now...
    return np.array([s for d in ll for m in d for s in m]).reshape(len(draws), -1)

def get_weights(draws):
    """
    Extract (intrinsic) weight samples from a list of draws.
    
    Arguments:
        :iterable draws: instances of het_mixture class
    
    Returns:
        :np.ndarray: samples
    """
    return np.array([d.intrinsic_weights for d in draws])

def get_samples_and_weights(draws):
    """
    Extract parameter and (intrinsic) weight samples from a list of draws.
    
    Arguments:
        :iterable draws: instances of het_mixture class
    
    Returns:
        :np.ndarray: samples
    """
    return np.block([get_samples(draws), get_weights(draws)])

def get_labels(draws, kind, pars_labels = None, par_models_labels = None):
    """
    Returns a list of labels for parameters and/or weights.
    
    Arguments:
        :iterable draws:               instances of het_mixture class
        :str kind:                     whether to produce labels for plots ('pars', 'weights', 'joint') or for output ('txt')
        :list-of-str pars_labels:      labels for parameters
        :list-of-str par_model_labels: labels for models (for weights)
    
    Return:
        :list-of-str: labels
    """
    # Number of parameters and models
    if draws[0].n_shared_pars > 0:
        shared_pars_idx = -draws[0].n_shared_pars
    else:
        shared_pars_idx = None
    n_parameters  = int(np.sum([len(m.pars[:shared_pars_idx]) for m in draws[0].models[draws[0].augment:]]) + draws[0].n_shared_pars)
    n_par_models  = len(draws[0].models)-draws[0].augment
    # Parameter labels
    if pars_labels is None or not (len(pars_labels) == n_parameters):
        parameter_labels = ['$\\theta_{0}$'.format(i+1) for i in range(n_parameters)]
    else:
        parameter_labels = ['${0}$'.format(l) for l in pars_labels]
    # Model lalbels
    if par_models_labels is None or not (len(par_models_labels) == n_par_models):
        weights_labels = ['$w_{'+'{0}'.format(i+1)+'}$' for i in range(n_par_models)]
    else:
        weights_labels = ['$w_\\mathrm{'+'{0}'.format(l)+'}$' for l in par_models_labels]
    if np.array([d.augment for d in draws]).all():
        weights_labels = ['$w_\\mathrm{np}$'] + weights_labels
    # Decide what to return
    if kind == 'pars':
        return parameter_labels
    elif kind == 'weights':
        return weights_labels
    elif kind == 'joint':
        return parameter_labels + weights_labels
    elif kind == 'txt':
        # Remove LaTeX characters from string
        labels = [s.translate(str.maketrans({st:'' for st in '$\{}'})) for s in parameter_labels + weights_labels]
        labels = [s.replace('mathrm', '') for s in labels]
        return labels
    else:
        raise ANUBISException("Kind not supported. Please pick from ['pars', 'weights', 'joint', 'txt']")
