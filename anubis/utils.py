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

def get_labels(draws, kind, models = None):
    """
    Returns a list of labels for parameters and/or weights.
    
    Arguments:
        :iterable draws:               instances of het_mixture class
        :str kind:                     whether to produce labels for plots ('pars', 'weights', 'joint') or for output ('save')
    
    Return:
        :list-of-str: labels
    """
    if models is not None:
        # Check that all models have the correct attributes
        for model in models:
            if 'par_names' not in model.keys() or 'parameters' in model.keys():
                model['par_names'] = []
            if 'par_labels' not in model.keys() or 'parameters' in model.keys():
                model['par_labels'] = []
        all_pars_names  = [lab for model in models for lab in model['par_names']]
        all_pars_labels = [lab for model in models for lab in model['par_labels']]
        # Fancy way of getting all unique instances while preserving order
        unique_names  = list(dict.fromkeys(all_pars_names).keys())
        unique_labels = list(dict.fromkeys(all_pars_labels).keys())
        # Identify items appearing once
        d_names = {}
        for i in all_pars_names: d_names[i] = i in d_names
        # List of names and labels
        pars_names        = [k for k in all_pars_names if not d_names[k]] + [k for k in unique_names if d_names[k]]
        pars_labels       = [x for x, k in zip(all_pars_labels, all_pars_names) if not d_names[k]] + [x for x, k in zip(unique_labels, unique_names) if d_names[k]]
        par_models_labels = [model['name'] for model in models]
    # Number of parameters and models
    if draws[0].n_shared_pars > 0:
        shared_pars_idx = -draws[0].n_shared_pars
    else:
        shared_pars_idx = None
    n_parameters  = int(np.sum([len(m.pars[:shared_pars_idx]) for m in draws[0].models[draws[0].augment:]]) + draws[0].n_shared_pars)
    n_par_models  = len(draws[0].models)-draws[0].augment
    # Parameter labels
    if models is None:
        parameter_labels = ['$\\theta_{0}$'.format(i+1) for i in range(n_parameters)]
    else:
        parameter_labels = ['${0}$'.format(l) for l in pars_labels]
    # Model labels
    if models is None:
        weights_labels = ['$w_{'+'{0}'.format(i+1)+'}$' for i in range(n_par_models)]
    else:
        weights_labels = ['$w_\\mathrm{'+'{0}'.format(l)+'}$' for l in par_models_labels]
    if draws[0].augment:
        weights_labels = ['$w_\\mathrm{np}$'] + weights_labels
    # Decide what to return
    if kind == 'pars':
        return parameter_labels
    elif kind == 'weights':
        return weights_labels
    elif kind == 'joint':
        return parameter_labels + weights_labels
    elif kind == 'save':
        # Remove LaTeX characters from string
        labels = pars_names + [s.translate(str.maketrans({st:'' for st in '$\{}'})) for s in weights_labels]
        labels = [s.replace('mathrm', '') for s in labels]
        return labels
    else:
        raise ANUBISException("Kind not supported. Please pick from ['pars', 'weights', 'joint', 'save']")
