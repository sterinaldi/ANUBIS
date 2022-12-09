import numpy as np

def get_samples(draws)
    """
    Extract parameter samples from a list of draws.
    
    Arguments:
        :iterable draws: instances of het_mixture class
    
    Returns:
        :np.ndarray: samples
    """
    return np.array([np.array([d.models[i+1].pars for i in range(len(d.par_models))]).flatten() for d in draws])

def get_weights(draws)
    """
    Extract weight samples from a list of draws.
    
    Arguments:
        :iterable draws: instances of het_mixture class
    
    Returns:
        :np.ndarray: samples
    """
    return np.array([d.weights for d in draws])


def get_samples_and_weights(draws)
    """
    Extract parameter and weight samples from a list of draws.
    
    Arguments:
        :iterable draws: instances of het_mixture class
    
    Returns:
        :np.ndarray: samples
    """
    return np.block([get_samples(draws), get_weights(draws)])
